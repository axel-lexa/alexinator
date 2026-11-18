"""Moteur de jeu : niveaux, partie en cours et chaîne de défenses.

Un niveau = un fichier YAML dans levels/ (prompt du gardien + liste des défenses actives).
Une partie = un objet GameSession : un joueur, un niveau, un film secret, un historique.
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import defenses
from .config import LEVELS_DIR
from .films import Film
from .journal import Journal
from .validation import GuessResult, check_answer

KNOWN_DEFENSES = {"input_detector", "output_filter", "censor"}

# Messages montrés au joueur quand une défense bloque (ils ne révèlent rien du secret).
REFUSALS = {
    "input_detector": "Tentative de manipulation détectée. Pose-moi plutôt une vraie question sur le film.",
    "output_filter": "J'allais en dire trop... Ma réponse a été interceptée. Essaie autrement.",
    "censor": "Mon censeur trouve cette réponse trop révélatrice : elle a été supprimée.",
    "limite": "Tu as utilisé tous tes messages pour ce niveau. Propose une réponse ou recommence.",
}

# Niveau 7 : le gardien appelle son outil en écrivant seul sur une ligne « VERIFIER: <titre> ».
# (Protocole en texte brut : il marche avec n'importe quel modèle, même sans « tool calling » natif.)
TOOL_RE = re.compile(r"^[ \t>*-]*VERIFIER\s*:\s*(.+?)\s*$", flags=re.I | re.M)


@dataclass
class Level:
    id: int
    name: str
    description: str
    guardian_prompt: str  # contient {fiche}, remplacé par la fiche du film
    defenses: list[str] = field(default_factory=list)
    filter_entities: bool = True  # le filtre bloque-t-il aussi acteurs, réalisateur, personnages ?
    max_messages: int = 15
    temperature: float = 0.7
    response_mode: str = "libre"  # "libre" ou "oui_non" (niveau 6 : réponses contraintes)
    fiche_mode: str = "complete"  # "complete" ou "anonyme" (niveau 7 : ni titre ni noms)
    tool_calls: int = 0  # niveau 7 : nombre d'appels à l'outil VERIFIER


def load_levels(directory: Path = LEVELS_DIR) -> list[Level]:
    levels = []
    for path in sorted(Path(directory).glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        level = Level(**data)
        unknown = set(level.defenses) - KNOWN_DEFENSES
        if unknown:
            raise ValueError(f"{path.name} : défense(s) inconnue(s) {sorted(unknown)}")
        levels.append(level)
    return sorted(levels, key=lambda lv: lv.id)


@dataclass
class TurnResult:
    reply: str  # ce que le joueur voit
    blocked_by: str | None  # nom de la défense qui a bloqué, sinon None
    raw_reply: str | None  # réponse du gardien avant les défenses (None s'il n'a pas été appelé)


class GameSession:
    def __init__(
        self,
        level: Level,
        film: Film,
        llm,
        aux_llm=None,
        journal: Journal | None = None,
        player: str = "anonyme",
        judge=None,
    ):
        self.level = level
        self.film = film
        self.llm = llm  # LLM du gardien
        self.aux_llm = aux_llm or llm  # LLM des défenses (détecteur, censeur)
        self.judge = judge  # LLM juge pour la validation (optionnel)
        self.journal = journal
        self.player = player
        self.session_id = uuid.uuid4().hex
        self.history: list[dict[str, str]] = []  # échanges tels que le joueur les a vus
        self.messages_used = 0
        self.solved = False
        self.tool_calls_left = level.tool_calls
        self.tool_notes: list[str] = []  # résultats des vérifications déjà faites (niveau 7)

    @property
    def messages_left(self) -> int:
        return max(0, self.level.max_messages - self.messages_used)

    def system_prompt(self) -> str:
        fiche = self.film.fiche(anonyme=self.level.fiche_mode == "anonyme")
        prompt = self.level.guardian_prompt.replace("{fiche}", fiche).replace(
            "{verifications}", str(self.tool_calls_left)
        )
        if self.tool_notes:
            prompt += "\nVérifications déjà faites avec ton outil :\n" + "\n".join(self.tool_notes)
        return prompt

    def _use_tool(self, messages: list[dict[str, str]], raw_reply: str) -> str:
        """Niveau 7 : exécute l'appel « VERIFIER: <titre> » du gardien, puis le laisse répondre."""
        call = TOOL_RE.search(raw_reply)
        if call:
            proposal = call.group(1)
            if self.tool_calls_left > 0:
                self.tool_calls_left -= 1
                verdict = "VRAI" if check_answer(proposal, self.film).correct else "FAUX"
                self.tool_notes.append(f"- « {proposal} » : {verdict}")
                tool_result = (
                    f"[RÉSULTAT DE L'OUTIL] « {proposal} » : {verdict}. "
                    f"Vérifications restantes : {self.tool_calls_left}."
                )
            else:
                tool_result = "[RÉSULTAT DE L'OUTIL] Refusé : tu n'as plus de vérifications."
            followup = messages + [
                {"role": "assistant", "content": raw_reply},
                {
                    "role": "user",
                    "content": tool_result + " Réponds maintenant au joueur, sans écrire de titre.",
                },
            ]
            raw_reply = self.llm.chat(followup, temperature=self.level.temperature)
        # Le joueur ne voit jamais les lignes d'appel à l'outil.
        return TOOL_RE.sub("", raw_reply).strip() or "..."

    # ------------------------------------------------------------------
    # Un tour de jeu : le message traverse la chaîne de défenses du niveau
    # ------------------------------------------------------------------
    def send(self, user_message: str) -> TurnResult:
        start = time.perf_counter()
        blocked_by, detail, raw_reply = None, "", None

        if self.messages_left == 0:
            blocked_by = "limite"
        else:
            self.messages_used += 1

            # Étape 1 - défense en ENTRÉE : le message est-il une manipulation ?
            if "input_detector" in self.level.defenses:
                verdict = defenses.input_detector(user_message, self.aux_llm)
                if verdict.blocked:
                    blocked_by, detail = "input_detector", verdict.reason

            # Étape 2 - le gardien répond (prompt système + historique + nouveau message)
            if blocked_by is None:
                messages = (
                    [{"role": "system", "content": self.system_prompt()}]
                    + self.history
                    + [{"role": "user", "content": user_message}]
                )
                raw_reply = self.llm.chat(messages, temperature=self.level.temperature)
                if self.level.tool_calls:
                    raw_reply = self._use_tool(messages, raw_reply)
                if self.level.response_mode == "oui_non":
                    raw_reply = defenses.force_oui_non(raw_reply)

                # Étape 3 - défenses en SORTIE : la réponse laisse-t-elle fuiter le secret ?
                if "output_filter" in self.level.defenses:
                    verdict = defenses.output_filter(raw_reply, self.film, self.level.filter_entities)
                    if verdict.blocked:
                        blocked_by, detail = "output_filter", verdict.reason
                if blocked_by is None and "censor" in self.level.defenses:
                    verdict = defenses.censor(raw_reply, self.film, self.aux_llm)
                    if verdict.blocked:
                        blocked_by, detail = "censor", verdict.reason

        reply = REFUSALS[blocked_by] if blocked_by else raw_reply

        # L'historique contient ce que le joueur a VU : une réponse bloquée n'y figure pas,
        # sinon le gardien pourrait la citer plus tard et contourner la défense.
        if blocked_by != "limite":
            self.history += [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": reply},
            ]

        if self.journal:
            self.journal.log_turn(
                session_id=self.session_id,
                joueur=self.player,
                niveau=self.level.id,
                film_id=self.film.id,
                modele=getattr(self.llm, "model", "?"),
                message_joueur=user_message,
                reponse_brute=raw_reply,
                reponse_affichee=reply,
                bloque_par=blocked_by,
                detail=detail,
                latence_ms=int((time.perf_counter() - start) * 1000),
            )
        return TurnResult(reply=reply, blocked_by=blocked_by, raw_reply=raw_reply)

    # ------------------------------------------------------------------
    # Le joueur propose une réponse dans le champ dédié
    # ------------------------------------------------------------------
    def guess(self, answer: str) -> GuessResult:
        result = check_answer(answer, self.film, judge=self.judge)
        if result.correct:
            self.solved = True
        if self.journal:
            self.journal.log_guess(
                session_id=self.session_id,
                joueur=self.player,
                niveau=self.level.id,
                film_id=self.film.id,
                proposition=answer,
                correcte=result.correct,
                methode=result.method,
                messages_utilises=self.messages_used,
            )
        return result
