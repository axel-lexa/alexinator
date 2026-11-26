"""Logique d'affichage de l'interface web, sans Streamlit (donc testable avec pytest).

Les pages de vues/ s'en servent pour construire les écrans : noms et icônes des étapes de la chaîne
de défenses, badges d'un niveau, issue d'un tour de jeu, questions suggérées, contrôle du pseudo.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .engine import Level, TurnResult


@dataclass(frozen=True)
class Badge:
    label: str
    icon: str
    color: str  # couleur de st.badge : red, orange, yellow, blue, green, violet, gray, primary


@dataclass(frozen=True)
class Etape:
    nom: str  # court, pour le résumé sous une réponse (« Filtre »)
    nom_long: str  # pour les badges et la page « Comment ça marche »
    icone: str
    en_cours: str  # affiché pendant que l'étape tourne
    explication: str


# Étapes que traverse un message, dans l'ordre de GameSession.send. Les clés des défenses sont
# celles des fichiers YAML (engine.KNOWN_DEFENSES) ; « guardian » et « tool » sont les appels au gardien.
ETAPES = {
    "input_detector": Etape(
        "Détecteur",
        "Détecteur d'attaques",
        "🕵️",
        "Le détecteur examine ton message...",
        "Un second LLM lit ton message avant le gardien. S'il y voit une manipulation, "
        "le gardien ne le reçoit jamais.",
    ),
    "guardian": Etape(
        "Gardien",
        "Gardien",
        "🧙",
        "Le gardien réfléchit...",
        "Le LLM principal connaît le film (sauf au niveau 7) et répond à partir de sa fiche, "
        "en suivant les consignes du niveau.",
    ),
    "tool": Etape(
        "Outil",
        "Outil de vérification",
        "🔍",
        "Le gardien vérifie une hypothèse...",
        "Le gardien ne connaît pas le titre : il peut tester un titre auprès du jeu, "
        "qui lui répond VRAI ou FAUX, un nombre limité de fois.",
    ),
    "output_filter": Etape(
        "Filtre",
        "Filtre de sortie",
        "🧹",
        "Le filtre relit la réponse...",
        "Du code Python, sans IA, cherche dans la réponse le titre, ses variantes et les noms de la "
        "fiche, même épelés ou espacés.",
    ),
    "censor": Etape(
        "Censeur",
        "Censeur",
        "✂️",
        "Le censeur relit la réponse...",
        "Un second LLM relit chaque réponse et la supprime si elle rend le film trop facile à deviner.",
    ),
}
ORDRE = tuple(ETAPES)


def chaine(level: Level) -> list[str]:
    """Étapes actives au niveau `level`, dans l'ordre où un message les traverse."""
    actives = set(level.defenses) | {"guardian"}
    if level.tool_calls:
        actives.add("tool")
    return [key for key in ORDRE if key in actives]


def badges_niveau(level: Level) -> list[Badge]:
    """Ce qui protège le secret à ce niveau, en badges."""
    badges = []
    if level.fiche_mode == "anonyme":
        badges.append(Badge("Ne connaît pas le titre", "🙈", "violet"))
    if level.tool_calls:
        s = "s" if level.tool_calls > 1 else ""
        badges.append(Badge(f"{level.tool_calls} vérification{s}", ETAPES["tool"].icone, "violet"))
    if level.response_mode == "oui_non":
        badges.append(Badge("Oui / non / je ne sais pas", "🤐", "violet"))
    for key in ORDRE:
        if key in level.defenses:
            badges.append(Badge(ETAPES[key].nom_long, ETAPES[key].icone, "blue"))
    return badges or [Badge("Consignes seulement", "📜", "gray")]


def issue(result: TurnResult) -> str:
    """Ce qui s'est passé pendant un tour : victoire, bloque, brule, chauffe, rate, limite ou reponse."""
    if result.guess is not None:
        if result.guess.correct:
            return "victoire"
        return {"partielle": "brule", "proche": "chauffe"}.get(result.guess.method, "rate")
    if result.blocked_by == "limite":
        return "limite"
    if result.blocked_by:
        return "bloque"
    return "reponse"


def badge_issue(issue_: str, blocked_by: str | None = None) -> Badge | None:
    """Badge affiché sous la réponse du gardien (aucun pour une réponse normale)."""
    if issue_ == "bloque" and blocked_by in ETAPES:
        etape = ETAPES[blocked_by]
        return Badge(f"Bloqué · {etape.nom_long}", "🛡️", "red")
    return {
        "brule": Badge("Tu brûles", "🔥", "orange"),
        "chauffe": Badge("Tu chauffes", "♨️", "yellow"),
        "rate": Badge("Mauvais titre", "❄️", "blue"),
        "limite": Badge("Plus de messages", "⌛", "gray"),
    }.get(issue_)


# Phrase du magicien (visuels.PHRASES) selon l'issue du tour.
MOMENT_SCENE = {
    "reponse": "attente",
    "bloque": "bloque",
    "rate": "rate",
    "brule": "brule",
    "chauffe": "chauffe",
}


def resume_etapes(etapes: list[str], blocked_by: str | None) -> str:
    """Résumé d'un tour : « 🕵️ Détecteur ✓ → 🧙 Gardien ✓ → 🧹 Filtre ✗ »."""
    parts = []
    for key in etapes:
        etape = ETAPES[key]
        parts.append(f"{etape.icone} {etape.nom} {'✗' if key == blocked_by else '✓'}")
    return " → ".join(parts)


# Questions proposées en début de partie, selon le mode de réponse du niveau.
SUGGESTIONS = {
    "libre": [
        "C'est quel genre de film ?",
        "Il est sorti à quelle époque ?",
        "Quelle est l'ambiance du film ?",
    ],
    "oui_non": [
        "Est-ce un film français ?",
        "Est-il sorti avant l'an 2000 ?",
        "Est-ce une comédie ?",
    ],
}

PSEUDO_MAX = 30


def verifier_pseudo(pseudo: str) -> str | None:
    """Renvoie un message d'erreur si le pseudo est refusé, sinon None."""
    pseudo = pseudo.strip()
    if not pseudo:
        return "Choisis un pseudo pour commencer."
    if len(pseudo) > PSEUDO_MAX:
        return f"{PSEUDO_MAX} caractères au maximum."
    if pseudo.lower().startswith("auto:"):
        return "Les pseudos « auto: » sont réservés aux campagnes automatiques."
    return None


def echapper(texte: str) -> str:
    """Texte saisi par un joueur, affiché tel quel dans du Markdown (sans gras, liens, titres...)."""
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|<>~:$])", r"\\\1", texte)
