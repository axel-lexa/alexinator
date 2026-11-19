"""Base cinéma : chargement des fiches de data/films.json."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from .config import FILMS_PATH


@dataclass
class Film:
    id: str
    titre: str
    annee: int
    alias: list[str] = field(default_factory=list)
    realisateurs: list[str] = field(default_factory=list)
    acteurs: list[str] = field(default_factory=list)
    personnages: list[str] = field(default_factory=list)
    genre: str = ""
    mots_cles: list[str] = field(default_factory=list)
    resume: str = ""

    @property
    def titres_acceptes(self) -> list[str]:
        """Le titre et tous ses alias : les réponses considérées comme justes."""
        return [self.titre] + self.alias

    def fiche(self, anonyme: bool = False) -> str:
        """Texte donné au gardien pour qu'il donne des indices exacts (et n'invente rien).

        anonyme=True (niveau 7) : ni titre ni noms propres, seulement année, genre et éléments
        marquants. Le gardien ne peut pas divulguer ce qu'il ne connaît pas.
        """
        lines = [
            f"Année : {self.annee}",
            f"Genre : {self.genre}",
            f"Éléments marquants : {', '.join(self.mots_cles)}",
        ]
        if not anonyme:
            lines = [
                f"Titre : {self.titre}",
                f"Autres titres : {', '.join(self.alias) or 'aucun'}",
                f"Réalisation : {', '.join(self.realisateurs)}",
                f"Acteurs : {', '.join(self.acteurs) or 'non renseigné'}",
                f"Personnages : {', '.join(self.personnages)}",
            ] + lines
        return "\n".join(lines)


def load_films(path: Path = FILMS_PATH) -> list[Film]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Film(**entry) for entry in data]


# Ordre des champs dans data/films.json (différent de l'ordre de la classe, qui exige les défauts en fin).
FIELD_ORDER = ("id", "titre", "alias", "annee", "realisateurs", "acteurs", "personnages", "genre")
FIELD_ORDER += ("mots_cles",)


def save_films(films: list[Film], path: Path = FILMS_PATH) -> None:
    """Écrit les fiches dans le style du fichier (JSON indenté de 2, champs dans l'ordre habituel)."""
    data = []
    for film in films:
        entry = {key: vars(film)[key] for key in FIELD_ORDER}
        if film.resume:
            entry["resume"] = film.resume
        data.append(entry)
    text = json.dumps(data, ensure_ascii=False, indent=2)
    Path(path).write_text(text, encoding="utf-8", newline="\n")


def pick_film(films: list[Film], exclude: Iterable[str] = (), rng: random.Random | None = None) -> Film:
    """Tire un film au hasard, en évitant ceux déjà joués (ids dans `exclude`) si possible."""
    rng = rng or random
    excluded = set(exclude)
    candidates = [f for f in films if f.id not in excluded] or films
    return rng.choice(candidates)
