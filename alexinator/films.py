"""Base cinéma : chargement des fiches de data/films.json."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

from .config import FILMS_PATH


@dataclass
class Film:
    id: str
    titre: str
    annee: int
    alias: List[str] = field(default_factory=list)
    realisateurs: List[str] = field(default_factory=list)
    acteurs: List[str] = field(default_factory=list)
    personnages: List[str] = field(default_factory=list)
    genre: str = ""
    mots_cles: List[str] = field(default_factory=list)

    @property
    def titres_acceptes(self) -> List[str]:
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


def load_films(path: Path = FILMS_PATH) -> List[Film]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Film(**entry) for entry in data]


def pick_film(films: List[Film], exclude: Iterable[str] = (), rng: Optional[random.Random] = None) -> Film:
    """Tire un film au hasard, en évitant ceux déjà joués (ids dans `exclude`) si possible."""
    rng = rng or random
    excluded = set(exclude)
    candidates = [f for f in films if f.id not in excluded] or films
    return rng.choice(candidates)
