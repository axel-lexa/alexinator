"""Validation de la réponse du joueur.

Trois étapes, de la plus sûre à la plus souple :
1. exacte    : après normalisation, la réponse est le titre ou l'un de ses alias ;
2. approchée : même nombre de mots et très forte ressemblance (tolère les fautes de frappe) ;
3. juge LLM  : en dernier recours et seulement si un LLM est fourni.

On veut accepter « star wars 4 » ou « amelie poulin », mais refuser « Star Wars » tout court
quand le secret est l'épisode IV : c'est pourquoi l'étape 2 exige le même nombre de mots.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional

from .films import Film

ARTICLES = ("le ", "la ", "les ", "l ", "un ", "une ", "the ", "a ")
SEUIL_APPROX = 0.85


def normalize(text: str) -> str:
    """minuscules, sans accents, sans ponctuation, espaces simples."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip()


def strip_article(normalized: str) -> str:
    """Retire un article en tête : « le parrain » -> « parrain »."""
    for article in ARTICLES:
        if normalized.startswith(article):
            return normalized[len(article):]
    return normalized


def canonical(text: str) -> str:
    return strip_article(normalize(text))


@dataclass
class GuessResult:
    correct: bool
    method: str  # "exacte", "approchee", "juge" ou "refusee"


def check_answer(guess: str, film: Film, judge=None) -> GuessResult:
    g = canonical(guess)
    if not g:
        return GuessResult(False, "refusee")
    targets = [canonical(t) for t in film.titres_acceptes]

    # 1) Correspondance exacte avec le titre ou un alias
    if g in targets:
        return GuessResult(True, "exacte")

    # 2) Correspondance approchée (fautes de frappe), à nombre de mots égal
    for target in targets:
        same_word_count = len(g.split()) == len(target.split())
        if same_word_count and SequenceMatcher(None, g, target).ratio() >= SEUIL_APPROX:
            return GuessResult(True, "approchee")

    # 3) Juge LLM (optionnel)
    if judge is not None and _judge_says_yes(guess, film, judge):
        return GuessResult(True, "juge")
    return GuessResult(False, "refusee")


def _judge_says_yes(guess: str, film: Film, judge) -> bool:
    prompt = (
        "Tu arbitres un jeu de devinettes sur le cinéma.\n"
        f"Film à trouver : {film.titre} ({film.annee}). Autres titres acceptés : {', '.join(film.alias) or 'aucun'}.\n"
        f"Réponse du joueur : « {guess} »\n"
        "La réponse désigne-t-elle précisément ce film (et pas seulement sa saga ou un autre volet) ?\n"
        "Réponds par un seul mot : OUI ou NON."
    )
    verdict = judge.chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=5)
    return normalize(verdict).startswith("oui")
