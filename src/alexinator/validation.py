"""Validation de la réponse du joueur.

Trois étapes, de la plus sûre à la plus souple :
1. exacte    : après normalisation, la réponse est le titre ou l'un de ses alias ;
2. approchée : même nombre de mots et très forte ressemblance (tolère les fautes de frappe) ;
3. juge LLM  : en dernier recours et seulement si un LLM est fourni.

On veut accepter « star wars 4 » ou « amelie poulin », mais refuser « Star Wars » tout court
quand le secret est l'épisode IV, ou « star wars 1 » : c'est pourquoi l'étape 2 exige le même
nombre de mots et compare aussi les mots un à un.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache

from .films import Film

ARTICLES = ("le ", "la ", "les ", "l ", "un ", "une ", "the ", "a ")
SEUIL_APPROX = 0.85  # ressemblance de la phrase entière
SEUIL_MOT = 0.5  # ressemblance minimale de chaque mot avec le mot à la même place
SEUIL_PROCHE = 0.7  # en dessous de SEUIL_APPROX mais au-dessus : « tu chauffes »


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
            return normalized[len(article) :]
    return normalized


@lru_cache(maxsize=4096)  # play() compare un même message à tous les titres de la base
def canonical(text: str) -> str:
    return strip_article(normalize(text))


@dataclass
class GuessResult:
    correct: bool
    method: str  # "exacte", "approchee", "juge", "partielle" ou "refusee"
    matched: str = ""  # titre ou alias reconnu, bien orthographié (pour corriger le joueur)


def check_answer(guess: str, film: Film, judge=None) -> GuessResult:
    g = canonical(guess)
    if not g:
        return GuessResult(False, "refusee")
    targets = [(canonical(t), t) for t in film.titres_acceptes]

    # 1) Correspondance exacte avec le titre ou un alias
    for target, original in targets:
        if g == target:
            return GuessResult(True, "exacte", original)

    # 2) Correspondance approchée (fautes de frappe), à nombre de mots égal
    for target, original in targets:
        if _is_typo_of(g, target):
            return GuessResult(True, "approchee", original)

    # 3) Juge LLM (optionnel)
    if judge is not None and _judge_says_yes(guess, film, judge):
        return GuessResult(True, "juge")
    return GuessResult(False, "refusee")


# ----------------------------------------------------------------------------------------
# Un seul champ de saisie : reconnaître qu'un message est une proposition de titre
# ----------------------------------------------------------------------------------------
# Formules que le joueur met devant un titre (« c'est Titanic ? »), sous forme normalisée.
GUESS_PREFIXES = sorted(
    [
        "c est",
        "c etait",
        "est ce",
        "est ce que c est",
        "est ce que ce serait",
        "est ce que ce n est pas",
        "ce serait",
        "ca serait",
        "serait ce",
        "ce ne serait pas",
        "ne serait ce pas",
        "ce n est pas",
        "je pense que c est",
        "je pense que",
        "je pense a",
        "je crois que c est",
        "je crois que",
        "je dirais",
        "je propose",
        "ma reponse",
        "ma reponse est",
        "reponse",
        "le film c est",
        "le film est",
        "c est le film",
        "il s agit de",
        "tu penses a",
        "tu parles de",
        "alors",
        "donc",
        "euh",
        "hmm",
        # tournures négatives ou hésitantes : « c'est pas Titanic ? » propose bien Titanic
        "c est pas",
        "c etait pas",
        "ce serait pas",
        "ca serait pas",
        "ca ne serait pas",
        "ce n etait pas",
        "est ce pas",
        "c est peut etre",
        "peut etre",
        "ca doit etre",
        "c est surement",
        "surement",
        "je parie sur",
        "je parie que c est",
        "et si c etait",
        "j ai trouve",
        "je sais",
        "ok",
        "oui",
        "bon",
        "ah",
    ],
    key=len,
    reverse=True,
)
# Ce que le joueur ajoute après le titre (« Titanic, non ? »), sous forme normalisée.
GUESS_SUFFIXES = sorted(
    [
        "non",
        "nan",
        "peut etre",
        "alors",
        "du coup",
        "hein",
        "ou pas",
        "je pense",
        "je crois",
        "j imagine",
        "c est ca",
        "c est bien ca",
        "non peut etre",
        "lol",
        "mdr",
    ],
    key=len,
    reverse=True,
)
MIN_PARTIAL_CHARS = 7  # « Harry Potter » ou « Astérix » : oui ; « star » ou « le » : non


def extract_title(message: str) -> str:
    """Retire les formules autour du titre : « Est-ce que c'est pas Titanic, non ? » -> « titanic »."""
    text = normalize(message)
    stripped = True
    while stripped and text:
        stripped = False
        for prefix in GUESS_PREFIXES:
            if text == prefix:
                return ""
            if text.startswith(prefix + " "):
                text, stripped = text[len(prefix) + 1 :], True
                break
        for suffix in GUESS_SUFFIXES:
            if text.endswith(" " + suffix):
                text, stripped = text[: -len(suffix) - 1], True
                break
    return text


def is_partial_title(candidate: str, film: Film) -> bool:
    """Le message est-il le début d'un titre du film (sa saga) ? « Star Wars » pour l'épisode IV."""
    c = canonical(candidate)
    if len(c) < MIN_PARTIAL_CHARS:
        return False
    return any(canonical(t).startswith(c + " ") for t in film.titres_acceptes)


def _is_typo_of(g: str, target: str) -> bool:
    """Fautes de frappe seulement : même nombre de mots, et chaque mot ressemble à celui d'en face.

    La comparaison mot à mot empêche un mot en trop et un mot en moins de se compenser :
    « pas retour vers le futur » n'est pas « retour vers le futur 1 », ni « star wars 1 » « star wars 4 ».
    """
    words, target_words = g.split(), target.split()
    if len(words) != len(target_words) or SequenceMatcher(None, g, target).ratio() < SEUIL_APPROX:
        return False
    pairs = zip(words, target_words, strict=True)  # longueurs égales, vérifiées juste au-dessus
    return all(SequenceMatcher(None, w, t).ratio() >= SEUIL_MOT for w, t in pairs)


def is_near_miss(candidate: str, film: Film) -> bool:
    """Le message ressemble fortement à un titre du film sans être correct : « Retour vers le passé »."""
    c = canonical(candidate)
    if len(c) < MIN_PARTIAL_CHARS:
        return False
    for target in (canonical(t) for t in film.titres_acceptes):
        close_length = abs(len(c.split()) - len(target.split())) <= 1
        if close_length and SequenceMatcher(None, c, target).ratio() >= SEUIL_PROCHE:
            return True
    return False


def _judge_says_yes(guess: str, film: Film, judge) -> bool:
    prompt = (
        "Tu arbitres un jeu de devinettes sur le cinéma.\n"
        f"Film à trouver : {film.titre} ({film.annee}). "
        f"Autres titres acceptés : {', '.join(film.alias) or 'aucun'}.\n"
        f"Réponse du joueur : « {guess} »\n"
        "La réponse désigne-t-elle précisément ce film (et pas seulement sa saga ou un autre volet) ?\n"
        "Réponds par un seul mot : OUI ou NON."
    )
    verdict = judge.chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=5)
    return normalize(verdict).startswith("oui")
