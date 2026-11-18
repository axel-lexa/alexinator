"""Images d'Alexinator : le magicien qui réfléchit, celui qui a trouvé, et leurs bulles.

Les dessins se trouvent dans assets/. Les versions « vierge » ont une bulle vide :
on y écrit le texte du moment (titre du film, réaction...) au lieu d'un texte figé.
Seule dépendance : Pillow (installé avec Streamlit).
"""

from __future__ import annotations

import random
from functools import cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .config import ROOT

ASSETS = ROOT / "assets"
AVATAR = ASSETS / "avatar.png"  # visage du magicien, pour la bulle de chat
AFFICHE = ASSETS / "affiche_reflechit.png"  # affiche « Il lit dans vos pensées... »
AFFICHE_VICTOIRE = ASSETS / "affiche_solution.png"  # affiche « ... et il ne se trompe jamais. »

_FONT_BOLD = ASSETS / "fonts" / "Inter-Bold.otf"
_FONT_BLACK = ASSETS / "fonts" / "Inter-ExtraBold.otf"
_INK = (30, 30, 46)  # texte sombre, comme les contours du dessin
_RED = (230, 57, 70)  # rouge du dessin original (« INCEPTION ! »)

# Zone utilisable de chaque bulle, en pixels de l'image d'origine (1000 x 1250) :
# (gauche, haut, droite, bas). L'ellipse de la bulle de pensée est plus étroite en haut et en bas.
_THOUGHT_BOX = (625, 62, 945, 207)
_SPEECH_BOX = (92, 66, 570, 206)

# Phrases du magicien selon le moment du jeu.
PHRASES = {
    "attente": [
        "Hmm... voyons voir...",
        "Laisse-moi réfléchir...",
        "Intéressant...",
        "Je consulte ma boule...",
    ],
    "bloque": ["Bien tenté...", "Je t'ai vu venir !", "Pas si vite...", "Ma boule a tout vu."],
    "rate": ["Raté ! Ce n'est pas ça...", "Non, non, non...", "Tu chauffes ? Même pas."],
    "debut": ["J'ai un film en tête...", "Devine à quoi je pense...", "Prêt ? Je pense à un film..."],
}


def phrase(moment: str, rng: random.Random | None = None) -> str:
    return (rng or random).choice(PHRASES[moment])


@cache
def _base(name: str) -> Image.Image:
    return Image.open(ASSETS / name).convert("RGB")


def _font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size)


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    """Coupe le texte en lignes qui tiennent dans max_width pixels (sans couper les mots)."""
    lines, current = [], ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if draw.textlength(candidate, font=font) <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _fit(draw, blocks: list[tuple[str, Path, int, tuple]], box) -> list[tuple[list[str], object, tuple]]:
    """Réduit la taille des polices jusqu'à ce que tous les blocs tiennent dans la bulle.

    blocks : liste de (texte, police, taille maximale, couleur). Renvoie les lignes et polices retenues.
    """
    width, height = box[2] - box[0], box[3] - box[1]
    scale = 1.0
    while True:
        laid_out, total = [], 0
        for text, font_path, size, color in blocks:
            font = _font(font_path, max(14, int(size * scale)))
            lines = _wrap(draw, text, font, width)
            total += len(lines) * int(font.size * 1.18)
            laid_out.append((lines, font, color))
        too_wide = any(draw.textlength(line, font=f) > width for lines, f, _ in laid_out for line in lines)
        if (total <= height and not too_wide) or scale < 0.3:
            return laid_out
        scale *= 0.92


def _draw_centered(image: Image.Image, blocks, box) -> Image.Image:
    image = image.copy()
    draw = ImageDraw.Draw(image)
    laid_out = _fit(draw, blocks, box)
    line_heights = [int(font.size * 1.18) for lines, font, _ in laid_out for _ in lines]
    y = box[1] + ((box[3] - box[1]) - sum(line_heights)) // 2
    cx = (box[0] + box[2]) // 2
    for lines, font, color in laid_out:
        for line in lines:
            draw.text((cx, y), line, font=font, fill=color, anchor="ma")
            y += int(font.size * 1.18)
    return image


def magicien_reflechit(texte: str, largeur: int | None = None) -> Image.Image:
    """Le magicien pensif, avec `texte` dans sa bulle de pensée."""
    image = _draw_centered(
        _base("cartoon_reflechit_vierge.png"), [(texte, _FONT_BOLD, 46, _INK)], _THOUGHT_BOX
    )
    return _resize(image, largeur)


def magicien_trouve(
    titre: str, accroche: str = "Bien joué ! C'était...", largeur: int | None = None
) -> Image.Image:
    """Le magicien triomphant, avec l'accroche et le titre du film (en rouge) dans sa bulle."""
    blocks = [(accroche, _FONT_BOLD, 40, _INK), (titre.upper() + " !", _FONT_BLACK, 66, _RED)]
    image = _draw_centered(_base("cartoon_solution_vierge.png"), blocks, _SPEECH_BOX)
    return _resize(image, largeur)


def _resize(image: Image.Image, largeur: int | None) -> Image.Image:
    if not largeur:
        return image
    hauteur = round(image.height * largeur / image.width)
    return image.resize((largeur, hauteur), Image.LANCZOS)
