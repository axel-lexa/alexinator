"""Ce que partagent les pages de l'interface : ressources, état du joueur, images, parties."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from alexinator import visuels
from alexinator.config import JOURNAL_PATH, get_settings
from alexinator.engine import GameSession, Level, load_levels
from alexinator.films import Film, load_films, pick_film
from alexinator.journal import Journal
from alexinator.llm import make_llm

AVATAR = str(visuels.AVATAR)


@dataclass
class Ressources:
    levels: list[Level]
    films: list[Film]
    journal: Journal
    llm: object
    aux_llm: object

    @property
    def par_id(self) -> dict[int, Level]:
        return {lv.id: lv for lv in self.levels}

    @property
    def dernier(self) -> int:
        return self.levels[-1].id


# --- Ressources partagées par tous les joueurs (créées une seule fois) ---------------------
@st.cache_resource
def ressources() -> Ressources:
    settings = get_settings()
    llm = make_llm(settings)
    aux_llm = make_llm(settings, settings.aux_model)
    # Mode démo : les fausses parties restent à part, hors des données des playtests.
    journal = Journal(JOURNAL_PATH.with_name("journal_demo.db")) if settings.backend == "demo" else Journal()
    return Ressources(load_levels(), load_films(), journal, llm, aux_llm)


# Les images avec bulle sont fabriquées une fois par texte, puis gardées en cache.
@st.cache_data(show_spinner=False)
def image_reflechit(texte: str):
    return visuels.magicien_reflechit(texte, largeur=500)


@st.cache_data(show_spinner=False)
def image_trouve(titre: str):
    return visuels.magicien_trouve(titre, largeur=600)


def video_reflechit_html() -> str:
    """Le magicien qui réfléchit, sans lecteur (st.video affiche toujours pause, barre de lecture...).

    Servie depuis static/ : le navigateur la télécharge une fois et la garde en cache.
    muted : les navigateurs ne lancent seules que les vidéos muettes ; playsinline : idem sur iPhone.
    """
    return (
        f'<video src="{visuels.VIDEO_REFLECHIT_URL}" autoplay muted loop playsinline '
        'disablepictureinpicture style="pointer-events: none; width: 100%; border-radius: 1rem"></video>'
    )


# --- État propre à chaque joueur ------------------------------------------------------------
# st.session_state survit aux interactions, mais pas au rechargement de la page (F5) : le pseudo est
# donc aussi gardé dans l'adresse (?joueur=...) et la progression relue dans le journal.
def init_etat() -> None:
    state = st.session_state
    state.setdefault("joueur", None)  # pseudo, None tant que le joueur n'est pas passé par l'accueil
    state.setdefault("meilleur", 0)  # plus haut niveau gagné
    state.setdefault("played", [])  # ids des films déjà joués
    state.setdefault("game", None)  # partie en cours (GameSession)
    state.setdefault("chat", [])  # messages affichés : {"role", "text", "issue", "blocked_by", "etapes"}
    state.setdefault(
        "scene", ("reflechit", "")
    )  # magicien affiché : ("reflechit", phrase) ou ("trouve", titre)
    state.setdefault("celebrate", False)  # lancer les ballons une seule fois
    state.setdefault("win_text", "")  # message de victoire (avec correction d'orthographe)
    state.setdefault("abandon", False)  # le joueur a donné sa langue au chat
    state.setdefault("erreur", "")  # dernière erreur du LLM, affichée une fois
    state.setdefault("toast", "")  # notification à montrer au prochain affichage

    if state.joueur is None and st.query_params.get("joueur"):
        connecter(st.query_params["joueur"])


def connecter(joueur: str) -> None:
    """Le joueur choisit (ou retrouve) son pseudo : sa progression est relue dans le journal."""
    state = st.session_state
    state.joueur = joueur.strip()
    st.query_params["joueur"] = state.joueur
    # « anonyme » est partagé par tout le monde : pas de progression à reprendre.
    state.meilleur = 0 if state.joueur == "anonyme" else ressources().journal.niveau_atteint(state.joueur)
    state.game = None


def deconnecter() -> None:
    state = st.session_state
    state.joueur, state.meilleur, state.game = None, 0, None
    st.query_params.pop("joueur", None)


def niveau_debloque() -> int:
    """Plus haut niveau jouable : celui qui suit le dernier niveau gagné."""
    return min(st.session_state.meilleur + 1, ressources().dernier)


def nouvelle_partie(level_id: int) -> None:
    r, state = ressources(), st.session_state
    film = pick_film(r.films, exclude=state.played)
    state.played.append(film.id)
    state.game = GameSession(
        r.par_id[level_id],
        film,
        r.llm,
        aux_llm=r.aux_llm,
        journal=r.journal,
        player=state.joueur or "anonyme",
        catalogue=r.films,
    )
    state.chat = []
    state.win_text = ""
    state.abandon = False
    state.erreur = ""
    state.scene = ("reflechit", visuels.phrase("debut"))


def partie_finie(game: GameSession) -> bool:
    return game.solved or game.messages_left == 0 or st.session_state.abandon
