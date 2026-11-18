"""Interface web d'Alexinator.

Lancer depuis la racine du projet :  streamlit run app.py
"""

import streamlit as st

from alexinator import visuels
from alexinator.config import get_settings
from alexinator.engine import GameSession, load_levels
from alexinator.films import load_films, pick_film
from alexinator.journal import Journal
from alexinator.llm import make_llm

st.set_page_config(page_title="Alexinator", page_icon=str(visuels.AVATAR))


# --- Ressources partagées par tous les joueurs (créées une seule fois) ---------------------
@st.cache_resource
def get_resources():
    settings = get_settings()
    llm = make_llm(settings)
    aux_llm = make_llm(settings, settings.aux_model)
    return load_levels(), load_films(), Journal(), llm, aux_llm


# Les images avec bulle sont fabriquées une fois par texte, puis gardées en cache.
@st.cache_data(show_spinner=False)
def image_reflechit(texte: str):
    return visuels.magicien_reflechit(texte, largeur=500)


@st.cache_data(show_spinner=False)
def image_trouve(titre: str):
    return visuels.magicien_trouve(titre, largeur=600)


levels, films, journal, llm, aux_llm = get_resources()
levels_by_id = {lv.id: lv for lv in levels}
AVATAR = str(visuels.AVATAR)

# --- État propre à chaque joueur (st.session_state survit aux rechargements de la page) ----
state = st.session_state
state.setdefault("unlocked", levels[0].id)  # plus haut niveau débloqué
state.setdefault("played", [])  # ids des films déjà joués
state.setdefault("game", None)  # partie en cours (GameSession)
state.setdefault("chat", [])  # messages affichés : {"role", "text", "pensee"}
state.setdefault("intro", "")  # phrase du magicien en début de partie
state.setdefault("celebrate", False)  # lancer les ballons une seule fois
state.setdefault("win_text", "")  # message de victoire (avec correction d'orthographe)


def new_game(level_id: int, player: str) -> None:
    film = pick_film(films, exclude=state.played)
    state.played.append(film.id)
    state.game = GameSession(
        levels_by_id[level_id],
        film,
        llm,
        aux_llm=aux_llm,
        journal=journal,
        player=player or "anonyme",
        catalogue=films,
    )
    state.chat = []
    state.win_text = ""
    state.intro = visuels.phrase("debut")


# --- Barre latérale : affiche, pseudo, niveau, nouvelle partie ------------------------------
with st.sidebar:
    st.image(str(visuels.AFFICHE), width=260)
    player = st.text_input("Ton pseudo", value="anonyme", max_chars=30)
    available = [lv.id for lv in levels if lv.id <= state.unlocked]
    # Par défaut : le niveau de la partie en cours (sinon le plus haut niveau débloqué).
    # Ainsi, débloquer un niveau ne fait pas quitter l'écran de victoire.
    current = state.game.level.id if state.game else available[-1]
    level_id = st.selectbox(
        "Niveau",
        available,
        index=available.index(current),
        format_func=lambda i: f"{i}. {levels_by_id[i].name}",
        filter_mode=None,  # dropdown classique : pas de saisie au clavier
    )
    if st.button("Nouveau film"):
        new_game(level_id, player)
    st.caption(f"Niveaux débloqués : {state.unlocked} / {levels[-1].id}")

# Première visite ou changement de niveau : on démarre une partie.
if state.game is None or state.game.level.id != level_id:
    new_game(level_id, player)
game = state.game
game.player = player or "anonyme"

# --- Zone principale -------------------------------------------------------------------------
st.header(f"Niveau {game.level.id} : {game.level.name}")
st.write(game.level.description)
st.caption(
    f"Messages restants : {game.messages_left} / {game.level.max_messages}  ·  "
    "Tu penses avoir trouvé ? Écris simplement le titre dans la discussion."
)

# Début de partie : le magicien annonce qu'il a un film en tête.
if not state.chat and not game.solved:
    st.image(image_reflechit(state.intro), width=320)

for entry in state.chat:
    with st.chat_message(entry["role"], avatar=AVATAR if entry["role"] == "assistant" else None):
        st.write(entry["text"])
        # Défense qui bloque ou mauvaise proposition : le magicien prend un air entendu.
        if entry.get("pensee"):
            st.image(image_reflechit(entry["pensee"]), width=220)

if game.solved:
    st.image(image_trouve(game.film.titre), width=420)
    if state.celebrate:
        st.balloons()
        state.celebrate = False
    st.success(state.win_text or f"Bravo ! C'était bien « {game.film.titre} » ({game.film.annee}).")
    if game.level.id < levels[-1].id:
        st.info("Le niveau suivant est débloqué : choisis-le dans la barre latérale.")
    else:
        st.image(str(visuels.AFFICHE_VICTOIRE), width=320)
        st.info("Tu as battu tous les niveaux. Clique sur « Nouveau film » pour rejouer.")
else:
    # Un seul champ : le moteur distingue lui-même une question d'une proposition de titre.
    message = st.chat_input(
        "Pose une question au gardien, ou tape le titre du film...", disabled=game.messages_left == 0
    )
    if message:
        with st.chat_message("user"):
            st.write(message)
        # Pendant que le LLM répond, le magicien réfléchit (l'image disparaît ensuite).
        thinking = st.empty()
        thinking.image(image_reflechit(visuels.phrase("attente")), width=260)
        try:
            result = game.play(message)
        except Exception as error:  # LLM injoignable, mauvais nom de modèle, clé invalide...
            thinking.empty()
            st.error(
                f"Impossible de joindre le LLM. Vérifie le fichier .env et que le serveur tourne.\n\n{error}"
            )
        else:
            thinking.empty()
            state.chat.append({"role": "user", "text": message})
            if result.guess and result.guess.correct:
                state.unlocked = min(max(state.unlocked, game.level.id + 1), levels[-1].id)
                state.celebrate = True
                state.win_text = result.reply
            else:
                blocked = result.blocked_by not in (None, "limite")
                missed = result.guess is not None
                pensee = visuels.phrase("rate") if missed else visuels.phrase("bloque") if blocked else ""
                state.chat.append({"role": "assistant", "text": result.reply, "pensee": pensee})
            st.rerun()
    if game.messages_left == 1:
        st.warning("Dernier message : c'est le moment de proposer ton titre !")
    elif game.messages_left == 0:
        st.warning("Plus de messages. Clique sur « Nouveau film » pour recommencer.")
