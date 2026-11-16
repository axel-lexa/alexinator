"""Interface web d'Alexinator.

Lancer depuis la racine du projet :  streamlit run app.py
"""
import streamlit as st

from alexinator.config import get_settings
from alexinator.engine import GameSession, load_levels
from alexinator.films import load_films, pick_film
from alexinator.journal import Journal
from alexinator.llm import make_llm

st.set_page_config(page_title="Alexinator", page_icon="🎬")


# --- Ressources partagées par tous les joueurs (créées une seule fois) ---------------------
@st.cache_resource
def get_resources():
    settings = get_settings()
    llm = make_llm(settings)
    aux_llm = make_llm(settings, settings.aux_model)
    return load_levels(), load_films(), Journal(), llm, aux_llm


levels, films, journal, llm, aux_llm = get_resources()
levels_by_id = {lv.id: lv for lv in levels}

# --- État propre à chaque joueur (st.session_state survit aux rechargements de la page) ----
state = st.session_state
state.setdefault("unlocked", levels[0].id)   # plus haut niveau débloqué
state.setdefault("played", [])               # ids des films déjà joués
state.setdefault("game", None)               # partie en cours (GameSession)
state.setdefault("chat", [])                 # messages affichés : (rôle, texte)


def new_game(level_id: int, player: str) -> None:
    film = pick_film(films, exclude=state.played)
    state.played.append(film.id)
    state.game = GameSession(levels_by_id[level_id], film, llm, aux_llm=aux_llm,
                             journal=journal, player=player or "anonyme")
    state.chat = []


# --- Barre latérale : pseudo, niveau, nouvelle partie --------------------------------------
with st.sidebar:
    st.title("🎬 Alexinator")
    player = st.text_input("Ton pseudo", value="anonyme", max_chars=30)
    available = [lv.id for lv in levels if lv.id <= state.unlocked]
    # Par défaut : le niveau de la partie en cours (sinon le plus haut niveau débloqué).
    # Ainsi, débloquer un niveau ne fait pas quitter l'écran de victoire.
    current = state.game.level.id if state.game else available[-1]
    level_id = st.selectbox(
        "Niveau", available, index=available.index(current),
        format_func=lambda i: f"{i}. {levels_by_id[i].name}",
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
st.caption(f"Messages restants : {game.messages_left} / {game.level.max_messages}")

for role, text in state.chat:
    with st.chat_message(role):
        st.write(text)

if game.solved:
    st.success(f"Bravo ! C'était bien « {game.film.titre} » ({game.film.annee}).")
    if game.level.id < levels[-1].id:
        st.info("Le niveau suivant est débloqué : choisis-le dans la barre latérale.")
    else:
        st.info("Tu as battu tous les niveaux. Clique sur « Nouveau film » pour rejouer.")
else:
    # Proposition de réponse (ne passe pas par le gardien : validée par le code)
    with st.form("reponse", clear_on_submit=True):
        answer = st.text_input("Ta réponse : quel est le film ?")
        submitted = st.form_submit_button("Valider")
    if submitted and answer.strip():
        result = game.guess(answer)
        if result.correct:
            state.unlocked = min(max(state.unlocked, game.level.id + 1), levels[-1].id)
            st.balloons()
            st.rerun()
        else:
            st.error("Ce n'est pas ça. Continue à interroger le gardien.")

    # Discussion avec le gardien
    message = st.chat_input("Parle au gardien...", disabled=game.messages_left == 0)
    if message:
        try:
            with st.spinner("Alexinator réfléchit..."):
                result = game.send(message)
        except Exception as error:  # LLM injoignable, mauvais nom de modèle, clé invalide...
            st.error(f"Impossible de joindre le LLM. Vérifie le fichier .env et que le serveur tourne.\n\n{error}")
        else:
            state.chat.append(("user", message))
            state.chat.append(("assistant", result.reply))
            st.rerun()
    if game.messages_left == 0:
        st.warning("Plus de messages : propose une réponse ou clique sur « Nouveau film ».")
