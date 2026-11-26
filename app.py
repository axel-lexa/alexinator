"""Interface web d'Alexinator.

Lancer depuis la racine du projet :  streamlit run app.py
Sans serveur LLM (pour voir l'interface) :  ALEXINATOR_BACKEND=demo streamlit run app.py

Ce fichier règle la page et la barre latérale communes, puis affiche la page choisie (dossier vues/).
Le thème est dans .streamlit/config.toml, la couche CSS dans assets/style.css.
"""

from pathlib import Path

import streamlit as st

from alexinator import interface, visuels
from vues import commun

st.set_page_config(
    page_title="Alexinator",
    page_icon=str(visuels.AVATAR),
    layout="wide",
    initial_sidebar_state="auto",  # ouverte sur ordinateur, repliée sur mobile
)
st.html(Path(__file__).parent / "assets" / "style.css")

commun.init_etat()

page = st.navigation(
    [
        st.Page("vues/jouer.py", title="Jouer", icon="🔮", default=True),
        st.Page("vues/comment_ca_marche.py", title="Comment ça marche", icon="🛡️"),
        st.Page("vues/classement.py", title="Classement", icon="🏆"),
    ],
    position="top",
)

# Sur l'écran d'accueil, l'affiche est déjà au centre : la barre latérale reste vide.
if st.session_state.joueur:
    with st.sidebar:
        st.image(str(visuels.AFFICHE), width="stretch", alt="Affiche : Alexinator, il lit dans vos pensées")
        with st.container(horizontal=True, vertical_alignment="center"):
            st.markdown(f"👤 **{interface.echapper(st.session_state.joueur)}**")
            if st.button("Changer", type="tertiary", help="Changer de pseudo"):
                commun.deconnecter()
                st.switch_page("vues/jouer.py")

page.run()
