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

# Barre latérale commune : la marque, puis la carte du joueur (la page Jouer y ajoute les niveaux).
# Sur l'écran d'accueil, l'affiche est déjà au centre : la barre latérale reste vide.
if st.session_state.joueur:
    r = commun.ressources()
    with st.sidebar:
        with st.container(key="marque", horizontal=True, vertical_alignment="center", gap="small"):
            st.image(commun.AVATAR, width=52)
            st.markdown("**Alexinator**  \n*Il lit dans vos pensées...*")
        with st.container(key="joueur"):
            with st.container(horizontal=True, vertical_alignment="center"):
                pseudo = interface.echapper(st.session_state.joueur)
                st.markdown(f":material/person: **{pseudo}**", width="stretch")
                if st.button("", icon=":material/logout:", type="tertiary", help="Changer de pseudo"):
                    commun.deconnecter()
                    st.switch_page("vues/jouer.py")
            gagnes = st.session_state.meilleur
            st.progress(gagnes / len(r.levels), text=f"{gagnes} / {len(r.levels)} niveaux gagnés")

page.run()
