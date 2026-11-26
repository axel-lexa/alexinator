"""Page « Classement » : les meilleurs joueurs, et ce que le gardien a bloqué.

Seuls des chiffres agrégés par pseudo sont montrés, jamais le contenu du journal (CONTRIBUTING.md).
"""

from __future__ import annotations

import streamlit as st

from alexinator.interface import ETAPES
from vues import commun

r = commun.ressources()

st.title("Classement")

lignes = r.journal.classement(limit=20)
if not lignes:
    st.info("Personne n'a encore trouvé de film. La première place t'attend !", icon="🏆")
else:
    for rang, ligne in enumerate(lignes, start=1):
        ligne["rang"] = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rang, str(rang))
    st.dataframe(
        lignes,
        hide_index=True,
        column_order=["rang", "joueur", "meilleur_niveau", "films_trouves", "messages_moyens"],
        column_config={
            "rang": st.column_config.TextColumn("", width="small"),
            "joueur": st.column_config.TextColumn("Joueur"),
            "meilleur_niveau": st.column_config.ProgressColumn(
                "Meilleur niveau", min_value=0, max_value=r.dernier, format="%d / " + str(r.dernier)
            ),
            "films_trouves": st.column_config.NumberColumn("Films trouvés"),
            "messages_moyens": st.column_config.NumberColumn(
                "Messages pour trouver", help="Moyenne des messages envoyés avant de trouver le film"
            ),
        },
    )
    if st.session_state.joueur and st.session_state.joueur not in {ligne["joueur"] for ligne in lignes}:
        st.caption("Trouve ton premier film pour apparaître ici.")

st.subheader("Le gardien en chiffres")
blocages = r.journal.blocages()
total = sum(n for issue, n in blocages.items() if issue != "limite")
colonnes = st.columns(4)
colonnes[0].metric("Messages reçus", total, border=True)
for colonne, key in zip(colonnes[1:], ("input_detector", "output_filter", "censor"), strict=True):
    etape = ETAPES[key]
    colonne.metric(
        f"{etape.icone} {etape.nom_long}",
        blocages.get(key, 0),
        help=f"Messages bloqués par cette défense. {etape.explication}",
        border=True,
    )
st.caption("Parties des joueurs humains uniquement (les campagnes automatiques sont exclues).")
