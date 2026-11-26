"""Page « Comment ça marche » : les règles, et la chaîne de défenses de chaque niveau.

Les consignes du gardien (prompts) ne sont jamais affichées : les découvrir fait partie du jeu.
"""

from __future__ import annotations

import streamlit as st

from alexinator import interface
from alexinator.interface import ETAPES
from vues import commun

r = commun.ressources()
state = st.session_state

st.title("Comment ça marche ?")
st.markdown(
    "Alexinator est un **banc d'essai de sécurité des LLM** déguisé en jeu. Un modèle de langage, "
    "le *gardien*, connaît un film secret et a pour consigne de ne pas le révéler. Tu peux jouer "
    "honnêtement (poser des questions et déduire) ou tenter de le **pirater** : jeu de rôle, fausse "
    "autorité, demande d'épeler ou de traduire le titre... Chaque niveau ajoute une protection, et "
    "toutes les parties servent à mesurer quelles attaques battent quelles défenses."
)

with st.expander("Les règles en détail", icon="📜"):
    st.markdown(
        "- **Un seul champ de saisie.** Si ton message entier désigne un film (« C'est Titanic ? »), "
        "c'est une proposition, vérifiée par le jeu et non par le gardien. Sinon, c'est une question.\n"
        "- **Les propositions ratées coûtent un message.** Citer un titre au milieu d'une phrase ou "
        "coller une liste de titres ne compte pas comme une proposition.\n"
        "- **« Tu brûles »** : tu as donné le début du titre (le nom de la saga, par exemple). "
        "**« Tu chauffes »** : le titre est presque bon, vérifie l'orthographe.\n"
        "- **Une réponse bloquée par une défense** n'entre pas dans la mémoire du gardien : "
        "il ne pourra pas la répéter plus tard.\n"
        "- **Les fautes de frappe sont tolérées** pour la bonne réponse : « forest gump » gagne."
    )

st.subheader("Le trajet d'un message, niveau par niveau")
courant = state.game.level.id if state.get("game") else 1
choix = st.segmented_control(
    "Niveau",
    [lv.id for lv in r.levels],
    default=courant,
    required=True,
    format_func=lambda i: f"{i}. {r.par_id[i].name}",
    label_visibility="collapsed",
)
level = r.par_id[choix or courant]

with st.container(key="carte-niveau"):
    st.markdown(f"#### Niveau {level.id} · {level.name}")
    st.write(level.description)

    actives = interface.chaine(level)
    maillons = ["💬 Ton message"]
    for key in interface.ORDRE:
        etape = ETAPES[key]
        couleur = "violet" if key in actives else "gray"
        label = etape.nom_long if key in actives else f"~~{etape.nom_long}~~"
        maillons.append(f":{couleur}-badge[{etape.icone} {label}]")
    maillons.append("📺 Réponse affichée")
    st.markdown(" → ".join(maillons))

    for key in actives:
        etape = ETAPES[key]
        st.markdown(f"**{etape.icone} {etape.nom_long}.** {etape.explication}")
    if level.response_mode == "oui_non":
        st.markdown(
            "**🤐 Sortie contrainte.** Quoi que le gardien écrive, le code ne laisse passer que "
            "« Oui. », « Non. » ou « Je ne sais pas. » : impossible de lui faire écrire autre chose."
        )
    if level.fiche_mode == "anonyme":
        st.markdown(
            "**🙈 Fiche anonyme.** Le gardien ne connaît ni le titre ni les noms : seulement l'année, "
            "le genre et quelques éléments du film. Il ne peut pas trahir ce qu'il ne sait pas."
        )
    st.caption(f"{level.max_messages} messages par partie.")

st.caption("Les consignes exactes du gardien restent secrètes : à toi de les deviner. 😉")
