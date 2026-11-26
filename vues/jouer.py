"""Page de jeu : accueil, en-tête du niveau, scène du magicien et discussion avec le gardien."""

from __future__ import annotations

import streamlit as st

from alexinator import interface, visuels
from alexinator.interface import ETAPES
from vues import commun

r = commun.ressources()
state = st.session_state


# --- Accueil : pseudo et avertissement sur l'enregistrement des messages ----------------------
def accueil() -> None:
    gauche, droite = st.columns([2, 3], gap="large", vertical_alignment="center")
    gauche.image(str(visuels.AFFICHE), width="stretch", alt="Affiche : Alexinator, il lit dans vos pensées")
    with droite, st.container(key="accueil"):
        st.title("Devine le film du gardien")
        st.markdown(
            "Alexinator, le gardien, pense à un film. Discute avec lui pour le deviner :\n\n"
            "- **pose-lui des questions**, ou **essaie de le pirater** pour qu'il lâche le titre ;\n"
            "- **pour proposer un film, écris simplement son titre** dans la discussion ;\n"
            f"- **chaque niveau ajoute une défense** : {len(r.levels)} niveaux, de plus en plus coriaces."
        )
        with st.form("pseudo", border=False):
            pseudo = st.text_input(
                "Ton pseudo", max_chars=interface.PSEUDO_MAX, placeholder="ex. Cinéphile du 3e rang"
            )
            st.caption(
                "🔒 Tes messages, les réponses du gardien et ton pseudo sont enregistrés pour étudier "
                "les attaques et entraîner notre propre gardien. Choisis un pseudo qui ne t'identifie pas "
                "et n'écris rien de personnel."
            )
            if st.form_submit_button("Jouer", type="primary", icon="🔮"):
                erreur = interface.verifier_pseudo(pseudo)
                if erreur:
                    st.error(erreur)
                else:
                    commun.connecter(pseudo)
                    st.rerun()


if not state.joueur:
    accueil()
    st.stop()

if state.game is None:
    commun.nouvelle_partie(commun.niveau_debloque())
game = state.game
level = game.level
finie = commun.partie_finie(game)

if state.toast:
    st.toast(state.toast, icon="🔓")
    state.toast = ""


# --- Changer de partie : confirmation si une partie est en cours ------------------------------
def executer(action: str, level_id: int | None = None) -> None:
    if action == "abandon":
        state.abandon = True
        state.scene = ("reflechit", visuels.phrase("perdu", titre=game.film.titre))
    else:
        commun.nouvelle_partie(level_id or level.id)


@st.dialog("Abandonner la partie en cours ?")
def confirmer(action: str, level_id: int | None = None) -> None:
    if action == "abandon":
        st.write("Le gardien va te révéler le film. Ta partie s'arrête là.")
    else:
        st.write("Tu vas perdre ta discussion avec le gardien sur ce film.")
    with st.container(horizontal=True):
        if st.button("Oui, j'abandonne", type="primary"):
            executer(action, level_id)
            st.rerun()
        if st.button("Non, je continue"):
            st.rerun()


def demander(action: str, level_id: int | None = None) -> None:
    if game.messages_used > 0 and not finie:
        confirmer(action, level_id)
    else:
        executer(action, level_id)
        st.rerun()


# --- Barre latérale : progression sur les niveaux ---------------------------------------------
with st.sidebar:
    st.subheader("Progression")
    debloque = commun.niveau_debloque()
    for lv in r.levels:
        gagne, verrouille = lv.id <= state.meilleur, lv.id > debloque
        if st.button(
            f"{lv.id}. {lv.name}",
            key=f"niveau_{lv.id}",
            icon="✅" if gagne else "🔒" if verrouille else "🔮",
            type="primary" if lv.id == level.id else "secondary",
            disabled=verrouille,
            width="stretch",
        ):
            demander("niveau", lv.id)
    st.caption(f"{state.meilleur} niveau(x) gagné(s) sur {len(r.levels)}")
    if st.button("Nouveau film", icon="🎬", width="stretch"):
        demander("film")


# --- En-tête : niveau, défenses actives, messages restants ------------------------------------
with st.container(key="entete"):
    st.title(f"Niveau {level.id} · {level.name}")
    st.write(level.description)
    with st.container(horizontal=True, gap="small"):
        for badge in interface.badges_niveau(level):
            st.badge(badge.label, icon=badge.icon, color=badge.color)
    restants = game.messages_left
    st.progress(
        restants / level.max_messages,
        text=f"💬 {restants} message{'s' if restants > 1 else ''} restant{'s' if restants > 1 else ''} "
        f"sur {level.max_messages}",
    )
    if level.tool_calls:
        st.caption(
            f"{ETAPES['tool'].icone} Vérifications du gardien : {game.tool_calls_left} / {level.tool_calls}"
        )


# --- Scène du magicien et discussion ----------------------------------------------------------
def afficher_scene(slot) -> None:
    genre, texte = state.scene
    image = commun.image_trouve(texte) if genre == "trouve" else commun.image_reflechit(texte)
    alt = (
        f"Le magicien : « Bien joué ! C'était {texte} »"
        if genre == "trouve"
        else f"Le magicien pense : {texte}"
    )
    slot.image(image, width="stretch", alt=alt)


def afficher_entree(entry: dict) -> None:
    assistant = entry["role"] == "assistant"
    with st.chat_message(entry["role"], avatar=commun.AVATAR if assistant else "🧑"):
        st.write(entry["text"])
        if not assistant:
            return
        badge = interface.badge_issue(entry.get("issue", ""), entry.get("blocked_by"))
        if badge:
            st.badge(badge.label, icon=badge.icon, color=badge.color)
        etapes = entry.get("etapes") or []
        if etapes and len(interface.chaine(level)) > 1:  # le niveau a des défenses : trajet du message
            st.caption(interface.resume_etapes(etapes, entry.get("blocked_by")))


def jouer_tour(message: str, discussion, scene, suggestions) -> None:
    """Envoie le message au moteur en montrant chaque étape de la chaîne de défenses."""
    suggestions.empty()
    with discussion:
        afficher_entree({"role": "user", "text": message})
        with st.chat_message("assistant", avatar=commun.AVATAR):
            status = st.status(ETAPES["guardian"].en_cours, expanded=True)
    scene.markdown(commun.video_reflechit_html(), unsafe_allow_html=True)

    etapes: list[str] = []

    def on_step(key: str) -> None:
        etapes.append(key)
        etape = ETAPES[key]
        status.update(label=etape.en_cours)
        status.write(f"{etape.icone} {etape.en_cours}")

    try:
        result = game.play(message, on_step=on_step)
    except Exception as error:  # LLM injoignable, mauvais nom de modèle, clé invalide...
        status.update(label="Le gardien ne répond pas", state="error")
        state.erreur = f"{type(error).__name__} : {error}"
        st.rerun()

    issue = interface.issue(result)
    state.chat.append({"role": "user", "text": message})
    if issue == "victoire":
        if state.meilleur < level.id < r.dernier:
            state.toast = f"Niveau {level.id + 1} débloqué !"
        state.meilleur = max(state.meilleur, level.id)
        state.celebrate = True
        state.win_text = result.reply
        state.scene = ("trouve", game.film.titre)
    else:
        state.chat.append(
            {
                "role": "assistant",
                "text": result.reply,
                "issue": issue,
                "blocked_by": result.blocked_by,
                "etapes": etapes,
            }
        )
        if game.messages_left == 0:
            state.scene = ("reflechit", visuels.phrase("perdu", titre=game.film.titre))
        elif issue in interface.MOMENT_SCENE:
            state.scene = ("reflechit", visuels.phrase(interface.MOMENT_SCENE[issue]))
    st.rerun()


def envoyer_suggestion(texte: str) -> None:
    state.a_envoyer = texte


def panneau_fin() -> None:
    film = game.film
    if game.solved:
        if state.celebrate:
            st.balloons()
            state.celebrate = False
        st.success(state.win_text or f"Bravo ! C'était bien « {film.titre} » ({film.annee}).", icon="🎉")
        bloques = sum(1 for e in state.chat if e.get("issue") == "bloque")
        st.caption(
            f"Trouvé en {game.messages_used} message(s) · {bloques} réponse(s) bloquée(s) par les défenses"
        )
        if level.id < r.dernier:
            with st.container(horizontal=True):
                if st.button("Niveau suivant", icon="➡️", type="primary"):
                    executer("niveau", level.id + 1)
                    st.rerun()
                if st.button("Rejouer ce niveau", icon="🎬"):
                    executer("niveau", level.id)
                    st.rerun()
        else:
            st.image(str(visuels.AFFICHE_VICTOIRE), width=280, alt="Affiche : et il ne se trompe jamais")
            st.info(f"Tu as battu les {len(r.levels)} niveaux. Chapeau, magicien !", icon="🏆")
            if st.button("Rejouer", icon="🎬", type="primary"):
                executer("niveau", level.id)
                st.rerun()
    else:
        raison = "Tu as donné ta langue au chat." if state.abandon else "Plus de messages !"
        st.info(f"{raison} Le film était « **{film.titre}** » ({film.annee}).", icon="🎬")
        if st.button("Réessayer avec un autre film", icon="🔁", type="primary"):
            executer("niveau", level.id)
            st.rerun()


col_scene, col_chat = st.columns([2, 3], gap="large")
with col_scene, st.container(key="scene"):
    scene = st.empty()
    afficher_scene(scene)

with col_chat:
    if state.erreur:
        st.error(
            "Le gardien ne répond pas : vérifie le fichier .env et que le serveur du LLM tourne. "
            "Ton message n'a pas été compté, tu peux le renvoyer.",
            icon="📡",
        )
        with st.expander("Détail technique"):
            st.code(state.erreur, language=None)

    discussion = st.container(height=460, key="discussion")
    with discussion:
        if not state.chat:
            st.caption("Le gardien a choisi son film. À toi de jouer !")
        for entry in state.chat:
            afficher_entree(entry)

    if finie:
        panneau_fin()
    else:
        suggestions = st.empty()
        if not state.chat:
            with suggestions.container(horizontal=True, gap="small"):
                for texte in interface.SUGGESTIONS.get(level.response_mode, interface.SUGGESTIONS["libre"]):
                    st.button(texte, on_click=envoyer_suggestion, args=(texte,), icon="💡", type="secondary")
        message = st.chat_input(
            "Pose une question au gardien, ou tape le titre du film...",
            key="saisie",
            submit_mode="disable",
        )
        message = message or state.pop("a_envoyer", None)
        if restants == 1:
            st.warning("Dernier message : c'est le moment de proposer ton titre !", icon="⏳")
        if st.button("Je donne ma langue au chat", icon="🏳️", type="tertiary"):
            demander("abandon")
        if message:
            state.erreur = ""
            jouer_tour(message, discussion, scene, suggestions)
