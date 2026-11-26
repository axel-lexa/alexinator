"""Journal SQLite : chaque message et chaque proposition de réponse sont enregistrés.

C'est la matière première de l'évaluation (taux de réussite par niveau, attaques qui marchent,
défense déclenchée, latence...). Lecture dans un notebook :

    import sqlite3, pandas as pd
    df = pd.read_sql("SELECT * FROM tours", sqlite3.connect("data/journal.db"))
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .config import JOURNAL_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS tours (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    horodatage TEXT, session_id TEXT, joueur TEXT, niveau INTEGER, film_id TEXT, modele TEXT,
    message_joueur TEXT,   -- ce que le joueur a écrit
    reponse_brute TEXT,    -- ce que le gardien a répondu avant les défenses (peut contenir une fuite)
    reponse_affichee TEXT, -- ce que le joueur a réellement vu
    bloque_par TEXT,       -- NULL, 'input_detector', 'output_filter', 'censor' ou 'limite'
    detail TEXT,           -- raison donnée par la défense
    latence_ms INTEGER
);
CREATE TABLE IF NOT EXISTS propositions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    horodatage TEXT, session_id TEXT, joueur TEXT, niveau INTEGER, film_id TEXT,
    proposition TEXT, correcte INTEGER, methode TEXT, messages_utilises INTEGER
);
-- Résultats des campagnes automatiques (campagne.py) : une ligne par message testé.
CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    horodatage TEXT, campagne TEXT, modele TEXT, niveau INTEGER, film_id TEXT,
    type TEXT,              -- 'attaque' ou 'honnete'
    categorie TEXT,         -- catégorie d'attaque (ou 'honnete')
    message TEXT, reponse_affichee TEXT,
    bloque_par TEXT,        -- défense déclenchée, sinon NULL
    fuite_titre INTEGER,    -- 1 si le titre est écrit dans la réponse affichée
    fuite_devineur INTEGER, -- 1 si le LLM devineur retrouve le film à partir de la réponse
    proposition_devineur TEXT
);
-- Questions pertinentes posées par les utilisateurs humains (joueur != 'auto:%')
CREATE TABLE IF NOT EXISTS questions_pertinentes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    horodatage TEXT, session_id TEXT, joueur TEXT, niveau INTEGER, film_id TEXT,
    question TEXT UNIQUE,
    reponse_affichee TEXT,
    source TEXT DEFAULT 'playtest'
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Journal:
    def __init__(self, path: Path = JOURNAL_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        # Une connexion par écriture : simple et sans souci avec les threads de Streamlit.
        return sqlite3.connect(self.path)

    def log_turn(
        self,
        session_id: str,
        joueur: str,
        niveau: int,
        film_id: str,
        modele: str,
        message_joueur: str,
        reponse_brute: str | None,
        reponse_affichee: str,
        bloque_par: str | None,
        detail: str,
        latence_ms: int,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO tours (horodatage, session_id, joueur, niveau, film_id, modele, message_joueur,"
                " reponse_brute, reponse_affichee, bloque_par, detail, latence_ms)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    _now(),
                    session_id,
                    joueur,
                    niveau,
                    film_id,
                    modele,
                    message_joueur,
                    reponse_brute,
                    reponse_affichee,
                    bloque_par,
                    detail,
                    latence_ms,
                ),
            )

    def log_guess(
        self,
        session_id: str,
        joueur: str,
        niveau: int,
        film_id: str,
        proposition: str,
        correcte: bool,
        methode: str,
        messages_utilises: int,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO propositions (horodatage, session_id, joueur, niveau, film_id, proposition,"
                " correcte, methode, messages_utilises) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    _now(),
                    session_id,
                    joueur,
                    niveau,
                    film_id,
                    proposition,
                    int(correcte),
                    methode,
                    messages_utilises,
                ),
            )

    def log_eval(
        self,
        campagne: str,
        modele: str,
        niveau: int,
        film_id: str,
        type: str,
        categorie: str,
        message: str,
        reponse_affichee: str,
        bloque_par: str | None,
        fuite_titre: bool,
        fuite_devineur: bool,
        proposition_devineur: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO evaluations (horodatage, campagne, modele, niveau, film_id, type, categorie,"
                " message, reponse_affichee, bloque_par, fuite_titre, fuite_devineur, proposition_devineur)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    _now(),
                    campagne,
                    modele,
                    niveau,
                    film_id,
                    type,
                    categorie,
                    message,
                    reponse_affichee,
                    bloque_par,
                    int(fuite_titre),
                    int(fuite_devineur),
                    proposition_devineur,
                ),
            )

    @staticmethod
    def is_pertinent_question(message: str, blocked_by: str | None = None) -> bool:
        """Détermine si un message d'un joueur est une vraie question pertinente sur le film.

        Critères :
        - non bloqué par une défense (pas une attaque de prompt injection) ;
        - longueur raisonnable (entre 10 et 250 caractères) ;
        - se termine par '?' ou contient une tournure interrogative ;
        - ne ressemble pas à une simple salutation ou une insulte courte.
        """
        if blocked_by is not None:
            return False
        msg = message.strip()
        if len(msg) < 10 or len(msg) > 300:
            return False
        interrogatifs = (
            "est-ce", "est ce", "qui", "quoi", "quel", "quelle", "quels", "quelles",
            "comment", "pourquoi", "où", "ou", "quand", "combien", "peux-tu", "peut-on",
            "raconte", "donne-moi", "donne moi", "dis-moi", "dis moi", "y a-t-il", "y a t il"
        )
        msg_lower = msg.lower()
        has_mark = "?" in msg
        has_keyword = any(msg_lower.startswith(k) or f" {k} " in msg_lower for k in interrogatifs)
        return has_mark or has_keyword

    def save_pertinent_question(
        self,
        session_id: str,
        joueur: str,
        niveau: int,
        film_id: str,
        question: str,
        reponse_affichee: str,
    ) -> bool:
        """Enregistre la question si elle est nouvelle et non vide. Renvoie True si insérée."""
        q = question.strip()
        if not q:
            return False
        try:
            with self._connect() as conn:
                cur = conn.execute(
                    "INSERT OR IGNORE INTO questions_pertinentes "
                    "(horodatage, session_id, joueur, niveau, film_id, question, reponse_affichee) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (_now(), session_id, joueur, niveau, film_id, q, reponse_affichee),
                )
                return cur.rowcount > 0
        except sqlite3.Error:
            return False

    def get_pertinent_questions(self, limit: int | None = None) -> list[str]:
        """Retourne la liste des questions pertinentes uniques posées par les utilisateurs."""
        query = "SELECT question FROM questions_pertinentes ORDER BY id DESC"
        if limit is not None:
            query += f" LIMIT {int(limit)}"
        with self._connect() as conn:
            rows = conn.execute(query).fetchall()
        return [r[0] for r in rows]

