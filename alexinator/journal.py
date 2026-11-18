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


def journal_messages(path: Path = JOURNAL_PATH, max_length: int = 600) -> list[str]:
    """Messages écrits par de vrais joueurs pendant les playtests, sans doublons.

    Les parties des campagnes automatiques (pseudo en `auto:`) sont exclues. Les messages de plus de
    `max_length` caractères sont écartés : ce sont surtout des collages de texte, peu représentatifs.
    Renvoie une liste vide si le journal n'existe pas encore.
    """
    if not Path(path).exists():
        return []
    with sqlite3.connect(path) as conn:
        rows = conn.execute(
            "SELECT DISTINCT message_joueur FROM tours WHERE joueur NOT LIKE 'auto:%'"
        ).fetchall()
    return [r[0] for r in rows if r[0] and len(r[0]) < max_length]
