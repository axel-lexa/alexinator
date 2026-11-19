"""Fusion de plusieurs fichiers journal.db dans le journal principal.

Chaque collègue envoie son fichier data/journal.db (par Drive, Discord…).
Ce script importe toutes les lignes manquantes sans créer de doublons.

Usage :
    python tools/fusionner_journaux.py journal_alice.db journal_bob.db ...
    python tools/fusionner_journaux.py journal_alice.db -o data/journal_fusionne.db
    python tools/fusionner_journaux.py journal_alice.db --stats-only

Les fichiers sources ne sont jamais modifiés.
La base de destination est data/journal.db par défaut.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEST = ROOT / "data" / "journal.db"

# ---------------------------------------------------------------------------
# Schéma minimal (repris de alexinator/journal.py)
# ---------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS tours (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    horodatage TEXT, session_id TEXT, joueur TEXT, niveau INTEGER, film_id TEXT, modele TEXT,
    message_joueur TEXT,
    reponse_brute TEXT,
    reponse_affichee TEXT,
    bloque_par TEXT,
    detail TEXT,
    latence_ms INTEGER
);
CREATE TABLE IF NOT EXISTS propositions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    horodatage TEXT, session_id TEXT, joueur TEXT, niveau INTEGER, film_id TEXT,
    proposition TEXT, correcte INTEGER, methode TEXT, messages_utilises INTEGER
);
CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    horodatage TEXT, campagne TEXT, modele TEXT, niveau INTEGER, film_id TEXT,
    type TEXT, categorie TEXT,
    message TEXT, reponse_affichee TEXT,
    bloque_par TEXT,
    fuite_titre INTEGER,
    fuite_devineur INTEGER,
    proposition_devineur TEXT
);
CREATE TABLE IF NOT EXISTS questions_pertinentes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    horodatage TEXT, session_id TEXT, joueur TEXT, niveau INTEGER, film_id TEXT,
    question TEXT UNIQUE,
    reponse_affichee TEXT,
    source TEXT DEFAULT 'playtest'
);
"""

# ---------------------------------------------------------------------------
# Clés de déduplication par table
# Pour chaque table, on définit quels champs identifient une ligne unique.
# ---------------------------------------------------------------------------
DEDUP_KEYS: dict[str, list[str]] = {
    "tours": ["session_id", "horodatage", "message_joueur"],
    "propositions": ["session_id", "horodatage", "proposition"],
    "evaluations": ["horodatage", "campagne", "niveau", "film_id", "message"],
    "questions_pertinentes": ["question"],
}

# Colonnes dans l'ordre d'insertion (sans l'id auto-incrémenté)
TABLE_COLUMNS: dict[str, list[str]] = {
    "tours": [
        "horodatage", "session_id", "joueur", "niveau", "film_id", "modele",
        "message_joueur", "reponse_brute", "reponse_affichee", "bloque_par", "detail", "latence_ms",
    ],
    "propositions": [
        "horodatage", "session_id", "joueur", "niveau", "film_id",
        "proposition", "correcte", "methode", "messages_utilises",
    ],
    "evaluations": [
        "horodatage", "campagne", "modele", "niveau", "film_id", "type", "categorie",
        "message", "reponse_affichee", "bloque_par", "fuite_titre", "fuite_devineur", "proposition_devineur",
    ],
    "questions_pertinentes": [
        "horodatage", "session_id", "joueur", "niveau", "film_id",
        "question", "reponse_affichee", "source",
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _connect(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(path)


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def _existing_keys(conn: sqlite3.Connection, table: str, keys: list[str]) -> set[tuple]:
    """Retourne un set de tuples (val_key1, val_key2, ...) déjà présents dans la table."""
    cols = ", ".join(keys)
    rows = conn.execute(f"SELECT {cols} FROM {table}").fetchall()
    return set(rows)


def _get_rows(conn: sqlite3.Connection, table: str, columns: list[str]) -> list[tuple]:
    """Lit toutes les lignes de la table source, dans l'ordre des colonnes."""
    # Certaines sources peuvent ne pas avoir toutes les colonnes (ancienne version du schéma)
    existing_cols = {
        row[1]
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    safe_cols = []
    for col in columns:
        if col in existing_cols:
            safe_cols.append(col)
        else:
            safe_cols.append(f"NULL AS {col}")  # colonne manquante → NULL
    select = ", ".join(safe_cols)
    return conn.execute(f"SELECT {select} FROM {table}").fetchall()


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return bool(row and row[0])


# ---------------------------------------------------------------------------
# Fusion d'une source dans la destination
# ---------------------------------------------------------------------------
def merge_source(
    dest_conn: sqlite3.Connection,
    src_path: Path,
    stats_only: bool = False,
) -> dict[str, int]:
    """Importe les lignes nouvelles du fichier source dans la destination.

    Renvoie un dict {table: n_lignes_ajoutées}.
    """
    counts: dict[str, int] = {}
    try:
        src_conn = _connect(src_path)
    except sqlite3.Error as e:
        print(f"  ✗ Impossible d'ouvrir {src_path} : {e}", file=sys.stderr)
        return counts

    for table, keys in DEDUP_KEYS.items():
        if not _table_exists(src_conn, table):
            counts[table] = 0
            continue

        columns = TABLE_COLUMNS[table]
        src_rows = _get_rows(src_conn, table, columns)
        if not src_rows:
            counts[table] = 0
            continue

        # Index des colonnes-clés dans le tuple
        key_indices = [columns.index(k) for k in keys if k in columns]
        existing = _existing_keys(dest_conn, table, [columns[i] for i in key_indices])

        new_rows = [
            row for row in src_rows
            if tuple(row[i] for i in key_indices) not in existing
        ]

        counts[table] = len(new_rows)

        if not stats_only and new_rows:
            placeholders = ", ".join(["?"] * len(columns))
            cols_str = ", ".join(columns)
            dest_conn.executemany(
                f"INSERT OR IGNORE INTO {table} ({cols_str}) VALUES ({placeholders})",
                new_rows,
            )

    src_conn.close()
    return counts


# ---------------------------------------------------------------------------
# Affichage du rapport
# ---------------------------------------------------------------------------
def print_report(src_path: Path, counts: dict[str, int], stats_only: bool) -> None:
    total = sum(counts.values())
    action = "trouverait" if stats_only else "ajouté"
    print(f"\n  📂 {src_path.name}")
    if not counts:
        print("    ✗ Fichier illisible ou vide.")
        return
    for table, n in counts.items():
        icon = "+" if n > 0 else "·"
        print(f"    {icon} {table:<25} {n:>5} ligne(s) {action}(s)")
    print(f"    {'TOTAL':<25} {total:>5} ligne(s)")


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fusionne des fichiers journal.db dans la base principale.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Exemples :
  python tools/fusionner_journaux.py journal_alice.db journal_bob.db
  python tools/fusionner_journaux.py journal_alice.db -o data/journal_fusionne.db
  python tools/fusionner_journaux.py journal_alice.db --stats-only
        """,
    )
    parser.add_argument(
        "sources",
        nargs="+",
        type=Path,
        help="Fichiers .db à importer (chemin absolu ou relatif au répertoire courant).",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Base de destination (défaut : {DEFAULT_DEST}).",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Affiche combien de lignes seraient ajoutées sans rien modifier.",
    )
    args = parser.parse_args()

    dest_path: Path = args.output
    sources: list[Path] = args.sources
    stats_only: bool = args.stats_only

    # Vérification des fichiers sources
    missing = [s for s in sources if not s.exists()]
    if missing:
        for s in missing:
            print(f"✗ Fichier introuvable : {s}", file=sys.stderr)
        sys.exit(1)

    # Ouverture / création de la base destination
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_conn = _connect(dest_path)
    _ensure_schema(dest_conn)

    action_label = "Simulation" if stats_only else "Fusion"
    print(f"{'='*60}")
    print(f"  {action_label} vers : {dest_path}")
    print(f"  Sources : {len(sources)} fichier(s)")
    print(f"{'='*60}")

    grand_total: dict[str, int] = {t: 0 for t in DEDUP_KEYS}

    for src_path in sources:
        counts = merge_source(dest_conn, src_path, stats_only=stats_only)
        print_report(src_path, counts, stats_only=stats_only)
        for table, n in counts.items():
            grand_total[table] += n

    if not stats_only:
        dest_conn.commit()

    dest_conn.close()

    # Résumé global
    total_all = sum(grand_total.values())
    print(f"\n{'='*60}")
    print(f"  {'BILAN TOTAL':}")
    for table, n in grand_total.items():
        if n:
            print(f"    + {table:<25} {n:>5} ligne(s)")
    verb = "seraient ajoutées" if stats_only else "ajoutées"
    print(f"\n  → {total_all} ligne(s) {verb} au total.")
    if stats_only:
        print("  (mode simulation : aucune modification effectuée)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
