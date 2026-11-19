"""Construit data/films.json depuis un export Kaggle/TMDB et IMDb.

Les fichiers sources restent locaux dans data/raw/. Les formats Kaggle varient,
donc le script accepte les colonnes usuelles par ordre de préférence.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import unicodedata
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alexinator.defenses import output_filter  # noqa: E402
from alexinator.films import Film, load_films, save_films  # noqa: E402
from alexinator.tmdb import TMDBClient, TMDBError, to_film_entry  # noqa: E402
from alexinator.validation import canonical, check_answer  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
FILMS_PATH = ROOT / "data" / "films.json"
HONEST_QUESTION = "Ce film parle-t-il d'amour, d'aventure ou de famille ?"


def first_value(row: dict[str, str], names: Iterable[str]) -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def parse_year(row: dict[str, str]) -> int:
    for name in ("release_date", "year", "startYear"):
        value = first_value(row, (name,))
        if value:
            try:
                return int(value[:4])
            except ValueError:
                continue
    return 0


def popularity_score(row: dict[str, str]) -> float:
    for name in ("vote_count", "popularity", "numVotes", "averageRating"):
        value = first_value(row, (name,))
        if value:
            try:
                return float(value)
            except ValueError:
                continue
    return 0.0


def parse_list(value: str) -> list[str]:
    if not value:
        return []
    try:
        items = json.loads(value)
        if isinstance(items, list):
            return [item.get("name", "") if isinstance(item, dict) else str(item) for item in items if item]
    except json.JSONDecodeError:
        pass
    return [item.strip().strip("'\"") for item in value.replace("|", ",").split(",") if item.strip()]


def is_latin(title: str) -> bool:
    return bool(canonical(title)) and all(
        not character.isalpha() or "LATIN" in unicodedata.name(character, "") for character in title
    )


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    csv.field_size_limit(2**31 - 1)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_french_aliases(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    csv.field_size_limit(2**31 - 1)
    aliases: dict[str, list[str]] = {}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row.get("region") != "FR":
                continue
            imdb_id = row.get("titleId", "")
            title = (row.get("title") or "").strip()
            if imdb_id and title and canonical(title):
                aliases.setdefault(imdb_id, [])
                if title not in aliases[imdb_id]:
                    aliases[imdb_id].append(title)
    return aliases


def make_film(row: dict[str, str], aliases: dict[str, list[str]]) -> Film | None:
    title = first_value(row, ("title", "primaryTitle", "original_title"))
    year = parse_year(row)
    imdb_id = first_value(row, ("imdb_id", "tconst"))
    if not title or not year or not is_latin(title):
        return None

    original = first_value(row, ("original_title", "originalTitle"))
    title_aliases = aliases.get(imdb_id, [])
    if original and canonical(original) != canonical(title):
        title_aliases.insert(0, original)
    unique_aliases = [alias for alias in dict.fromkeys(title_aliases) if is_latin(alias)]

    genres = parse_list(first_value(row, ("genres", "genre")))
    return Film(
        id="_".join(canonical(title).split()[:5]),
        titre=title,
        alias=unique_aliases[:8],
        annee=year,
        realisateurs=parse_list(first_value(row, ("director", "directors"))),
        acteurs=parse_list(first_value(row, ("cast", "actors"))),
        personnages=[],
        genre=", ".join(genres).lower(),
        mots_cles=[],
    )


def same_film(candidate: Film, existing: Iterable[Film]) -> bool:
    existing = list(existing)
    for title in candidate.titres_acceptes:
        if any(check_answer(title, other).correct for other in existing):
            return True
    return any(
        check_answer(title, candidate).correct for other in existing for title in other.titres_acceptes
    )


def build_films(
    csv_path: Path,
    akas_path: Path,
    limit: int,
    min_popularity: float,
) -> tuple[list[Film], dict]:
    existing = load_films(FILMS_PATH)
    aliases = load_french_aliases(akas_path)
    added: list[Film] = []
    rejected = {"missing": 0, "not_popular": 0, "duplicate": 0, "unsafe": 0}

    rows = load_csv(csv_path)
    rows.sort(key=popularity_score, reverse=True)
    ids = {film.id for film in existing}

    for row in rows:
        if len(added) >= limit:
            break
        score = popularity_score(row)
        if score < min_popularity:
            rejected["not_popular"] += 1
            continue
        film = make_film(row, aliases)
        if film is None:
            rejected["missing"] += 1
            continue
        if same_film(film, [*existing, *added]):
            rejected["duplicate"] += 1
            continue
        if output_filter(HONEST_QUESTION, film).blocked:
            rejected["unsafe"] += 1
            continue
        if not output_filter(f"Bravo, c'est {film.titre}.", film).blocked:
            rejected["unsafe"] += 1
            continue
        if film.id in ids:
            film.id = f"{film.id}_{film.annee}"
        ids.add(film.id)
        added.append(film)

    films = [*existing, *added]
    report = {
        "existing": len(existing),
        "added": len(added),
        "total": len(films),
        "aliases_fr": sum(len(film.alias) for film in added),
        "rejected": rejected,
    }
    return films, report


def enrich_with_tmdb(films: list[Film], token: str, log=print) -> dict:
    """Remplace les métadonnées par leur version française depuis TMDB."""
    client = TMDBClient(token, language="fr-FR")
    if not client.check():
        raise TMDBError("Le jeton TMDB a été refusé.")

    enriched = 0
    missing = 0
    no_resume = 0
    for index, film in enumerate(films, start=1):
        try:
            results = client.search(film.titre, year=film.annee)
        except TMDBError:
            missing += 1
            continue
        if not results:
            missing += 1
            continue
        try:
            entry = to_film_entry(client.movie(results[0]["id"]))
        except TMDBError:
            missing += 1
            continue

        aliases = list(dict.fromkeys([*film.alias, *entry["alias"]]))
        aliases = [alias for alias in aliases if is_latin(alias)]
        film.titre = entry["titre"] or film.titre
        film.alias = aliases
        film.realisateurs = entry["realisateurs"]
        film.acteurs = entry["acteurs"]
        film.personnages = entry["personnages"]
        film.genre = entry["genre"]
        film.mots_cles = entry["mots_cles"]
        film.resume = entry["resume"]
        if film.resume:
            enriched += 1
        else:
            no_resume += 1
        log(f"[{index}/{len(films)}] {film.titre}")

    return {"enriched": enriched, "missing": missing, "no_resume": no_resume}


def main() -> None:
    parser = argparse.ArgumentParser(description="Construire data/films.json depuis Kaggle et IMDb")
    parser.add_argument("--csv", type=Path, default=RAW_DIR / "tmdb.csv", help="CSV Kaggle")
    parser.add_argument("--akas", type=Path, default=RAW_DIR / "title.akas.tsv.gz", help="IMDb AKAS")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--min-popularity", type=float, default=1000.0)
    parser.add_argument("--report", action="store_true", help="affiche le rapport JSON")
    parser.add_argument("--tmdb-enrich", action="store_true", help="localise toutes les fiches avec TMDB")
    args = parser.parse_args()

    films, report = build_films(args.csv, args.akas, args.limit, args.min_popularity)
    if args.tmdb_enrich:
        from alexinator.config import get_settings

        settings = get_settings()
        if not settings.tmdb_token:
            raise TMDBError("Aucun jeton TMDB : ajoute ALEXINATOR_TMDB_TOKEN dans .env.")
        report["tmdb"] = enrich_with_tmdb(films, settings.tmdb_token)
    save_films(films, FILMS_PATH)
    if args.report:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"{report['added']} films ajoutés : {report['total']} fiches au total.")


if __name__ == "__main__":
    main()
