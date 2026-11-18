"""Accès à l'API de TMDB (The Movie Database), pour compléter la base cinéma data/films.json.

Il faut une clé personnelle de https://www.themoviedb.org/settings/api (la courte « API Key » ou le long
« API Read Access Token »), à copier dans .env sous le nom ALEXINATOR_TMDB_TOKEN.
Aucun paquet externe n'est nécessaire.

    python -m alexinator.tmdb --test                   vérifie que le jeton fonctionne
    python -m alexinator.tmdb "Forrest Gump"           fiche au format de data/films.json
    python -m alexinator.tmdb "Le Parrain" --annee 1972
    python -m alexinator.tmdb --lot 130                ajoute 130 films très connus à data/films.json

Une fiche seule (avec un titre) est un brouillon : ses mots-clés restent en anglais. Le mode --lot
les traduit avec le LLM du .env et ne garde que les fiches qui passent les contrôles des tests.
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable

from .config import get_settings
from .defenses import find_leak, forbidden_terms, output_filter
from .films import Film, load_films, save_films
from .validation import canonical, check_answer, extract_title, is_near_miss, is_partial_title

BASE_URL = "https://api.themoviedb.org/3"
MAX_ACTEURS = 4
MAX_MOTS_CLES = 8  # demandés à la traduction ; au plus MAX_MOTS_CLES_FICHE gardés ensuite
MAX_MOTS_CLES_FICHE = 6
MAX_ALIAS = 4
ALIAS_COUNTRIES = {"FR", "BE", "CA", "US", "GB"}  # « Lokaj » (Pologne) ne doit pas être une bonne réponse
# Rôles qui ne sont pas des personnages (documentaires, caméos, voix off).
NON_PERSONNAGES = {"", "self", "himself", "herself", "lui meme", "elle meme", "narrator", "narrateur"}
# Mots-clés TMDB sans valeur d'indice.
MOTS_CLES_INUTILES = {
    "duringcreditsstinger", "aftercreditsstinger", "woman director", "based on novel or book", "sequel",
    "remake", "live action remake", "3d", "imax", "based on comic", "based on young adult novel",
}  # fmt: skip
# Même phrase que le test test_output_filter_lets_honest_hints_through.
HONEST_HINT = "C'est un film très connu, plutôt émouvant, et oui il est en couleur et il y a de la musique."


class TMDBError(RuntimeError):
    """Erreur lisible par le joueur : jeton absent ou refusé, film introuvable, réseau..."""


class TMDBClient:
    def __init__(self, token: str, language: str = "fr-FR", timeout: float = 10.0):
        if not token:
            raise TMDBError(
                "Aucun jeton TMDB : ajoute ALEXINATOR_TMDB_TOKEN dans .env "
                "(clé de https://www.themoviedb.org/settings/api)."
            )
        self.token = token
        self.language = language
        self.timeout = timeout

    @property
    def is_short_key(self) -> bool:
        """« API Key » (32 caractères) plutôt que le long « API Read Access Token » (avec des points)."""
        return "." not in self.token

    def _get(self, path: str, **params) -> dict:
        headers = {"Accept": "application/json"}
        # TMDB accepte les deux : la clé courte en paramètre, le jeton long dans l'en-tête.
        if self.is_short_key:
            params["api_key"] = self.token
        else:
            headers["Authorization"] = f"Bearer {self.token}"
        query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        url = f"{BASE_URL}{path}" + (f"?{query}" if query else "")
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            messages = {
                401: "Jeton TMDB refusé : vérifie ALEXINATOR_TMDB_TOKEN dans .env (copie complète, "
                "sans espace ni guillemet ; une clé toute neuve peut mettre quelques minutes à marcher).",
                404: f"Introuvable sur TMDB : {path}",
                429: "TMDB limite le nombre de requêtes : réessaie dans quelques secondes.",
            }
            raise TMDBError(messages.get(error.code, f"Erreur TMDB {error.code} sur {path}")) from error
        except urllib.error.URLError as error:
            raise TMDBError(f"TMDB injoignable : {error.reason}") from error

    def check(self) -> bool:
        """Vrai si le jeton est accepté."""
        return bool(self._get("/authentication").get("success"))

    def search(self, query: str, year: int | None = None) -> list[dict]:
        """Films dont le titre correspond, du plus pertinent au moins pertinent."""
        data = self._get("/search/movie", query=query, year=year, language=self.language)
        return data.get("results", [])

    def movie(self, movie_id: int) -> dict:
        """Détails d'un film, avec distribution, mots-clés et titres alternatifs en un seul appel."""
        return self._get(
            f"/movie/{movie_id}",
            language=self.language,
            append_to_response="credits,keywords,alternative_titles",
        )

    def most_voted(self, page: int = 1) -> list[dict]:
        """Les films qui ont le plus de votes sur TMDB : les plus connus du grand public."""
        params = {"sort_by": "vote_count.desc", "include_adult": "false", "language": self.language}
        return self._get("/discover/movie", page=page, **params).get("results", [])


def clean_text(text: str) -> str:
    """TMDB met des espaces insécables avant « : » (« Star Wars : ... ») : on les remplace."""
    return " ".join(text.replace(" ", " ").replace(" ", " ").split())


def is_latin(title: str) -> bool:
    """« 기생충 » (Parasite) ne peut pas être tapé par le joueur, et la validation le réduit à rien."""
    return bool(canonical(title)) and all(
        not c.isalpha() or "LATIN" in unicodedata.name(c, "") for c in title
    )


def to_film_entry(movie: dict) -> dict:
    """Convertit les détails TMDB d'un film en fiche au format de data/films.json."""
    movie = json.loads(clean_text(json.dumps(movie, ensure_ascii=False)))
    titre = movie.get("title") or movie.get("original_title", "")
    original = movie.get("original_title", "")
    credits = movie.get("credits", {})
    cast = sorted(credits.get("cast", []), key=lambda person: person.get("order", 999))[:MAX_ACTEURS]

    personnages = []
    for person in cast:
        # « Forrest Gump / Narrator (voice) » -> « Forrest Gump »
        role = (person.get("character") or "").split("/")[0].split("(")[0].strip()
        if canonical(role) not in NON_PERSONNAGES and role not in personnages:
            personnages.append(role)

    alias = []
    others = [original] + [
        t["title"]
        for t in movie.get("alternative_titles", {}).get("titles", [])
        if t.get("iso_3166_1") in ALIAS_COUNTRIES
    ]
    for title in others:
        known = {canonical(titre)} | {canonical(a) for a in alias}
        # Versions de projection (« The Avengers 3D », « An IMAX 3D Experience ») : pas de vrais titres.
        technical = any(word in canonical(title).split() for word in ("3d", "imax", "experience"))
        usable = is_latin(title) and canonical(title) not in known and not technical
        if usable and len(alias) < MAX_ALIAS:
            alias.append(title)

    keywords = [k["name"] for k in movie.get("keywords", {}).get("keywords", [])]
    return {
        "id": "_".join(canonical(titre).split()[:5]),
        "titre": titre,
        "alias": alias,
        "annee": int(movie["release_date"][:4]) if movie.get("release_date") else 0,
        "realisateurs": [p["name"] for p in credits.get("crew", []) if p.get("job") == "Director"],
        "acteurs": [person["name"] for person in cast],
        "personnages": personnages,
        "genre": ", ".join(g["name"].lower() for g in movie.get("genres", [])),
        "mots_cles": [k for k in keywords if k.lower() not in MOTS_CLES_INUTILES][:MAX_MOTS_CLES],
        **popularity(movie),
    }


def popularity(movie: dict) -> dict:
    """Note, nombre de votes et langue : les champs qui décident si un film est jouable."""
    return {
        "note": round(float(movie.get("vote_average") or 0), 1),
        "votes": int(movie.get("vote_count") or 0),
        "langue": movie.get("original_language", ""),
    }


def find_movie(client: TMDBClient, film: Film) -> dict | None:
    """Retrouve le film sur TMDB : un de ses titres et son année (à un an près) doivent correspondre."""

    def key(title: str) -> str:  # « Astérix & Obélix » = « Astérix et Obélix »
        return " ".join(w for w in canonical(title).split() if w not in {"et", "and"})

    names = {key(t) for t in film.titres_acceptes}
    years = {str(film.annee + delta) for delta in (-1, 0, 1)}

    def matches(result: dict) -> bool:
        titles = {key(result.get("title", "")), key(result.get("original_title", ""))}
        return bool(titles & names) and (result.get("release_date") or "")[:4] in years

    for query in film.titres_acceptes:  # titre, puis alias (« Nightcall » est « Nightcrawler » sur TMDB)
        for year in (film.annee, None):
            found = [r for r in client.search(query, year=year) if matches(r)]
            if found:
                return max(found, key=lambda r: r.get("vote_count", 0))
    return None


# ----------------------------------------------------------------------------------------
# Mode --lot : ajouter d'un coup des films très connus, traduits et contrôlés
# ----------------------------------------------------------------------------------------
TRANSLATE_PROMPT = (
    "Traduis en français ces mots-clés qui décrivent un film. Garde-les courts (quelques mots), en "
    "minuscules sauf les noms propres. Réponds uniquement par une liste JSON de {n} chaînes, dans le "
    "même ordre, sans rien d'autre.\n{keywords}"
)


def translate_keywords(keywords: list[str], llm) -> list[str] | None:
    """Traduit les mots-clés avec le LLM ; None si la réponse reste inutilisable après deux essais."""
    if not keywords:
        return []
    prompt = TRANSLATE_PROMPT.format(n=len(keywords), keywords=json.dumps(keywords, ensure_ascii=False))
    for _ in range(2):
        reply = llm.chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=400)
        try:
            items = json.loads(reply[reply.find("[") : reply.rfind("]") + 1])
        except ValueError:
            continue
        if isinstance(items, list) and len(items) == len(keywords) and all(isinstance(i, str) for i in items):
            return [i.strip() for i in items]
    return None


def _same_film(a: Film, b: Film) -> bool:
    return any(check_answer(t, b).correct for t in a.titres_acceptes) or any(
        check_answer(t, a).correct for t in b.titres_acceptes
    )


def _check_film(entry: dict, films: list[Film], questions: list[str], llm) -> tuple[Film | None, str]:
    """Construit la fiche définitive, ou explique pourquoi le film est écarté."""
    if not entry["annee"] or not entry["realisateurs"] or not entry["acteurs"]:
        return None, "fiche incomplète"
    main = Film(**{**entry, "alias": []})
    if any(_same_film(main, other) for other in films):
        return None, "déjà dans la base (ou titre trop proche d'un film de la base)"
    # Un alias qui valide un autre film (« Le Seigneur des anneaux » pour toute la saga) est retiré.
    entry["alias"] = [
        a for a in entry["alias"] if not any(_same_film(Film(**{**entry, "alias": [a]}), o) for o in films)
    ]

    translated = translate_keywords(entry["mots_cles"], llm)
    if translated is None:
        return None, "traduction des mots-clés impossible"
    film = Film(**{**entry, "mots_cles": []})
    terms = forbidden_terms(film, include_entities=True)
    # Un mot-clé qui contient un titre ou un nom serait bloqué par le filtre : ce n'est plus un indice.
    entry["mots_cles"] = [k for k in translated if k and not find_leak(k, terms)][:MAX_MOTS_CLES_FICHE]
    if len(entry["mots_cles"]) < 2:
        return None, "pas assez de mots-clés utilisables"

    ids = {f.id for f in films}
    if entry["id"] in ids or not entry["id"]:
        entry["id"] = f"{entry['id']}_{entry['annee']}".strip("_")
    film = Film(**entry)
    if output_filter(HONEST_HINT, film).blocked:
        return None, "un titre ou un nom bloquerait des indices honnêtes"
    if not output_filter(f"Bravo, c'est {film.titre}.", film).blocked:
        return None, "titre trop court pour que le filtre du niveau 3 le protège"
    for question in questions:
        candidate = extract_title(question)
        if (
            check_answer(candidate, film).correct
            or is_partial_title(candidate, film)
            or is_near_miss(candidate, film)
        ):
            return None, f"une question honnête ou une attaque serait prise pour ce titre : {question!r}"
    return film, ""


def build_batch(client: TMDBClient, llm, films: list[Film], count: int, log: Callable = print) -> list[Film]:
    """Ajoute jusqu'à `count` films parmi les plus votés de TMDB, en sautant ceux qui posent problème."""
    from .redteam import load_attacks, load_honest_questions

    questions = load_honest_questions() + [a["texte"] for a in load_attacks()]
    films, added, page = list(films), [], 1
    while len(added) < count:
        results = client.most_voted(page)
        if not results:
            break
        page += 1
        for summary in results:
            if len(added) >= count:
                break
            film, reason = _check_film(to_film_entry(client.movie(summary["id"])), films, questions, llm)
            if film is None:
                log(f"  écarté  {summary.get('title')} : {reason}")
                continue
            films.append(film)
            added.append(film)
            log(f"[{len(added)}/{count}] {film.titre} ({film.annee}) : {', '.join(film.mots_cles)}")
    return added


def main() -> None:
    parser = argparse.ArgumentParser(description="Fiches de films depuis TMDB")
    parser.add_argument("titre", nargs="?", help="titre du film à chercher")
    parser.add_argument("--annee", type=int, help="année de sortie, pour départager les homonymes")
    parser.add_argument("--test", action="store_true", help="vérifie seulement le jeton")
    parser.add_argument("--lot", type=int, help="ajoute N films très connus à data/films.json")
    parser.add_argument("--notes", action="store_true", help="note et votes des fiches qui n'en ont pas")
    args = parser.parse_args()
    if not args.test and not args.titre and not args.lot and not args.notes:
        parser.error("donne un titre de film, --test, --lot N ou --notes")

    try:
        settings = get_settings()
        client = TMDBClient(settings.tmdb_token)
        if args.test:
            print("Connexion à TMDB réussie." if client.check() else "TMDB a refusé le jeton.")
            return
        if args.lot:
            from .llm import make_llm

            films = load_films()
            print(f"{len(films)} films dans la base, traduction avec {settings.model}.", flush=True)
            log = lambda message: print(message, flush=True)  # noqa: E731
            added = build_batch(client, make_llm(settings), films, args.lot, log=log)
            save_films(films + added)
            print(f"{len(added)} films ajoutés : {len(films) + len(added)} dans data/films.json.")
            return
        if args.notes:
            films, missing = load_films(), []
            for i, film in enumerate(films):
                if film.votes:
                    continue
                movie = find_movie(client, film)
                if movie is None:
                    missing.append(film.titre)
                    continue
                films[i] = new = Film(**{**vars(film), **popularity(movie)})
                print(f"  {new.titre} ({new.annee}) : {new.note}/10, {new.votes} votes", flush=True)
            save_films(films)
            print(f"{sum(f.jouable for f in films)} films jouables sur {len(films)}.")
            if missing:
                print("Introuvables (vérifier titre et année) : " + "; ".join(missing), file=sys.stderr)
            return
        results = client.search(args.titre, year=args.annee)
        if not results:
            raise TMDBError(f"Aucun film trouvé pour « {args.titre} ».")
        entry = to_film_entry(client.movie(results[0]["id"]))
    except TMDBError as error:
        sys.exit(str(error))

    print(json.dumps(entry, ensure_ascii=False, indent=2))
    others = [f"{r.get('title')} ({(r.get('release_date') or '????')[:4]})" for r in results[1:4]]
    if others:
        print(f"\nAutres résultats : {', '.join(others)}  (préciser avec --annee)", file=sys.stderr)


if __name__ == "__main__":
    main()
