"""Alexinator dans le terminal : pratique pour tester vite un modèle ou un prompt.

    python cli.py --niveau 2
    python cli.py --niveau 3 --film titanic     (film imposé, pour comparer des modèles)

Dans la partie : pose tes questions ou tape directement le titre du film quand tu l'as trouvé.
/r <titre> force une proposition (avec juge LLM),  /brut montre la dernière réponse non filtrée
(triche de développeur),  /q pour quitter.
"""
import argparse

from alexinator.config import get_settings
from alexinator.engine import GameSession, load_levels
from alexinator.films import load_films, pick_film
from alexinator.journal import Journal
from alexinator.llm import make_llm


def main() -> None:
    parser = argparse.ArgumentParser(description="Alexinator en ligne de commande")
    parser.add_argument("--niveau", type=int, default=1)
    parser.add_argument("--film", help="id d'un film de data/films.json (sinon tirage au hasard)")
    parser.add_argument("--joueur", default="cli")
    args = parser.parse_args()

    settings = get_settings()
    levels = {lv.id: lv for lv in load_levels()}
    films = load_films()
    film = next((f for f in films if f.id == args.film), None) if args.film else pick_film(films)
    if film is None:
        raise SystemExit(f"Film inconnu : {args.film}")
    if args.niveau not in levels:
        raise SystemExit(f"Niveau inconnu : {args.niveau} (disponibles : {sorted(levels)})")

    game = GameSession(
        levels[args.niveau], film,
        llm=make_llm(settings),
        aux_llm=make_llm(settings, settings.aux_model),
        journal=Journal(), player=args.joueur, catalogue=films,
    )
    print(f"Niveau {game.level.id} : {game.level.name} (modèle : {settings.model})")
    print(game.level.description)
    print("Pose tes questions, ou tape le titre quand tu l'as trouvé. Commandes : /r <titre>, /brut, /q.\n")

    last = None
    while not game.solved:
        text = input(f"[{game.messages_left} messages] Toi > ").strip()
        if not text:
            continue
        if text == "/q":
            break
        if text == "/brut":
            print(f"(brut) {last.raw_reply if last else 'rien pour l instant'}\n")
        elif text.startswith("/r "):
            if game.guess(text[3:]).correct:
                print(f"Bravo ! C'était « {film.titre} » ({film.annee}).")
            else:
                print("Ce n'est pas ça.\n")
        else:
            result = game.play(text)
            if result.guess is None:
                last = result
            tag = f" [bloqué par {result.blocked_by}]" if result.blocked_by else ""
            print(f"Alexinator{tag} > {result.reply}\n")


if __name__ == "__main__":
    main()
