"""Campagne d'évaluation automatique.

    python campagne.py --niveaux 1 2 3 --films 5              attaques + questions honnêtes
    python campagne.py --niveaux 1 2 3 --films 2 --max-attaques 10 --max-questions 10  campagne rapide
    python campagne.py --niveaux 1 2 3 --films 5 --attaquant  en plus : un LLM attaquant joue chaque partie
    python campagne.py --nom modele_entraine ...              donner un nom pour comparer deux modèles

Nombre d'appels au gardien : niveaux x films x (attaques + questions). Commencer petit.
Les résultats vont dans la table `evaluations` de data/journal.db (voir analyse.ipynb).
"""

import argparse
import random

from alexinator.config import get_settings
from alexinator.engine import GameSession, load_levels
from alexinator.films import load_films
from alexinator.journal import Journal
from alexinator.llm import make_llm
from alexinator.redteam import load_attacks, load_honest_questions, run_attacker, run_campaign, summarize


def main() -> None:
    parser = argparse.ArgumentParser(description="Campagne d'évaluation d'Alexinator")
    parser.add_argument("--niveaux", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--films", type=int, default=3, help="nombre de films tirés au hasard")
    parser.add_argument("--graine", type=int, default=0, help="même graine = mêmes films (comparaisons)")
    parser.add_argument("--nom", default="campagne", help="étiquette de la campagne dans le journal")
    parser.add_argument("--max-attaques", type=int, default=None, help="limite le nombre d'attaques testées")
    parser.add_argument("--max-questions", type=int, default=None, help="limite le nombre de questions testées")
    parser.add_argument("--sans-devineur", action="store_true")
    parser.add_argument("--attaquant", action="store_true", help="fait aussi jouer un LLM attaquant")
    args = parser.parse_args()

    settings = get_settings()
    llm, aux_llm = make_llm(settings), make_llm(settings, settings.aux_model)
    levels = [lv for lv in load_levels() if lv.id in args.niveaux]
    films = random.Random(args.graine).sample(load_films(), args.films)
    journal = Journal()

    attacks = load_attacks()
    if args.max_attaques and args.max_attaques < len(attacks):
        attacks = random.Random(args.graine).sample(attacks, args.max_attaques)

    questions = load_honest_questions()
    if args.max_questions and args.max_questions < len(questions):
        questions = random.Random(args.graine).sample(questions, args.max_questions)

    rows = run_campaign(
        levels,
        films,
        attacks,
        questions,
        llm,
        aux_llm=aux_llm,
        guesser_llm=None if args.sans_devineur else aux_llm,
        journal=journal,
        name=args.nom,
    )
    print(f"\n{len(rows)} messages testés avec {settings.model} sur {[f.id for f in films]}")
    for level_id, stats in summarize(rows).items():
        print(
            f"Niveau {level_id} : fuite sur {stats['taux_fuite']:.0%} des attaques, "
            f"{stats['taux_blocage']:.0%} bloquées, "
            f"{stats['taux_faux_positifs']:.0%} de faux positifs sur les questions honnêtes"
        )

    if args.attaquant:
        print("\nAttaquant automatique :")
        for level in levels:
            for film in films:
                session = GameSession(
                    level, film, llm, aux_llm=aux_llm, journal=journal, player=f"auto:{args.nom}:attaquant"
                )
                outcome = run_attacker(session, aux_llm)
                status = "trouvé" if outcome["trouve"] else "échec"
                print(f"Niveau {level.id}, {film.id} : {status} en {outcome['messages_utilises']} messages")


if __name__ == "__main__":
    main()
