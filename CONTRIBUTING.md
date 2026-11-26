# Travailler à plusieurs sur Alexinator

## Les chantiers

| Chantier | Fichiers principaux |
| --- | --- |
| A. Moteur et gardien | `alexinator/engine.py`, `alexinator/llm.py`, `levels/*.yaml` |
| B. Défenses (blue team) | `alexinator/defenses.py` |
| C. Contenu cinéma | `data/films.json` |
| D. Interface et déploiement | `app.py`, `vues/`, `alexinator/interface.py`, `.streamlit/`, `assets/style.css`, `cli.py` |
| E. Red team et évaluation | `alexinator/redteam.py`, `campagne.py`, `analyse.ipynb`, `data/attaques.json` |
| Entraînement | `finetune/` |

## Façon de travailler

1. Ne jamais travailler directement sur `main`. Créer une branche par tâche :
   `git switch -c defenses/censeur-plus-strict`.
2. Lancer les tests avant chaque commit : `python -m pytest`. Ils tournent sans LLM, en moins d'une seconde.
   Vérifier aussi le style avec `ruff check .` et `ruff format --check .` (configuration dans `pyproject.toml`).
3. Ajouter un test pour chaque nouvelle fonctionnalité dans `tests/test_alexinator.py`, avec `MockLLM`
   pour remplacer le vrai modèle.
4. Ouvrir une pull request et la faire relire par quelqu'un d'un autre chantier.

## Règles

- Ne jamais committer le fichier `.env` (il contient les clés d'API) ni `data/journal.db` (données des joueurs).
  Les deux sont dans `.gitignore`.
- Les messages enregistrés pendant les playtests sont des données personnelles : prévenir les joueurs,
  utiliser des pseudos, ne pas publier le journal brut.
- Un changement de prompt peut casser un niveau sans casser un test : rejouer le niveau concerné
  (`python cli.py --niveau N`) et, si possible, relancer une petite campagne.
- Après une modification de `data/films.json`, lancer les tests : ils vérifient qu'aucun titre n'en valide
  un autre et que le filtre bloque chaque titre.
