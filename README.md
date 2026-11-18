# Alexinator

<p align="center">
  <img src="assets/affiche_reflechit.png" alt="Affiche : Alexinator, il lit dans vos pensées" width="300">
  <img src="assets/affiche_solution.png" alt="Affiche : et il ne se trompe jamais" width="300">
</p>

Un LLM « gardien » cache un film. Le joueur discute avec lui pour le deviner, en jouant le jeu
ou en le piratant. Chaque niveau ajoute une défense.

Projet de groupe du cours *Deep Learning et LLM* (IMT, CI3). Le jeu sert aussi de banc d'essai pour
mesurer quelles attaques par prompt battent quelles défenses, et se termine par l'entraînement de notre
propre gardien.

## Les niveaux

| Niveau | Nom | Ce qui protège le secret |
| --- | --- | --- |
| 1 | Le Bavard | Une simple consigne : « ne dis pas le titre » |
| 2 | Le Discret | Prompt système renforcé (aucun nom, aucune ruse acceptée) |
| 3 | Le Filtré | + filtre de sortie en Python (titre, alias, noms de la fiche) |
| 4 | Le Censeur | + un 2e LLM relit chaque réponse |
| 5 | Le Méfiant | + un détecteur d'attaques examine chaque message du joueur |
| 6 | Le Mime | Le gardien ne peut répondre que oui, non ou je ne sais pas |
| 7 | Le Boss | Le gardien ne connaît pas le titre : il a 3 vérifications et toutes les défenses sont actives |

## Documentation

| Document | Contenu |
| --- | --- |
| [docs/architecture.md](docs/architecture.md) | Trajet d'un message, modules, format des niveaux, des fiches et du journal |
| [docs/evaluation.md](docs/evaluation.md) | Mesures, catégories d'attaques, campagnes automatiques |
| [docs/entrainement.md](docs/entrainement.md) | Choix du jeu de données, entraînement QLoRA, évaluation du modèle entraîné |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Chantiers, branches, tests, règles sur les données |

## État du développement

| Élément | État |
| --- | --- |
| Niveaux 1 à 7 | Écrits |
| Validation des réponses, journal SQLite | Écrits et testés |
| Red team : campagne d'attaques, devineur, attaquant automatique (`campagne.py`) | Écrits et testés avec un faux LLM |
| Analyse (`analyse.ipynb`) | Écrit, exécuté sur un journal factice |
| Fabrication du jeu de données (`finetune/construire_dataset.py`) | Écrit et testé avec un faux LLM |
| Entraînement (`finetune/entrainement_qlora.ipynb`) | Écrit, **jamais exécuté** (pas de GPU à la rédaction) |
| Interface Streamlit (avec les images du magicien), mode terminal, backend local | Écrits, **jamais lancés** |
| Base cinéma | 287 fiches : 200 dans la base commune (dont 40 % de films français), 87 générées depuis TMDB (mots-clés relus) |

Les 36 tests automatiques passent, mais ils utilisent un faux LLM : aucun prompt n'a encore été
essayé sur un vrai modèle. C'est le travail de la phase 0.

## Installation

```
python -m venv .venv
.venv\Scripts\activate          (Windows)   ou   source .venv/bin/activate   (macOS, Linux)
pip install -r requirements.txt
```

Puis choisir un LLM :

- **en local avec Ollama** : installer Ollama, télécharger un modèle (`ollama pull llama3.1:8b`),
  laisser Ollama tourner. Les réglages par défaut conviennent.
- **avec une API en ligne** : copier `.env.example` en `.env` et renseigner l'adresse, la clé et le modèle.

## Lancer

```
python -m pytest                              les tests (sans LLM)
python cli.py --niveau 1                      jouer dans le terminal
streamlit run app.py                          jouer dans le navigateur
python campagne.py --niveaux 1 2 3 --films 3  campagne d'évaluation automatique
jupyter notebook analyse.ipynb                chiffres et graphiques du rapport
python -m alexinator.tmdb "Forrest Gump"      brouillon de fiche depuis TMDB (voir ci-dessous)
```

### Compléter la base cinéma avec TMDB

1. Créer un compte gratuit sur [themoviedb.org](https://www.themoviedb.org/signup), puis demander une clé
   d'API dans [Paramètres → API](https://www.themoviedb.org/settings/api) (usage personnel / éducatif).
2. Copier la clé dans `.env` : `ALEXINATOR_TMDB_TOKEN=...`. La courte « API Key » comme le long
   « API Read Access Token » conviennent.
3. `python -m alexinator.tmdb --test` vérifie la connexion.
4. `python -m alexinator.tmdb "Le Parrain" --annee 1972` affiche une fiche au format de `data/films.json`.
   C'est un brouillon : traduire les mots-clés (TMDB les donne en anglais), vérifier qu'ils restent des
   indices, puis lancer les tests.
5. `python -m alexinator.tmdb --lot 50` ajoute directement à `data/films.json` les 50 films suivants parmi
   les plus votés de TMDB. Les mots-clés sont traduits par le LLM du `.env` ; un film est écarté s'il est
   déjà dans la base, si un de ses titres en valide un autre, ou s'il ferait échouer les contrôles des tests
   (nom propre dans les mots-clés, question honnête prise pour son titre...). Relire le diff avant de committer.

*Ce produit utilise l'API TMDB mais n'est ni approuvé ni certifié par TMDB.*

## Organisation du code

```
app.py                  interface web Streamlit
cli.py                  jeu dans le terminal (pratique pour tester un modèle ou un prompt)
campagne.py             campagne d'évaluation automatique
analyse.ipynb           analyse du journal (playtests et campagnes)
alexinator/
  config.py             chemins, lecture du fichier .env
  llm.py                LLMClient (API), TransformersLLM (modèle local entraîné), MockLLM (tests)
  films.py              chargement des fiches, tirage d'un film
  validation.py         la réponse du joueur est-elle juste ?
  visuels.py            images du magicien et texte de ses bulles
  defenses.py           filtre de sortie, censeur, détecteur d'attaques, réponses oui/non
  engine.py             niveaux, partie en cours, chaîne de défenses, outil du niveau 7
  redteam.py            devineur, campagne d'attaques, attaquant automatique
  journal.py            enregistrement SQLite
  tmdb.py               client de l'API TMDB : brouillons de fiches de films
assets/                 affiches, dessins du magicien (bulles vides), avatar, police Inter
levels/*.yaml           un fichier par niveau : prompt du gardien + défenses actives
data/films.json         les fiches des films
data/attaques.json      24 attaques classées en 8 catégories
data/questions_honnetes.json   30 questions qui ne doivent pas être bloquées
data/journal.db         créé au premier lancement
finetune/               jeu de données et notebook d'entraînement de notre gardien
tests/                  tests automatiques
```

Le détail du fonctionnement est dans [docs/architecture.md](docs/architecture.md).

## Entraîner notre propre gardien (dernière étape)

Sur le serveur équipé du GPU de 24 Go :

```
pip install -r requirements.txt -r requirements-finetune.txt
python finetune/construire_dataset.py --journal --lakera 150 --devineur
jupyter notebook finetune/entrainement_qlora.ipynb
```

1. `construire_dataset.py` fabrique `finetune/data/train.jsonl` et `val.jsonl` : un modèle « professeur »
   (variable `ALEXINATOR_TEACHER_MODEL`, le meilleur modèle disponible) joue le gardien idéal face à nos
   attaques, à nos questions honnêtes, aux tentatives des playtests et à des injections réelles du jeu
   Gandalf ; nos défenses rejettent toute réponse qui laisse fuiter le film.
2. Le notebook affine un modèle open source de 7 milliards de paramètres avec QLoRA et compare le
   taux de fuite avant et après.
3. Pour jouer avec le résultat, mettre dans `.env` : `ALEXINATOR_BACKEND=local` et
   `ALEXINATOR_ADAPTER=finetune/gardien-lora` (voir la fin du notebook).

Tout le détail, dont le choix du jeu de données, est dans [docs/entrainement.md](docs/entrainement.md).

## Comment contribuer

Méthode de travail en groupe : voir [CONTRIBUTING.md](CONTRIBUTING.md).

- **Ajouter un film** : ajouter une fiche dans `data/films.json`. Soigner les `alias` (ils servent à la
  validation et au filtre). Éviter les titres de moins de 4 lettres : le filtre les ignore.
- **Modifier un niveau** : éditer son fichier YAML (prompt, défenses, nombre de messages).
- **Ajouter un niveau** : créer un nouveau fichier YAML avec un `id` plus grand.
- **Ajouter une défense** : écrire la fonction dans `defenses.py`, ajouter son nom à `KNOWN_DEFENSES`
  et son appel dans `GameSession.send` (`engine.py`), plus un message dans `REFUSALS`.
- **Analyser les parties** : `pd.read_sql("SELECT * FROM tours", sqlite3.connect("data/journal.db"))`.

## Prochaines étapes

1. Phase 0 : jouer les niveaux 1 et 2 avec 2 ou 3 modèles (`python cli.py --niveau 2 --film titanic`)
   et choisir celui qui garde le secret sans inventer d'indices.
2. Ajuster les prompts des niveaux, du censeur et du détecteur d'après ces essais et une première campagne.
3. Ouvrir le jeu à la promo (playtest).
4. Fabriquer le jeu de données, entraîner le gardien, comparer modèle de base et modèle entraîné.
