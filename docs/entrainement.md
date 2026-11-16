# Entraîner notre propre gardien

Dernière étape du projet : spécialiser un LLM open source pour qu'il tienne le rôle du gardien, puis le
comparer au modèle de base avec la même campagne d'attaques.

## Ce que veut dire « entraîner » ici

Entraîner un LLM à partir de zéro demande des milliers d'heures de GPU. On fait donc un **affinage**
(fine-tuning) avec **QLoRA** :

- le modèle de base (environ 7 milliards de paramètres) est chargé en 4 bits, soit environ 6 Go ;
- ses poids sont gelés ; on entraîne seulement de petits adaptateurs LoRA (moins de 1 % des paramètres) ;
- l'ensemble tient sur la carte graphique de 24 Go du serveur.

Modèle de base par défaut : `Qwen/Qwen2.5-7B-Instruct` (licence Apache 2.0, multilingue, accepte un
message système). Il se change en une ligne dans le notebook.

## Le jeu de données

**Choix : un jeu de données fabriqué par distillation**, complété par des attaques publiques.

Aucun jeu public ne contient « un gardien qui donne des indices sans trahir son film ». Les jeux publics
de prompt injection ne contiennent que des attaques, pas les bonnes réponses. On fabrique donc les
réponses nous-mêmes avec `finetune/construire_dataset.py` :

1. **Messages de joueurs**, de quatre sources :

   | Source | Option | Intérêt |
   | --- | --- | --- |
   | `data/attaques.json` (24 attaques, 8 catégories) | toujours | Couvrir toutes les catégories |
   | `data/questions_honnetes.json` (30 questions) | toujours | Apprendre à rester jouable, pas seulement à refuser |
   | Tentatives réelles des playtests (journal) | `--journal` | Les attaques qui visent vraiment notre jeu, en français |
   | [Lakera/gandalf_ignore_instructions](https://huggingface.co/datasets/Lakera/gandalf_ignore_instructions) | `--lakera N` | Injections réelles du jeu Gandalf (1 000 exemples, licence MIT, en anglais) |

2. **Un modèle professeur** (`ALEXINATOR_TEACHER_MODEL`, le meilleur disponible) répond à chaque message
   comme le gardien idéal. Il reçoit des conseils supplémentaires qui ne figurent pas dans les exemples :
   le modèle élève apprend donc à bien se comporter avec le prompt normal du jeu.
3. **Contrôle qualité par nos défenses** : une réponse qui contient le titre ou un nom de la fiche est
   rejetée et régénérée (3 essais). Avec `--devineur`, une réponse qui permet à un LLM de deviner le
   film est aussi rejetée.
4. **Séparation par film** : un film sur cinq part en validation et n'est jamais vu à l'entraînement.
   On vérifie ainsi que le modèle apprend un comportement, pas des fiches par cœur.

Taille : avec 54 messages par film, 21 films donnent au plus 1 134 exemples (918 en entraînement), peu variés. **Portez la base
à une centaine de fiches avant l'entraînement**, et ajoutez les playtests et 150 attaques Lakera.

```
python finetune/construire_dataset.py --journal --lakera 150 --devineur
```

## Lancer l'entraînement

Sur le serveur :

```
pip install -r requirements.txt -r requirements-finetune.txt
jupyter notebook finetune/entrainement_qlora.ipynb
```

Le notebook enchaîne : chargement des données, modèle en 4 bits avec LoRA, masquage du contexte (on
n'apprend que la réponse du gardien), mesure avant entraînement, boucle d'entraînement écrite à la main
(comme dans les labs), courbes, mesure après entraînement, enregistrement de l'adaptateur dans
`finetune/gardien-lora/`.

Premier lancement : cellule par cellule, avec `EPOCHS = 1`.

| Hyperparamètre | Valeur | Commentaire |
| --- | --- | --- |
| `MAX_LENGTH` | 1024 | Longueur maximale d'un exemple, en tokens |
| `EPOCHS` | 3 | On garde l'époque avec la meilleure loss de validation |
| `BATCH_SIZE` × `GRAD_ACCUM` | 4 × 4 | Batch effectif de 16 |
| `LEARNING_RATE` | 2e-4 | Valeur courante pour LoRA |
| `LORA_R` / `LORA_ALPHA` | 16 / 32 | Rang et échelle des adaptateurs |

## Problèmes courants

| Symptôme | Solution |
| --- | --- |
| `CUDA out of memory` | Baisser `BATCH_SIZE` à 2 ou 1 (et monter `GRAD_ACCUM` d'autant), ou `MAX_LENGTH` à 768 |
| Loss de validation qui remonte | Sur-apprentissage : moins d'époques, plus de données, ou `LORA_R` plus petit |
| Le modèle refuse tout | Trop d'attaques par rapport aux questions honnêtes : rééquilibrer le jeu de données |
| Erreur sur `target_modules` | Autre architecture de modèle : afficher `print(model)` et adapter les noms des couches |
| Téléchargement du modèle refusé | Certains modèles demandent d'accepter une licence sur Hugging Face et un `huggingface-cli login` |

## Évaluer le modèle entraîné

Dans `.env`, sur le serveur :

```
ALEXINATOR_BACKEND=local
ALEXINATOR_MODEL=Qwen/Qwen2.5-7B-Instruct
ALEXINATOR_ADAPTER=finetune/gardien-lora
```

Puis lancer la même campagne avec et sans adaptateur (même `--graine`), et ouvrir `analyse.ipynb`, section B.3 :

```
python campagne.py --niveaux 2 --films 4 --nom modele_entraine
python campagne.py --niveaux 2 --films 4 --nom modele_de_base     (ligne ALEXINATOR_ADAPTER retirée)
```

Comparer au niveau 2, qui n'a aucune défense technique : c'est la résistance du modèle seul.
