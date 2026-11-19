# Entraîner notre propre gardien

Deux approches complémentaires, comparées dans le rapport :

| Approche | Fichiers | Ce que c'est |
| --- | --- | --- |
| **Alexinator-110M** (approche principale) | `alexinator/gpt.py`, `entrainement/` | Notre propre GPT décodeur, conçu et entraîné à partir de zéro, comme au lab 5 |
| Affinage QLoRA d'un modèle de 7 milliards de paramètres | `finetune/entrainement_qlora.ipynb` | Point de comparaison : ce qu'apporte un modèle 60 fois plus gros |

## Alexinator-110M : notre modèle

La conception (architecture, taille, budget de calcul et de mémoire, données) est expliquée dans le
document « Alexinator-110M — Modélisation de notre LLM ». Résumé : 12 blocs, largeur 768, 12 têtes,
contexte de 1 024 tokens, environ 110 M de paramètres, pré-entraîné sur 3 milliards de tokens de
français (Wikipédia + FineWeb-2), puis spécialisé sur les conversations du gardien.

### Phase 1 : tokenizer et pré-entraînement

Sur le serveur GPU, depuis la racine du dépôt :

```
bash entrainement/lancer_phase1.sh
```

Le script installe les dépendances, vérifie le GPU, lance un **test rapide de toute la chaîne**
(mini-corpus, mini-modèle, mini-spécialisation : 2 à 3 minutes), puis lance la vraie phase 1 en
arrière-plan. Elle continue même si la connexion SSH se coupe.

| Commande | Rôle |
| --- | --- |
| `tail -f entrainement/logs/phase1.log` | Suivre l'avancement (loss, tokens par seconde, temps restant) |
| `kill $(cat entrainement/logs/phase1.pid)` | Arrêter |
| `bash entrainement/lancer_phase1.sh --reprendre` | Reprendre au dernier checkpoint après une coupure |
| `cat entrainement/runs/alexinator-110m/echantillons.txt` | Textes générés au fil de l'entraînement |

Étapes enchaînées :

1. `preparer_donnees.py` : tokenizer BPE de 32 000 tokens, puis tokenisation de Wikipédia FR et de
   FineWeb-2 jusqu'à 3 milliards de tokens (`entrainement/donnees/train.bin`, `val.bin`). Plusieurs
   heures selon le réseau et le processeur ; environ 6 Go de disque, plus le cache Hugging Face.
2. `pretrain.py` : environ 5 700 pas de 512 000 tokens. Sauvegarde toutes les 1 000 étapes
   (`dernier.pt`) et à chaque meilleure loss de validation (`meilleur.pt`), journal dans `journal.csv`.

Si la mémoire déborde (`CUDA out of memory`) : `--micro-batch 8 --accumulation 64` garde le même
nombre de tokens par pas. Si `torch.compile` pose problème : `--no-compile`.

### Phase 2 : spécialisation en gardien

Quand le jeu de données du gardien est prêt (`finetune/construire_dataset.py`, section suivante) :

```
python entrainement/sft.py
```

Le gardien est enregistré dans `entrainement/runs/alexinator-110m-gardien/gardien.pt`, avec
l'évolution de la loss de validation et du taux de fuite dans `resultats.json`.

### Jouer contre notre modèle

Dans `.env` :

```
ALEXINATOR_BACKEND=alexinator
ALEXINATOR_CHECKPOINT=entrainement/runs/alexinator-110m-gardien/gardien.pt
```

Le gardien est alors notre modèle ; le censeur, le détecteur et le devineur restent sur l'API
configurée (`ALEXINATOR_AUX_BACKEND`, par défaut `api`), car un modèle de 110 M est trop petit pour
ces rôles. La comparaison se fait avec `campagne.py --niveaux 2 --nom alexinator-110m`.

## Affinage QLoRA (point de comparaison)

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
   | `data/attaques.json` (128 attaques, 8 catégories) | toujours | Couvrir toutes les catégories d'attaques (OWASP LLM01 et benchmark Lakera Gandalf traduit) |
   | `data/questions_honnetes.json` (228 questions) | toujours | Apprendre à rester jouable et informatif sans fuiter |
   | Tentatives réelles des playtests (journal) | `--journal` | Les attaques qui visent vraiment notre jeu, en français |
   | [Lakera/gandalf_ignore_instructions](https://huggingface.co/datasets/Lakera/gandalf_ignore_instructions) | `--lakera N` | (Optionnel) Injections Gandalf brutes supplémentaires depuis HF (licence MIT) |

2. **Un modèle professeur** (`ALEXINATOR_TEACHER_MODEL`, le meilleur disponible) répond à chaque message
   comme le gardien idéal. Il reçoit des conseils supplémentaires qui ne figurent pas dans les exemples :
   le modèle élève apprend donc à bien se comporter avec le prompt normal du jeu.
3. **Contrôle qualité par nos défenses** : une réponse qui contient le titre ou un nom de la fiche est
   rejetée et régénérée (3 essais). Avec `--devineur`, une réponse qui permet à un LLM de deviner le
   film est aussi rejetée.
4. **Séparation par film** : un film sur cinq part en validation et n'est jamais vu à l'entraînement.
   On vérifie ainsi que le modèle apprend un comportement, pas des fiches par cœur.

Taille : la base contient désormais **200 films** (160 en entraînement, 40 en validation). Pour éviter un temps de génération excessif et préserver un équilibre parfait entre refus et indices, le script applique par défaut un équilibrage 50/50 entre attaques et questions. Avec `--max-par-film 16`, on obtient un jeu de données idéal de ~2 560 exemples d'entraînement et ~640 de validation :

```
python finetune/construire_dataset.py --max-par-film 16 --journal --devineur
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
