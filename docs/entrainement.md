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

### Variante Alexinator-240M sur une Radeon AI PRO R9700

Le profil `r9700` adapte la phase 1 à une carte AMD Radeon AI PRO R9700 : RDNA4 (`gfx1201`), 32 Go,
bfloat16 matériel (191 TFLOPS en crête selon AMD). Cette carte est assez rapide pour entraîner, sur les
mêmes 3 milliards de tokens, un modèle **deux fois plus gros** :

| | Alexinator-110M | Alexinator-240M (`--profil r9700`) |
| --- | --- | --- |
| Blocs / largeur / têtes | 12 / 768 / 12 | 16 / 1 024 / 16 (têtes de 64) |
| MLP SwiGLU | 2 048 | 2 816 (multiple de 128 pour les unités matricielles) |
| Paramètres | 109,5 M | 238,3 M |
| Calcul par token | 0,77 GFLOP | 1,63 GFLOP |
| Précision | selon la carte (float32 sur la M40) | bfloat16 |
| Micro-batch | 16 (fixé à la main) | le plus grand qui tient dans 32 Go, trouvé au démarrage |
| Learning rate max / min | 6e-4 / 6e-5 | 4e-4 / 4e-5 |
| Sauvegarde | tous les 1 000 pas | tous les 250 pas |

Le pas reste de 524 288 tokens quelle que soit la taille du micro-batch (l'accumulation s'ajuste), donc
environ 5 700 pas pour 3 milliards de tokens. Durée estimée : **environ un jour** si la carte tient 25 à
35 % de sa puissance de crête. C'est une estimation : la mesurer avec `--banc` avant de lancer.

**Installation** (Linux, ROCm 6.4.2 ou plus récent, utilisateur dans les groupes `render` et `video`) :

```
pip install torch --index-url https://download.pytorch.org/whl/rocm6.4
```

Sous ROCm, PyTorch garde le nom `cuda` : `torch.cuda.is_available()` et `device="cuda"` désignent la carte AMD.

**Lancement dans un notebook** : ouvrir `entrainement/alexinator_r9700.ipynb` et exécuter les cellules dans
l'ordre. Le notebook récupère le code du dépôt (clone s'il n'est pas déjà dedans), vérifie la carte, prépare le
corpus s'il manque, choisit le micro-batch, mesure la vitesse (cellule « Banc »), puis entraîne avec une barre de
progression, des échantillons de texte et les courbes. `TEST = True` dans la première cellule vérifie toute la
chaîne en 2 à 3 minutes. Le bouton ■ interrompt proprement : `dernier.pt` est enregistré et relancer la cellule
reprend. Les checkpoints ont le même format que ceux du script.

**Lancement en ligne de commande**, depuis la racine du dépôt :

```
bash entrainement/lancer_r9700.sh
```

Comme `lancer_phase1.sh` : vérifications, test rapide de toute la chaîne, puis préparation du corpus
(sautée si `entrainement/donnees/stats.json` existe déjà) et pré-entraînement en arrière-plan.

| Commande | Rôle |
| --- | --- |
| `bash entrainement/lancer_r9700.sh --banc` | Mesurer la vitesse sur 20 pas et estimer la durée totale (rien n'est sauvé) |
| `tail -f entrainement/logs/r9700.log` | Suivre l'avancement |
| `kill $(cat entrainement/logs/r9700.pid)` | Arrêter |
| `bash entrainement/lancer_r9700.sh --reprendre` | Reprendre au dernier checkpoint |
| `python entrainement/sft.py --run alexinator-240m` | Phase 2 sur ce modèle |

Le corpus n'a pas besoin d'être refait : on peut copier le dossier `entrainement/donnees/` (environ 6 Go)
depuis le serveur, par exemple `scp -r serveur:~/alexinator/entrainement/donnees entrainement/`.

Ce qui est fait automatiquement pour cette carte (`entrainement/materiel.py`) :

- **bfloat16** : détecté sous ROCm avec `torch.cuda.is_bf16_supported()` ;
- **attention rapide** : `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` active les noyaux AOTriton de
  `scaled_dot_product_attention` sur les cartes Radeon (encore expérimentaux sur RDNA) ;
- **torch.compile** : essayé sur un vrai pas dès le démarrage ; s'il échoue, l'entraînement continue
  sans compilation au lieu de planter au premier pas ;
- **micro-batch** : essais de 64 à 1 séquences, en gardant le premier dont le pic de mémoire reste sous
  90 % de la carte.

Toutes les options restent modifiables : `--profil r9700 --tokens 2e9` ou `--micro-batch 16` l'emportent
sur le profil.

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
