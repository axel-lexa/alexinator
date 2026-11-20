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

**Choix : un jeu de données de 12 350 exemples fabriqué par distillation**, complété par des attaques de référence et les playtests humains.

Au début du projet, l'intention première était de **composer le dataset uniquement à partir de fiches et résumés de films**. Cette intuition s'est vite heurtée à une réalité fondamentale : une base de fiches enseigne des connaissances cinématographiques statiques, mais n'apprend absolument pas le **rôle conversationnel d'un gardien**. Un modèle entraîné uniquement sur des fiches a tendance à réciter passivement son texte dès qu'un joueur l'interroge. Pour incarner un gardien équilibré, le modèle doit apprendre à distinguer et arbitrer entre deux interactions humaines diamétralement opposées :
1. **Les attaques adversariales** : refuser les tentatives de manipulation (jailbreak, persona, fausse autorité) avec humour et fermeté ;
2. **Les questions honnêtes** : coopérer avec les questions légitimes d'un joueur sincère en délivrant des indices subtils sans trahir le secret.

La solution a donc été d'opérer un **croisement systématique** entre le catalogue de films et ces deux flux de dialogue : chaque film est croisé avec un échantillon d'attaques et de questions honnêtes (produit cartésien contrôlé). Aucun benchmark public n'existant pour ce rôle spécifique, nous avons conçu un pipeline de distillation automatisé via [`finetune/construire_dataset.py`](../finetune/construire_dataset.py) :

### 1. Sources des messages d'entrée (Inputs)
Le générateur puise dans trois réservoirs complémentaires :
* **Benchmark Adversarial (`data/attaques.json` — 128 attaques)** : Couvre exhaustivement les 8 catégories OWASP LLM01 et intègre les véritables injections du jeu public Lakera Gandalf traduites en français.
* **Benchmark de Jouabilité (`data/questions_honnetes.json` — 228 questions)** : Questions légitimes d'analyse filmique, relances courtes et questions courantes permettant d'apprendre à formuler des indices captivants sans rien divulguer.
* **Télémétrie des Playtests Humains (`data/journal.db` — 202 messages réels)** : Extraits de parties réelles de joueurs humains (questions pertinentes et tentatives d'attaques spontanées en français), consolidés via [`tools/fusionner_journaux.py`](../tools/fusionner_journaux.py).

### 2. Le Modèle Professeur (Teacher Model)
Un modèle frontière propriétaire de grande taille (ex: GPT-4o, Claude 3.5 Sonnet, Mistral Large configuré via `ALEXINATOR_TEACHER_MODEL`) incarne le gardien idéal.
Il est guidé par une consigne masquée (`TEACHER_GUIDE`) qui lui intime de refuser toute tentative de manipulation avec humour et concision, et de fournir un indice vague et véridique pour toute question honnête. Cette consigne n'est jamais injectée dans les exemples finaux : l'élève apprend ainsi à bien se comporter à partir du prompt standard du jeu.

### 3. Problèmes rencontrés et solutions apportées lors de la génération
Lors du développement du pipeline de génération, plusieurs difficultés majeures ont été identifiées et résolues :

* **Le piège du dataset purement factuel (Besoin du croisement)** :
  * *Problème* : Vouloir utiliser seulement des films n'enseigne ni le refus des manipulations, ni le dosage des indices, rendant le gardien trivial à contourner.
  * *Solution* : Croisement relationnel obligatoire entre fiches de films, attaques et questions honnêtes.
* **Autonomie hors-ligne et standardisation de Lakera Gandalf** :
  * *Problème* : L'extraction dynamique depuis l'API Hugging Face créait une dépendance réseau instable, avec des textes bruts en anglais centrés sur un mot de passe informatique (`password`).
  * *Solution* : Traduction en français, contextualisation cinématographique et intégration directe au format canonique dans [`data/attaques.json`](../data/attaques.json). Le pipeline est 100 % reproductible hors-ligne.
* **Risque de sur-refus (*Over-Refusal*)** :
  * *Problème* : Un modèle entraîné majoritairement sur des attaques devient paranoïaque et rejette même les questions sincères des joueurs, rendant le jeu injouable.
  * *Solution* : Échantillonnage stratifié strict garantissant une parité 50/50 entre attaques et questions honnêtes par film (`sample_items`). Le dataset final compte 55,0 % de réponses d'aide/indices et 45,0 % de refus pédagogiques.
* **Fuites inter-partitions et contamination des sagas** :
  * *Problème* : Un découpage classique par exemple entraîne une mémorisation brute des fiches. De plus, scinder des œuvres d'une même saga (*Blade Runner* / *Blade Runner 2049*, *Alien* / *Aliens*) entre train et val fausse l'évaluation.
  * *Solution* : Partitionnement strict par film (`split_by_film`), avec regroupement forcé des franchises du côté entraînement. La validation (40 films inédits) teste la réelle généralisation du comportement.
* **Temps de calcul et coupures réseau (Résilience de génération)** :
  * *Problème* : Générer plus de 12 000 exemples via API prend plusieurs heures. Une déconnexion réseau ou un `Ctrl+C` provoquait la perte de tout le tampon en mémoire.
  * *Solution* : Écriture incrémentale en continu sur disque avec flush immédiat (`f_out.flush()`), reprise automatique transparente (`seen_keys`) évitant tout recalcul en cas de relance, et affichage dynamique de l'avancement (vitesse et ETA).
* **Qualité des réponses et fuites d'indices** :
  * *Problème* : Le modèle professeur peut occasionnellement halluciner ou laisser échapper le titre ou le nom d'un acteur.
  * *Solution* : Double garde-fou automatique rejetant la réponse avec retry (jusqu'à 3 essais) via le filtre de sortie regex (`defenses.output_filter`) et le devineur sémantique (`--devineur`).

### 4. Commande de génération

Le dataset complet présent dans le dépôt (9 882 exemples de train, 2 468 de val) a été produit par la commande :
```bash
python finetune/construire_dataset.py --journal --devineur
```
*(Grâce au mode reprise automatique, relancer cette commande inspecte les fichiers existants sans écraser le travail déjà accompli, sauf si l'option `--ecraser` est expressément fournie).*

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
