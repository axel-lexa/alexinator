# Architecture

Ce document décrit comment le code est organisé et comment chaque fichier de configuration est structuré.
Pour l'installation et les commandes, voir le [README](../README.md).

## Trajet d'un message

Le joueur n'a qu'un champ de saisie. `GameSession.play` (`alexinator/engine.py`) regarde d'abord si le
message est une proposition de titre, sinon il le passe à `GameSession.send`. Chaque étape entre crochets
ne s'exécute que si le niveau l'active dans son fichier YAML.

```
message du joueur
  -> proposition de titre ?   engine.play               code Python, sans LLM (voir plus bas)
  -> [détecteur d'attaques]   defenses.input_detector   LLM auxiliaire, refuse les manipulations
  -> gardien                  LLM principal, prompt système + fiche du film + historique
  -> [outil VERIFIER]         engine._use_tool          niveau 7 : le gardien teste un titre (VRAI/FAUX)
  -> [oui / non]              defenses.force_oui_non    niveau 6 : seules 3 réponses possibles
  -> [filtre de sortie]       defenses.output_filter    code Python : titre, alias, noms de la fiche
  -> [censeur]                defenses.censor           LLM auxiliaire : la réponse rend-elle le film évident ?
  -> réponse affichée au joueur
```

Points importants :

- **Une réponse bloquée n'entre pas dans l'historique.** Le gardien ne voit que ce que le joueur a vu, sinon il pourrait citer plus tard une réponse censurée.
- **Une proposition de titre ne passe pas par le gardien.** `play` retire les formules d'introduction (« est-ce que c'est … ? », `validation.extract_title`), puis compare **le message entier** aux titres. Citer un titre au milieu d'une phrase ou coller une liste de titres ne fait donc pas gagner.
  - le film secret (exact ou approché, via `validation.check_answer`) : victoire, avec l'orthographe correcte si le joueur a fait une faute ;
  - le début d'un titre du film secret (« Le Seigneur des anneaux » pour *La Communauté de l'anneau*) : « Tu brûles », la proposition est refusée ;
  - un titre très proche sans être correct (« Retour vers le passé ») : « Tu chauffes » (`validation.is_near_miss`) ;
  - un autre film de `data/films.json` (titre exact, début ou titre proche) : « Non, ce n'est pas *X* » ou « Non, ce n'est pas ça » ;
  - sinon : c'est une question, elle part au gardien.

  Une proposition refusée coûte un message et entre dans l'historique du gardien. La comparaison se fait en deux étapes : correspondance exacte (après normalisation), puis approchée à nombre de mots égal, mot à mot (fautes de frappe : « forest gump » oui, « star wars 1 » pour l'épisode IV non). Le juge LLM n'intervient que pour la commande `/r` du terminal. Un test vérifie qu'aucune question honnête ni aucune attaque de `data/` n'est prise pour une proposition.
- **Le gardien ne valide jamais un titre.** `engine.GUARDIAN_NO_VERDICT` est ajouté au prompt de chaque niveau : sans cette règle, un petit modèle félicitait le joueur pour un titre faux qui lui était parvenu.
- **Tout est enregistré** dans `data/journal.db`, y compris la réponse brute du gardien avant les défenses.

## Les modules

| Fichier | Rôle |
| --- | --- |
| `alexinator/config.py` | Chemins du projet, lecture du fichier `.env` |
| `alexinator/llm.py` | `LLMClient` (API compatible OpenAI), `TransformersLLM` (modèle local + adaptateur LoRA), `MockLLM` (tests), `make_llm` (choix selon `.env`) |
| `alexinator/films.py` | Classe `Film`, chargement de `data/films.json`, fiche complète ou anonyme |
| `alexinator/validation.py` | Normalisation et vérification de la réponse du joueur |
| `alexinator/defenses.py` | Les quatre défenses |
| `alexinator/engine.py` | Chargement des niveaux, classe `GameSession` |
| `alexinator/visuels.py` | Images du magicien : bulle de pensée ou de victoire remplie avec le texte du moment |
| `alexinator/redteam.py` | Devineur, campagne d'attaques, attaquant automatique |
| `alexinator/journal.py` | Écriture dans SQLite |
| `app.py` | Interface Streamlit |
| `cli.py` | Jeu dans le terminal |
| `campagne.py` | Campagne d'évaluation automatique |

Tout le code ne connaît qu'une méthode des LLM : `chat(messages, temperature, max_tokens) -> str`.
On peut donc remplacer un modèle par un autre, ou par `MockLLM` dans les tests, sans rien changer d'autre.

## Format d'un niveau (`levels/*.yaml`)

| Clé | Obligatoire | Valeur par défaut | Rôle |
| --- | --- | --- | --- |
| `id` | oui | | Numéro du niveau (ordre de progression) |
| `name` | oui | | Nom affiché |
| `description` | oui | | Texte affiché au joueur |
| `guardian_prompt` | oui | | Prompt système du gardien. `{fiche}` est remplacé par la fiche du film, `{verifications}` par le nombre d'appels d'outil restants |
| `defenses` | non | `[]` | Parmi `input_detector`, `output_filter`, `censor` |
| `filter_entities` | non | `true` | Le filtre bloque aussi acteurs, réalisateurs et personnages |
| `max_messages` | non | `15` | Messages autorisés par partie |
| `temperature` | non | `0.7` | Température du gardien |
| `response_mode` | non | `libre` | `oui_non` pour le niveau 6 |
| `fiche_mode` | non | `complete` | `anonyme` : ni titre ni noms propres (niveau 7) |
| `tool_calls` | non | `0` | Nombre d'appels à l'outil `VERIFIER` (niveau 7) |

Un nom de défense inconnu provoque une erreur au chargement.

## Format d'une fiche de film (`data/films.json`)

| Champ | Exemple | Utilisé par |
| --- | --- | --- |
| `id` | `"titanic"` | Journal, tirage, séparation entraînement/validation |
| `titre` | `"Titanic"` | Validation, filtre, fiche |
| `alias` | `["The Lion King"]` | Validation (réponses acceptées) et filtre (termes interdits) |
| `annee` | `1997` | Fiche |
| `realisateurs`, `acteurs`, `personnages` | listes de noms | Fiche complète et filtre |
| `genre` | `"drame romantique"` | Fiche (complète et anonyme) |
| `mots_cles` | `["paquebot", "iceberg"]` | Fiche (complète et anonyme), indices du gardien |

Conseils : soigner les alias (titre original, numéro de volet), éviter les titres de moins de 4 lettres
(le filtre les ignore pour ne pas bloquer des mots courants), ne rien mettre dans `mots_cles` qui donne le
film à lui seul, comme un nom de personnage ou de lieu imaginaire (ces mots-clés sont donnés au gardien du
niveau 7, qui ne doit pas pouvoir trahir le film).

## Le journal (`data/journal.db`)

| Table | Une ligne = | Colonnes principales |
| --- | --- | --- |
| `tours` | un message du joueur | `session_id`, `joueur`, `niveau`, `film_id`, `modele`, `message_joueur`, `reponse_brute`, `reponse_affichee`, `bloque_par`, `detail`, `latence_ms` |
| `propositions` | une proposition de réponse | `session_id`, `proposition`, `correcte`, `methode`, `messages_utilises` |
| `evaluations` | un message testé par `campagne.py` | `campagne`, `modele`, `niveau`, `type`, `categorie`, `bloque_par`, `fuite_titre`, `fuite_devineur` |

Les parties jouées par les campagnes automatiques ont un pseudo qui commence par `auto:`, ce qui permet
de les séparer des parties humaines dans l'analyse.

## Les images du magicien (`assets/`)

| Fichier | Où il apparaît |
| --- | --- |
| `affiche_reflechit.png` | Barre latérale de l'interface, README |
| `affiche_solution.png` | Écran final, quand le joueur a battu tous les niveaux ; README |
| `avatar.png` | Avatar du gardien dans la discussion (visage découpé du dessin) |
| `cartoon_reflechit_vierge.png` | Début de partie, réponse bloquée par une défense, mauvaise proposition |
| `video_reflechit.mp4` | Attente de la réponse du LLM (lue en boucle, sans le son) |
| `cartoon_solution_vierge.png` | Victoire, avec le titre du film trouvé |
| `cartoon_*_original.png` | Dessins d'origine, avec leur texte figé (archives) |
| `fonts/` | Police Inter (licence SIL OFL, voir `fonts/OFL.txt`) pour écrire dans les bulles |

Les versions « vierge » ont une bulle vide. `visuels.magicien_reflechit(texte)` et
`visuels.magicien_trouve(titre)` y écrivent le texte en ajustant la taille de la police, ce qui permet
d'afficher n'importe quel titre, même long. Les phrases du magicien se changent dans `visuels.PHRASES`.
