# Architecture

Ce document décrit comment le code est organisé et comment chaque fichier de configuration est structuré.
Pour l'installation et les commandes, voir le [README](../README.md).

## Trajet d'un message

Tout se passe dans `GameSession.send` (`alexinator/engine.py`). Chaque étape entre crochets ne s'exécute
que si le niveau l'active dans son fichier YAML.

```
message du joueur
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
- **La proposition de réponse ne passe pas par le gardien.** Le champ « Ta réponse » est vérifié par `validation.check_answer`, en trois étapes : correspondance exacte (après normalisation), correspondance approchée à nombre de mots égal (fautes de frappe), puis juge LLM optionnel.
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
| `cartoon_reflechit_vierge.png` | Début de partie, attente de la réponse du LLM, réponse bloquée par une défense, mauvaise proposition |
| `cartoon_solution_vierge.png` | Victoire, avec le titre du film trouvé |
| `cartoon_*_original.png` | Dessins d'origine, avec leur texte figé (archives) |
| `fonts/` | Police Inter (licence SIL OFL, voir `fonts/OFL.txt`) pour écrire dans les bulles |

Les versions « vierge » ont une bulle vide. `visuels.magicien_reflechit(texte)` et
`visuels.magicien_trouve(titre)` y écrivent le texte en ajustant la taille de la police, ce qui permet
d'afficher n'importe quel titre, même long. Les phrases du magicien se changent dans `visuels.PHRASES`.
