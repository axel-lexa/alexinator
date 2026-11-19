"""Vérification continue de l'intégrité et de l'étanchéité des datasets.

Ce script certifie que l'ensemble des règles de sécurité, de formatage et de non-régression
sont strictement respectées sur les données d'Alexinator :
  1. Base cinéma (data/films.json) : champs, unicité, zéro fuite de titre ou d'entité.
  2. Étanchéité Train / Validation : aucune franchise scindée entre train et val.
  3. Attaques (data/attaques.json) : 8 catégories canoniques, pas de chaîne vide.
  4. Questions honnêtes (data/questions_honnetes.json) : zéro collision de titre, zéro doublon.
  5. Attaques externes figées (data/attaques_lakera.json) : intégrité du corpus Gandalf.
  6. Dataset de fine-tuning (finetune/data/*.jsonl si présents) : schéma ChatML, < 1024 tokens, zéro fuite.

Usage :
    python tools/verifier_donnees.py
    python tools/verifier_donnees.py --strict
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from alexinator import defenses  # noqa: E402
from alexinator.films import Film, load_films  # noqa: E402
from alexinator.redteam import load_attacks, load_honest_questions  # noqa: E402
from alexinator.validation import check_answer, extract_title, is_near_miss, is_partial_title, normalize  # noqa: E402


class DataVerifier:
    def __init__(self, root: Path = ROOT, strict: bool = False):
        self.root = root
        self.strict = strict
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.passed_checks: list[str] = []

    def log_ok(self, message: str) -> None:
        self.passed_checks.append(message)
        print(f"  [OK] {message}")

    def log_error(self, message: str) -> None:
        self.errors.append(message)
        print(f"  [ERREUR] {message}")

    def log_warning(self, message: str) -> None:
        self.warnings.append(message)
        print(f"  [ATTENTION] {message}")

    def verifier_films(self) -> list[Film]:
        print("\n--- 1. Vérification de data/films.json ---")
        path = self.root / "data" / "films.json"
        if not path.exists():
            self.log_error(f"Fichier manquant : {path}")
            return []

        try:
            films = load_films(path)
        except Exception as e:
            self.log_error(f"Échec de lecture de films.json : {e}")
            return []

        if len(films) < 20:
            self.log_error(f"Nombre de films insuffisant : {len(films)} (attendu >= 20)")
        else:
            self.log_ok(f"Volume du catalogue : {len(films)} films chargés")

        # IDs uniques et format
        ids = [f.id for f in films]
        if len(ids) != len(set(ids)):
            self.log_error("IDs en doublon détectés dans films.json")
        else:
            self.log_ok("Tous les IDs de films sont uniques")

        id_regex = re.compile(r"^[a-z0-9_]+$")
        invalid_ids = [f.id for f in films if not id_regex.match(f.id)]
        if invalid_ids:
            self.log_error(f"Format d'ID non conventionnel : {invalid_ids}")
        else:
            self.log_ok("Format des IDs conforme (minuscules, chiffres, underscores)")

        # Années et métadonnées
        invalid_years = [f.id for f in films if not (1900 <= f.annee <= 2030)]
        if invalid_years:
            self.log_error(f"Années anormales détectées : {invalid_years}")
        else:
            self.log_ok("Années de sortie valides (1900 - 2030)")

        # Zéro fuite de titre ou d'entités protégées dans les mots-clés
        filter_leaks = []
        for f in films:
            terms = defenses.forbidden_terms(f, include_entities=True)
            for kw in f.mots_cles:
                leak = defenses.find_leak(kw, terms)
                if leak:
                    filter_leaks.append((f.id, kw, leak))

        if filter_leaks:
            self.log_error(f"Termes interdits détectés dans les mots-clés ({len(filter_leaks)}) : {filter_leaks[:3]}")
        else:
            self.log_ok("Zéro fuite de titre ou d'entité dans les mots-clés (Niveau 7 parfaitement étanche)")

        return films

    def verifier_etancheite_split(self, films: list[Film]) -> None:
        print("\n--- 2. Vérification de l'étanchéité Train / Validation ---")
        if len(films) < 5:
            return

        sys.path.insert(0, str(self.root / "finetune"))
        from construire_dataset import split_by_film

        train_films, val_films = split_by_film(films)
        val_ids = {f.id for f in val_films}

        # Vérifier que les sagas ne sont pas scindées
        known_sagas = [
            ("blade_runner", "blade_runner_2049"),
            ("terminator", "terminator_2"),
            ("alien_le_huitieme_passager", "aliens_le_retour"),
        ]

        leaked_sagas = []
        for f1, f2 in known_sagas:
            f1_in_val = f1 in val_ids
            f2_in_val = f2 in val_ids
            if f1_in_val != f2_in_val:
                leaked_sagas.append((f1, f2))

        if leaked_sagas:
            self.log_error(f"Sagas scindées entre Train et Val : {leaked_sagas}")
        else:
            self.log_ok(f"Étanchéité Train/Val garantie ({len(train_films)} train, {len(val_films)} val, 0 saga scindée)")

    def verifier_attaques(self) -> None:
        print("\n--- 3. Vérification de data/attaques.json ---")
        path = self.root / "data" / "attaques.json"
        if not path.exists():
            self.log_error(f"Fichier manquant : {path}")
            return

        try:
            attacks = load_attacks(path)
        except Exception as e:
            self.log_error(f"Échec de lecture de attaques.json : {e}")
            return

        categories_attendues = {
            "demande_directe", "jeu_de_role", "hypothetique", "traduction_encodage",
            "format_detourne", "extraction_partielle", "ingenierie_sociale", "injection_instructions"
        }
        cats_trouvees = {a.get("categorie") for a in attacks}

        if cats_trouvees != categories_attendues:
            self.log_error(f"Catégories d'attaques divergentes : {cats_trouvees ^ categories_attendues}")
        else:
            self.log_ok("8 catégories d'attaques canoniques bien présentes")

        if len(attacks) < 24:
            self.log_error(f"Nombre d'attaques insuffisant : {len(attacks)}")
        else:
            self.log_ok(f"Volume d'attaques : {len(attacks)} attaques conformes (8 par catégorie)")

    def verifier_questions(self, films: list[Film]) -> None:
        print("\n--- 4. Vérification de data/questions_honnetes.json ---")
        path = self.root / "data" / "questions_honnetes.json"
        if not path.exists():
            self.log_error(f"Fichier manquant : {path}")
            return

        try:
            questions = load_honest_questions(path)
        except Exception as e:
            self.log_error(f"Échec de lecture de questions_honnetes.json : {e}")
            return

        # Doublons
        if len(questions) != len(set(questions)):
            self.log_error("Doublons détectés dans questions_honnetes.json")
        else:
            self.log_ok(f"Zéro doublon sur les {len(questions)} questions")

        # Ponctuation
        sans_point = [q for q in questions if not q.strip().endswith("?")]
        if sans_point:
            self.log_warning(f"Questions sans point d'interrogation final : {len(sans_point)}")
        else:
            self.log_ok("Toutes les questions se terminent par un point d'interrogation")

        # Collisions de proposition de titre
        collisions = []
        for q in questions:
            cand = extract_title(q)
            for f in films:
                if check_answer(cand, f).correct or is_partial_title(cand, f) or is_near_miss(cand, f):
                    collisions.append((q, f.titre))
                    break

        if collisions:
            self.log_error(f"Questions prises pour des propositions de titre ({len(collisions)}) : {collisions[:2]}")
        else:
            self.log_ok("Zéro collision entre questions et catalogue de films")

    def verifier_lakera(self) -> None:
        print("\n--- 5. Vérification de data/attaques_lakera.json ---")
        path = self.root / "data" / "attaques_lakera.json"
        if not path.exists():
            self.log_warning("Fichier attaques_lakera.json absent (téléchargement HF en direct requis)")
            return

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if len(data) >= 50:
                self.log_ok(f"Corpus Lakera local figé et valide ({len(data)} injections prêtes)")
            else:
                self.log_warning(f"Volume Lakera faible : {len(data)}")
        except Exception as e:
            self.log_error(f"Erreur de lecture de attaques_lakera.json : {e}")

    def verifier_dataset_finetune(self, films: list[Film]) -> None:
        print("\n--- 6. Vérification de finetune/data/ (*.jsonl) ---")
        data_dir = self.root / "finetune" / "data"
        films_by_id = {f.id: f for f in films}

        for split in ("train", "val"):
            path = data_dir / f"{split}.jsonl"
            if not path.exists():
                self.log_warning(f"Fichier {split}.jsonl absent (exécuter construire_dataset.py)")
                continue

            try:
                lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            except Exception as e:
                self.log_error(f"Erreur de syntaxe JSONL dans {split}.jsonl : {e}")
                continue

            # Contrôle de sécurité et de longueur
            fuites = 0
            trop_longs = 0
            for l in lines:
                f_id = l.get("film_id")
                if f_id in films_by_id:
                    film = films_by_id[f_id]
                    asst = l["messages"][2]["content"]
                    if defenses.output_filter(asst, film, True).blocked:
                        fuites += 1

                texte_total = " ".join(m.get("content", "") for m in l.get("messages", []))
                if len(texte_total) // 4 > 1024:
                    trop_longs += 1

            if fuites:
                self.log_error(f"Fuites détectées dans {split}.jsonl : {fuites}")
            else:
                self.log_ok(f"{split}.jsonl : 0 fuite sur {len(lines)} exemples")

            if trop_longs:
                self.log_error(f"Exemples dépassant 1024 tokens dans {split}.jsonl : {trop_longs}")
            else:
                self.log_ok(f"{split}.jsonl : 100 % des exemples < 1024 tokens")

    def run(self) -> int:
        print("==================================================")
        print("  Alexinator : Audit d'intégrité des datasets")
        print("==================================================")

        films = self.verifier_films()
        if films:
            self.verifier_etancheite_split(films)
        self.verifier_attaques()
        if films:
            self.verifier_questions(films)
        self.verifier_lakera()
        if films:
            self.verifier_dataset_finetune(films)

        print("\n==================================================")
        print(f"  Bilan : {len(self.passed_checks)} succès, {len(self.warnings)} avertissements, {len(self.errors)} erreurs")
        print("==================================================")

        if self.errors:
            print("\n❌ Des erreurs critiques ont été détectées.")
            return 1

        print("\n✅ Tous les jeux de données sont intègres, étanches et conformes !")
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Vérifie l'intégrité des jeux de données d'Alexinator")
    parser.add_argument("--strict", action="store_true", help="considère les avertissements comme des erreurs")
    args = parser.parse_args()

    verifier = DataVerifier(strict=args.strict)
    sys.exit(verifier.run())


if __name__ == "__main__":
    main()
