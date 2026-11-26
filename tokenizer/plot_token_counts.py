"""Trace trois graphiques à partir des rapports de `tokenizer/tests/` (écrits par `tokenizer_test.py`).

- `vocabulary_sizes.png` : taille réelle du vocabulaire de chaque tokenizer. Les corpus
  d'entraînement ne sont pas les mêmes d'un tokenizer à l'autre.
- `tokens_per_word.png` : tokens par mot (fertility) sur la validation Allociné. Plus petit = texte
  plus compact.
- `letters_per_token.png` : lettres par token. Plus grand = tokens plus longs en moyenne.

En bleu, les tokenizers entraînés en local. En vert, ceux dont la source commence par `hf:` (téléchargés
du Hub). Le script relit aussi la validation Allociné : les rapports ne contiennent pas le nombre de
lettres, et il faut vérifier que tous les rapports portent sur les mêmes textes.

À lancer depuis la racine du projet. `--group` (et non `--only-group`) ajoute le groupe `app`, qui
contient matplotlib :

    uv run --group tokenizer python tokenizer/plot_token_counts.py
"""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from datasets import load_dataset

DATASET_NAME = "tblard/allocine"
DATASET_SPLIT = "validation"


@dataclass(frozen=True)
class Benchmark:
    """Les chiffres d'un rapport dont les graphiques ont besoin.

    `name` est la ligne `Tokenizer:` du rapport, `vocabulary_size` la taille réelle du vocabulaire et
    `tokens` le nombre total de tokens sur toute la validation.
    """

    name: str
    vocabulary_size: int
    reviews: int
    words: int
    tokens: int

    @property
    def label(self) -> str:
        """Nom affiché sur le graphique : la source sans le préfixe `tokenizer/`."""
        return self.name.removeprefix("tokenizer/")


def report_value(contents: str, pattern: str, path: Path) -> str:
    """Renvoie le premier groupe capturé par `pattern` dans le rapport.

    `re.MULTILINE` permet à `^` et `$` de viser une ligne du rapport. Une ligne absente, dans un
    rapport d'un ancien format par exemple, donne une erreur qui nomme le fichier.
    """
    match = re.search(pattern, contents, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"Missing benchmark field in {path}: {pattern}")
    return match.group(1)


def read_benchmarks(directory: Path) -> list[Benchmark]:
    """Lit tous les rapports `*-vocab-*.txt` de `directory/tests`, triés par taille de vocabulaire.

    La fonction refuse un rapport mesuré sur un autre dataset ou un autre split. Elle refuse aussi des
    rapports qui n'ont pas tous le même nombre de critiques et de mots : les comparer n'aurait pas de
    sens. Les rapports ont deux formats (avec ou sans exemples de découpage) : seules les lignes
    communes aux deux sont lues.
    """
    reports_dir = directory / "tests"
    text_paths = sorted(reports_dir.glob("*-vocab-*.txt"))
    if not text_paths:
        raise ValueError(f"No benchmark reports found in {directory / 'tests'}")

    benchmarks: list[Benchmark] = []
    for path in text_paths:
        contents = path.read_text(encoding="utf-8")
        if f"{DATASET_NAME} / {DATASET_SPLIT}:" not in contents:
            raise ValueError(f"Unexpected dataset or split in {path}")
        benchmark = Benchmark(
            name=report_value(contents, r"^Tokenizer: (.+)$", path),
            vocabulary_size=int(
                report_value(contents, r"^Vocabulary size: ([\d,]+)$", path).replace(",", "")
            ),
            reviews=int(report_value(contents, r"validation: (\d+) reviews;", path)),
            words=int(report_value(contents, r"reviews; (\d+) whitespace-separated words", path)),
            tokens=int(report_value(contents, r"^Total tokens: ([\d,]+)$", path).replace(",", "")),
        )
        benchmarks.append(benchmark)

    if len({(benchmark.reviews, benchmark.words) for benchmark in benchmarks}) != 1:
        raise ValueError("Benchmark reports do not use the same validation texts")
    return sorted(benchmarks, key=lambda benchmark: (benchmark.vocabulary_size, benchmark.name))


def count_corpus_letters() -> tuple[int, int, int]:
    """Compte critiques, mots et lettres de la validation, avec les règles de `tokenizer_test.py`.

    Comme dans `tokenizer_test.py`, les critiques vides sont ignorées et les mots sont séparés par les
    espaces. Une lettre est un caractère pour lequel `str.isalpha` est vrai : les lettres accentuées
    comptent, les chiffres et la ponctuation non. Le nombre de lettres ne dépend pas du tokenizer :
    on le compte une seule fois pour tous.
    """
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT, streaming=True)
    reviews = words = letters = 0
    for row in dataset:
        review = row.get("review")
        if not isinstance(review, str) or not review.strip():
            continue
        reviews += 1
        words += len(review.split())
        letters += sum(character.isalpha() for character in review)
    return reviews, words, letters


def save_chart(
    benchmarks: list[Benchmark], values: list[float], title: str, xlabel: str, output: Path, *, decimals: int
) -> None:
    """Enregistre un graphique en barres horizontales, avec une barre par tokenizer et sa valeur au bout.

    La hauteur de la figure suit le nombre de tokenizers. `invert_yaxis` place le premier de la liste
    (le plus petit vocabulaire) en haut.
    """
    labels = [benchmark.label for benchmark in benchmarks]
    colors = ["#43a047" if benchmark.name.startswith("hf:") else "#1e88e5" for benchmark in benchmarks]
    fig, ax = plt.subplots(figsize=(11, max(6, 0.48 * len(labels) + 1.5)))
    bars = ax.barh(labels, values, color=colors)
    ax.bar_label(bars, labels=[f"{value:,.{decimals}f}" for value in values], padding=4)
    ax.set_xlim(0, max(values) * 1.2)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Saved {output}")


def main() -> None:
    """Lit les rapports, vérifie qu'ils portent sur le dataset actuel et trace les trois graphiques.

    Lettres par token = lettres de toute la validation / tokens émis. Le dénominateur compte aussi les
    tokens de ponctuation et d'espaces, qui n'ont aucune lettre. La valeur est donc plus basse que la
    longueur moyenne d'un token de mot.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="dossier du tokenizer qui contient les rapports dans tests/",
    )
    parser.add_argument("--show", action="store_true", help="affiche aussi les trois graphiques")
    args = parser.parse_args()

    benchmarks = read_benchmarks(args.directory)
    reviews, words, letters = count_corpus_letters()
    expected = (benchmarks[0].reviews, benchmarks[0].words)
    if (reviews, words) != expected:
        parser.error(f"Dataset has {reviews} reviews and {words} words; reports expect {expected}")

    title_suffix = f"Allociné validation ({reviews:,} reviews)"
    save_chart(
        benchmarks,
        [benchmark.vocabulary_size for benchmark in benchmarks],
        "Tokenizer vocabulary size (training corpora vary)",
        "Tokens in vocabulary",
        args.directory / "vocabulary_sizes.png",
        decimals=0,
    )
    save_chart(
        benchmarks,
        [benchmark.tokens / words for benchmark in benchmarks],
        f"Average tokens per word - {title_suffix}",
        "Tokens / whitespace-separated word",
        args.directory / "tokens_per_word.png",
        decimals=2,
    )
    save_chart(
        benchmarks,
        [letters / benchmark.tokens for benchmark in benchmarks],
        f"Average letters per token - {title_suffix}",
        "Unicode letters / emitted token (including punctuation tokens)",
        args.directory / "letters_per_token.png",
        decimals=2,
    )
    print("Blue: locally trained tokenizer; green: open source tokenizer from Hugging Face.")
    if args.show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
