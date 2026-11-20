"""Plot tokenizer benchmarks from the reports in tokenizer/tests.

Run from the project root with ``uv run --group tokenizer python tokenizer/plot_token_counts.py``.
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
    name: str
    vocabulary_size: int
    reviews: int
    words: int
    tokens: int

    @property
    def label(self) -> str:
        return self.name.removeprefix("tokenizer/")


def report_value(contents: str, pattern: str, path: Path) -> str:
    match = re.search(pattern, contents, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"Missing benchmark field in {path}: {pattern}")
    return match.group(1)


def read_benchmarks(directory: Path) -> list[Benchmark]:
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
    """Match tokenizer_test.py: ignore empty reviews and split words on whitespace."""
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Tokenizer directory containing the tests/ benchmark reports",
    )
    parser.add_argument("--show", action="store_true", help="Display the three charts")
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
