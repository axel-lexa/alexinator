"""Benchmark a tokenizer on Allociné validation reviews and save a text report."""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tokenizers import Tokenizer

TOKENIZER_SOURCE = "tokenizer/alexinator-bpe-200000-f2.json"
DATASET_NAME = "tblard/allocine"
DATASET_SPLIT = "validation"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(__file__).resolve().parent / "tests"


def load_tokenizer(source: str) -> Tokenizer:
    from tokenizers import Tokenizer

    if source.startswith("hf:"):
        repository = source.removeprefix("hf:")
        if not repository:
            raise ValueError("Specify a Hugging Face repository after 'hf:'.")
        return Tokenizer.from_pretrained(repository)

    path = Path(source)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"Tokenizer not found: {path}. Train or download it first.")
    return Tokenizer.from_file(str(path))


def benchmark_ids(encode_ids: Callable[[str], list[int]], unknown_id: int | None = None) -> dict[str, int]:
    from datasets import load_dataset

    reviews = words = total_tokens = unknown_tokens = 0
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT, streaming=True)
    for row in dataset:
        review = row.get("review")
        if not isinstance(review, str) or not review.strip():
            continue
        reviews += 1
        words += len(review.split())
        ids = encode_ids(review)
        total_tokens += len(ids)
        if unknown_id is not None:
            unknown_tokens += ids.count(unknown_id)

    if reviews == 0 or words == 0:
        raise ValueError(f"No nonempty review text in {DATASET_NAME} ({DATASET_SPLIT}).")
    return {
        "reviews": reviews,
        "words": words,
        "total_tokens": total_tokens,
        "unknown_tokens": unknown_tokens,
    }


def result_path(source: str, vocab_size: int) -> Path:
    name = source.removeprefix("hf:") if source.startswith("hf:") else Path(source).stem
    name = re.sub(r"[^a-zA-Z0-9._-]+", "-", name).strip("-.")
    return RESULTS_DIR / f"{name}-vocab-{vocab_size}.txt"


def main(source: str = TOKENIZER_SOURCE) -> Path:
    tokenizer = load_tokenizer(source)
    vocab_size = tokenizer.get_vocab_size()
    metrics = benchmark_ids(
        lambda text: tokenizer.encode(text, add_special_tokens=False).ids,
        tokenizer.token_to_id("[UNK]"),
    )

    report = (
        f"Tokenizer: {source}\n"
        f"Vocabulary size: {vocab_size}\n\n"
        f"{DATASET_NAME} / {DATASET_SPLIT}: {metrics['reviews']} reviews; "
        f"{metrics['words']} whitespace-separated words\n"
        "Special tokens disabled; no truncation.\n"
        f"Total tokens: {metrics['total_tokens']:,}\n"
        f"Tokens/review: {metrics['total_tokens'] / metrics['reviews']:.2f}\n"
        f"Tokens/word: {metrics['total_tokens'] / metrics['words']:.2f}\n"
        f"[UNK] tokens: {metrics['unknown_tokens']:,}\n"
    )
    path = result_path(source, vocab_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"\nSaved results: {path}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", nargs="?", default=TOKENIZER_SOURCE, help="local tokenizer JSON or hf:repository"
    )
    args = parser.parse_args()
    main(args.source)
