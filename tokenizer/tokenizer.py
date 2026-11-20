"""Train a local BPE or retrain a pretrained tokenizer on Allociné reviews."""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterator
from itertools import chain
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

VOCAB_SIZE = 200_000
MIN_FREQUENCY = 2
SPECIAL_TOKENS = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]

OUTPUT = Path(__file__).resolve().parent / "alexinator-bpe-200000-f2.json"


def local_corpus() -> list[str] | None:
    """Placeholder until the intended local training data is selected."""
    ...


def allocine_corpus() -> Iterator[str]:
    """Stream every review from the Hugging Face train split as plain text."""
    from datasets import load_dataset

    dataset = load_dataset("tblard/allocine", split="train", streaming=True)
    for row in dataset:
        review = row.get("review")
        if isinstance(review, str) and review.strip():
            yield review


def build_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def main() -> None:
    local = local_corpus() or []
    texts = chain(local, allocine_corpus())
    first_text = next(texts, None)
    if first_text is None:
        raise ValueError("The training corpus is empty")

    tokenizer = build_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(chain((first_text,), texts), trainer=trainer)
    tokenizer.save(str(OUTPUT))

    print(f"Local texts: {len(local)}; Allociné train split: all available reviews")
    print(f"Requested vocabulary: {VOCAB_SIZE}; actual vocabulary: {tokenizer.get_vocab_size()}")
    print(f"Saved tokenizer: {OUTPUT}")


def train_pretrained(source: str, vocab_size: int) -> Path:
    """Retrain a fast Hugging Face tokenizer on the same Allociné train corpus."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install transformers or run with `uv run --with transformers`.") from exc

    tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError(f"{source} does not provide a fast tokenizer")

    local = local_corpus() or []
    trained = tokenizer.train_new_from_iterator(chain(local, allocine_corpus()), vocab_size=vocab_size)
    name = re.sub(r"[^a-zA-Z0-9._-]+", "-", source).strip("-.")
    output = Path(__file__).resolve().parent / f"{name}-allocine-{vocab_size}.json"
    trained.backend_tokenizer.save(str(output))

    print(f"Local texts: {len(local)}; Allociné train split: all available reviews")
    print(f"Requested vocabulary: {vocab_size}; actual vocabulary: {len(trained)}")
    print(f"Saved tokenizer: {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-pretrained", help="Fast Hugging Face tokenizer to retrain, e.g. almanach/camembert-base"
    )
    parser.add_argument("--vocab-size", type=int, help="Vocabulary size for --from-pretrained")
    args = parser.parse_args()
    if args.from_pretrained:
        if args.vocab_size is None or args.vocab_size < 1:
            parser.error("--from-pretrained requires a positive --vocab-size")
        train_pretrained(args.from_pretrained, args.vocab_size)
    elif args.vocab_size is not None:
        parser.error("--vocab-size requires --from-pretrained")
    else:
        main()
