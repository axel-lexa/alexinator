"""Train BPE with BertPreTokenizer for comparison with the byte-level version."""

from __future__ import annotations

from itertools import chain
from pathlib import Path

from tokenizers import Tokenizer, pre_tokenizers, trainers

from tokenizer import (
    MIN_FREQUENCY,
    SPECIAL_TOKENS,
    VOCAB_SIZE,
    allocine_corpus,
    build_tokenizer,
    local_corpus,
)

OUTPUT = Path(__file__).resolve().parent / "alexinator-bert-bpe-80000.json"


def build_bert_tokenizer() -> Tokenizer:
    tokenizer = build_tokenizer()
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.decoder = None
    return tokenizer


def main() -> None:
    local = local_corpus() or []
    texts = chain(local, allocine_corpus())
    first_text = next(texts, None)
    if first_text is None:
        raise ValueError("The training corpus is empty")

    tokenizer = build_bert_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS,
    )
    tokenizer.train_from_iterator(chain((first_text,), texts), trainer=trainer)
    tokenizer.save(str(OUTPUT))

    print(f"Local texts: {len(local)}; Allociné train split: all available reviews")
    print(f"Requested vocabulary: {VOCAB_SIZE}; actual vocabulary: {tokenizer.get_vocab_size()}")
    print(f"Saved tokenizer: {OUTPUT}")


if __name__ == "__main__":
    main()
