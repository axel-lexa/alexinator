"""Train a byte-level BPE tokenizer."""

from __future__ import annotations

from itertools import islice
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

VOCAB_SIZE = 200_000
MIN_FREQUENCY = 2
ALLOCINE_REVIEWS = 100_000
SPECIAL_TOKENS = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]

OUTPUT = Path(__file__).resolve().parent / "alexinator-bpe.json"


def local_corpus() -> list[str] | None:
    """Placeholder until the intended local training data is selected."""
    ...


def allocine_corpus(limit: int = ALLOCINE_REVIEWS) -> list[str]:
    """Read only the requested number of reviews from the Hugging Face train split."""
    from datasets import load_dataset

    dataset = load_dataset("tblard/allocine", split="train", streaming=True)
    return [row["review"] for row in islice(dataset, limit) if isinstance(row.get("review"), str)]


def build_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def main() -> None:
    local = local_corpus() or []
    reviews = allocine_corpus()
    texts = local + reviews
    if not texts:
        raise ValueError("The training corpus is empty")

    tokenizer = build_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(texts, trainer=trainer, length=len(texts))
    tokenizer.save(str(OUTPUT))

    print(f"Local texts: {len(local)}; Allociné reviews: {len(reviews)}")
    print(f"Requested vocabulary: {VOCAB_SIZE}; actual vocabulary: {tokenizer.get_vocab_size()}")
    print(f"Saved tokenizer: {OUTPUT}")


if __name__ == "__main__":
    main()
