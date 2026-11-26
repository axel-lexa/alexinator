"""Variante de notre BPE avec `BertPreTokenizer`, à comparer à la version byte-level.

Tout le reste est identique à `tokenizer.py` : normalizer, special tokens, corpus, `VOCAB_SIZE` et
`MIN_FREQUENCY`. Seul le pre-tokenizer change, donc tout écart de découpage vient de lui.

`BertPreTokenizer` coupe aux espaces et à la ponctuation, puis jette les espaces. Conséquences
visibles dans `tokenizer/tests/alexinator-bert-bpe-80000-f3-vocab-80000.txt` :

- les tokens ne contiennent jamais d'espace, donc pas de doublon « quel » / « Ġquel » ;
- sans alphabet d'octets, un caractère absent du corpus d'entraînement (un emoji rare, par exemple)
  devient [UNK] ;
- sans decoder, `decode` recolle les tokens avec une espace : « titan ik », « l ' histoire ».
  Le texte d'origine n'est pas restitué.

À lancer depuis la racine du projet, comme un fichier :

    uv run --only-group tokenizer python tokenizer/tokenizer-bert.py

Python place alors le dossier du script en tête de `sys.path` : `from tokenizer import ...` charge
`tokenizer/tokenizer.py`, et non le package `tokenizer/`. Le tiret du nom de fichier empêche de toute
façon d'importer ce script comme un module.
"""

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

# Nom écrit à la main, qui ne suit pas les réglages importés : avec VOCAB_SIZE = 200 000 et
# MIN_FREQUENCY = 2, le fichier écrit garde ce nom en « 80000 ». Le fichier versionné,
# alexinator-bert-bpe-80000-f3.json, a bien un vocabulaire de 80 000 tokens.
OUTPUT = Path(__file__).resolve().parent / "alexinator-bert-bpe-80000.json"


def build_bert_tokenizer() -> Tokenizer:
    """Reprend le pipeline de `build_tokenizer` et remplace le pre-tokenizer par `BertPreTokenizer`.

    Le decoder `ByteLevel` est retiré : il attend des tokens écrits avec l'alphabet d'octets, alors
    qu'ici les tokens sont du texte ordinaire.
    """
    tokenizer = build_tokenizer()
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.decoder = None
    return tokenizer


def main() -> None:
    """Entraîne la variante et l'enregistre dans `OUTPUT`, comme `tokenizer.main`.

    Une différence avec `tokenizer.main` : pas d'`initial_alphabet`. Les symboles de départ sont les
    seuls caractères vus dans le corpus d'entraînement, d'où les [UNK] sur des caractères nouveaux.
    """
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
