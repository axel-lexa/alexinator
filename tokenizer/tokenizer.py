"""Entraîne le tokenizer BPE d'Alexinator sur les critiques de films Allociné.

Un tokenizer découpe un texte en tokens (mots ou morceaux de mots) et donne à chacun un ID entier.
Un LLM ne voit que ces IDs. Ce script entraîne un tokenizer BPE (Byte-Pair Encoding) avec la
bibliothèque Hugging Face `tokenizers`, sur le split `train` du dataset `tblard/allocine`.

Chaque texte traverse le pipeline monté par `build_tokenizer` :

    encodage : texte -> normalizer -> pre-tokenizer -> model (BPE) -> IDs
    décodage : IDs -> decoder -> texte (normalisé)

Deux modes :

- sans option, `main` entraîne notre BPE byte-level et l'enregistre dans `OUTPUT` ;
- `--from-pretrained REPO --vocab-size N` réentraîne un tokenizer du Hub (par exemple
  `almanach/camembert-base`) sur le même corpus, en gardant son algorithme et son pipeline.

À lancer depuis la racine du projet :

    uv run --only-group tokenizer python tokenizer/tokenizer.py
    uv run --only-group tokenizer --with transformers python tokenizer/tokenizer.py \\
        --from-pretrained almanach/camembert-base --vocab-size 100000

Les messages affichés restent en anglais, comme les rapports de `tokenizer/tests/`.
Explications détaillées et résultats : `docs/tokenizer.md`.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterator
from itertools import chain
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

# Taille maximale du vocabulaire : special tokens + 256 symboles d'octets + un token par merge.
# L'entraînement s'arrête plus tôt si aucune paire n'atteint MIN_FREQUENCY : 200 000 demandés donnent
# 123 367 tokens avec MIN_FREQUENCY = 2, et 99 252 avec MIN_FREQUENCY = 3.
VOCAB_SIZE = 200_000
# Nombre minimal d'occurrences d'une paire de symboles pour qu'elle devienne un merge. Plus la valeur
# est haute, moins on apprend de tokens rares (fautes de frappe, noms propres cités une seule fois).
MIN_FREQUENCY = 2
# Special tokens : jamais découpés, ils reçoivent les IDs 0 à 4 dans cet ordre. Les noms viennent de BERT :
# [UNK] (token inconnu), [CLS] (début de séquence), [SEP] (séparateur), [PAD] (remplissage d'un batch),
# [MASK] (masked language modeling). Le pipeline n'a pas de post-processor : l'encodage ne les ajoute
# jamais tout seul, ils sont seulement réservés dans le vocabulaire.
SPECIAL_TOKENS = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]

# Convention de nom : alexinator-bpe-{VOCAB_SIZE}-f{MIN_FREQUENCY}.json. Le nom n'est pas calculé :
# le mettre à jour avec VOCAB_SIZE et MIN_FREQUENCY, sinon on écrase un tokenizer déjà versionné.
OUTPUT = Path(__file__).resolve().parent / "alexinator-bpe-200000-f2.json"


def local_corpus() -> list[str] | None:
    """Textes du projet à ajouter au corpus d'entraînement, avant les critiques Allociné.

    Pas encore choisis : la fonction renvoie `None` (son corps est `...`), et les appelants
    utilisent alors seulement Allociné (`local_corpus() or []`).
    """
    ...


def allocine_corpus() -> Iterator[str]:
    """Renvoie une à une les critiques non vides du split `train` de `tblard/allocine`.

    Avec `streaming=True`, le dataset est lu au fil de l'itération au lieu d'être chargé en entier
    en mémoire. `datasets` est importé ici pour que ce module se charge même sans cette bibliothèque.
    """
    from datasets import load_dataset

    dataset = load_dataset("tblard/allocine", split="train", streaming=True)
    for row in dataset:
        review = row.get("review")
        if isinstance(review, str) and review.strip():
            yield review


def build_tokenizer() -> Tokenizer:
    """Monte le pipeline du tokenizer, avec un model BPE encore vide.

    - model `BPE` : vide ici. L'entraînement le remplit avec le vocabulaire et la liste des merges.
      `unk_token` sert aux caractères absents du vocabulaire, ce qui n'arrive pas en byte-level.
    - normalizer : `NFD` sépare chaque lettre de son accent (« é » devient « e » + accent),
      `Lowercase` passe en minuscules et `StripAccents` retire les accents. « AMÉLIE » et « amelie »
      donnent donc les mêmes tokens. Cela fait moins de variantes à apprendre, mais l'information est
      perdue : le décodage rend « j'ai adore amelie », pas le texte d'origine.
    - pre-tokenizer `ByteLevel` : passe le texte en octets UTF-8 et écrit chaque octet avec un
      caractère imprimable (l'espace devient « Ġ », un emoji devient plusieurs symboles comme « ðŁ »).
      Il découpe ensuite en mots avec la regex de GPT-2, et chaque espace reste collée au mot qui la suit.
      Avec `add_prefix_space=False`, le premier mot du texte n'a pas d'espace devant : « quel » en
      début de texte (ID 1170) et « Ġquel » après une espace (ID 544) sont deux tokens différents.
    - decoder `ByteLevel` : fait l'inverse, des symboles aux octets, puis au texte UTF-8.
    """
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def main() -> None:
    """Entraîne le BPE byte-level sur le corpus local et Allociné, puis l'enregistre dans `OUTPUT`.

    Le premier texte est lu à part pour échouer tout de suite si le corpus est vide. `chain` le
    remet ensuite en tête de l'itérateur.

    Le `BpeTrainer` part d'un vocabulaire de symboles de base. Il compte les paires de symboles
    voisins dans les mots du corpus, fusionne la plus fréquente en un nouveau token (un merge),
    et recommence. Il s'arrête à `VOCAB_SIZE` tokens, ou plus tôt si plus aucune paire
    n'apparaît `MIN_FREQUENCY` fois.

    `initial_alphabet=ByteLevel.alphabet()` met d'office les 256 symboles d'octets dans le
    vocabulaire. Tout texte s'encode alors sans [UNK] : un emoji jamais vu est découpé en octets.

    Ordre des IDs dans le fichier : 0 à 4 pour les special tokens, 5 à 260 pour les octets, puis un
    token par merge, dans l'ordre où ils ont été appris (les premiers sont « Ġd », « es », « en »...).
    """
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
    """Réentraîne un tokenizer du Hub sur le même corpus que notre BPE et renvoie le fichier écrit.

    `train_new_from_iterator` garde tout le pipeline d'origine : normalizer, pre-tokenizer,
    special tokens et algorithme (Unigram pour CamemBERT, BPE pour GPT-2). Il n'apprend qu'un
    nouveau vocabulaire de `vocab_size` tokens. Comparé au tokenizer d'origine, on mesure l'effet
    du corpus seul. Comparé à notre BPE, on mesure l'effet du pipeline et de l'algorithme.

    Seul un fast tokenizer (écrit en Rust, adossé à `tokenizers`) se réentraîne ainsi.
    `backend_tokenizer.save` écrit le même format JSON que notre BPE, donc `tokenizer_test.py`
    peut le lire. Le fichier s'appelle `{source}-allocine-{vocab_size}.json`, avec les caractères
    spéciaux de `source` remplacés par des tirets.
    """
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
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--from-pretrained", help="fast tokenizer du Hub à réentraîner, par exemple almanach/camembert-base"
    )
    parser.add_argument("--vocab-size", type=int, help="taille du vocabulaire pour --from-pretrained")
    args = parser.parse_args()
    # La taille de notre BPE se règle avec VOCAB_SIZE : --vocab-size ne vaut que pour --from-pretrained.
    if args.from_pretrained:
        if args.vocab_size is None or args.vocab_size < 1:
            parser.error("--from-pretrained requires a positive --vocab-size")
        train_pretrained(args.from_pretrained, args.vocab_size)
    elif args.vocab_size is not None:
        parser.error("--vocab-size requires --from-pretrained")
    else:
        main()
