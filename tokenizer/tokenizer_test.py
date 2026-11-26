"""Mesure un tokenizer sur les critiques de validation Allociné et écrit un rapport texte.

Malgré son nom, ce n'est pas un fichier de tests pytest : c'est un benchmark. Pour chaque critique
non vide du split `validation` de `tblard/allocine` (20 000 critiques, jamais vues à
l'entraînement, qui utilise le split `train`), il compte les mots et les tokens. Le rapport donne :

- Tokens/word : la fertility, nombre moyen de tokens par mot. Plus elle est basse, moins un texte
  coûte de tokens, et plus il en tient dans la context window d'un LLM ;
- Tokens/review : nombre moyen de tokens par critique ;
- [UNK] tokens : nombre de tokens inconnus, c'est-à-dire d'information perdue.

Le tokenizer mesuré est :

- un fichier JSON local, avec un chemin relatif à la racine du projet (par défaut `TOKENIZER_SOURCE`) ;
- ou `hf:REPO` pour un tokenizer du Hub Hugging Face, par exemple `hf:gpt2` ou `hf:almanach/camembert-base`.

    uv run --only-group tokenizer python tokenizer/tokenizer_test.py
    uv run --only-group tokenizer python tokenizer/tokenizer_test.py hf:almanach/camembert-base

Le rapport est écrit dans `tokenizer/tests/{nom}-vocab-{taille}.txt`, puis lu par
`plot_token_counts.py`. Il reste en anglais : `plot_token_counts.py` le lit avec des regex.
"""

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
    """Charge un `Tokenizer` depuis un fichier JSON local ou depuis le Hub (préfixe `hf:`).

    `Tokenizer.from_pretrained` télécharge le fichier `tokenizer.json` du dépôt : seuls les dépôts
    qui fournissent un fast tokenizer marchent. Un chemin relatif part de la racine du projet, et
    non du dossier courant.
    """
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
    """Encode chaque critique non vide de la validation et additionne critiques, mots, tokens et inconnus.

    La fonction reçoit une fonction d'encodage plutôt qu'un `Tokenizer`. Elle peut donc mesurer
    n'importe quelle bibliothèque qui renvoie une liste d'IDs.

    Les mots sont ceux de `review.split()`, séparés par les espaces, avec la ponctuation collée.
    Le découpage est le même pour tous les tokenizers : Tokens/word se compare d'un tokenizer à
    l'autre. Si `unknown_id` vaut `None`, les tokens inconnus ne sont pas comptés.
    """
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
    """Chemin du rapport : `tokenizer/tests/{nom}-vocab-{vocab_size}.txt`.

    Le nom est celui du dépôt (`hf:almanach/camembert-base` donne `almanach-camembert-base`) ou du
    fichier sans `.json`. Les caractères hors de `[a-zA-Z0-9._-]` deviennent des tirets.
    `vocab_size` est la taille réelle du vocabulaire, qui peut être plus petite que celle demandée à
    l'entraînement (99 252 pour `alexinator-bpe-100000-f3`).
    """
    name = source.removeprefix("hf:") if source.startswith("hf:") else Path(source).stem
    name = re.sub(r"[^a-zA-Z0-9._-]+", "-", name).strip("-.")
    return RESULTS_DIR / f"{name}-vocab-{vocab_size}.txt"


def main(source: str = TOKENIZER_SOURCE) -> Path:
    """Charge le tokenizer, le mesure sur la validation, écrit le rapport et renvoie son chemin.

    `add_special_tokens=False` : on ne compte pas les special tokens que certains post-processors
    ajoutent autour du texte ([CLS] et [SEP] pour BERT, <s> et </s> pour CamemBERT). On ne mesure
    que le texte.

    Limite : le token inconnu est cherché sous le nom [UNK]. CamemBERT appelle le sien <unk>, et
    GPT-2 n'en a pas. Pour ces tokenizers, la ligne « [UNK] tokens: 0 » veut dire « non compté », et
    non « aucun token inconnu ».
    """
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
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "source", nargs="?", default=TOKENIZER_SOURCE, help="fichier JSON local ou hf:dépôt du Hub"
    )
    args = parser.parse_args()
    main(args.source)
