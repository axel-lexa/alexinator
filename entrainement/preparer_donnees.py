"""Phase 1, étape 1 : corpus français, tokenizer BPE et fichiers de tokens.

    python entrainement/preparer_donnees.py          corpus complet (3 Md de tokens, plusieurs heures)
    python entrainement/preparer_donnees.py --test     mini-corpus pour vérifier la chaîne (1 à 2 minutes)

Ce que fait le script :
1. entraîne un tokenizer BPE au niveau des octets (32 000 tokens) sur un échantillon du corpus ;
2. tokenise Wikipédia en français en entier, puis complète avec FineWeb-2 (français) jusqu'à
   atteindre le nombre de tokens voulu ;
3. écrit les tokens dans train.bin et val.bin (entiers sur 16 bits : 32 000 < 65 536), lus ensuite
   sans tout charger en mémoire (np.memmap).

Les documents sont lus en flux (streaming) : rien n'est téléchargé en entier à l'avance.
Disque nécessaire pour le corpus complet : environ 6 Go pour les tokens, plus le cache Hugging Face.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SPECIAL_TOKENS = ["<|fin_texte|>", "<|pad|>", "<|systeme|>", "<|joueur|>", "<|gardien|>", "<|fin|>"]
# (mêmes tokens, dans le même ordre, que alexinator/gpt.py ; ce script n'importe pas torch)


def wikipedia_fr():
    from datasets import load_dataset

    return (
        r["text"] for r in load_dataset("wikimedia/wikipedia", "20231101.fr", split="train", streaming=True)
    )


def fineweb2_fr():
    from datasets import load_dataset

    return (
        r["text"]
        for r in load_dataset("HuggingFaceFW/fineweb-2", name="fra_Latn", split="train", streaming=True)
    )


def corpus():
    """Wikipédia d'abord (texte propre, culture cinéma), puis FineWeb-2 (français plus varié)."""
    return itertools.chain(wikipedia_fr(), fineweb2_fr())


def interleaved_sample(n_docs: int):
    """Échantillon pour le tokenizer : moitié Wikipédia, moitié FineWeb-2, pour couvrir les deux styles."""
    half = n_docs // 2
    yield from itertools.islice(wikipedia_fr(), half)
    yield from itertools.islice(fineweb2_fr(), n_docs - half)


def train_tokenizer(texts, vocab_size: int, n_docs: int, out_path: Path):
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

    tokenizer = Tokenizer(models.BPE())
    # NFC (et non NFD comme au lab 4) : « é » reste UN caractère au lieu de « e » + accent,
    # ce qui économise des tokens en français.
    tokenizer.normalizer = normalizers.NFC()
    # Niveau octets : tout texte est encodable (emojis compris), aucun token inconnu.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer, length=n_docs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_path))
    return tokenizer


def tokenize_corpus(
    tokenizer, texts, target_tokens: int, out_dir: Path, val_every: int = 200, batch_size: int = 1000
) -> dict:
    """Écrit train.bin / val.bin. Un document sur `val_every` part en validation (0,5 %)."""
    eot = tokenizer.token_to_id("<|fin_texte|>")
    n_train = n_val = n_docs = 0
    start = time.time()
    with open(out_dir / "train.bin", "wb") as f_train, open(out_dir / "val.bin", "wb") as f_val:
        while n_train + n_val < target_tokens:
            batch = list(itertools.islice(texts, batch_size))
            if not batch:
                print("Corpus épuisé avant d'atteindre l'objectif.")
                break
            # encode_batch tokenise en parallèle sur tous les cœurs du processeur
            for enc in tokenizer.encode_batch(batch):
                ids = np.array(enc.ids + [eot], dtype=np.uint16)
                if n_docs % val_every == 0:
                    ids.tofile(f_val)
                    n_val += len(ids)
                else:
                    ids.tofile(f_train)
                    n_train += len(ids)
                n_docs += 1
            if n_docs % (batch_size * 50) == 0:
                total = n_train + n_val
                speed = total / (time.time() - start)
                print(
                    f"{n_docs:>10,} documents | {total / 1e9:6.3f} Md de tokens | "
                    f"{speed / 1e6:.2f} M tokens/s | reste ~{(target_tokens - total) / speed / 3600:.1f} h"
                )
    return {"documents": n_docs, "train_tokens": n_train, "val_tokens": n_val}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prépare le corpus de pré-entraînement")
    parser.add_argument("--test", action="store_true", help="mini-corpus pour vérifier la chaîne")
    parser.add_argument("--tokens", type=float, default=3e9, help="nombre de tokens visé")
    parser.add_argument("--vocab", type=int, default=32000)
    parser.add_argument(
        "--docs-tokenizer", type=int, default=400_000, help="documents utilisés pour entraîner le tokenizer"
    )
    parser.add_argument("--sortie", default=None, help="dossier de sortie (défaut : entrainement/donnees)")
    args = parser.parse_args()

    if args.test:
        args.tokens, args.vocab, args.docs_tokenizer = 2e6, 4000, 2000
    out_dir = (
        Path(args.sortie)
        if args.sortie
        else ROOT / "entrainement" / ("donnees_test" if args.test else "donnees")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    tok_path = out_dir / "tokenizer.json"

    print(f"=== 1/2 Tokenizer BPE ({args.vocab} tokens, {args.docs_tokenizer} documents) ===")
    if tok_path.exists():
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(tok_path))
        print(f"Déjà entraîné : {tok_path}")
    else:
        tokenizer = train_tokenizer(
            interleaved_sample(args.docs_tokenizer), args.vocab, args.docs_tokenizer, tok_path
        )
    sample = "Le gardien cache un film culte des années 1990, réalisé par un cinéaste américain."
    enc = tokenizer.encode(sample)
    print(f"Exemple : {len(sample.split())} mots -> {len(enc.ids)} tokens : {enc.tokens[:12]}...")

    print(f"=== 2/2 Tokenisation du corpus ({args.tokens / 1e9:.3f} Md de tokens visés) ===")
    stats = tokenize_corpus(tokenizer, corpus(), int(args.tokens), out_dir)
    stats.update(vocab_size=tokenizer.get_vocab_size(), tokenizer=str(tok_path))
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"Terminé : {stats}")


if __name__ == "__main__":
    main()
