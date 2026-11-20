"""Phase 1, étape 2 : pré-entraînement d'Alexinator-110M (prédiction du token suivant).

    python entrainement/pretrain.py                 entraînement complet (14 à 22 h estimées)
    python entrainement/pretrain.py --test          mini-modèle sur le mini-corpus (1 à 2 minutes)
    python entrainement/pretrain.py --reprendre     repart du dernier checkpoint après une coupure
    python entrainement/pretrain.py --profil r9700  Alexinator-240M, réglé pour une Radeon AI PRO R9700 (32 Go)
    python entrainement/pretrain.py --profil r9700 --banc 20   mesure la vitesse sur 20 pas, sans rien sauver

Même boucle que le lab 5 (get_batch -> loss -> backward -> step), avec en plus :
accumulation de gradients, précision mixte bfloat16, learning rate warmup + cosinus,
gradient clipping, sauvegardes régulières et reprise.
Journal : entrainement/runs/<nom>/journal.csv (+ échantillons de texte dans echantillons.txt).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "entrainement"))

import materiel  # noqa: E402

materiel.preparer_environnement()  # avant `import torch` : variables lues au démarrage de PyTorch

import numpy as np  # noqa: E402
import torch  # noqa: E402

from alexinator.gpt import FIN_TEXTE, GPT, GPTConfig  # noqa: E402

# Réglages par carte graphique. Une option donnée en ligne de commande l'emporte toujours sur le profil.
PROFILS = {
    # AMD Radeon AI PRO R9700 : RDNA4 (gfx1201), 32 Go, bfloat16 matériel (191 TFLOPS en crête).
    # Assez rapide pour un modèle 2 fois plus gros que la version 110M sur les mêmes 3 Md de tokens :
    # 16 couches, d_model 1024, têtes de 64 (taille gérée par les noyaux d'attention AOTriton de RDNA4),
    # dimensions multiples de 128 pour les unités matricielles (WMMA). 238 M de paramètres.
    "r9700": dict(
        n_layer=16,
        n_head=16,
        d_model=1024,
        d_ff=2816,
        micro_batch="auto",  # plus grand micro-batch qui tient dans les 32 Go
        lr=4e-4,  # un peu plus bas que pour 110M : les modèles plus larges supportent moins bien 6e-4
        lr_min=4e-5,
        eval_batchs=40,
        sauver_tous=250,
        nom="alexinator-240m",
    ),
}


def parse_args():
    p = argparse.ArgumentParser(description="Pré-entraînement d'Alexinator-110M")
    p.add_argument(
        "--profil", choices=sorted(PROFILS), default=None, help="réglages prêts pour une carte graphique"
    )
    p.add_argument("--test", action="store_true", help="mini-modèle, quelques pas, pour vérifier la chaîne")
    p.add_argument("--donnees", default=None, help="dossier produit par preparer_donnees.py")
    p.add_argument("--nom", default=None, help="nom du dossier de sortie dans entrainement/runs/")
    p.add_argument("--reprendre", action="store_true")
    # Modèle (valeurs du document de modélisation)
    p.add_argument("--n-layer", type=int, default=12)
    p.add_argument("--n-head", type=int, default=12)
    p.add_argument("--d-model", type=int, default=768)
    p.add_argument("--d-ff", type=int, default=2048)
    p.add_argument("--block-size", type=int, default=1024)
    # Entraînement
    p.add_argument(
        "--micro-batch",
        default="16",
        help="séquences par passage sur le GPU, ou « auto » : la plus grande taille qui tient en mémoire",
    )
    p.add_argument(
        "--accumulation",
        type=int,
        default=32,
        help="micro-batchs cumulés par pas (16 x 1024 x 32 ≈ 512 000 tokens) ; calculé si --micro-batch auto",
    )
    p.add_argument(
        "--tokens-par-pas",
        type=int,
        default=16 * 1024 * 32,
        help="avec --micro-batch auto : tokens par pas d'optimisation (fixe l'accumulation)",
    )
    p.add_argument("--tokens", type=float, default=3e9, help="tokens à voir au total (fixe le nombre de pas)")
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--lr-min", type=float, default=6e-5)
    p.add_argument("--warmup", type=int, default=200, help="pas de montée progressive du learning rate")
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--eval-tous", type=int, default=250, help="évaluation + échantillons tous les N pas")
    p.add_argument("--eval-batchs", type=int, default=50)
    p.add_argument("--sauver-tous", type=int, default=1000)
    p.add_argument("--no-compile", action="store_true", help="désactive torch.compile")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument(
        "--banc", type=int, default=0, help="mesure la vitesse sur N pas puis s'arrête (rien n'est sauvé)"
    )
    # Le profil change les valeurs par défaut ; les options écrites explicitement restent prioritaires.
    profil, _ = p.parse_known_args()
    if profil.profil:
        p.set_defaults(**{k: v for k, v in PROFILS[profil.profil].items() if k != "nom"})
    args = p.parse_args()
    default_name = PROFILS[args.profil]["nom"] if args.profil else "alexinator-110m"

    if args.test:
        args.n_layer, args.n_head, args.d_model, args.d_ff, args.block_size = 2, 4, 128, 352, 128
        args.micro_batch, args.accumulation, args.tokens = 8, 2, 8 * 2 * 128 * 40  # 40 pas
        args.warmup, args.eval_tous, args.eval_batchs, args.sauver_tous = 5, 20, 5, 20
        args.no_compile = True
    args.donnees = (
        Path(args.donnees)
        if args.donnees
        else ROOT / "entrainement" / ("donnees_test" if args.test else "donnees")
    )
    args.nom = args.nom or ("test" if args.test else default_name)
    if args.banc:
        args.banc = max(args.banc, 5)  # les 3 premiers pas ne sont pas chronométrés
        args.nom = f"banc-{args.nom}"
    if str(args.micro_batch) != "auto":
        args.micro_batch = int(args.micro_batch)
    return args


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and not args.test:
        sys.exit("Aucun GPU détecté : le pré-entraînement complet doit tourner sur le serveur.")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Précision mixte selon la carte (NVIDIA ou AMD) : voir materiel.choisir_precision
    print(f"Carte : {materiel.decrire_carte(device)}")
    amp_dtype, scaler = materiel.choisir_precision(device)
    autocast = (lambda: torch.autocast("cuda", dtype=amp_dtype)) if amp_dtype else nullcontext

    # ----------------------------------------------------------------- données
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(args.donnees / "tokenizer.json"))
    train_data = np.memmap(args.donnees / "train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap(args.donnees / "val.bin", dtype=np.uint16, mode="r")
    T = args.block_size

    def get_batch(split: str):
        """Comme au lab 5 : fenêtres prises au hasard ; la cible est la fenêtre décalée d'un token."""
        data = train_data if split == "train" else val_data
        ix = torch.randint(len(data) - T - 1, (args.micro_batch,))
        x = torch.stack([torch.from_numpy(data[i : i + T].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + T].astype(np.int64)) for i in ix])
        if device == "cuda":
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x, y

    # ------------------------------------------------------------------ modèle
    run_dir = ROOT / "entrainement" / "runs" / args.nom
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "dernier.pt"
    cfg = GPTConfig(
        vocab_size=tokenizer.get_vocab_size(),
        block_size=T,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        d_ff=args.d_ff,
        dropout=0.0,
    )
    if args.micro_batch == "auto":
        # Avant de charger le modèle : les essais tournent dans des processus séparés sur la même carte.
        if device != "cuda":
            sys.exit("--micro-batch auto demande un GPU.")
        sequences = max(1, args.tokens_par_pas // T)
        # On ne garde que les tailles qui divisent le nombre de séquences par pas : le pas reste
        # exactement de --tokens-par-pas tokens, même si une reprise choisit une autre taille.
        candidats = [c for c in (64, 32, 16, 8, 4, 2, 1) if sequences % c == 0]
        args.micro_batch = materiel.trouver_micro_batch(cfg.to_dict(), amp_dtype, candidats)
        args.accumulation = sequences // args.micro_batch
    model = GPT(cfg).to(device)
    optimizer = model.configure_optimizer(args.lr, args.weight_decay)
    start_step, best_val = 0, float("inf")
    if args.reprendre and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step, best_val = ckpt["step"], ckpt.get("best_val", float("inf"))
        print(f"Reprise au pas {start_step}")
        del ckpt

    tokens_per_step = args.micro_batch * T * args.accumulation
    max_steps = math.ceil(args.tokens / tokens_per_step)
    if args.banc:
        max_steps = min(max_steps, start_step + args.banc)

    print(f"Modèle : {model.num_params() / 1e6:.1f} M de paramètres | {device} | précision {amp_dtype}")
    print(f"Micro-batch {args.micro_batch} x accumulation {args.accumulation} x {T} tokens")
    print(f"Données : {len(train_data) / 1e9:.3f} Md tokens (train), {len(val_data) / 1e6:.1f} M (val)")
    print(
        f"{max_steps} pas de {tokens_per_step:,} tokens = "
        f"{max_steps * tokens_per_step / 1e9:.2f} Md de tokens"
    )
    if max_steps * tokens_per_step > len(train_data):
        print("Attention : plus de tokens demandés que dans le corpus, certains seront revus.")

    raw_model = model  # le modèle compilé partage ses poids avec raw_model (utilisé pour sauver et générer)
    possible, raison = materiel.compilation_possible(device)
    if device == "cuda" and not possible and not args.no_compile:
        print(f"torch.compile désactivé ({raison}).")
        args.no_compile = True
    if device == "cuda" and not args.no_compile:
        # torch.compile ne compile vraiment qu'au premier appel : une erreur (Triton absent, carte non
        # gérée...) n'apparaîtrait qu'au premier pas. On fait donc un passage d'essai tout de suite.
        print("Compilation du modèle (torch.compile, 1 à 3 minutes)...")
        try:
            model = torch.compile(raw_model)
            x, y = get_batch("train")
            with autocast():
                _, loss = model(x, y)
            loss.backward()
            print("Compilation réussie.")
        except Exception as error:  # compilation indisponible : on continue sans
            print(f"torch.compile indisponible ({type(error).__name__}: {error}), entraînement sans compilation.")
            model = raw_model
        finally:
            raw_model.zero_grad(set_to_none=True)

    def lr_at(step: int) -> float:
        """Warmup linéaire, puis décroissance en cosinus jusqu'à lr_min."""
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        progress = min(1.0, (step - args.warmup) / max(1, max_steps - args.warmup))
        return args.lr_min + 0.5 * (args.lr - args.lr_min) * (1 + math.cos(math.pi * progress))

    @torch.no_grad()
    def evaluate() -> dict:
        model.eval()
        out = {}
        for split in ("train", "val"):
            losses = []
            for _ in range(args.eval_batchs):
                x, y = get_batch(split)
                with autocast():
                    losses.append(model(x, y)[1].item())
            out[split] = sum(losses) / len(losses)
        model.train()
        return out

    def samples(n: int = 3, length: int = 60) -> list:
        eot = tokenizer.token_to_id(FIN_TEXTE)
        texts = []
        for prompt in ["Le film", "En 1997,", "Le réalisateur"][:n]:
            idx = torch.tensor([tokenizer.encode(prompt).ids], device=device)
            with autocast():
                out = raw_model.generate(idx, length, temperature=0.8, top_k=50, stop_id=eot)
            texts.append(tokenizer.decode(out[0].tolist()))
        return texts

    def save(step: int, path: Path) -> None:
        torch.save(
            {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": cfg.to_dict(),
                "step": step,
                "best_val": best_val,
                "tokenizer": str(args.donnees / "tokenizer.json"),
            },
            path,
        )

    # ------------------------------------------------------------ entraînement
    log_path = run_dir / "journal.csv"
    new_log = not log_path.exists() or not args.reprendre
    log_file = open(log_path, "w" if new_log else "a", newline="")
    log = csv.writer(log_file)
    if new_log:
        log.writerow(["pas", "tokens_vus", "loss_train", "loss_val", "lr", "tokens_par_s", "memoire_go"])
    (run_dir / "config.json").write_text(
        json.dumps({**cfg.to_dict(), **{k: str(v) for k, v in vars(args).items()}}, indent=2)
    )

    model.train()
    t0 = time.time()
    for step in range(start_step, max_steps):
        lr = lr_at(step)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Évaluation, échantillons et sauvegarde régulières
        if not args.banc and (step % args.eval_tous == 0 or step == max_steps - 1):
            losses = evaluate()
            mem = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
            print(
                f"[pas {step:>6}/{max_steps}] loss train {losses['train']:.3f} | val {losses['val']:.3f} | "
                f"lr {lr:.2e} | mémoire max {mem:.1f} Go"
            )
            log.writerow(
                [
                    step,
                    step * tokens_per_step,
                    f"{losses['train']:.4f}",
                    f"{losses['val']:.4f}",
                    f"{lr:.3e}",
                    "",
                    f"{mem:.2f}",
                ]
            )
            log_file.flush()
            with open(run_dir / "echantillons.txt", "a", encoding="utf-8") as f:
                f.write(
                    f"\n=== pas {step} (loss val {losses['val']:.3f}) ===\n"
                    + "\n---\n".join(samples())
                    + "\n"
                )
            if losses["val"] < best_val and step > start_step:
                best_val = losses["val"]
                save(step, run_dir / "meilleur.pt")
        if not args.banc and step > start_step and step % args.sauver_tous == 0:
            save(step, ckpt_path)

        # Un pas d'optimisation = `accumulation` micro-batchs (gradients additionnés)
        for _ in range(args.accumulation):
            x, y = get_batch("train")
            with autocast():
                _, loss = model(x, y)
            loss = loss / args.accumulation
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
        if scaler:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.clip)
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if args.banc and step == start_step + 2:  # les 3 premiers pas (compilation, caches) sont exclus
            if device == "cuda":
                torch.cuda.synchronize()
            t_banc = time.time()

        if step % 10 == 0:
            if device == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t0
            t0 = time.time()
            speed = tokens_per_step * (10 if step > start_step else 1) / dt
            remaining = (max_steps - step) * tokens_per_step / speed / 3600
            print(
                f"pas {step:>6} | loss {loss.item() * args.accumulation:.3f} | "
                f"{speed / 1e3:.0f} k tokens/s | "
                f"reste ~{remaining:.1f} h"
            )
            log.writerow(
                [
                    step,
                    step * tokens_per_step,
                    f"{loss.item() * args.accumulation:.4f}",
                    "",
                    f"{lr:.3e}",
                    f"{speed:.0f}",
                    "",
                ]
            )
            log_file.flush()

    if args.banc:
        if device == "cuda":
            torch.cuda.synchronize()
        measured = max(1, max_steps - start_step - 3)
        speed = measured * tokens_per_step / (time.time() - t_banc)
        mem = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
        log_file.close()
        print(
            f"Banc : {speed / 1e3:.1f} k tokens/s | mémoire max {mem:.1f} Go | "
            f"{args.tokens / 1e9:.1f} Md de tokens en ~{args.tokens / speed / 3600:.1f} h"
        )
        return

    save(max_steps, ckpt_path)
    if best_val == float("inf"):  # entraînement très court : pas encore de « meilleur » modèle
        save(max_steps, run_dir / "meilleur.pt")
    log_file.close()
    print(f"Terminé. Modèles dans {run_dir} (meilleur.pt = plus petite loss de validation).")


if __name__ == "__main__":
    main()
