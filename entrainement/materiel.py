"""Adaptation des scripts d'entraînement à la carte graphique : NVIDIA (CUDA) ou AMD (ROCm).

Sous ROCm, PyTorch garde le nom « cuda » : torch.cuda.* et device="cuda" désignent la carte AMD.
On distingue les deux avec torch.version.hip (None sous CUDA).

    preparer_environnement()   à appeler AVANT `import torch` (variables lues au démarrage)
    decrire_carte()            nom, plateforme, mémoire
    choisir_precision()        bfloat16, float16 + GradScaler, ou float32 selon la carte
    compilation_possible()     torch.compile a-t-il une chance de marcher sur cette carte ?
    trouver_micro_batch()      plus grand micro-batch qui tient en mémoire (essais successifs)
"""

from __future__ import annotations

import os
import sys


def preparer_environnement() -> None:
    """Variables d'environnement utiles, sans écraser celles déjà définies par l'utilisateur."""
    # ROCm sur cartes Radeon (RDNA3/RDNA4) : active les noyaux d'attention AOTriton
    # (FlashAttention / mem-efficient de scaled_dot_product_attention), encore « expérimentaux » sur RDNA.
    # Sans effet sous CUDA.
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
    # Pas de PYTORCH_CUDA_ALLOC_CONF=expandable_segments par défaut : après un manque de mémoire
    # (essais de micro-batch), l'allocateur peut se retrouver dans un état cassé (INTERNAL ASSERT).


def est_rocm() -> bool:
    import torch

    return getattr(torch.version, "hip", None) is not None


def decrire_carte(device: str = "cuda") -> str:
    import torch

    if device != "cuda" or not torch.cuda.is_available():
        return "processeur (pas de GPU)"
    p = torch.cuda.get_device_properties(0)
    plateforme = f"ROCm {torch.version.hip}" if est_rocm() else f"CUDA {torch.version.cuda}"
    arch = getattr(p, "gcnArchName", "") if est_rocm() else f"{p.major}.{p.minor}"
    return f"{p.name} ({arch}), {p.total_memory / 1e9:.0f} Go, {plateforme}"


def choisir_precision(device: str = "cuda"):
    """Renvoie (dtype de l'autocast ou None, GradScaler ou None).

    - AMD (ROCm) : bfloat16 si la carte le gère (RDNA3, RDNA4 comme la R9700, Instinct MI200+) ;
    - NVIDIA Ampere et plus récent (8.x+) : bfloat16 ;
    - NVIDIA Volta/Turing (7.x) : float16 + GradScaler ;
    - plus ancien (ex. Tesla M40, 5.2) : float32. PyTorch y annonce le bfloat16 « disponible »,
      mais il est émulé (lent) et ces cartes n'accélèrent pas non plus le float16.
    """
    import torch

    if device != "cuda" or not torch.cuda.is_available():
        return None, None
    if est_rocm():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, None
        return torch.float16, torch.amp.GradScaler("cuda")
    capability = torch.cuda.get_device_capability()
    if capability >= (8, 0):
        return torch.bfloat16, None
    if capability >= (7, 0):
        return torch.float16, torch.amp.GradScaler("cuda")
    return None, None


def compilation_possible(device: str = "cuda") -> tuple[bool, str]:
    """torch.compile s'appuie sur Triton : NVIDIA 7.0+ ou AMD sous ROCm."""
    import torch

    if device != "cuda" or not torch.cuda.is_available():
        return False, "pas de GPU"
    if est_rocm():
        return True, ""
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) < (7, 0):
        return False, f"carte {major}.{minor} : Triton exige 7.0+"
    return True, ""


SONDE_OK, SONDE_MEMOIRE = 0, 3  # codes de sortie de la sonde


def trouver_micro_batch(
    config: dict,
    amp_dtype,
    candidats=(64, 32, 16, 8, 4, 2, 1),
    marge: float = 0.90,
) -> int:
    """Essaie un pas complet (avant, arrière, AdamW) avec des tailles décroissantes ; garde la première qui tient.

    Chaque essai tourne dans un processus séparé : après un manque de mémoire, certains pilotes laissent
    le GPU dans un état inutilisable pour le reste du processus (constaté sous Windows, signalé sous ROCm).
    L'essai se fait sans torch.compile (qui consomme plutôt moins) et le pic doit rester sous
    `marge` × mémoire de la carte, pour laisser de la place à l'évaluation et à la génération.
    """
    import json
    import subprocess
    import sys

    dtype = {None: "float32"}.get(amp_dtype, str(amp_dtype).replace("torch.", ""))
    derniere_erreur = ""
    for mb in candidats:
        result = subprocess.run(
            [sys.executable, __file__, "--sonde", json.dumps(config), dtype, str(mb), str(marge)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        last = (result.stdout.strip().splitlines() or [""])[-1]
        if result.returncode == SONDE_OK:
            print(f"Micro-batch {mb} : {last} -> retenu")
            return mb
        if result.returncode == SONDE_MEMOIRE:
            print(f"Micro-batch {mb} : {last or 'mémoire insuffisante'}")
            continue
        # Autre échec : le plus souvent un manque de mémoire sous une autre forme
        # (CUBLAS_STATUS_ALLOC_FAILED, erreur HIP...). On essaie plus petit et on garde le message.
        erreurs = (result.stderr.strip().splitlines() or ["?"])
        derniere_erreur = result.stdout + result.stderr
        print(f"Micro-batch {mb} : échec ({erreurs[-1][:120]}), probablement la mémoire")
    raise SystemExit(
        "Aucun micro-batch ne passe, même 1 : réduire le modèle ou --block-size.\n"
        f"Dernière erreur :\n{derniere_erreur[-3000:]}"
    )


def _sonde(config_json: str, dtype_name: str, micro_batch: int, marge: float) -> int:
    """Processus d'essai : un pas d'entraînement complet sur des tokens aléatoires, puis le pic de mémoire."""
    import json
    import sys
    from contextlib import nullcontext
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import torch

    from alexinator.gpt import GPT, GPTConfig

    cfg = GPTConfig(**json.loads(config_json))
    dtype = None if dtype_name == "float32" else getattr(torch, dtype_name)
    autocast = (lambda: torch.autocast("cuda", dtype=dtype)) if dtype else nullcontext
    total = torch.cuda.get_device_properties(0).total_memory
    try:
        model = GPT(cfg).to("cuda")
        optimizer = model.configure_optimizer(1e-4, 0.1)
        x = torch.randint(cfg.vocab_size, (micro_batch, cfg.block_size), device="cuda")
        y = torch.randint(cfg.vocab_size, (micro_batch, cfg.block_size), device="cuda")
        for _ in range(2):  # le 2e pas inclut l'état d'AdamW, créé au 1er
            with autocast():
                _, loss = model(x, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
    except Exception as error:  # OutOfMemoryError, ou RuntimeError « out of memory » selon le pilote
        if isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower():
            print("mémoire insuffisante")
            return SONDE_MEMOIRE
        raise
    pic = torch.cuda.max_memory_allocated()
    print(f"pic {pic / 1e9:.1f} Go sur {total / 1e9:.0f} Go")
    return SONDE_OK if pic <= marge * total else SONDE_MEMOIRE


if __name__ == "__main__" and len(sys.argv) == 6 and sys.argv[1] == "--sonde":
    sys.exit(_sonde(sys.argv[2], sys.argv[3], int(sys.argv[4]), float(sys.argv[5])))
