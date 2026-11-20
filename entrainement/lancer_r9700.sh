#!/usr/bin/env bash
# Pré-entraînement d'Alexinator-240M sur une AMD Radeon AI PRO R9700 (32 Go, RDNA4, ROCm).
#
#   bash entrainement/lancer_r9700.sh              vérifications + test rapide, puis entraînement en arrière-plan
#   bash entrainement/lancer_r9700.sh --sans-test  sans le test rapide
#   bash entrainement/lancer_r9700.sh --banc       mesure la vitesse sur 20 pas (corpus déjà préparé)
#   bash entrainement/lancer_r9700.sh --reprendre  reprend au dernier checkpoint après une coupure
#
# Prérequis : Linux avec ROCm 6.4.2 ou plus récent et le PyTorch ROCm (voir docs/entrainement.md).
# L'entraînement continue même si la connexion SSH se coupe (nohup).
# Suivre :   tail -f entrainement/logs/r9700.log
# Arrêter :  kill $(cat entrainement/logs/r9700.pid)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p entrainement/logs
PY=${PYTHON:-python3}
MODE="${1:-}"

echo "=== PyTorch ROCm et carte graphique ==="
$PY - <<'PYCHECK'
import sys
try:
    import torch
except ImportError:
    sys.exit("PyTorch absent. Installer la version ROCm :\n"
             "  pip install torch --index-url https://download.pytorch.org/whl/rocm6.4")
if torch.version.hip is None:
    sys.exit(f"PyTorch {torch.__version__} est la version CUDA (NVIDIA). Installer la version ROCm :\n"
             "  pip uninstall -y torch\n"
             "  pip install torch --index-url https://download.pytorch.org/whl/rocm6.4")
if not torch.cuda.is_available():
    sys.exit("Aucune carte AMD visible par PyTorch : vérifier `rocm-smi` et que l'utilisateur est\n"
             "dans les groupes render et video (sudo usermod -aG render,video $USER, puis se reconnecter).")
p = torch.cuda.get_device_properties(0)
arch = getattr(p, "gcnArchName", "?")
print(f"PyTorch {torch.__version__} | ROCm {torch.version.hip}")
print(f"Carte : {p.name} ({arch}), {p.total_memory / 1e9:.0f} Go, bfloat16 : {torch.cuda.is_bf16_supported()}")
if not arch.startswith("gfx12"):
    print("Attention : ce n'est pas une carte RDNA4 (gfx12xx) ; le profil r9700 suppose 32 Go de mémoire.")
PYCHECK

echo "=== Dépendances (sans toucher au PyTorch ROCm) ==="
$PY -m pip install -q tokenizers datasets numpy

PRETRAIN="$PY -u entrainement/pretrain.py --profil r9700"

if [[ "$MODE" == "--banc" ]]; then
    [[ -f entrainement/donnees/stats.json ]] || { echo "Corpus absent : lancer d'abord ce script sans --banc."; exit 1; }
    $PRETRAIN --banc 20
    exit 0
fi

if [[ "$MODE" != "--sans-test" && "$MODE" != "--reprendre" ]]; then
    echo "=== Test rapide de toute la chaîne (mini-corpus, mini-modèle) ==="
    $PY -u entrainement/preparer_donnees.py --test
    $PY -u entrainement/pretrain.py --test
    $PY -u entrainement/sft.py --test
    echo "Test réussi."
fi

# Le corpus (tokenizer + 3 Md de tokens, ~6 Go) se prépare une seule fois. On peut aussi le copier
# depuis une autre machine qui l'a déjà : dossier entrainement/donnees/ complet, avec stats.json.
if [[ -f entrainement/donnees/stats.json ]]; then
    PREPARER="true"
    echo "Corpus déjà prêt : $(tr -d '\n ' < entrainement/donnees/stats.json)"
else
    PREPARER="$PY -u entrainement/preparer_donnees.py"
fi

REPRENDRE=""
[[ "$MODE" == "--reprendre" ]] && REPRENDRE="--reprendre"

echo "=== Entraînement lancé en arrière-plan ==="
nohup bash -c "$PREPARER && $PRETRAIN $REPRENDRE" > entrainement/logs/r9700.log 2>&1 < /dev/null &
echo $! > entrainement/logs/r9700.pid
echo "PID $(cat entrainement/logs/r9700.pid). Suivre avec :  tail -f entrainement/logs/r9700.log"
