#!/usr/bin/env bash
# Lance la phase 1 d'Alexinator-110M sur le serveur GPU : tokenizer, corpus, pré-entraînement.
#
#   bash entrainement/lancer_phase1.sh            test rapide (2 à 3 min), puis vraie phase 1 en arrière-plan
#   bash entrainement/lancer_phase1.sh --sans-test
#
# La vraie phase 1 continue même si la connexion SSH se coupe (nohup).
# Suivre :   tail -f entrainement/logs/phase1.log
# Arrêter :  kill $(cat entrainement/logs/phase1.pid)
# Reprendre après une coupure :  bash entrainement/lancer_phase1.sh --reprendre
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p entrainement/logs
PY=${PYTHON:-python3}

echo "=== Dépendances ==="
$PY -m pip install -q -r requirements-entrainement.txt
$PY - <<'PYCHECK'
import torch
print("PyTorch", torch.__version__, "| CUDA :", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"GPU : {p.name}, {p.total_memory / 1e9:.0f} Go, bfloat16 : {torch.cuda.is_bf16_supported()}")
else:
    raise SystemExit("Aucun GPU visible : vérifier le serveur (nvidia-smi) avant de lancer la phase 1.")
PYCHECK

if [[ "${1:-}" != "--sans-test" && "${1:-}" != "--reprendre" ]]; then
    echo "=== Test rapide de toute la chaîne (mini-corpus, mini-modèle) ==="
    $PY entrainement/preparer_donnees.py --test
    $PY entrainement/pretrain.py --test
    $PY entrainement/sft.py --test
    echo "Test réussi."
fi

REPRENDRE=""
[[ "${1:-}" == "--reprendre" ]] && REPRENDRE="--reprendre"

echo "=== Phase 1 lancée en arrière-plan ==="
nohup bash -c "$PY entrainement/preparer_donnees.py && $PY entrainement/pretrain.py $REPRENDRE" \
    > entrainement/logs/phase1.log 2>&1 &
echo $! > entrainement/logs/phase1.pid
echo "PID $(cat entrainement/logs/phase1.pid). Suivre avec :  tail -f entrainement/logs/phase1.log"
