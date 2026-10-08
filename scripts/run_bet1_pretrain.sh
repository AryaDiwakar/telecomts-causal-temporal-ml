#!/usr/bin/env bash
# Bet 1 pipeline: self-supervised pretraining -> fine-tune all 5 seeds.
# ALSO re-runs the 5-seed fresh (non-pretrained) TAGN under the corrected
# scaler so the comparison is fair.
set -euo pipefail
cd "$(dirname "$0")/.."

export TAGN_DEVICE="${TAGN_DEVICE:-mps}"

echo "== exp_008: self-supervised pretraining (seed 42) =="
python -m src.experiments.exp_008_pretrain --epochs 12 --batch 256

for s in 42 123 2024 2026 3407; do
  echo "== exp_002 fresh seed $s (corrected scaler) =="
  python -m src.experiments.exp_002_tagn --seed "$s"
  echo "== exp_002 pretrained seed $s =="
  python -m src.experiments.exp_002_tagn --seed "$s" --pretrained --freeze-epochs 3
done

echo "ALL DONE"