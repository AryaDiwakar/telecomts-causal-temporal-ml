#!/bin/bash
# Run Phase 1 backbone across seeds
for seed in 42 123 456; do
  echo "=== seed $seed ==="
  python3 -m src.experiments.exp_014_backbone --seed $seed --epochs 20
done
echo "done"
EOF
chmod +x /Users/aryadiwakar/telecomts-causal-temporal-ml/scripts/run_phase1_multiseed.sh
echo done