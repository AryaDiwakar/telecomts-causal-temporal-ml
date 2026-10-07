#!/bin/zsh
# Full post-sweep evaluation. Run from the project root after exp_002 finishes.
set -e

python3 -m src.models.graphs
python3 -m src.evaluation.graph_validation
python3 -m src.experiments.exp_003_ablations --epochs 10 --patience 3
python3 -m src.experiments.exp_004_robustness --seed 42
python3 -m src.experiments.exp_006_explainability
python3 -m src.experiments.exp_007_early_warning
python3 -m src.experiments.exp_005_loro --seed 42 --epochs 8 --patience 3
python3 -m src.report.recompute_rca
python3 -m src.report.summarize_results
python3 -m src.report.summarize_ablations
python3 -m src.report.generate_report
