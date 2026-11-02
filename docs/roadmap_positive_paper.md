# Roadmap: Positive (TAGN beats prior art) Paper

Goal: make TAGN genuinely outperform TCN/LogReg/KAC on detection and RCA, while keeping the honest-evaluation machinery (cluster bootstrap, LORO, paired tests, repaired split).

| Phase | Title | Gate (must pass to proceed) |
|---|---|---|
| 0 | Freeze honest baseline (DONE) | all experiments on repaired split + registry provenance |
| 1 | Stronger temporal backbone (TCN+Transformer) | SCAFFOLDED (exp_014, smoke PR=0.79); full training pending |
| 2 | Multimodal observability fusion | SCAFFOLDED (exp_015, smoke PR=0.74); full training pending |
| 3 | Causal graph as intervention (core novelty) | SCAFFOLDED (exp_016, smoke fidelity=0.0); full eval pending |
| 4 | Self-supervised pretraining at scale | ≥0.03 PR-AUC gain, consistent across seeds |
| 5 | Cross-vendor transfer (C3) | zero-shot F1 >0.75; fine-tuned beats KAC bar |
| 6 | Deferral-aware, calibrated RCA | precision-at-coverage ≥0.4 at 10% coverage |
| 7 | Ensembling & thresholding | ensemble PR-AUC > best single by ≥0.02 |
| 8 | Paper (positive, credible) | every claim has registered citable artifact + passing test |

Critical path: **1 → 3 → 8**. Phases 2, 4, 5, 6, 7 parallelizable.
Honesty rule: a failed gate is iterated, never published as success.
