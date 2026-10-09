# Causal-Temporal Machine Learning for 5G KPI Root-Cause Localization

> **Demo status:** Evidence-first pilot package. Results marked as pilot predate TAGN-v2 and must be regenerated before submission.

## Abstract

This demonstration presents a reproducible framework for anomaly detection, anomaly-type classification, and KPI-level root-cause localization in 5G RAN telemetry. The protocol explicitly audits recording memorization and template shortcuts, reports both chronological and leave-recording-out evaluation, and treats learned graphs as predictive candidates rather than identified causal DAGs. In the completed baseline artifact, logistic regression reaches PR-AUC 0.957 on the chronological test split, while the TAGN pilot reaches 0.574. This gap is reported as a research finding, not concealed as a state-of-the-art claim.

## Core Results

| Model | ROC-AUC | PR-AUC | TPR @ 1% FPR | Interpretation |
|---|---:|---:|---:|---|
| Logistic regression | 0.986 | 0.957 | 0.951 | strong engineered-feature baseline |
| XGBoost | 0.921 | 0.668 | 0.552 | supervised nonlinear baseline |
| TAGN pilot | 0.886 | 0.574 | 0.480 | pre-TAGN-v2 artifact; rerun required |

## Generalization And Audit

- Leave-recording-out PR-AUC: **0.606**.
- Leave-recording-out ROC-AUC: **0.893**.
- Recording-prior RCA Jaccard: **0.818**.
- Input-shift RCA Jaccard: **0.362**.
- Template-attribution agreement AUC: **0.852**.
- Dynamic graph stability: **0.550**.

## Seed Stability (exp_002)

| Seed | ROC-AUC | PR-AUC | Type acc |
|---:|---:|---:|---:|
| 42 | 0.886 | 0.574 | 0.247 |
| 123 | 0.936 | 0.841 | 0.592 |
| 2024 | 0.826 | 0.489 | 0.300 |
| 2026 | 0.946 | 0.719 | 0.251 |
| 3407 | 0.888 | 0.615 | 0.614 |

- Mean PR-AUC = **0.647 ± 0.136** over 5 seeds.
- Mean ROC-AUC = **0.896 ± 0.048** over 5 seeds.

## Ablations (exp_003)

| Variant | PR-AUC | ROC-AUC | Type acc | Δ vs full |
|---|---:|---:|---:|---:|
| static graph | 0.832 | 0.953 | 0.695 | +0.279 |
| undirected | 0.568 | 0.896 | 0.552 | +0.015 |
| full | 0.553 | 0.884 | 0.229 | reference |
| no graph | 0.522 | 0.834 | 0.377 | -0.032 |
| no nodeconv | 0.510 | 0.837 | 0.215 | -0.044 |
| hazard only | 0.035 | 0.361 | 0.040 | -0.518 |

## Robustness (exp_004, seed 42)

| Scenario | PR-AUC | ROC-AUC |
|---|---:|---:|
| baseline | 0.504 | 0.825 |
| noise 0.05 | 0.504 | 0.825 |
| noise 0.1 | 0.504 | 0.825 |
| noise 0.2 | 0.504 | 0.825 |
| channel drop 0.2 | 0.488 | 0.838 |
| time shift 0.1 | 0.509 | 0.827 |
| scale 0.2 | 0.513 | 0.844 |

- Worst PR-AUC drop vs baseline: **0.016**.

## Early Warning (exp_007)

- Warned before onset: **56.7%** on 60 chunks.
- Median lead: **-13.0** steps (negative = early).
- Per-step average precision: **0.627**.

## Contributions

1. Leakage-controlled chronological and leave-recording-out protocols.
2. Candidate dynamic KPI graphs with held-out predictive validation.
3. Joint anomaly, type, root-cause, and causal early-warning heads.
4. Explicit shortcut baselines for recording memorization and constant templates.
5. Reproducible artifacts, tests, figures, and a browser-ready demonstration.

## Dataset Snapshot

- Total windows: **31729**.
- Anomalous windows: **1130**.
- Signals: **18 channels**, including 16 numeric KPIs and 2 protocol channels.
- Fault types: **11**.

## Submission Guardrails

- Do not claim universal state-of-the-art performance from these artifacts.
- Rerun TAGN-v2, all ablations, robustness, LORO, explainability, and early-warning experiments.
- Report mean, standard deviation, confidence intervals, and per-recording results.
- Compare against published baselines only when split, labels, and metrics are directly compatible.
- Describe learned graphs as candidate predictive dependencies, not causal truth.

## Reproduction

```bash
python3 -m src.models.graphs
python3 -m src.experiments.exp_002_tagn --seed 42 --epochs 20 --patience 6
python3 -m src.report.generate_demo
open outputs/scopus_demo.html
```
