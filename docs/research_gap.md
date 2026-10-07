# Research Gap & Motivation

## What the dataset is
TelecomTS is a 5G RAN benchmarking suite: 32 000 labeled 12.8 s windows over
18 KPIs, 11 anomaly types, and root-cause metadata. It is large enough for
neural baselines and ships crowd-free ground truth for *window-level* anomaly
flags, *type* labels, and `affected_kpis` sets.

## What we want to verify (hypotheses H1-H5)
- **H1 — The baseline reality.** On TelecomTS, classical non-temporal
  detectors (Isolation Forest, LOF, one-class SVM) underperform their
  published expectations on telecom data; a supervised temporal model is
  materially better. (exp_001 already shows ISF/LOF ~ random, one-class SVM
  0.69 ROC on the provided features, and supervised gradient boosting / LogReg
  reach 0.92-0.97 ROC — a strong *feature-extraction* effect, not a deep-model
  effect.)
- **H2 — Graph structure helps.** Endowing a temporal network with a
  data-driven dynamic KPI graph improves anomaly/type/RCA metrics over the
  same temporal backbone with no graph (ablations A1-A5).
- **H3 — Type is the lever, templates follow.** Because `affected_kpis` is a
  per-type constant, *anomaly-type classification* is the true perceptual
  task; template-based RCA inherits its quality. We predict type-first
  accuracy dominates naive input-shift localization (already supported
  directionally by exp_001: input-shift Jaccard 0.37 vs template 0.76+).
- **H4 — Recording memorization shortcut exists.** A recording-prior RCA
  (which anomaly type appears in that recording during training) reaches
  Jaccard ~0.82 on test — ABOVE the input-driven 0.37 — demonstrating a
  shortcut that any per-recording-split methodology would hide. Leave-one-
  recording-out (Setting II) must therefore be the honest generalization test.
- **H5 — Early warning is feasible pre-onset.** 362 anomalous windows contain
  a within-window pre-onset region; a causal hazard head should detect onset
  before step `onset` (evaluated as lead-time statistics).

## How this project fills the gap (hedged)
- Brings a *directed, dynamic, per-chunk* KPI dependency graph derived from
  TelecomTS itself (GraphicalLasso skeleton + transfer-entropy orientation;
  `src/models/graphs.py`), packaged as small prototype adjacency matrices.
- Provides a full **honesty instrument**: leakage audit, split protocol with
  burst integrity, matching-prior baselines, leave-one-recording-out,
  and recording-level aggregation (docs/).
- Reuses only verified, public baselines/methods; no unverifiable claims.

## Explicit non-claims
We do NOT claim: (a) the learned graphs recover the true unobserved causal
DAG of a 5G RAN; (b) superiority over every published 5G-RAN RCA method —
those use their own splits and metrics and are not directly comparable;
(c) that `affected_kpis` anomalies are causally independent — they are
templates (`docs/data_leakage.md`).