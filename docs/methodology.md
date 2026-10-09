# Methodology

This file records the *mechanism* of every model and every experiment, so the
`outputs/` numbers can be reproduced and audited.

## 0. Data and labels (recap, full detail in `docs/dataset_schema.md`)
- 32 000 chunks -> train/val/test 70/15/15 within-recording chronological,
  burst-cluster integrity, whole-recording jammer assignment
  (`docs/data_leakage.md`).
- Features fed to models: the 16 numeric KPI channels (robust-z scored with
  train-only statistics) + 2 protocol-code channels.

## 1. Dynamic dependency graphs (`src/models/graphs.py`)
- **Binning.** Consecutive chunks of one recording (one split) are pooled into
  sliding bins of 90 chunks (hop 45). Each bin supplies a multivariate sample
  of ~90x128 steps over 16 channels.
- **Skeleton.** GraphicalLasso (alpha=0.05, columns normalized) -> precision
  -> partial-correlation matrix; keep top-4 undirected edges per node above
  0.15 -> binary skeleton weighted by |partial corr|.
- **Orientation.** For each edge, transfer entropy (4-bin, history 1, from
  3D joint histogram) is computed both directions; if TE_ij > 1.15*TE_ji the
  edge is stored i->j (else j->i, or undirected when within tolerance).
  The adjacency implementation preserves this asymmetry when supported by the
  data; the current graph audit retained no directed prototype edges, so the
  reported model evidence should not be read as evidence for directional gain.
- **Dynamic topology.** Bins are clustered (KMeans, K=6) on train bins only;
  val/test bins are assigned to the nearest train prototype. Each chunk gets
  its bin's prototype adjacency => the model sees a time-varying graph id.
- **Sanity/limits.** Reported as exploratory, NOT ground-truth causality.
  Recording-level and chunk-level aggregation used to reduce auto-correlation
  artifacts. `src.evaluation.graph_validation` reports within-prototype mask
  stability and held-out lagged directional predictivity. The latter is an
  association audit, not causal identification.

## 2. Proposed model: temporal + graph network (TAGN; `src/models/tagn.py`)
1. `in_proj`: Linear(18->d) + GELU.
2. `tconv`: causal dilated depthwise-then-dense temporal conv stack (d),
   residual-corrected.
3. `node_enc`: independently grouped causal 1D convolutions preserve KPI
   identity and produce node embeddings per step [B,T,16,d].
4. `GATLayer x L`: GATv2-style attention over the chunk's prototype adjacency
   (masked to existing edges), producing contextualized node/step states.
5. `fuse`: concatenation of temporal context + mean node embedding -> Linear ->
   GRU -> causal step states.
6. Heads:
   - anomaly: a validated linear summary anchor plus a compact nonlinear
     residual over train-only chunk summaries, with the temporal/graph trunk
     added as a residual path -> Logistic (BCE).
   - type: same hybrid context -> 11-way softmax (class-weighted CE,
     anomalous chunks only).
   - kpi (input-driven RCA): time-mean node states -> one logit per KPI
     (multi-label BCE vs template, anomalous only).
   - hazard (early warning): per-step logistic on causal step states (masked
     BCE pre-onset-only).
7. Joint loss with weights 1.0 / 1.0 / 1.0 / 0.5; square-root inverse-frequency
   type weights; AdamW; grad clip 1.0;
   early stop on validation PR-AUC; train undersamples normals to 4:1 per
   epoch.

## 3. Experiment matrix (configs/experiments.yaml summaries)
- **exp_001** classical baselines (completed): ISF, LOF, OC-SVM (unsup);
  XGB, LogReg (sup); type XGB; RCA baselines (type->template, recording-prior,
  input-shift).
- **exp_002** full TAGN: five configured seeds completed; aggregate results are
  reported with mean and standard deviation.
- **exp_003** full-budget ablations completed for (A) no-graph, (B) static
  graph, (C) undirected graph, (D) no node convolution, and (E) hazard head
  only, each with early stopping from a 30-epoch maximum.
- **exp_004** robustness: noise injection into X (sigma 0.05/0.1/0.2),
  missing-channel masking, time-scale perturbation; retrain or infer under
  the perturbation with the same checkpoint.
- **exp_005** Setting-II: five-fold leave-one-recording-out GroupKFold RCA / AD
  completed and reported separately from the chronological split.
- **exp_006** explainability: model-graded (type logit) and input-shift
  attribution on the KPI head; per-type top-KPI summary; sanity against
  `description` text (counted qualitatively only).
- **exp_007** early-warning granularity: reporting threshold sweep,
  lead-time CDF, mean-time-to-detect vs false-alarm rate.

## 4. Metric conventions (`src/evaluation/metrics.py`)
- Anomaly: ROC-AUC, PR-AUC, F1 at the validation-selected threshold, TPR at
  FPR 0.01 / 0.05; the operating threshold is tuned on validation only and
  then applied unchanged to test.
- Type: accuracy + macro-F1 over test-present types (always reported with the
  number of test types present).
- RCA: chunk-level template Jaccard (type->template and input-driven
  top-K) + per-KPI micro-F1; recording-level aggregation also exported.
  - **`type_template_jaccard` uses the PREDICTED type's template**, not the
    ground truth (an earlier version used `y_true` which trivially equals 1.0
    because `affected_kpis` templates are per-type constants; fixed in
    `common.py` / `exp_002_tagn.py`; `tests/test_rca_metric.py` guards it).
  - **Template-overlap floor.** The 11 templates share a large common KPI core
    (UL_BLER, DL_BLER, TX_Bytes, RX_Bytes, UL_NPRB, PRB_Utilization_* appear in
    nearly every template), so Jaccard cannot go near 0: a uniform-random type
    guess over 11 classes already scores ~0.56 template Jaccard, and the
    train-distribution type-prior scores ~0.76 (exp_001). Interpret every RCA
    Jaccard relative to these floors, not absolute 0..1.
- Early warning: lead-time error vs ground-truth onset; fraction of warnings
  strictly before onset; lag distribution.

## 5. Reproducibility
- Seeds fixed (configs/experiments.yaml seeds list); all artifacts saved under
  outputs/ with the run seed in the filename; every script is a single `python
  -m` entry point; docs/data_leakage.md and the split scripts define the exact
  audit gates.
