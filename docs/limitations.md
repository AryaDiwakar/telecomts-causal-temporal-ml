# Limitations & Threats to Validity

Ordered by how much they could affect the headline conclusions.

## 1. Label structure: `affected_kpis` are per-type templates
The benchmark's KPI-level root-cause "ground truth" has one variant per
anomaly type. Consequences:
- Type accuracy *is* the perceptual ceiling for template-based RCA; per-type
  template Jaccard largely inherits type accuracy (exp_001 shows confusion
  overlaps between templates can raise Jaccard above type accuracy).
- We therefore evaluate RCA twice: (a) `type -> template` mapping, and
  (b) input-driven per-KPI localization that ignores the template entirely.
- Any claims of "KPI-level root-cause localization" must be read under this
  caveat.

## 2. Recording-origin confound
Normal and each anomalous class come from *different scenario recordings*;
no recording mixes normal and anomalous chunks. A model can partially solve
anomaly/type/RCA by recognizing recording/scenario (zone, app, mobility)
rather than fault signature. Evidence (exp_001): recording-prior RCA
(Jaccard 0.82) beats input-driven RCA (0.37). Mitigations: chronological
within-recording splits, LORO Setting II, and explicit reporting of both
setups; but residual scenario-shift remains a dataset-level threat.

## 3. Evaluation-domain realism
- Total test set = 223 anomalous chunks concentrated in a handful of bursts;
  per-type test counts are small (8-92). Macro metrics over types are noisy.
- Many anomalous windows overlap within bursts, making windows highly
  dependent; we aggregate at recording level and report burst-aware numbers,
  but chunk-level i.i.d. assumptions still break formal CIs.

## 4. Model & compute constraints
- The proposed model is a compact CPU-trained network (about 485k params, causal
  conv + GAT + GRU), not a large pretrained representation; results reflect
  its capacity. No hyperparameter search beyond the config defaults.
- Reported inference/lead-time numbers come from one checkpoint per seed and
  no uncertainty calibration on hazard outputs (we report empirical lead-time
  error, not calibrated probabilities).

## 5. Graph inference caveats
- GraphicalLasso + transfer entropy on pooled windows is a simple estimator;
  edges are thresholds-dependent; the current audit retained no directed
  prototype edges; graphs are prototypes of bin-level topologies, so
  fine-grained per-milli-second structure is discarded.
- We report graphs as *dependencies*, not verified physical causality.

## 6. Early-warning evaluation scope
- Only 362 anomalous chunks (52 in test) have onset>0, so early-warning
  numbers rest on a small support; jammer chunks (onset=0) carry no early
  signal and are excluded from lead-time statistics.

## 7. Freshness
- All results reflect a single session; scripts are deterministic given
  seeds, but hardware/software differences (Apple-Arc, arm64 wheels) can
  change CPU ordering; we pin the library versions in requirements/notes to
  limit this.

## 8. What we did NOT do
- No human re-annotation; no ensembling beyond per-seed averaging; no
  external validation on another 5G dataset (out of scope); no release of
  model weights beyond the checkpoints in this repo.
