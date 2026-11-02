# Benchmark protocol — v3

Status: **authoritative for the corrected evaluation.** Supersedes the "pre-v2"
tables in `docs/integrated_results_ci.md` (marked invalid: template ground truth +
split bug). Do not compare numbers across different ``benchmark_version`` values.

Machine-readable companion: `outputs/benchmark_manifest.json`
(built by `scripts/build_benchmark_table.py`).

## B1. Dataset & ingestion

| item | value |
|---|---|
| dataset | TelecomTS (AliMaatouk/TelecomTS), 5G RAN multivariate KPI telemetry |
| channels | 16 numeric KPI + 2 protocol-code channels |
| window | 128 steps, stride 32 (overlapping) |
| interim chunks | `data/interim/chunks_X.npy` (N, 128, 18) + `chunks_meta.parquet` |
| Track 2 | instance-varying synthetic injection (`src/data/synthetic_inject.py`) |

## B2. Split protocol (`temporal_within_recording`)

- Per recording: chronological 0.70 / 0.15 / 0.15 (train / val / test).
- Chunks that straddle a boundary are dropped (implicit embargo).
- Atomic recordings (jammer, Track-2 `synthetic_gen`) are assigned **whole**
  (0.5/0.15/0.35 for synthetic_gen) so one injected event never spans splits.
- **Track-2 background sources are bound to their injected twin's split**
  (finding G5): a `normal/<rel>` source is never in train unless the synthetic
  recording injected onto it is also in train
  (`split_stats.json::audit.background_source_pairs`).
- **No retained window shares a raw timestep across splits** (regression test:
  `tests/test_audit_regressions.py::test_no_cross_split_timestep_overlap`).
- Persisted: `data/processed/split_indices.npz`, `split_stats.json`.

### v3 split sizes

| split | chunks | anomalous | recordings |
|---|---|---|---|
| train | 22,009 | 1,235 | 31 |
| val | 4,841 | 185 | 18 |
| test | **5,689** | **424** | 26 |

Positive prevalence on test = 7.45%. Anomaly families on test: shipped
"template" windows (**223**) and Track-2 "instance-varying" windows (**201**).
The test split grew relative to the pre-G5-fix number (4,969) because the four
test-held synthetic *sources* now contribute their clean background alongside
their injected twins.

## B3. Preprocessing (train-only)

- Robust z-score scaler fit on **train chunks only** (`prepare.py:fit_scaler`),
  frozen and applied to all splits. Zero-inflated channels use IQR→MAD→std
  escalation.
- RCA templates: **train-only majority vote per type** (`rca_templates.json`
  carries `rca_template_source: train`). Rebuilt by
  `scripts/rebuild_rca_templates.py`.
- No imputation, no data-driven feature selection.
- Graph KMeans prototypes fit on **train bins only** (`graphs.py`); per-bin
  GraphicalLasso + transfer entropy are split-specific.

## B4. Evaluation protocols

| id | unit | metrics | notes |
|---|---|---|---|
| P-det | pooled window | ROC-AUC, PR-AUC, TPR@FPR{0.01,0.05}, F1 at val threshold | threshold selected on **val**, applied to test; 95% **cluster-bootstrap** CI over recordings (M5) |
| P-rca-template | anomalous test window | `type_template_jaccard`, micro-F1 | uses train-majority templates |
| P-rca-instance | anomalous test window | `topk_shift_jaccard`, `recall@1/3/5`, `mrr`, `map` | fixed-k ranking metrics are oracle-free |
| P-ew | anomalous test window | warning-before-onset frac, median lag | hazard head |
| P-loro | recording-grouped 5-fold | detection + RCA per fold | leave-one-recording-out; **per-fold scaler + graph prototypes** (R4 fixed) |

## B5. Validity rules

1. **Same examples or no comparison.** A detection number may only be compared
   against another value measured on the **current v3 test set (n = 5,689,
   pos = 424)**. Any number measured before the 2026-10-14 G5 split repair
   (n = 4,969/424) or on the older template-only set (n = 4,768/223) is not
   citable against these results.
2. **Val-tuned operating points.** Reported F1/precision/recall use a threshold
   chosen on val. The test-optimal F1, where retained, is recorded as
   `f1_max_test_optimal_optimistic` and must not be cited as a headline.
3. **Provenance required.** A result is `COMPLETE` only if the registry row
   carries `code_revision`, `config_hash`, `data_fingerprint`, `status`,
   `timestamp`. Legacy rows without provenance are labelled `legacy`.
4. **Uncertainty is recording-level.** Single-model 95% CIs are *cluster*
   bootstraps that resample recordings with replacement
   (`scripts/bootstrap_ci.py`, finding M5). Model-vs-baseline claims use a
   **paired** cluster bootstrap of the shared-chunk metric difference
   (`scripts/paired_test.py`); the union of two independent CIs is not used as a
   significance test.
5. **Leak-free generalization.** Setting II (LORO) fits the scaler and graph
   prototypes on each fold's training recordings only
   (`src/models/graph_fold.py`); the legacy global-artifact path
   (`--legacy-global-artifacts`) is retained for continuity only and its
   numbers are not citable.

## B6. Known limitations (v3)

1. **Track-2 background-signal duplication (G5).** Track-2 injected windows
   share the source recording's background, so detection can be aided by
   recording identity. **Fixed (2026-10-14):** each source is now bound
   atomically to its injected twin's split, so a test window's background is
   never in train; all results retrained on the repaired split.
2. **Injection labels (G2/G3/G4).** Identity/inert effects were labelled as
   affected (G2), multi-event windows were under-labelled (G3) and `end_time`
   was 10× too large (G4). **Fixed and regenerated (2026-10-14):** numeric data
   changed by <6e-11 (JSON round-trip only); labels/metadata corrected (280
   spurious labels removed across 237/714 windows).
3. **RCA cardinality.** Ground-truth root sets are large (median 9 of 16 KPIs),
   so `recall@1` is bounded near 1/|gt|. Report `mrr`/`map` as the primary RCA
   numbers; fixed-k metrics (M4) are oracle-free.
4. **Single-seed ablation.** The component ablation (exp_003) is run at one seed
   with the shared headline anchor recipe; treat its ordering as indicative, not
   a significance result.
5. **Archaeology.** `outputs/results/exp_003_ablation_*.json` from 2026-10-07
   were trained without the stage-A anchor, so their `full` variant is far
   weaker than the headline model; they are archived under
   `outputs/results/archive/stale_2026-10-07/` and are not citable.
