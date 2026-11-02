# Benchmark audit v3 — split, leakage, labels, metrics

Status: **complete (code-level audit).** This document records the Phase 0/1
audit of the evaluation pipeline. It is intentionally conservative: every
finding cites `file:line`, states severity, and separates *confirmed defects*
from *by-design* behaviour and *uncertain* items.

Scope: `src/preprocessing/{prepare,prepare_loro}.py`, `src/data/{loader,synthetic_inject}.py`,
`src/models/{graphs,tagn,pretrain}.py`, `src/experiments/{common,exp_001..exp_012}`,
`src/evaluation/{metrics,graph_validation}.py`, `scripts/{run_ci_sweep,shortcut_audit,subset_breakdown}.py`,
`configs/*`, and the produced artifacts under `data/processed/` and `outputs/`.

---

## 0. Working-tree inventory

- 3 commits; working tree carries large uncommitted changes (48 modified tracked
  files + untracked docs/scripts/experiments). **Do not reset/checkout.** This
  audit is read-only plus additive fixes.
- Datasets: `data/` ~1.9 GB (raw HF snapshot + interim chunks + processed
  artifacts). No redownload needed.
- Checkpoints: `checkpoints/exp_002/tagn_seed{42,123,1337}.pt` are tracked;
  `tagn_seed{456,789}.pt` and `exp_012_grl_seed42.pt` are untracked.
- Registry: `outputs/results_registry.json` has **61 rows, 6 fields only**
  (`experiment, seed, task, model, metrics, [variant/note]`). **No provenance**
  (git rev, config hash, split fingerprint, status, timestamps) — Phase 6 gap.

---

## 1. Split integrity

**Protocol** (`configs/data.yaml:49`, `data/processed/split_stats.json`):
`temporal_within_recording`, ratios 0.7/0.15/0.15 per recording; jammer + Track-2
`/synthetic_gen/` recordings are *atomic* (whole recording assigned by
chronological rank; synthetic_gen uses 0.5/0.15/0.35, `prepare.py:84-116`).

| id | Severity | Finding | Status |
|----|----------|---------|--------|
| S1 | INFO | Same recording appears in >1 split (30/45 recordings). Intended and documented (`docs/data_leakage.md:10-24`). Measured: **0 retained chunks share a raw timestep across splits**; overlapping/straddling chunks are dropped (`prepare.py:145,161`). | By design |
| S2 | LOW | `configs/data.yaml:53 embargo_s: 12.8` is **never read** by code. The implicit embargo is only "drop straddling chunks"; nearest train/test chunks can still share ~1 window of context. | Confirmed doc/code drift |
| S3 | MEDIUM | `chunk_id` order is **not chronological** (`loader.py:35,91` sorts by `rglob` file order). A naive "split by contiguous id range" would silently leak. Split *membership* is temporal; ids are not. | Confirmed (metadata hazard) |

## 2. Preprocessing leakage

| id | Severity | Finding | Status |
|----|----------|---------|--------|
| P1 | — | Scaler is **train-only** (`prepare.py:176,268`; applied frozen at `common.py:35-39`). No imputation; no data-driven feature selection. | Verified clean |
| P2 | MEDIUM | Categorical protocol **codebook built over the full dataset incl. val/test** (`loader.py:95-101`). Minor unsupervised vocabulary leakage. | Confirmed defect |
| P3 | LOW | `type_map` label space derived from full meta (`prepare.py:213`). Train happens to contain all 11 types, so no practical effect. | Confirmed defect (minor) |

## 3. Ground-truth validity

| id | Severity | Finding | Status |
|----|----------|---------|--------|
| G1 | **HIGH** | `rca_templates.json` is built with **last-write-wins over all splits** (`prepare.py:221-225`, called on full meta at `:269`). Verified: artifact equals the **last** anomalous chunk's set for **11/11** types, and equals the train **majority** set for only **3/11**. For 8/11 types the template is an arbitrary instance (e.g. `Resource Allocation Bugs` = the *least* common 4-KPI set vs. majority 9-KPI set). Defining chunk is in val/test for several types. Consumed by `common.py:417`, `exp_001:209-235`, `exp_002:179-188`, `exp_005:62`, `exp_006:103`, `recompute_rca.py:58`. | **Confirmed defect** |
| G2 | MEDIUM | Synthetic-injection labels include **identity/inert effects** (`synthetic_inject.py:87-100,307-324,378,390`) — a KPI can be labelled affected with no injected change. | Confirmed defect (label noise) |
| G3 | MEDIUM | Multi-event windows are **under-labelled**: only `overlap[0]` contributes `affected_kpis` (`synthetic_inject.py:375,378`) although all overlapping events were applied to the data. | Confirmed defect |
| G4 | MEDIUM | Synthetic `end_time` is 10× too large (`synthetic_inject.py:382`: `w0 + 10*(wlen-1)` = 127 s vs. true 12.7 s; verified in `chunks_meta.parquet`). | Confirmed defect (metadata) |
| G5 | **HIGH** | Track-2 synthetic recordings **duplicate the background signal of Setting-I train recordings** (each injected recording is derived from a `normal/...` source recording that contributes 215 chunks to train; `manifest.json`). Non-injected timesteps of test synthetic windows are bit-identical to train data. Unacknowledged in `docs/data_leakage.md`. | Confirmed defect (structural) |

## 4. Evaluation metrics & protocol

| id | Severity | Finding | Status |
|----|----------|---------|--------|
| M1 | — | PR-AUC = `average_precision_score`, ROC = `roc_auc_score`, on continuous scores, pos_label=1, pooled across windows (`metrics.py:45-74`). | Verified clean |
| M2 | **HIGH** | **Unsupervised scores are sign-inverted**: `exp_001:120-121,129-130,295` uses `score_samples` directly (higher = normal). Artifact confirms ROC 0.236/0.196/0.049 for IF/OCSVM/LOF (`exp_001_...json`), all far below 0.5; inverting gives plausible 0.76/0.80/0.95. F1 degenerates to predict-all-positive (recall 1.0, precision = base rate). | **Confirmed defect** |
| M3 | MEDIUM | `f1_max` is **test-set-optimised** whenever `summarize(threshold=None)` is used (`metrics.py:60-61`): exp_001 (`:122,124`), exp_009 (`:92-99`), exp_011 (`:97-102`). Contradicts the val-tuned convention (`docs/data_leakage.md:58-59`, `configs/training.yaml:19-20`) used by exp_002/exp_005 (`common.py:464,472`). Ranking metrics unaffected. | Confirmed defect |
| M4 | MEDIUM | top-k RCA uses **k = |ground-truth set|** (`common.py:419`, `exp_001:246-248`, `subset_breakdown.py:53`) — prediction set size chosen with label information. | Confirmed defect |
| M5 | HIGH | Pooled windows treated as **i.i.d.** everywhere; cluster-bootstrap CIs are planned (`docs/novelty_plan.md:107-120`) but **not implemented** (`bootstrap_samples` unused). | Confirmed defect |
| M6 | LOW | `run_ci_sweep.py:44-58` uses z=1.645 for a 90% CI with n=5 (t quantile ≈ 2.132) → over-narrow. `stats_agg` silently drops `None`. | Confirmed defect |
| M7 | — | No `best-seed`/`best-config`-on-test selection anywhere (`run_ci_sweep` aggregates mean/std; exp_003/005 report all variants). exp_011's test-optimal weight is explicitly flagged `LEAKY` (`:121`). | Verified clean |
| M8 | **HIGH** | `exp_003_ablations.py` trained every variant with a bare `C.fit` loop and **no stage-A summary anchor**. Its `full` variant reached test PR 0.536 while the headline model (exp_002, same split/seed) reaches 0.9255, so the old bake-off confounded *training protocol* with *component contribution*, and `hazard_only` collapsed to PR 0.035. | **Fixed**: stage-A anchor + frozen `summary_head` + `trunk_anomaly_weight=1.0`, shared recipe; old JSONs archived to `outputs/results/archive/stale_2026-10-07/` |
| M9 | MEDIUM | exp_002 has its own `evaluate`, so saved results predate the fixed-k RCA ranking metrics (M4) and cannot be read from its JSON. | **Fixed**: `scripts/rca_from_checkpoint.py` recovers Recall@k/MRR/MAP from the checkpoint without retraining |

## 5. Cross-experiment comparability

| id | Severity | Finding | Status |
|----|----------|---------|--------|
| C1 | **HIGH** | exp_001 reports `n=4768, pos=223` (`exp_001_...json`), **not** the v3 test set `n=4969, pos=424`. So the "acceptance gate PR-AUC ≥ 0.9523" (`docs/integrated_results_ci.md:21`) was measured on a different, template-only test set than TAGN v3. | **Confirmed defect** |
| C2 | **CRITICAL** | **exp_009 is mixed-vintage**: seeds 42/123 → `n=4969/424`; seeds 456/789/1337 → `n=4768/223` (verified across all five `exp_009_hybrid_seed*.json`). Any 5-seed aggregate silently mixes test sets. | **Confirmed defect** |
| C3 | **HIGH** | exp_009 does **not register** results (no `registry.append`); `run_ci_sweep.py` rewrites per-seed JSONs but aborts before the aggregate on failure, leaving mixed-vintage state without a split fingerprint (no `n`/`pos` in aggregate rows). | Confirmed defect |
| C4 | MEDIUM | `ci_sweep_5seed.json` contains keys (`threshold_free`, `convention_note`) that `run_ci_sweep.py` does not emit; `ci_sweep_5seed_corrected.json` has a schema no script writes, yet `docs/integrated_results_ci.md:76` claims reproduction via `run_ci_sweep.py`. | Confirmed defect (provenance) |

## 6. Graph estimation / model training

| id | Severity | Finding | Status |
|----|----------|---------|--------|
| R1 | — | KMeans prototypes fit on **train bins only** (`graphs.py:209-213`); per-bin GraphicalLasso+TE are split-specific (no train-only filter, but no cross-split mixing). Prototypes stored non-trainable. | By design (transductive) |
| R2 | — | Pretraining is train-only, label-free (`exp_008:46-63`, `pretrain.py:157-193`). Anchor scaler + summary/residual heads fit on train, selected on val (`common.py:126-291`). exp_002 early stopping/checkpoint selection val-only (`exp_002:378-386`), threshold val-tuned (`:405-406`). | Verified clean |
| R3 | MEDIUM | exp_002 `--anchor-only` reports test-tuned F1 (`exp_002:335-337`). exp_012 `TCRL` probe fits on half of **test** (`exp_012:166-178`) — documented diagnostic, not model selection. | Confirmed defect (reporting) |
| R4 | **HIGH** | LORO reuses the Setting-I-train scaler + prototypes (`exp_005:55-61`); 69-70% of each fold's test chunks are also Setting-I train chunks. LORO "unseen-recording" numbers are not leakage-free. | Confirmed defect |

---

## Priority queue

**P0 (correctness, cheap):** M2 unsup sign · M3 val-threshold operating points · G1 train-only majority
templates · C2/C3 exp_009 vintage + registry provenance · C1/C4 provenance. **Done.**

**P1 (correctness, moderate compute):** R4 per-fold LORO scaling · G5 dominant-signal control ·
G2/G3/G4 injection-label fixes (require regeneration). **R4 done; G5 mitigated
(recording-level CIs + paired test); G2/G3/G4 open (needs synthetic regeneration).**

**P2 (measurement, cheap):** M4 fixed-k RCA · M5 cluster-bootstrap CIs · M6 t-quantile CI ·
RCA ranking metrics (Recall@k/MRR/MAP). **Done**, plus a paired cluster-bootstrap
significance test (`scripts/paired_test.py`) and checkpoint-recovered RCA metrics
(`scripts/rca_from_checkpoint.py`).

**P3 (research, expensive):** type-invariant / shortcut-resistant RCA (C1 of
`docs/novelty_plan.md` — currently a confirmed negative) · TCN/graph-free
baselines. **TCN/GRU baselines done** (`exp_013`); the graph ablation (`exp_003`)
now shows the graph is detection-neutral. Consolidated results: `docs/results_v3.md`.
