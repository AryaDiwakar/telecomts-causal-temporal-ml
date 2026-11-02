# Benchmark audit — the type→template shortcut in KPI-level RCA

Status: **complete.** Script: `scripts/shortcut_audit.py`; output
`outputs/results/shortcut_audit.json`.

## Finding

TelecomTS labels each anomalous window with an `affected_kpis` set that is the
KPI-level root-cause ground truth. In the shipped dataset this set is **constant
per anomaly type**:

- `anomalous/synthetic`: 956 anomalous windows, 10 types, **1 distinct
  affected-KPI set per type (100%)**.
- `anomalous/jammer`: 279 windows, 1 type, 1 set.

Consequently the KPI-level RCA task is a *bijection with the type label*: a model
that predicts the type and emits that type's template scores near-perfect RCA
**without looking at the input**. Any reported RCA score that does not beat this
input-free baseline is not evidence of causal localization.

## Quantitative ceiling

`scripts/shortcut_audit.py` builds a train-majority template per type and scores
an **input-free "type-oracle"** predictor (emit the true type's template) against
`y_affected` with the same top-k Jaccard used by the model. On the corrected
split (see `docs/exp_012_typeinvariant_rca.md` for the split fix):

| test subset | windows | types | max distinct sets/type | type-oracle Jaccard | global-majority Jaccard |
|---|---|---|---|---|---|
| all_test | 424 | 11 | 4 | 0.824 | 0.552 |
| template (shipped) | 223 | 10 | **1** | **0.971** | 0.542 |
| instance-varying (Track 2) | 201 | 5 | 3 | 0.661 | 0.563 |

On the shipped (template) test windows an input-free lookup reaches **0.97**
Jaccard — essentially the reported model RCA. On genuinely instance-varying
ground truth the same input-free predictor drops to 0.66, so the KPI set must be
recovered from the signal.

## Model vs input-free oracle (corrected split, `exp_002`/`exp_012` seed 42)

The model's own top-k RCA never approaches the input-free ceiling, on either
subset:

| test subset | type-oracle Jaccard | model Jaccard (baseline) | model Jaccard (GRL) |
|---|---|---|---|
| template (shipped) | 0.971 | 0.325 | 0.627 |
| instance-varying | 0.661 | 0.253 | 0.508 |

On template windows the input-free oracle (0.971) is ~3× the model (0.325): the
model is only weakly echoing the type lookup, and the headline "RCA" reported on
such data is dominated by the shortcut. On instance-varying windows both the
oracle (0.661) and the model (0.253) drop, and the model again sits well below the
oracle — but here the oracle is *cheating* (it knows the true type), so the gap
measures localization skill that must come from the signal.

## Corrected benchmark

Track 2 (`src/data/synthetic_inject.py`) injects faults whose active KPI set is
a per-event random 50–100% subset of the type's symptom KPIs, producing 2–6
distinct affected sets per type. The split logic
(`src/preprocessing/prepare.py::assign_splits`) assigns injected recordings
atomically with a 0.5/0.15/0.35 train/val/test share so non-template RCA windows
appear in val (56 windows) and test (201 windows) — the within-recording temporal
split otherwise placed every injected window in train.

## Implications for the paper

1. **Audit contribution:** publish the shortcut and the ceiling numbers; they
   invalidate input-free RCA on TelecomTS.
2. **Corrected benchmark:** release the instance-varying injection + split.
3. **Honest evaluation:** report model RCA separately on template vs
   instance-varying test windows (see exp_012); report detection separately too,
   since the harder injected anomalies lower PR-AUC.
4. The type→template shortcut is also *why* the C1 type-invariance adversary
   cannot work (see exp_012): attribution quality is type-predictive.

## Caveats

- The type→template map is injective (one set per type) but not surjective onto
  KPI subsets; RCA in the shipped data is exactly 10-way type classification.
- `min_sustained_steps: 1` and the 2% window rule are unchanged from the main
  pipeline; only the affected-KPI subsetting is new.
