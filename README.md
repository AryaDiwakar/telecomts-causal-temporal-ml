# Causal-Temporal ML for Root-Cause Localization in 5G Networks

End-to-end, reproducible research on the **TelecomTS** dataset
(AliMaatouk/TelecomTS): anomaly detection, anomaly-type classification, and
KPI-level root-cause localization with leakage-controlled splits, dynamic
KPI dependency graphs, a temporal-GNN model, honest baselines, and
explainability/early-warning analysis.

## Project layout

```
configs/            data/model/training/experiment configs
data/raw/           TelecomTS huggingface snapshot
data/interim/       chunks_X.npy (32000,128,18), chunks_meta.parquet, kpi names
data/processed/     scaler, splits, targets, rca templates, graph artifacts
src/data/           download + loader
src/preprocessing/  leakage-safe splits (prepare, prepare_loro)
src/features/       classical per-chunk features
src/models/         graphs.py (dependency discovery), tagn.py (temporal-GNN)
src/experiments/    exp_001..007
src/evaluation/     metrics.py
src/dashboard/      streamlit app
src/report/         result aggregation
outputs/            results/, tables/, figures/, results_registry.json
docs/               schema, leakage, literature, gap, methodology,
                    novelty_audit, limitations
tests/              pipeline sanity tests
```

## Key results so far (exp_001, seed 42, test split)

The numbers below are the last completed **pre-TAGN-v2** run and are retained
as a baseline, not as results for the upgraded model. TAGN-v2 changes graph
message direction, uses edge strengths in attention, adds causal channel
normalization, and adds a chunk-statistics branch. All TAGN experiments must
be rerun before making comparisons.

| model | ROC-AUC | PR-AUC | notes |
|---|---|---|---|
| Isolation Forest | 0.25 | 0.03 | below random — anomaly manifold is dense |
| LOF (unsup.) | 0.24 | 0.03 | same |
| One-Class SVM | 0.69 | 0.08 | |
| XGBoost (sup.) | 0.92 | 0.67 | |
| Logistic Regr. (sup.) | 0.97 | 0.76 | |

Root-cause baselines: recording-prior Jaccard **0.82** > ML type-prior **0.76**
> input-driven top-K **0.37** — strong evidence of the recording-memorization
shortcut; motivates leave-recording-out (Setting II).

exp_002 (TAGN) has completed runs for all five configured seeds. The aggregate
results and full experiment matrix are in `outputs/report.md`.

## Latest verified anomaly result

The corrected chronological benchmark now reports logistic regression at
PR-AUC **0.9523** / ROC-AUC **0.9880**. The TAGN anomaly anchor uses the same
train-only normalization and an explicitly checked classical-statistics
prefix, followed by a frozen linear anchor and a nonlinear residual. Its
validated anchor-only artifact reaches PR-AUC **0.9730** / ROC-AUC **0.9921**
on the held-out test split. A fixed 50/50 logit ensemble with a deterministic
logistic expert reaches PR-AUC **0.9774** / ROC-AUC **0.9926**
(`outputs/results/exp_009_hybrid_seed42.json`).
These are verified chronological anomaly-head results; leave-recording-out and
external published-baseline comparisons are still required for an SOTA claim.

Run the fast benchmark with:

`python3 -m src.experiments.exp_002_tagn --seed 42 --summary-epochs 100 --summary-residual-epochs 40 --anchor-only`

`python3 -m src.experiments.exp_009_hybrid --seed 42 --weight 0.5`

The full multi-task run keeps the graph anomaly residual gated at zero by
default while it learns type, RCA, and early-warning representations. Tune
`--trunk-anomaly-weight` on validation before enabling that residual.

## Reproduce

1. `pip3 install "scipy==1.14.1" "numpy==1.26.4"` (see scipy wheel caveat in
   requirements.txt) plus the other packages listed in requirements.txt.
2. `python3 -m src.data.download`
3. `python3 -m src.data.loader`
4. `python3 -m src.preprocessing.prepare`
5. `python3 -m src.eda.run_eda`
6. `python3 -m src.models.graphs`
7. `python3 -m src.experiments.exp_001_classical_baselines`
8. `python3 -m src.experiments.exp_002_tagn --seed 42 --epochs 20 --patience 6`
9. `streamlit run src/dashboard/app.py`

The dashboard opens on **Live Operation & Simulation**. Select a held-out test
chunk to replay telemetry, graph context, anomaly score, predicted type, KPI
root causes, and early-warning hazard. The intervention panel applies a
predictive KPI perturbation and reruns the checkpoint; it is explicitly a
model what-if, not an identified causal intervention.

The TAGN checkpoint must be regenerated after changing the model architecture;
the training command above writes it to `checkpoints/exp_002/`. Because the
graph message direction and hazard objective were corrected, rerun exp_002,
the ablations, robustness, LORO, explainability, and early-warning analyses
before treating old output artifacts as evidence.
All experiments are honest `RESULT_PENDING` until actually run and written to
`outputs/results_registry.json`; the report records completed runs and any
remaining gaps.

## Novelty & validity
Only hedged claims; see `docs/novelty_audit.md` and `docs/data_leakage.md`.
