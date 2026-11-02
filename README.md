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

## Key results (v3, leak-free split)

The citable, current numbers are in **`docs/results_v3.md`** and
`outputs/tables/headline_v3.{md,csv}`: test set **n = 5,689**, 424 positives,
26 recordings, validation-tuned thresholds, recording-level cluster-bootstrap
CIs.

Setting I (pooled window), mean PR-AUC / ROC-AUC:

| model | PR-AUC | ROC-AUC |
|---|---|---|
| TCN (temporal deep) | **0.8606** | 0.9438 |
| LogReg (hand features) | 0.8196 | 0.9309 |
| GRU (temporal deep) | 0.7635 | 0.9399 |
| TAGN (graph + summary) | 0.7589 | **0.9543** |
| XGBoost (hand features) | 0.7427 | 0.8956 |
| LOF / OneClassSVM / IsolationForest | 0.5776 / 0.5278 / 0.1923 | 0.9307 / 0.8776 / 0.7956 |

After the G5 split repair (finding G5, `docs/benchmark_audit_v3.md`), **TAGN
does not beat the classical or temporal baselines on PR-AUC** (paired ΔPR is
negative on 4/5 seeds vs LogReg) and Setting I (0.7589) ≈ Setting II (0.7689).
The graph is detection-neutral in the ablation. This reverses the earlier
headline; see §6 of `docs/results_v3.md`.

The older "acceptance gate" numbers that previously appeared here (LogReg
0.9523 / anchor 0.9730 / ensemble 0.9774 PR-AUC) were measured on a different,
smaller test set and — for the Track-2 synthetic recordings — with the test
sessions' backgrounds present in training. They are **invalid and must not be
cited**; the pre-repair artefacts are archived under `outputs/results/archive/`.

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
