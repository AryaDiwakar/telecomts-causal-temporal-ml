# Results — v3 (leak-free protocol)

All numbers here are computed on the v3 test set (**n = 4,969, positives = 424,
34 recordings**) using validation-tuned operating points and 95% *cluster*
bootstrap intervals over recordings. Protocol and validity rules:
`docs/benchmark_protocol_v3.md`. Finding index: `docs/benchmark_audit_v3.md`.

Generated artefacts:

| artefact | script |
|---|---|
| `outputs/tables/headline_v3.{md,csv}` | `python -m scripts.build_headline_table` |
| `outputs/results/bootstrap_ci*.json` | `python -m scripts.bootstrap_ci ...` |
| `outputs/results/paired_test.json` | `python -m scripts.paired_test` |
| `outputs/results/rca_ranking_exp_002.json` | `python -m scripts.rca_from_checkpoint` |

---

## 1. Setting I — pooled-window detection

The whole setting is a single held-out chronological/recording split; the CI is a
bootstrap over the 34 test recordings, so it reflects *recording-level*
variability rather than the (much narrower, and misleading) chunk-level one
(finding M5).

| model | seeds | PR-AUC (mean ± sd) | PR 95% CI | ROC-AUC (mean ± sd) | ROC 95% CI |
|---|---|---|---|---|---|
| **TAGN (graph + summary)** | 5 | **0.9236 ± 0.0039** | [0.8317, 0.9768] | **0.9554 ± 0.0025** | [0.8991, 0.9918] |
| LogReg (hand features) | 1 | 0.8717 | [0.7822, 0.9304] | 0.9114 | [0.8389, 0.9609] |
| TCN (temporal deep) | 3 | 0.8447 ± 0.0634 | [0.5431, 0.9612] | 0.9298 ± 0.0151 | [0.8312, 0.9754] |
| GRU (temporal deep) | 3 | 0.8353 ± 0.0385 | [0.6310, 0.9509] | 0.9379 ± 0.0069 | [0.8538, 0.9850] |
| LOF (unsupervised) | 1 | 0.8167 | [0.5831, 0.9335] | 0.9462 | [0.8830, 0.9802] |
| XGBoost (hand features) | 1 | 0.7317 | [0.4565, 0.9523] | 0.8953 | [0.7714, 0.9880] |
| OneClassSVM (unsupervised) | 1 | 0.6839 | [0.4395, 0.8350] | 0.8418 | [0.7278, 0.9057] |
| IsolationForest (unsupervised) | 1 | 0.3869 | [0.1729, 0.7336] | 0.8505 | [0.6952, 0.9634] |

Reading the table:

- TAGN has the best **point** PR-AUC and the best mean ROC-AUC, and the smallest
  seed-to-seed spread (±0.004 PR).
- The two temporal deep baselines (TCN, GRU) exceed every classical baseline on
  ROC but do **not** beat the hand-feature LogReg on PR, and both are far more
  seed-sensitive (±0.06 and ±0.04) than TAGN.
- The single-model cluster CIs overlap between TAGN and LogReg — with 34
  recordings, a 4–5 point PR gap is not separable from recording-level noise by
  eye. The correct test is the **paired** difference (§4).
- IsolationForest is the weakest: its unsupervised score is essentially
  sign-inverted for this signal (finding M2).

## 2. Setting II — leave-recordings-out (honest generalization)

Each fold holds out entire recordings *and* refits the scaler and the KMeans
graph prototypes on the fold's training recordings only (finding R4 fixed;
`src/models/graph_fold.py`).

| model | protocol | PR-AUC (mean ± sd over folds) | fold range | ROC-AUC (mean ± sd) | fold range |
|---|---|---|---|---|---|
| TAGN | leave-recordings-out (5 folds) | 0.7535 ± 0.1499 | [0.5092, 0.9129] | 0.9053 ± 0.0613 | [0.8287, 0.9799] |

The headline generalization number is **0.754 PR-AUC** with an enormous fold
spread (0.51–0.91). That spread *is* the finding: cross-recording transfer is
unstable, and any single-fold claim would be misleading. Detection degrades from
0.92 (Setting I) to 0.75 (Setting II), i.e. roughly a third of the in-setting
PR-AUC is attributable to recording-specific structure rather than a
transferable anomaly signature.

## 3. Root-cause localization (RCA)

exp_002's own `evaluate` predates the fixed-k ranking metrics, so they are
recovered from the checkpoint (no retraining) with
`scripts/rca_from_checkpoint.py` (finding M9). Ground-truth root sets are large
(median 9 of 16 KPIs), so `recall@1` is bounded near 1/|gt| ≈ 0.11 and is not the
right headline; `MRR`/`MAP` are.

| seed | recall@1 | recall@3 | recall@5 | MRR | MAP | type acc |
|---|---|---|---|---|---|---|
| 42 | 0.0249 | 0.1070 | 0.2243 | 0.4825 | 0.4943 | 0.2123 |
| 123 | 0.1255 | 0.3441 | 0.4737 | 0.9239 | 0.7879 | 0.2193 |
| 456 | 0.0644 | 0.1706 | 0.3041 | 0.6886 | 0.5483 | 0.1981 |
| 789 | 0.0560 | 0.1335 | 0.2458 | 0.6508 | 0.5292 | 0.2075 |
| 1337 | 0.0946 | 0.2177 | 0.2844 | 0.8297 | 0.5476 | 0.1722 |
| **mean** | 0.0731 | 0.1946 | 0.3065 | **0.7151** | **0.5815** | 0.2019 |

Two honest observations:

- Type classification accuracy is ~0.20 (chance ≈ 0.09), i.e. weak. The per-KPI
  ranking is better than the type label: MAP ≈ 0.58.
- Detection quality and RCA quality are **uncorrelated across seeds**: seed 42
  has the best detection (0.9255) and the worst RCA (MRR 0.48), while seed 123
  has slightly worse detection and much better RCA (MRR 0.92). RCA variance is
  dominated by the shared trunk, not by the anomaly head.

## 4. Is TAGN significantly better than the baseline?

Because both models are scored on the *same* chunks, the right statistic is a
**paired** cluster bootstrap of the metric difference (recordings resampled with
replacement, both models re-scored on each resample). Positive Δ = TAGN better:

| seed | ΔPR (95% CI) | p(ΔPR ≤ 0) | ΔROC (95% CI) |
|---|---|---|---|
| 42 | +0.0538 [+0.0103, +0.0982] | 0.0060 | +0.0442 [+0.0162, +0.0777] |
| 123 | +0.0553 [+0.0203, +0.0859] | 0.0050 | +0.0457 [+0.0219, +0.0724] |
| 456 | +0.0517 [+0.0149, +0.0902] | 0.0020 | +0.0448 [+0.0199, +0.0788] |
| 789 | +0.0533 [+0.0137, +0.0944] | 0.0045 | +0.0455 [+0.0172, +0.0819] |
| 1337 | +0.0452 [+0.0148, +0.0728] | 0.0010 | +0.0397 [+0.0175, +0.0627] |
| **mean** | **+0.0519** | **≤0.006** | **+0.0440** |

**Conclusion.** For every seed the 95% CI on ΔPR *and* ΔROC lies strictly above
zero. The two independent single-model CIs in §1 overlap, but the paired test is
the correct (variance-reducing) one, and it shows a consistent ≈+5 PR-point /
≈+4 ROC-point advantage of TAGN over the strongest classical baseline. This is
the paper's central quantitative claim; the ablation (§5) then shows that the
gain is *not* attributable to the graph.

Overlap of two marginal CIs is not a significance test, and a paired test that
ignores recording clustering would be anti-conservative; this analysis avoids
both errors (findings M5, M9).

---

## 5. Component ablation

Same split, seed, anchor, optimiser and validation-tuned threshold for every
variant; `anchor PR` is the frozen summary readout *before* stage-B trunk
training. Because the trunk feeds the anomaly logit with weight 1.0 here, the
ablation can in principle move detection — it doesn't.

| variant | anchor PR | test PR-AUC | test ROC-AUC | type acc | RCA jaccard |
|---|---|---|---|---|---|
| full | 0.9286 | 0.9309 | 0.9612 | 0.1816 | 0.5264 |
| no_graph | 0.9286 | 0.9300 | 0.9602 | 0.1698 | 0.5144 |
| static_graph | 0.9286 | 0.9303 | 0.9616 | 0.1698 | 0.5284 |
| undirected | 0.9286 | 0.9301 | 0.9618 | 0.1722 | 0.5076 |
| no_nodeconv | 0.9286 | 0.9282 | 0.9609 | 0.1627 | 0.4969 |
| hazard_only | 0.9286 | 0.9286 | 0.9621 | 0.1604 | 0.5003 |

Interpretation (single seed, indicative):

- **Detection is anchor-dominated.** Removing the graph entirely (`no_graph`,
  PR 0.9300 vs full 0.9309) or training only the early-warning head
  (`hazard_only`, PR 0.9286 = the frozen anchor) changes PR-AUC by ≈0.001–0.003.
  The graph, the node convolution and the edge directionality are all
  *detection-neutral*; the ≈0.93 PR-AUC comes from the summary readout (stage-A
  anchor + residual).
- **RCA / type is also barely graph-dependent** at this operating point (RCA
  jaccard 0.50–0.53 across variants; type accuracy 0.16–0.18, near chance ≈0.09).
- The honest conclusion is therefore **not** "the graph helps" but "the graph
  does not hurt, and this dataset does not supply evidence that it helps for any
  head we measured." This is consistent with the leakage-free LORO degradation
  in §2 — most of the in-setting performance is not a transferable graph signal.

> Note: the earlier `exp_003` numbers (e.g. `full` PR 0.536, `hazard_only` PR
> 0.035) were produced without the stage-A anchor and are archived under
> `outputs/results/archive/stale_2026-10-07/`; they are not comparable and must
> not be cited (finding M8).

---

## 6. What changed relative to the old numbers (do not cite the old ones)

- Old classical numbers were on a different test set (n=4768/223) and carried a
  score-cache bug that made all unsupervised baselines identical; both fixed
  (findings C1, M2).
- Old exp_003 ablations were trained without the stage-A anchor, so their `full`
  variant (PR 0.536) did not match the headline model (finding M8); archived.
- LORO was refit leak-free (per-fold scaler + graph prototypes, finding R4); the
  legacy global-artifact number (0.7588) is not citable. The leak changed the
  LORO mean by ≈0.005 (0.7588 → 0.7535), so the leak was not the main driver of
  the old LORO figure — the fold spread is.
- New temporal deep baselines (TCN/GRU, `exp_013`) and checkpoint-recovered RCA
  metrics (`scripts/rca_from_checkpoint.py`) were added; both supersede the
  "no deep baseline / no ranking metrics" gaps in the earlier write-up.
- The pre-v2 "acceptance gate" and any table in `docs/integrated_results_ci.md`
  are invalid.
