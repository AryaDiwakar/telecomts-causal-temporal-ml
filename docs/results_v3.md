# Results — v3 (leak-free protocol)

All numbers here are computed on the v3 test set (**n = 5,689, positives = 424,
26 recordings**) using validation-tuned operating points and 95% *cluster*
bootstrap intervals over recordings. Protocol and validity rules:
`docs/benchmark_protocol_v3.md`. Finding index: `docs/benchmark_audit_v3.md`.
The split was repaired for finding **G5** (synthetic background sources are no
longer in train unless their injected twin is): every number below is retrained
on that repaired split. **The repair reverses the headline claim — see §6.**

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
bootstrap over the 26 test recordings, so it reflects *recording-level*
variability rather than the (much narrower, and misleading) chunk-level one
(finding M5).

| model | seeds | PR-AUC (mean ± sd) | PR 95% CI | ROC-AUC (mean ± sd) | ROC 95% CI |
|---|---|---|---|---|---|
| **TCN (temporal deep)** | 3 | **0.8606 ± 0.0589** | [0.5722, 0.9748] | 0.9438 ± 0.0100 | [0.8417, 0.9878] |
| **LogReg (hand features)** | 1 | 0.8196 | [0.6265, 0.9387] | 0.9309 | [0.8657, 0.9713] |
| GRU (temporal deep) | 3 | 0.7635 ± 0.0266 | [0.4657, 0.9308] | 0.9399 ± 0.0078 | [0.8417, 0.9808] |
| TAGN (graph + summary) | 5 | 0.7589 ± 0.0413 | [0.4713, 0.9777] | **0.9543 ± 0.0069** | [0.8767, 0.9941] |
| XGBoost (hand features) | 1 | 0.7427 | [0.4519, 0.9628] | 0.8956 | [0.7501, 0.9939] |
| LOF (unsupervised) | 1 | 0.5776 | [0.3145, 0.8195] | 0.9307 | [0.8631, 0.9704] |
| OneClassSVM (unsupervised) | 1 | 0.5278 | [0.2493, 0.8282] | 0.8776 | [0.7856, 0.9336] |
| IsolationForest (unsupervised) | 1 | 0.1923 | [0.0738, 0.5855] | 0.7956 | [0.6253, 0.9343] |

Reading the table:

- **On PR-AUC — the metric that matters at this positive rate — TAGN does not
  beat the baselines.** Both the hand-feature LogReg (0.8196) and the TCN
  (0.8606) rank test chunks better on average, and the paired test in §4 makes
  the LogReg gap explicit (negative on 4/5 seeds).
- **TAGN has the best ROC-AUC mean (0.9543)** and the best PR upper bound
  (0.9777): it separates cleanly on *some* recordings, but its per-seed spread
  (±0.041) is now large relative to the baseline gaps.
- The temporal deep baselines split: TCN is the strongest PR model, GRU is
  mid-field.
- IsolationForest is the weakest: its unsupervised score is essentially
  sign-inverted for this signal (finding M2).
- Single-model CIs overlap heavily, so the pooled ranks alone are not evidence;
  the paired analysis in §4 is the correct (variance-reducing) comparison because
  both models are scored on the *same* chunks.
- This is the **post-G5** picture and it reverses the pre-repair conclusion
  (TAGN 0.92 > LogReg 0.87). §6 explains why; **do not cite the older numbers.**

## 2. Setting II — leave-recordings-out (honest generalization)

Each fold holds out entire recordings *and* refits the scaler and the KMeans
graph prototypes on the fold's training recordings only (finding R4 fixed;
`src/models/graph_fold.py`).

| model | protocol | PR-AUC (mean ± sd over folds) | fold range | ROC-AUC (mean ± sd) | fold range |
|---|---|---|---|---|---|
| TAGN | leave-recordings-out (5 folds) | 0.7689 ± 0.0737 | [0.6831, 0.8475] | 0.9011 ± 0.0277 | [0.8581, 0.9283] |

The generalization number is **0.769 PR-AUC** with a large fold spread
(0.68–0.85). The key observation is that it is now **indistinguishable from the
pooled Setting-I number (0.7589)**: once the G5 background leak is removed,
holding entire *sessions* out costs nothing relative to the pooled split. The
pre-repair tables showed Setting I (0.92) comfortably above Setting II (0.75);
that gap was recording-specific memorisation — the model had seen each test
session's background in training — not a transferable anomaly signature.

## 3. Root-cause localization (RCA)

exp_002's own `evaluate` predates the fixed-k ranking metrics, so they are
recovered from the checkpoint (no retraining) with
`scripts/rca_from_checkpoint.py` (finding M9). Ground-truth root sets are large
(median 9 of 16 KPIs across anomalous training windows and 7 on the test split),
so `recall@1` is bounded near 1/|gt| ≈ 0.1 and is not the right headline;
`MRR`/`MAP` are.

| seed | recall@1 | recall@3 | recall@5 | MRR | MAP | type acc |
|---|---|---|---|---|---|---|
| 42 | 0.0227 | 0.1035 | 0.2268 | 0.4621 | 0.4883 | 0.2453 |
| 123 | 0.0841 | 0.1970 | 0.3513 | 0.7792 | 0.5776 | 0.2170 |
| 456 | 0.0656 | 0.1840 | 0.3232 | 0.6936 | 0.5437 | 0.2170 |
| 789 | 0.0531 | 0.1380 | 0.2593 | 0.6394 | 0.5245 | 0.2453 |
| 1337 | 0.0863 | 0.2051 | 0.2751 | 0.7806 | 0.5331 | 0.1934 |
| **mean** | 0.0624 | 0.1655 | 0.2871 | **0.6710** | **0.5334** | 0.2236 |

Two honest observations:

- Type classification accuracy is ~0.22 (chance ≈ 0.09), i.e. weak. The per-KPI
  ranking is a little better than the type label: MAP ≈ 0.53.
- Detection quality and RCA quality are **not monotone across seeds**: seed 42 is
  mid-field on detection (0.7534) but worst on RCA (MRR 0.4621), and seed 123 has
  the lowest detection (0.7300) but the second-best MRR (0.7792). RCA variance is
  dominated by the shared trunk, not by the anomaly head.

## 4. Does TAGN beat the baseline?

Because both models are scored on the *same* chunks, the right statistic is a
**paired** cluster bootstrap of the metric difference (recordings resampled with
replacement, both models re-scored on each resample). Positive ΔPR = TAGN better;
the baseline is LogReg, the strongest PR model in §1:

| seed | ΔPR (95% CI) | p(ΔPR ≤ 0) | ΔROC (95% CI) |
|---|---|---|---|
| 42 | -0.0662 [-0.1563, +0.0557] | 0.7860 | +0.0196 [-0.0218, +0.0559] |
| 123 | -0.0896 [-0.2000, +0.0570] | 0.8805 | +0.0208 [-0.0357, +0.0797] |
| 456 | -0.0605 [-0.1482, +0.0712] | 0.7745 | +0.0271 [-0.0185, +0.0800] |
| 789 | -0.0956 [-0.1981, +0.0526] | 0.8900 | +0.0160 [-0.0381, +0.0698] |
| 1337 | +0.0082 [-0.0721, +0.0698] | 0.3740 | +0.0334 [+0.0024, +0.0708] |

**Conclusion (negative).** TAGN is **not** significantly better than the
classical LogReg on PR-AUC, and on 4/5 seeds it is *worse* (ΔPR < 0); every 95%
CI nevertheless includes zero, so the honest statement is "no detectable
advantage", not "significantly worse". On ROC-AUC the sign is consistently
positive (+0.016 to +0.033), but only seed 1337 excludes zero. This is the
opposite of the pre-repair result; the earlier "TAGN beats the classical
baseline" claim was an artifact of the G5 split leak (§6) and **must be
retracted**.

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
| full | 0.7305 | 0.7439 | 0.9485 | 0.1651 | 0.5310 |
| no_graph | 0.7305 | 0.7581 | 0.9520 | 0.1840 | 0.5402 |
| static_graph | 0.7305 | 0.7482 | 0.9501 | 0.1934 | 0.5538 |
| undirected | 0.7305 | 0.7431 | 0.9499 | 0.1934 | 0.5484 |
| no_nodeconv | 0.7305 | 0.7445 | 0.9499 | 0.1769 | 0.5364 |
| hazard_only | 0.7305 | 0.7305 | 0.9497 | 0.1557 | 0.5640 |

Interpretation (single seed, indicative):

- **Detection is anchor-dominated.** Every variant sits within **0.028 PR** of the
  frozen anchor (0.7305); removing the graph entirely (`no_graph`, PR 0.7581)
  does not hurt — it is the highest PR here — while training only the
  early-warning head (`hazard_only`) reproduces the anchor exactly. The graph,
  the node convolution and the edge directionality are all *detection-neutral*.
- **RCA / type is also barely graph-dependent** at this operating point (RCA
  jaccard 0.531–0.564 across variants; type accuracy 0.156–0.193, near chance
  ≈0.09).
- The honest conclusion is therefore **not** "the graph helps" but "the graph
  does not help, and this dataset supplies no evidence that it does for any head
  we measured." This is consistent with the Setting-I ≈ Setting-II equality in
  §2: there is no transferable graph signal beyond the summary statistics.

> Note: the earlier `exp_003` numbers (e.g. `full` PR 0.536, `hazard_only` PR
> 0.035) were produced without the stage-A anchor and are archived under
> `outputs/results/archive/stale_2026-10-07/`; they are not comparable and must
> not be cited (finding M8).

---

## 6. What changed relative to the old numbers (do not cite the old ones)

- **The headline reverses (finding G5, 2026-10-14).** In the pre-repair split,
  every injected Track-2 recording's `normal/...` source was itself
  chunk-split across splits: e.g. `normal/mobile/File` contributed **215 chunks
  to train, 43 to val, 43 to test**, while its injected twin
  (`anomalous/synthetic_gen/mobile/File`) was atomically in test. The model
  therefore trained on the very session whose test anomalies it was scored on.
  Each source is now bound atomically to its injected twin's split (the four
  test sources add ~720 clean normal chunks to test; the test set grows from
  4,969 to 5,689 chunks, still 424 positives). **TAGN's pooled PR-AUC falls from
  ~0.92 to 0.7589**, below LogReg (0.8196) and TCN (0.8606), and Setting I
  collapses onto Setting II (0.7689). All numbers in this document are retrained
  on the repaired split; the pre-repair artefacts are archived under
  `outputs/results/archive/stale_2026-10-14_preG/`.
- **Synthetic label repair (findings G2/G3/G4, 2026-10-14).** Identity/inert
  effects were labelled as affected, multi-event windows were under-labelled,
  and `end_time` was 10× too large. The injector was fixed and the corpus
  regenerated: the KPI time series are unchanged to <6e-11 (JSON round-trip
  only); **280 spurious affected-KPI labels were removed** across 237/714
  windows. Detection labels are unaffected, so this changes RCA KPI ground
  truth, not detection.
- Old classical numbers were on a different test set and carried a score-cache
  bug that made all unsupervised baselines identical; both fixed (findings C1,
  M2).
- Old exp_003 ablations were trained without the stage-A anchor, so their `full`
  variant (PR 0.536) did not match the headline model (finding M8); archived.
- LORO was refit leak-free (per-fold scaler + graph prototypes, finding R4); the
  legacy global-artifact number is not citable.
- New temporal deep baselines (TCN/GRU, `exp_013`) and checkpoint-recovered RCA
  metrics (`scripts/rca_from_checkpoint.py`) were added; both supersede the
  "no deep baseline / no ranking metrics" gaps in the earlier write-up.
- The pre-v2 "acceptance gate" and any table in `docs/integrated_results_ci.md`
  are invalid.
