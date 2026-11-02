# Headline benchmark — v3 (leak-free)

Setting I — pooled window, validation-tuned threshold. CIs are 95%
*cluster-bootstrap* intervals (recordings resampled with replacement);
for multi-seed models the interval shown is the union of per-seed intervals.

| model | seeds | PR-AUC (mean ± sd) | PR 95% CI | ROC-AUC (mean ± sd) | ROC 95% CI |
|---|---|---|---|---|---|
| TAGN (graph + summary) | 5 | 0.9236 ± 0.0039 | [0.8317, 0.9768] | 0.9554 ± 0.0025 | [0.8991, 0.9918] |
| LogReg (hand features) | 1 | 0.8717 | [0.7822, 0.9304] | 0.9114 | [0.8389, 0.9609] |
| LOF (unsupervised) | 1 | 0.8167 | [0.5831, 0.9335] | 0.9462 | [0.8830, 0.9802] |
| IsolationForest (unsupervised) | 1 | 0.3869 | [0.1729, 0.7336] | 0.8505 | [0.6952, 0.9634] |
| OneClassSVM (unsupervised) | 1 | 0.6839 | [0.4395, 0.8350] | 0.8418 | [0.7278, 0.9057] |
| XGBoost (hand features) | 1 | 0.7317 | [0.4565, 0.9523] | 0.8953 | [0.7714, 0.9880] |
| TCN (temporal deep) | 3 | 0.8447 ± 0.0634 | [0.5431, 0.9612] | 0.9298 ± 0.0151 | [0.8312, 0.9754] |
| GRU (temporal deep) | 3 | 0.8353 ± 0.0385 | [0.6310, 0.9509] | 0.9379 ± 0.0069 | [0.8538, 0.9850] |

## Setting II — leave-recordings-out (honest generalization)

| model | protocol | PR-AUC (mean ± sd over folds) | fold range | ROC-AUC (mean ± sd) | fold range |
|---|---|---|---|---|---|
| TAGN (seed 42) | leave-recordings-out (5 folds) | 0.7535 ± 0.1499 | [0.5092, 0.9129] | 0.9053 ± 0.0613 | [0.8287, 0.9799] |

## Component ablation (exp_003, seed 42, shared stage-A anchor)

All variants share the summary anchor, split, seed and optimiser.
`anchor PR` is the frozen summary readout before stage-B trunk training.

| variant | anchor PR | test PR-AUC | test ROC-AUC | type acc | RCA jaccard |
|---|---|---|---|---|---|
| full | 0.9286 | 0.9309 | 0.9612 | 0.1816 | 0.5264 |
| no_graph | 0.9286 | 0.9300 | 0.9602 | 0.1698 | 0.5144 |
| static_graph | 0.9286 | 0.9303 | 0.9616 | 0.1698 | 0.5284 |
| undirected | 0.9286 | 0.9301 | 0.9618 | 0.1722 | 0.5076 |
| no_nodeconv | 0.9286 | 0.9282 | 0.9609 | 0.1627 | 0.4969 |
| hazard_only | 0.9286 | 0.9286 | 0.9621 | 0.1604 | 0.5003 |

## Paired cluster bootstrap vs `logreg` (same test chunks)

dPR = PR(TAGN) − PR(baseline) on shared resamples; `p` is the fraction
of resamples with dPR <= 0 (one-sided evidence for TAGN).

| seed | dPR (95% CI) | p | dROC |
|---|---|---|---|
| 42 | +0.0538 [+0.0103, +0.0982] | 0.0060 | +0.0442 [+0.0162, +0.0777] |
| 123 | +0.0553 [+0.0203, +0.0859] | 0.0050 | +0.0457 [+0.0219, +0.0724] |
| 456 | +0.0517 [+0.0149, +0.0902] | 0.0020 | +0.0448 [+0.0199, +0.0788] |
| 789 | +0.0533 [+0.0137, +0.0944] | 0.0045 | +0.0455 [+0.0172, +0.0819] |
| 1337 | +0.0452 [+0.0148, +0.0728] | 0.0010 | +0.0397 [+0.0175, +0.0627] |
