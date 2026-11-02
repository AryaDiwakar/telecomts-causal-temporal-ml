# Headline benchmark — v3 (leak-free)

Setting I — pooled window, validation-tuned threshold. CIs are 95%
*cluster-bootstrap* intervals (recordings resampled with replacement);
for multi-seed models the interval shown is the union of per-seed intervals.

| model | seeds | PR-AUC (mean ± sd) | PR 95% CI | ROC-AUC (mean ± sd) | ROC 95% CI |
|---|---|---|---|---|---|
| TAGN (graph + summary) | 5 | 0.7589 ± 0.0413 | [0.4713, 0.9777] | 0.9543 ± 0.0069 | [0.8767, 0.9941] |
| LogReg (hand features) | 1 | 0.8196 | [0.6265, 0.9387] | 0.9309 | [0.8657, 0.9713] |
| LOF (unsupervised) | 1 | 0.5776 | [0.3145, 0.8195] | 0.9307 | [0.8631, 0.9704] |
| IsolationForest (unsupervised) | 1 | 0.1923 | [0.0738, 0.5855] | 0.7956 | [0.6253, 0.9343] |
| OneClassSVM (unsupervised) | 1 | 0.5278 | [0.2493, 0.8282] | 0.8776 | [0.7856, 0.9336] |
| XGBoost (hand features) | 1 | 0.7427 | [0.4519, 0.9628] | 0.8956 | [0.7501, 0.9939] |
| TCN (temporal deep) | 3 | 0.8606 ± 0.0589 | [0.5722, 0.9748] | 0.9438 ± 0.0100 | [0.8417, 0.9878] |
| GRU (temporal deep) | 3 | 0.7635 ± 0.0266 | [0.4657, 0.9308] | 0.9399 ± 0.0078 | [0.8417, 0.9808] |

## Setting II — leave-recordings-out (honest generalization)

| model | protocol | PR-AUC (mean ± sd over folds) | fold range | ROC-AUC (mean ± sd) | fold range |
|---|---|---|---|---|---|
| TAGN (seed 42) | leave-recordings-out (5 folds) | 0.7689 ± 0.0737 | [0.6831, 0.8475] | 0.9011 ± 0.0277 | [0.8581, 0.9283] |

## Component ablation (exp_003, seed 42, shared stage-A anchor)

All variants share the summary anchor, split, seed and optimiser.
`anchor PR` is the frozen summary readout before stage-B trunk training.

| variant | anchor PR | test PR-AUC | test ROC-AUC | type acc | RCA jaccard |
|---|---|---|---|---|---|
| full | 0.7305 | 0.7439 | 0.9485 | 0.1651 | 0.5310 |
| no_graph | 0.7305 | 0.7581 | 0.9520 | 0.1840 | 0.5402 |
| static_graph | 0.7305 | 0.7482 | 0.9501 | 0.1934 | 0.5538 |
| undirected | 0.7305 | 0.7431 | 0.9499 | 0.1934 | 0.5484 |
| no_nodeconv | 0.7305 | 0.7445 | 0.9499 | 0.1769 | 0.5364 |
| hazard_only | 0.7305 | 0.7305 | 0.9497 | 0.1557 | 0.5640 |

## Paired cluster bootstrap vs `logreg` (same test chunks)

dPR = PR(TAGN) − PR(baseline) on shared resamples; `p` is the fraction
of resamples with dPR <= 0 (one-sided evidence for TAGN).

| seed | dPR (95% CI) | p | dROC |
|---|---|---|---|
| 42 | -0.0662 [-0.1563, +0.0557] | 0.7860 | +0.0196 [-0.0218, +0.0559] |
| 123 | -0.0896 [-0.2000, +0.0570] | 0.8805 | +0.0208 [-0.0357, +0.0797] |
| 456 | -0.0605 [-0.1482, +0.0712] | 0.7745 | +0.0271 [-0.0185, +0.0800] |
| 789 | -0.0956 [-0.1981, +0.0526] | 0.8900 | +0.0160 [-0.0381, +0.0698] |
| 1337 | +0.0082 [-0.0721, +0.0698] | 0.3740 | +0.0334 [+0.0024, +0.0708] |
