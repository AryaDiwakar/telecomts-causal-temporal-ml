# Integrated results: 5-seed CI + seed ensemble (TelecomTS, Track 2 folded in)

> **STATUS: the tables below are PRE-v2 (template GT, split bug) and are kept only
> for provenance. They are superseded by the "Corrected benchmark" section.**
> The Track 2 corpus was regenerated with the instance-varying injector and
> `assign_splits` was fixed so injected anomalies reach val/test
> (see `docs/benchmark_shortcut_audit.md`, `docs/exp_012_typeinvariant_rca.md`).

## Corrected benchmark (v3; instance-varying GT in test, n=4969, 424 positives)

5-seed sweep of TAGN (`exp_002`, seeds 42/123/456/789/1337) on the corrected
split. Raw: `outputs/results/ci_sweep_5seed_corrected.json`.

| model | ROC-AUC | PR-AUC | F1-max |
|---|---|---|---|
| TAGN (exp_002) | 0.9554 ± 0.0022 | 0.9236 ± 0.0035 | 0.8986 ± 0.0099 |

Per-seed (ROC / PR / F1): 42 → 0.9556/0.9255/0.9117; 123 → 0.9571/0.9270/0.9010;
456 → 0.9563/0.9234/0.8929; 789 → 0.9569/0.9250/0.9045; 1337 → 0.9511/0.9169/0.8828.

- Acceptance gate (PR-AUC ≥ 0.9523, ROC-AUC ≥ 0.9880): **FAIL** on the corrected
  benchmark (PR 0.924, ROC 0.955) — the harder injected anomalies are genuinely
  less separable than the shipped template anomalies.
- Input-free type-oracle RCA ceiling: template 0.971, instance-varying 0.661
  (model baseline 0.325 / 0.253). See `docs/benchmark_shortcut_audit.md`.

## Pre-v2 tables (INVALID — do not cite)

Dataset: 32,714 windows / 45 recordings (Track 2 synthetic injection folded
in). Seed sweep: 42, 123, 456, 789, 1337; each seed trains TAGN from scratch
(exp_002, ~12.5 min) then the hybrid (exp_009). Raw: `outputs/results/ci_sweep_5seed.json`,
`outputs/results/exp_011_seed_ensemble.json`, `outputs/results/ci_sweep_log.txt`.

## Threshold-free detection metrics (test split, mean ± std, n=5)
| model | ROC-AUC | PR-AUC | F1-max |
|---|---|---|---|
| TAGN (exp_002) | 0.9913 ± 0.0007 | 0.9699 ± 0.0027 | 0.9484 ± 0.0036 |
| Logistic reference | 0.9841 (det.) | 0.9624 (det.) | 0.9388 (det.) |
| Hybrid TAGN+ref (w=0.5) | 0.9886 ± 0.0010 | 0.9701 ± 0.0009 | **0.9526 ± 0.0018** |

90% CIs (hybrid): F1 [0.9513, 0.9539], PR [0.9694, 0.9708], ROC [0.9878, 0.9893].

## Seed ensemble (mean logits across the 5 checkpoints, inference only)
| model | ROC-AUC | PR-AUC | F1-max |
|---|---|---|---|
| TAGN ensemble | 0.9920 | 0.9728 | 0.9500 |
| Hybrid ensemble (w=0.5) | 0.9885 | 0.9704 | 0.9522 |

Ensembling lifts ranking quality (PR 0.9728, ROC 0.9920) but not the max-F1
ceiling.

## Acceptance gate vs. KAC
- Gate (F1 ≥ 0.80, PR-AUC ≥ 0.9523, ROC-AUC ≥ 0.9880): **PASS** for TAGN and
  hybrid on the means (hybrid ROC CI low 0.9878 is the only marginal cell).
- KAC TelecomTS SOTA F1 0.960: **not cleared.** Best honest numbers are hybrid
  mean 0.9526 [0.9513, 0.9539] and val-selected seed ensemble 0.9500.
- Fusion-weight sweep (`exp_011.weight_sweep`): val max-F1 is monotone in the
  TAGN weight and selects w=1.0 (test F1 0.9500); test max-F1 peaks at w=0.8
  (0.9565) but that weight is test-selected (leaky) and is reported only as an
  upper bound. Even that optimistic point is below 0.960.

## Interpretation
- The gap to 0.960 is ~0.007 F1 and is not closed by ensembling or fusion-weight
  selection. A genuine gain needs a different modelling lever (e.g. restoring
  the pre-injection F1 by re-balancing the harder instance-varying windows, or
  a better-calibrated operating point), not threshold/ensemble tricks.
- Val vs test max-F1 diverges sharply at w=1 (val 0.839 vs test 0.948), so the
  val-tuned threshold transfers poorly to test (this is why exp_002 reports a
  low F1 at the val threshold). Probability calibration is the likely culprit.
- Reported numbers are threshold-free (max-F1) unless stated; exp_002's own
  `f1_max` (0.880 ± 0.028) is F1 at the val-tuned threshold and is *not*
  comparable to the threshold-free figures above.

## Reproduce
```
python3 scripts/run_ci_sweep.py --seeds 42 123 456 789 1337
python3 -m src.experiments.exp_011_seed_ensemble
```
