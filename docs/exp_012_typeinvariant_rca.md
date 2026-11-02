# exp_012 — Type-invariant, deferral-capable root-cause analysis (C1)

Status: **complete (seed 42); negative on the type-invariance claim — confirmed
twice, v1 on template GT and v2 on corrected instance-varying GT.**
Code: `src/experiments/exp_012_typeinvariant_rca.py`
Model changes: `src/models/tagn.py` (`rca_adv`, `ident_head`, gradient-reversal
`grad_reverse`).
Results: `outputs/results/exp_012_typeinvariant_rca_seed42.json`,
`outputs/results/exp_012_baseline_seed42.json`,
`outputs/results/exp_012_grl_seed42.json`. Registry: 4 `exp_012` entries.

## Motivation (novelty claim C1)

`docs/novelty_plan.md` proposes C1: a *type-invariant, deferral-capable* root-
cause analysis head. Two mechanisms:

1. **Type-invariance via a gradient-reversal (GRL) adversary.** The adversary
   predicts the anomaly type from the KPI-attribution logits; the gradient is
   reversed into the encoder so attribution cannot reveal the type. Intended
   effect: lower type-conditional RCA leakage (TCRL).
2. **Deferral via an identifiability signal.** Attribution stability `s` under
   input perturbations and attribution entropy feed a calibrator
   `g([s, entropy, max_prob]) -> P(top-k correct)`; low-`g` windows are deferred.
   Intended effect: precision-at-coverage rises with the deferral signal.

## Method

The `TagnModel` gains:
- `rca_adv = MLP(16 -> d -> n_types)` applied to `grad_reverse(kpi_logits, lambda)`;
  trained with cross-entropy on anomalous windows, `lambda` linearly ramped to
  `grl_lambda` over the first 25% of epochs.
- `ident_head` is exercised only by the analysis code (post-hoc deferral
  calibrator fit on the validation split, never on test).

Baseline is the existing `exp_002` seed-42 checkpoint (no adversary). The GRL
variant is trained for 20 epochs with `grl_lambda=1.0`, `adv_weight=1.0`.

## Metrics

- **Detection**: ROC / PR / F1 from `src.evaluation.metrics.summarize`.
- **Attribution quality**: `topk_shift_jaccard` (top-k KPI attribution vs
  `y_affected`), same definition as `exp_002`.
- **TCRL** (type-conditional RCA leakage): fit a logistic type-probe on half of
  the anomalous test attributions, evaluate on the other half, repeated 5x;
  report probe accuracy, majority-class rate, and normalized leakage
  `TCRL = (acc - majority) / (1 - majority)`. *Lower is more type-invariant.*
  (Note: `val` contains only 5 of 11 types, so the probe is fit within `test`.)
- **RCAI**: mean pairwise Spearman correlation of attribution vectors under
  Gaussian input perturbations (`K=4`).
- **Coverage–risk**: precision-at-coverage curves (10–100%) ordered by
  (a) attribution stability, (b) negative entropy, (c) the learned calibrator.

## Results (seed 42)

| variant | ROC | PR | F1 | top-k RCA Jaccard | TCRL | RCAI (stability) |
|---|---|---|---|---|---|---|
| baseline (`exp_002`) | 0.9905 | 0.9708 | **0.9519** | 0.319 | **0.654** | 0.9965 |
| + GRL adversary | 0.9917 | 0.9708 | 0.9500 | **0.694** | 0.736 | 0.9998 |

Both variants pass the acceptance gate (F1 ≥ 0.80, PR ≥ 0.9523, ROC ≥ 0.9880).
GRL does **not** degrade detection and improves top-k attribution Jaccard
(0.319 → 0.694).

**Type-invariance failed.** TCRL increased under GRL (0.654 → 0.736): the
adversarial objective did not remove type information from the attribution, and
the type-probe decodes type *better* from the GRL attribution (probe accuracy
0.796 → 0.845).

**Deferral works only for the baseline.** With the learned calibrator, baseline
precision-at-coverage rises from 0.319 (100% coverage) to 0.440 (50%), 0.448
(25%), 0.503 (10%). For GRL the calibrator is degenerate because the validation
split is single-class (`jaccard ≥ 0.5` for every validation window), and the raw
stability signal is saturated (≈1.0) and uninformative.

## Retest on corrected data (v2, instance-varying GT)

Track 2 was regenerated with the instance-varying injector active (2–5 distinct
affected-KPI sets per type, window sizes 3–11 KPIs; `y_affected` now carries 38
distinct vectors over 1,949 anomalous windows), the pipeline was rebuilt, and
`exp_002` + `exp_012` were retrained (seed 42).

| variant | ROC | PR | F1 | top-k RCA Jaccard | TCRL | RCAI (stability) |
|---|---|---|---|---|---|---|
| baseline (`exp_002` v2) | 0.9918 | 0.9689 | **0.9502** | 0.295 | **0.666** | 0.9968 |
| + GRL adversary (v2) | 0.9927 | 0.9673 | 0.9443 | **0.659** | 0.711 | 0.9997 |

**The conclusion is unchanged.** GRL again *raises* TCRL (0.666 → 0.711) while
improving attribution Jaccard (0.295 → 0.659). Removing the template shortcut
did not make the adversary reduce type leakage: attribution *quality* itself is
type-predictive (a better localized KPI set is easier to type), so an adversary
that suppresses type while the supervised RCA objective rewards type-consistent
attributions is self-defeating. The type-probe still decodes type from the GRL
attribution (probe accuracy 0.804 → 0.830).

Positive v2 finding: with meaningful ground truth, baseline **deferral works**
(precision-at-coverage 0.295 → 0.477 at 10%, 0.473 at 25%). The GRL variant's
learned calibrator is degenerate (validation split is single-class).

## Retest on the corrected split (v3, instance-varying GT in test)

> **Stale — pre-G5 (2026-10-14).** The numbers in this section were measured on
> the 4,969-chunk split *before* the G5 background-leak repair. On the repaired
> split (n=5,689) the corresponding exp_002 detection is PR 0.7534 / ROC 0.9505
> (`docs/results_v3.md`). exp_012 (GRL / TCRL) has **not** been re-run on the
> repaired split, so these values are not citable against the v3 numbers.

The v2 rebuild exposed a split bug: with the generalized stratum, **all**
instance-varying `synthetic_gen` windows had landed in *train* (test carried only
shipped-template anomalies), so v2's "instance-varying" RCA was still measured
against template ground truth. `assign_splits` was fixed to stratify
`/synthetic_gen/` recordings with ratios (0.5, 0.15, 0.35). The corrected test set
is **n=4969, 424 positives** — 201 instance-varying (5 types) + 223 template.

| variant | ROC | PR | F1 | RCA all | RCA inst-var | RCA template | TCRL | TCRL probe acc |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.9556 | 0.9255 | **0.9177** | 0.291 | 0.253 | 0.325 | **0.623** | 0.705 |
| + GRL adversary | 0.9566 | 0.9226 | 0.9077 | **0.570** | 0.508 | 0.627 | 0.855 | 0.887 |

**The conclusion is unchanged and now robust to the split bug.** GRL again raises
type leakage (TCRL 0.623 → 0.855, probe accuracy 0.705 → 0.887) while nearly
doubling attribution Jaccard (0.291 → 0.570), on a test set that genuinely
contains instance-varying ground truth. Note the model's RCA on template windows
(0.325) is *far below* the input-free type-oracle ceiling on those same windows
(0.971; see `docs/benchmark_shortcut_audit.md`) — the reported RCA is a weak echo
of a type lookup, not evidence of localization skill.

Detection on the corrected split is also harder than on v1: PR 0.9255 vs 0.9689
and ROC 0.9556 vs 0.9918 for `exp_002`, i.e. the injected instance-varying
anomalies are less separable than the shipped template anomalies. **The old
acceptance gate (PR ≥ 0.9523) is no longer met** on the corrected benchmark.

## Why type-invariance failed (and what it means)

On the v1 (template) build the root-cause ground truth was type-determined, so
"localize the right KPIs" and "hide the type" were trivially in conflict. The v2
retest removed that explanation: with genuinely instance-varying ground truth the
conflict persists. The deeper reason is architectural — the anomaly-type head,
the type-conditioned summary readout, and the KPI-attribution head are trained
jointly, and **a well-localized attribution is inherently type-predictive**
(knowing *which* KPIs moved narrows the type). A gradient-reversal adversary on
the attribution logits therefore cannot win against the supervised RCA/type
objectives; it only improves localization, which *increases* type leakage. **C1's
type-invariance premise is unsupported on both builds.**

## Data-generation mismatch (v1 only; fixed for v2)

The Track 2 injector (`src/data/synthetic_inject.py`) samples an instance-varying
50–100% subset of a type's symptom KPIs *per event*. The v1 corpus in
`data/raw/TelecomTS/anomalous/synthetic_gen/` was produced
(`manifest.json` mtime 22:48) **before** that subsetting was added to the
injector (file mtime 22:58); the pipeline was then rebuilt (23:00) against the
stale raw data. Result: every injected window carries a per-type-constant
affected-KPI set. This is verifiable: each of the 12 injected recordings has a
single distinct `affected_kpis` per type.

## Comparability flags

- Single seed (42); not a multi-seed result.
- GRL trained 20 epochs vs `exp_002`'s full schedule — not epoch-matched.
- v3 uses the corrected split (instance-varying GT in train/val/test); v2 kept all
  instance-varying windows in train; v1 used template GT throughout.
- TCRL is a linear-probe lower bound on leakage, not an information-theoretic
  measure.

## State of the tree after v3

Track 2 data was regenerated and the pipeline rebuilt, then the split bug was
fixed, so `data/interim`, `data/processed`, and the `exp_002` seed-42 checkpoint
now reflect the corrected data **with instance-varying anomalies in test**. The
pre-v2 artifacts (template GT, old checkpoints, old results) are saved under
`/var/folders/.../opencode/pre_instance_backup/`. **All previously reported
CI/transfer numbers (5-seed CI, exp_010, exp_011, exp_005 LORO) were computed on
the pre-v2 data and are now invalid**; only `exp_002` seed 42 and `exp_012` seed
42 have been recomputed on the corrected split (5-seed `exp_002` CI in progress).

## Recommendation

C1's type-invariance mechanism should be reported as **unsupported** on both
builds. Two viable directions remain:

1. **Keep v2 and re-run the suite** (5-seed `exp_002`/`exp_009`, `exp_010`,
   `exp_011`, `exp_005` LORO) to refresh the headline and transfer numbers on the
   corrected data.
2. **Revert to the pre-v2 backup** if the CI/transfer numbers are to be preserved
   and C1 is dropped, since C1 fails either way.

If C1 is pursued further, the mechanism must change: a gradient-reversal
adversary on the attribution logits cannot remove type information while the
supervised RCA objective rewards type-consistent attributions. Alternatives
include enforcing invariance on a *separate* representation (not the attribution)
or reframing the contribution as **selective/deferral RCA**, which the v2 data
supports (precision-at-coverage 0.295 → 0.477).

## BibTeX

```bibtex
@inproceedings{sun2024spotlight,
  title     = {SpotLight: A Multi-Modal Anomaly Detection Dataset for 5G O-RAN},
  author    = {Sun, Chuanhao and Pawar, Ujjwal and others},
  booktitle = {Proceedings of the 30th Annual International Conference on Mobile
               Computing and Networking (MobiCom)},
  year      = {2024},
  doi       = {10.1145/3636534.364938}
}

@article{feng2025telecomts,
  title   = {TelecomTS: A Dataset and Benchmark for Time-Series Anomaly Detection
             in Telecom Networks},
  author  = {Feng, et al.},
  journal = {arXiv preprint arXiv:2510.06063},
  year    = {2025}
}
```
