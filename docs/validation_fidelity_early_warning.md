# Validation: interventional RCA fidelity and early warning

This note records the two validation studies added to the paper. Both are
reproducible from the released checkpoints and scripts; the machine-readable
outputs are:

- `outputs/results/fidelity_validation.json`
- `outputs/results/early_warning_validation.json`

```bash
python scripts/validate_fidelity.py
python scripts/validate_early_warning.py
```

## 1. Interventional RCA fidelity

**Definition.** For a window `x`, a candidate root set `S`, and an anomaly
score `s(.)`, intervene by setting the normalized KPI channels in `S` to zero
(the training mean), then compute the score drop

```
Delta(S) = s(x) - s(x_do(S)).
```

For a candidate set `R` and a matched random non-root set `C` of the same size,

```
F = mean( 1[Delta(R) > Delta(C)] + 0.5 * 1[Delta(R) == Delta(C)] ).
```

**Controls.** A valid metric must pass a positive control (ground-truth
`affected_kpis`) and a negative control (random root set). We also report
sensitivity to `k` and to the checkpoint seed.

**Setup.** Five TAGN checkpoints (seeds 42, 123, 456, 789, 1337), 424 anomalous
test windows from 13 recordings, matched random non-root sets for the top-$k$
controls. The full-set control uses a random set of the same size (which may
overlap the roots). This is a model-internal perturbation, not an identified
physical intervention.

**Results (mean ± SD over checkpoints; range in parentheses).**

| Control | k | Fidelity |
|---|---|---|
| Predicted roots | 3 | 0.567 ± 0.235 (0.304–0.976) |
| Predicted roots | 5 | 0.570 ± 0.217 (0.368–0.986) |
| Predicted roots | 7 | 0.561 ± 0.215 (0.377–0.976) |
| True roots (positive) | 3 | 0.496 ± 0.052 (0.420–0.559) |
| True roots (positive) | 5 | 0.476 ± 0.086 (0.351–0.616) |
| True roots (positive) | 7 | 0.519 ± 0.137 (0.349–0.738) |
| True roots (full set) | full | 0.508 ± 0.085 (0.394–0.639) |
| Random roots (negative) | 3 | 0.503 ± 0.016 (0.479–0.528) |
| Random roots (negative) | 5 | 0.506 ± 0.026 (0.469–0.542) |
| Random roots (negative) | 7 | 0.495 ± 0.025 (0.467–0.528) |

**Conclusion.** The negative control behaves as expected. The positive control
does not: the ground-truth affected-KPI set is indistinguishable from a random
set, both when the set is truncated to `k` and when the full ground-truth set
(median 7 KPIs) is used (0.508 ± 0.085). The metric therefore fails validation,
and the predicted-root score (0.567 ± 0.235) is not evidence of causal
localization. The earlier pilot value of 0.465 is withdrawn: it used one
checkpoint without matched controls and is not reproducible under the corrected
implementation.

## 2. Early warning

**Problem with the old metric.** Taking `argmax` hazard over the whole window
and comparing it with onset can score post-onset steps and is not calibrated to
a false-alarm budget.

**Event-level protocol.** For each anomalous window with `onset > 0`, find the
first step before onset whose hazard exceeds `tau`. Detection occurs if such a
step exists; lead time is `onset - first_alarm`. Choose `tau` on validation
normal windows for a target window-level false-alarm rate.

**Results (five seeds, 93 test events).**

| Metric | Mean | SD |
|---|---|---|
| Per-step AUC | 0.495 | 0.029 |
| Per-step AP | 0.006 | 0.001 |
| Detection at 1% FA | 0.037 | 0.009 |
| False alarms/window at 1% FA | 0.018 | 0.007 |
| Detection at 5% FA | 0.101 | 0.048 |
| False alarms/window at 5% FA | 0.096 | 0.029 |
| Argmax "warn before onset" | 0.652 | 0.134 |
| Argmax mean absolute lead error | 53.6 steps | 5.7 |

**Conclusion.** The hazard head is near chance under a calibrated event-level
evaluation. The old argmax rule reports 65.2% "warn before onset" but hides a
mean absolute lead error of 53.6 steps. The support is small (93 of 424 test
anomalies have `onset > 0`), so the numbers are indicative rather than
definitive.

## 3. What this changes in the paper

- The headline is no longer "TAGN beats baselines" or "fidelity 0.465".
- TAGN is reported as a testbed; the audit, the validation protocol, and the
  negative evidence are the contributions.
- The learned graph is described as a statistical dependency graph, not a
  verified causal DAG.
- Early warning is reported with an event-level, false-alarm-calibrated
  protocol.
