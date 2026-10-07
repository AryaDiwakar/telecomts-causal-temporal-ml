# Data-Leakage & Validity Analysis

This note documents every place where the TelecomTS benchmark carries a
built-in shortcut, and exactly which design choices we take (and report
honestly) in response. It is written before model training so the protocol is
fixed ahead of time.

## 1. Chronological split protocol (Setting I)

Per recording (raw file), we split **along that recording's own timeline** at
70 % / 15 % / 15 %:

- **Normal recordings:** chunk-level split. Any chunk that straddles a 70 % or
  85 % boundary is dropped (one full-chunk embargo of 12.8 s each side).
  Consequence: no raw timestep is ever shared between chunks in different
  splits.
- **Synthetic (anomalous) recordings:** overlapping chunks are grouped into
  **burst clusters** (`src/preprocessing/prepare.py::cluster_chunks`). A
  cluster is atomic — assigned to exactly one split by its time position, and
  **dropped entirely** if it straddles a boundary, so a single injected fault
  event never spans train and test.
- **Jammer recordings:** a single continuous fault spanning the whole
  recording is inherently atomic; these three recordings are assigned
  **whole**, chronologically — File -> train, Twitch -> val, YouTube -> test.

Verified invariants: no chunk appears in two splits; no two chunks from the
same recording in different splits share raw timesteps (checked in the
pipeline); train contains all 11 anomaly types, test 10/11 (Resource
Allocation Bugs is chronologically absent from test — reported as-is).

## 2. Built-in shortcuts we refuse to exploit

1. **Affected-KPI templates are type-derived constants.** Because
   `affected_kpis` has exactly one variant per anomaly type, a model that
   guesses the right *type* automatically gets the "right" KPI set. We
   therefore:
     - treat **anomaly-type classification** as the true perceptual task, and
       evaluate RCA both (a) as `type -> template` (mapping) and (b) as an
       **input-driven** localization (top-K shifted KPIs) that never sees any
       template.
1. **Recording-origin memorization.** Since each recording contains only one
   class of anomaly (normal or exactly one anomaly type), a model can cheat by
   recognizing the recording/zone/scenario rather than the fault signature.
   Evidence (exp_001): a recording-prior RCA (most frequent train type in that
   recording) reaches Jaccard 0.82, *above* the ML type-prior (0.76). We
   expose this with:
     - Setting II (**leave-one-recording-out** GroupKFold; `GroupKFold` on
       recording) that forces every test chunk to come from recordings unseen
       in training (implemented in a separate prepare script), and
     - a threats-to-validity section that reports seen-recording vs
       unseen-recording gap.
1. **Description / statistics / QnA / tickets leakage.** Not used as model
   input; `statistics` excluded by design; `troubleshooting_tickets` are
   labels, never inputs.

## 3. Evaluation-time pitfalls tracked

- **Threshold selection** is done on the validation split; all reported test
  numbers use the val-tuned operating point (reported alongside threshold).
- **Class imbalance** (3.86 % anomaly): we report ROC-AUC (threshold-free),
  PR-AUC, F1@best, and TPR@FPR 0.01 / 0.05 — not raw accuracy.
- **Anomaly types differing across splits:** macro metrics are computed over
  types *present* in the test split; the set is always reported
  (10/11 currently) so macro-F1 is not inflated by absent types.
- **Chunks within a burst are highly correlated.** We therefore always report
  cluster/recording-level summaries (e.g., RCA Jaccard per recording) in
  addition to chunk-level numbers, and per-type aggregated metrics.

## 4. What we deliberately do NOT do

- No dataset-level random shuffle split (would put the same burst / the same
  recording in train and test).
- No fixing of `affected_kpis` from test chunks.
- No templated RCA labels as supervision beyond the type label itself
  (the template is only used to *map* a predicted type to KPI indices for
  evaluation, matching how the dataset labels were generated).

## 5. Residual threats (documented, unfixable-at-data-level)

- Normal and anomalous recordings come from different scenario pipelines;
  Jammer/synthetic differ from normal in zone/app composition.
- The label "exists/type" is author-provided and may itself be noisy; we do
  not re-annotate.
- Sampling has burst-level auto-correlation; per-step independence
  assumptions are violated — clustering corrects this at the unit level.