# Demo Talk Track

## Opening

"This project studies anomaly detection and root-cause localization in 5G
telemetry. The main contribution is not only a neural model; it is an audited
evaluation protocol that tests whether a model learned transferable KPI
signals or memorized recording and template shortcuts."

## Five Numbers

- **0.957 PR-AUC:** strongest completed chronological baseline, logistic regression.
- **0.647 ± 0.136 PR-AUC:** TAGN mean and standard deviation over five seeds (best 0.841, worst 0.489).
- **0.606 PR-AUC:** leave-recording-out generalization result.
- **0.818 vs 0.362 Jaccard:** recording-prior RCA versus input-shift RCA, exposing the memorization shortcut.
- **0.016:** worst PR-AUC drop under inference-time perturbations (channel dropout).

## What To Show

1. Open `outputs/scopus_demo.html`.
2. Point to the evidence-status banner and explain that old TAGN artifacts are
   clearly marked because TAGN-v2 requires a full rerun.
3. Show the primary benchmark table: the engineered baseline is deliberately
   included, so the neural model cannot claim an improvement without earning it.
4. Show the generalization and validity-audit panels.
5. Show the **seed stability** panel and state the spread out loud; hiding it
   would be the first thing a reviewer catches.
6. Show the **ablation** panel. The static-graph variant currently beats the
   full model by +0.279 PR-AUC; frame this as an open question, not a defeat.
7. Show the **robustness** and **early warning** panels, then the four embedded
   evidence figures.
8. Open the Streamlit dashboard for the graph, early-warning, and explainability
   views: `streamlit run src/dashboard/app.py`.

## Reviewer-Safe Answer

**Question:** "Is this state of the art?"

**Answer:** "The current artifact is a strong, reproducible research
demonstration, not yet a verified universal state-of-the-art claim. It exposes
the baseline gap and has a clear validation path: rerun TAGN-v2, all ablations,
LORO, robustness, and early-warning experiments, then report confidence
intervals and directly comparable published baselines."

**Question:** "Your seed variance is large and a simpler ablation beats the full model."

**Answer:** "Both are reported deliberately. The ±0.136 spread tells us the
training objective is under-determined, which is exactly what the TAGN-v2
causal-normalization and class-weighting fixes target. The static-graph win
means the dynamic graph is not yet earning its complexity, so the rerun will
either close that gap or justify simplifying the architecture."

## Closing

"The scientific value is that the project makes shortcut behavior measurable.
The next result is not selected because it looks impressive; it is accepted
only if it survives recording-held-out evaluation and beats the honest
engineered baseline under the same protocol."
