# Faculty Demonstration Script

Use this as a 10-page, 10-12 minute demonstration. Do not begin by saying
"state of the art." Begin with the operational problem, disclose the data
limits, then let the live result earn attention.

## Page 1: The Question

Show: title page.

Say:

> 5G networks generate thousands of KPI signals, but an anomaly alarm alone
> does not tell an operator what failed or what to inspect first. This project
> builds a leakage-aware system that detects abnormal behavior, ranks candidate
> root-cause KPIs, identifies the fault pattern, and estimates early warning.

Faculty takeaway: this is an operational decision-support problem, not just a
classification exercise.

## Page 2: What The Data Represents

Show: dataset page or dashboard Dataset section.

Say:

> The data is aggregated 5G RAN telemetry: 16 numerical KPI channels and two
> protocol channels sampled at 10 Hz. It describes network or scenario behavior,
> not individual phones. There are no UE IDs, tower IDs, locations, packets, or
> subscriber traces.

Point out: 32,000 windows, 128 steps per window, 33 recordings, 1,235 labeled
              anomalies, and 11 anomaly types.

Faculty takeaway: the scope is clearly defined and the claims are not inflated.

## Page 3: The Hidden Validity Problem

Show: recording layout and anomaly distribution.

Say:

> The difficult part is not only training a neural network. The dataset has few
> long recordings, overlapping windows, and recording-level scenario structure.
> A model can appear excellent by recognizing a file or scenario instead of
> learning a fault pattern that works on a new file.

Then say:

> Root-cause labels also use constant KPI templates per anomaly type. If we do
> not audit that, RCA can quietly become type lookup.

Faculty takeaway: the project has a research question beyond model size.

## Page 4: The Contribution

Show: novelty slide.

Say:

> The new part is a set of checks for whether the model is using real fault
> patterns or easy clues. We test file recognition, fixed fault-to-KPI lookup,
> repeated overlapping windows, and whether KPI connections stay stable on new
> recordings. TAGN is the tool; the trust checks are the research contribution.

Do not say: "We discovered the causal network." Say: "The graph shows which KPIs
move together; it does not prove that one KPI caused another."

## Page 5: The Full Pipeline

Show: pipeline diagram.

Say:

> Each window passes through five stages: leakage-safe chronological splitting,
> train-only scaling, statistical and temporal representation learning, dynamic
> KPI graph reasoning, and four operational outputs: detection, type, RCA, and
> early warning.

Point to the flow:

`telemetry -> split audit -> features and graph -> TAGN -> diagnosis -> what-if replay`

Faculty takeaway: every output is connected to a visible stage in the pipeline.

## Page 6: The Model

Show: TAGN architecture page.

Say:

> The temporal convolution models local KPI dynamics. Separate KPI node states
> preserve channel identity. Graph attention passes information along candidate
> KPI dependencies. A GRU summarizes the sequence. The heads then produce the
> anomaly score, fault type, KPI ranking, and pre-onset hazard curve.

Then disclose:

> The anomaly head also contains an explicit statistical anchor because the
> strongest baseline exposes distributional shifts directly. The graph path is
> gated so multi-task training cannot destroy the validated detector.

## Page 7: The Benchmark Result

Show: benchmark table or dashboard Anomaly Detection section.

Say:

> On the chronological held-out split, corrected logistic regression reaches
> PR-AUC 0.9523. TAGN reaches 0.9730. A fixed 50/50 blend of TAGN and a
> deterministic logistic expert reaches PR-AUC 0.9774 and ROC-AUC 0.9926.

Point out: the blend weight was fixed before test evaluation. The result is not
              presented as universal SOTA without multi-seed and LORO confirmation.

Faculty takeaway: the proposed system beats the strongest local baseline while
              remaining honest about evaluation scope.

## Page 8: The Validity Audit

Show: robustness, LORO, graph, and RCA audit panels.

Say:

> We do not accept the chronological score as the whole story. The protocol
> includes leave-recording-out evaluation, recording-prior RCA, input-driven RCA,
> burst-aware splitting, graph-direction audits, perturbation tests, and ablations.

Use this sentence if challenged:

> A high in-distribution score is useful for the benchmark, but transfer to an
> unseen recording is the stronger deployment question and remains a required
> confirmation for the hybrid model.

## Page 9: Live Operation Replay

Show: `streamlit run src/dashboard/app.py`, then the Live Operation & Simulation
page.

Do this in order:

1. Select an anomalous held-out chunk.
2. Show the six most abnormal KPI traces.
3. Show the anomaly score and predicted type.
4. Show the ranked KPI root causes.
5. Show the active graph prototype and strongest edges.
6. Show the early-warning hazard curve.

Say:

> This is the transition from a benchmark number to an operator workflow. The
> system does not stop at "anomaly = yes"; it gives the operator a ranked next
> inspection list and a time-localized warning signal.

## Page 10: Predictive Simulation And Close

Show: intervention simulator.

Do this:

1. Select the top-ranked KPI.
2. Reduce it over the final 32 steps.
3. Click `Run what-if simulation`.
4. Compare baseline and counterfactual scores.

Say:

> This replay asks whether the trained predictor would change its decision if a
> KPI trajectory were modified. It is a predictive what-if, not proof of a
> physical causal intervention. That distinction is deliberate.

Close with:

> The contribution is a complete, inspectable chain from aggregated 5G telemetry
> to anomaly detection, candidate root cause, early warning, and simulation,
> evaluated against the shortcuts that can make RCA results look stronger than
> they really are.

## Questions To Expect

### Is this truly causal?

> No. The graphs are predictive dependency candidates. Causal identification
> would require interventions, stronger assumptions, or experimental data.

### Is the data per phone?

> No. It is aggregated RAN telemetry. The system diagnoses network-level patterns,
> not individual subscribers.

### Why not use logistic regression?

> We do use it and report it prominently. The proposed model beats it on the
> current chronological benchmark while adding temporal, graph, type, RCA, and
> early-warning outputs. The hybrid result still needs multi-seed and LORO
> confirmation before a broad SOTA claim.

### What is the main limitation?

> The dataset has 33 recordings, recording-level scenario confounding, synthetic
> anomalies, and template-based RCA labels. Those limitations are measured and
> exposed rather than hidden.

### What would make this publishable?

> Complete hybrid LORO evaluation, multi-seed confidence intervals, compatible
> published baselines, and an ablation proving which part of the hybrid improves
> transfer rather than only the chronological split.
