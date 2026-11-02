# Novelty Audit (honest)

This document states exactly what we claim is novel, and what we explicitly
do NOT claim. It is written to keep any write-up honest.

## Claims we can defend
1. **An open, reproducible dynamic-KPI-graph protocol for TelecomTS.**
   Given TelecomTS (a public benchmark), we publish the full pipeline that
   derives candidate-oriented, time-varying, per-chunk KPI dependency graphs
   (GraphicalLasso skeleton + transfer-entropy orientation, prototype
   clustering of bin-level topologies; `src/models/graphs.py`) and feeds them
   to a temporal-GNN model. To the best of our knowledge, no published
   TelecomTS study ships this (the dataset's own baseline suite does not
   provide per-chunk candidate-oriented graphs).
2. **An explicit leakage/interference audit for TelecomTS RCA.**
   We make visible the *recording-memorization* and *template-is-constant*
   shortcuts that plague type/RCA evaluation on this benchmark, and provide
   matching-prior baselines (recording-prior) that bound what any model must
   beat to show real generalization (exp_001: recording-prior Jaccard 0.82 >
   ML type-prior 0.76 > input-shift 0.37). This framing — measuring the
   shortcut explicitly — is, to our knowledge, not present in prior TelecomTS
   write-ups.
3. **A defined leave-one-recording-out (LORO) protocol for TelecomTS** with
   GroupKFold on recording, designed to neutralize the shortcut in (2). The
   five-fold results are now reported in `outputs/report.md`; performance is
   interpreted as recording-shift evidence, not as a universal estimate.

## What we DO NOT claim
- State-of-the-art accuracy on TelecomTS vs every published method. Other
  papers use their own splits (often random, which inflates via burst-sharing);
  direct numbers are not comparable across those protocols. We only claim
  *internally consistent* numbers under our audited protocol.
- That the extracted graphs equal the true (unknown) physical causal DAG of a
  5G RAN.
- That `affected_kpis` reflects independent causal root causes (it is a
  per-type template; `docs/data_leakage.md`).
- Any result or reference we have not ourselves run/verified this session.
  Every experiment with `RESULT_PENDING` is reported as pending, not implied.

## Non-verifiable references policy
Any paper found only on mirror/aggregator sites (not the primary venue) is
listed separately in `docs/literature_review.md` and is never used as
authoritative support.

## Citation verification (2026-10-13)

The three prior-work citations used to motivate the shortcut audit were checked
against primary sources and are confirmed real:

- **arXiv:2606.22450** — Fard, Komarov, Wunder, *Cross-Layer Intrusion Detection
  in 5G O-RAN: Gains and Limits of Fusing Radio Telemetry with Network Flow
  Records*, submitted 21 Jun 2026; accepted at IEEE CSR 2026 (Lisbon). Supports
  "the field measures evaluation artifacts in RAN telemetry."
- **arXiv:2603.09675** — Bello, Chiarlone, Fiori, García González, Larroca, *GNNs
  for Time Series Anomaly Detection: An Open-Source Framework and a Critical
  Evaluation*, 10 Mar 2026. Supports "metric design/thresholding obscures TSAD
  comparisons."
- **KAC** — Salavati, Wu, Wan, Darbari, Hong (University of Connecticut · Nokia),
  *KPI-Aware Multimodal Anomaly Detection for 5G and Open RAN Telemetry*, IEEE
  ICDM 2026 Applied Track. Reports TelecomTS F1 0.960 / SpotLight F1 0.950 on
  their own splits — these are the bars we reference, with the caveat that they
  are not measured under our audited split.

Related but not relied upon: **arXiv:2604.17616** — Mishra, *Conditional
Attribution for Root Cause Analysis in Time-Series Anomaly Detection*, 2026.
