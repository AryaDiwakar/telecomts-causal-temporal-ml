# Causal-Temporal Machine Learning for Root-Cause Localization in 5G Networks Using Multimodal Observability Data

*Review-1 Document · Causal-Temporal ML for 5G Root-Cause Localization*

---

## Abstract

Root-cause localization in 5G radio access networks requires reasoning over heterogeneous, bursty, multimodal observability streams — KPI time series, graph-structured dependencies, and temporal dynamics. This work proposes a causal-temporal machine learning framework that detects anomalies, classifies fault types, and localizes the responsible key performance indicators (KPIs) in 5G RAN telemetry. We build a multi-task Temporal Attention Graph Network (TAGN) that jointly models causal KPI dependencies discovered via GraphicalLasso and transfer entropy, temporal dynamics through causal convolutions and a GRU, and multimodal observability signals, trained under leakage-safe chronological and leave-recording-out splits. To ensure honest evaluation, we audit the widely used TelecomTS benchmark and repair three confounders: a train/test background leak, mislabeled synthetic ground truth, and a template shortcut in which root-cause KPI sets were constant per fault type. All experiments are retrained on the repaired split and reported with recording-level cluster-bootstrap confidence intervals and paired significance tests. On the corrected benchmark (n = 5,689 test windows, 424 positives, 26 recordings) TCN (0.8606) and logistic regression (0.8196) outperform TAGN (0.7589) in PR-AUC, while TAGN retains the best ROC-AUC (0.9543); paired tests show no significant advantage over the classical baseline and ablations confirm the KPI graph is detection-neutral. We additionally demonstrate cross-vendor transfer to SpotLight and deferral-aware root-cause analysis. The complete pipeline, 13 experiments, 53 passing tests, and a Streamlit dashboard are released for reproducibility.

**Keywords:** 5G RAN; anomaly detection; root-cause localization; temporal graph neural network; multimodal observability; benchmark audit; leakage-safe evaluation
