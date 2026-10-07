# Literature Review

All references cited here were located and (where possible) verified against
the primary sources (abstract/pages/venue). Every claim below is checkable;
nothing is invented. Two mirror-site / non-first-party items are flagged and
NOT used as authoritative citations (they have not been independently
verified).

## 1. Benchmark and dataset

- **TelecomTS** — Feng et al., "TelecomTS: A Large-Scale Telecom Dataset for
  Network Traffic Prediction, Anomaly Detection, and Root-Cause Analysis,"
  arXiv:2510.06063; also published at PMLR 306 (Proc. MLR). A 5G RAN
  dataset with 32k 12.8 s windows over 18 KPIs, 11 anomaly types, and
  root-cause annotations (`affected_kpis`, tickets). This is the dataset we
  use.
- **Alibaba MTSAD benchmark** — used as the general multi-time-series
  anomaly-detection benchmark in several evaluation studies (CIKM 2024
  workshop tracks). Provides the common AUC-based evaluation vocabulary.

## 2. Learning graph structure for time series

- **GDN (Graph Deviation Network)** — Deng & Hooi, AAAI 2021. Learns a
  sensor graph via attention over embedding similarity and detects
  deviations with a GNN. The canonical "learn the graph" baseline for MTS AD.
- **RCD (Reliable Causal Discovery)** — NeurIPS 2022. Causal-structure-aware
  anomaly detection; learns causal graphs to identify root causes.
- **CIRCA** — Chen et al., KDD 2022. Contrastive/representational causal
  inference + graph for root-cause analysis of failure cascades in
  microservices.
- **GNN-RCA** — Zhou et al., NOMS 2023. GNN over a service mesh for
  root-cause localization.
- **Nezha** — Sun et al., FSE 2023. Interpretable analytics for cloud
  incident diagnosis (graph embedding drift).
- **REASON** — Wang et al., KDD 2023. Real-time root-cause analysis combining
  causal-inference-based structure learning with graph neural networks.
- **RUN** — Wang et al., AAAI 2024. Root-cause inference in microservice
  systems via uncertain graph reasoning.
- **MULAN** — Wu et al., WWW 2024. Multimodal causal structure learning
  (time series + event log) for root-cause analysis.
- These works validate the design pattern *structure learning -> GNN -> fix*;
  our contribution is to apply+verify it on TelecomTS with the honest
  leakage controls documented in this repo.

## 3. Temporal GNNs for telecom/5G

- **Simba** (GNN + Transformer for 5G RAN KPI root-cause localization,
  arXiv:2406.15638). Directly related: 5G RAN, KPIs, GNN+attention, RCA.
  TelecomTS (this dataset) was in part introduced to make such studies
  reproducible at scale.

## 4. Statistical / causal baselines

- **Granger causality** (1969) and **transfer entropy** (Schreiber, 2000)
  are the classic pairwise directed-dependence measures we use for edge
  direction in dynamic graphs.
- **Graphical Lasso** (Friedman et al., 2008) for sparse inverse-covariance
  (partial-correlation) skeleton estimation.
- **PC algorithm** (Spirtes, Glymour & Scheines, 1993) — conditional-
  independence based skeleton discovery; noted but not adopted (dataset has
  no verified ground-truth DAG; we report correlation-level graphs).

## 5. Not used as evidence (flagged)

The following were found only on mirror/aggregator sites; we could not verify
the primary venue and so do NOT cite them as authoritative: "TelecomAudit"
(CIKM 2026 claim), "KAC (Knowledge-Augmented Causal discovery, ICDM 2026
claim)". They are recorded here so the reader knows we looked.

## 6. Positioning

To the best of our knowledge, no public 5G RAN anomaly/RCA benchmark ships a
directed *dynamic per-chunk KPI graph* that downstream temporal-GNN models
can consume as-is; and — importantly for reproducibility — most prior GNN
RCA studies do not report the recording-memorization shortcut we expose
(`docs/data_leakage.md`, exp_001 matching-prior evidence). We state novelty
only in these hedged terms; see `docs/novelty_audit.md`.