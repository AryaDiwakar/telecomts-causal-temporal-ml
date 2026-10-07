# Research Positioning

## What is already established

The model ingredients are not individually novel. GDN (AAAI 2021) established
attention-based graph learning for multivariate anomaly detection; RCD
(NeurIPS 2022), CIRCA (KDD 2022), REASON (KDD 2023), and RUN (AAAI 2024)
established causal or uncertain graph reasoning for diagnosis; MULAN (WWW
2024) established multimodal causal structure learning for RCA; and Simba
(arXiv:2406.15638) is directly close in 5G-RAN KPI localization.

Therefore this project should not be presented as inventing temporal GNNs,
transfer entropy, or causal RCA.

## Defensible contribution package

| Component | Prior art overlap | Defensible contribution here |
|---|---|---|
| Dynamic KPI graph | Graph learning and causal graph RCA are established | An auditable TelecomTS protocol that makes the graph construction, split fit, prototype assignment, and graph artifacts reproducible |
| Temporal-GNN | Temporal GNNs and GNN-RCA are established | A benchmarked integration of temporal context, graph prototypes, KPI localization, and hazard prediction |
| Leakage audit | Usually underreported in benchmark demonstrations | Explicit recording-prior, template-overlap, burst-cluster, and cross-recording audits |
| Generalization | Standard group/leave-one-group-out idea | A concrete recording-held-out protocol for TelecomTS, with five-fold results |
| Causal interpretation | Not justified by observational KPI data alone | Predictive graph validation and explicit non-causal language rather than unsupported DAG claims |

## Acceptance bar

The paper should claim a useful evaluation protocol and empirical findings, not
state-of-the-art architecture novelty, unless all of the following hold:

1. Five-seed TAGN confidence intervals are reported.
2. Full-budget ablations are reported, not one-epoch screening runs.
3. LORO results show whether performance survives recording shift.
4. TAGN is compared with strong classical and temporal baselines under the
   same split protocol.
5. Graph stability and held-out directional predictivity are reported, with
   the explicit caveat that neither proves causal identification.
6. Results are reported against recording-prior and template-overlap floors.

If TAGN does not beat the classical baselines or LORO is weak, the honest
paper contribution is still valuable: it demonstrates that apparent RCA
performance on TelecomTS can be explained by recording/template shortcuts and
that a more complex graph model does not automatically solve them.
