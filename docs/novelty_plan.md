# Novelty Plan

Working title: **Type–Template Shortcuts in KPI-Level Root-Cause Benchmarks:
Audit, Corrected Benchmark, and Deferral-Aware RCA for 5G RAN.**

## Abstract (draft)

KPI-level root-cause analysis (RCA) on 5G RAN telemetry is evaluated by scoring a
predicted set of affected KPIs against a ground-truth set. We show that in the
widely used TelecomTS benchmark this ground truth is a **constant template per
fault type**: across 956 shipped anomalous windows and 10 types there is exactly
one affected-KPI set per type, so RCA reduces to a fault-type lookup. An
**input-free "type-oracle"** that emits a type's template reaches **0.971** top-k
Jaccard on these windows — above every trained model, whose reported RCA (0.325)
is thus a weak echo of the shortcut. We (i) quantify this ceiling, (ii) release a
**corrected benchmark** in which Track 2 synthetic injection produces
instance-varying affected-KPI sets per event and the split is stratified so these
windows reach validation and test (type-oracle ceiling drops to 0.661), and
(iii) report detection and RCA **separately on template vs instance-varying
windows**. On the corrected split, detection drops from ROC 0.988/PR 0.963
(template) to ROC 0.920/PR 0.855 (instance-varying), and the acceptance gate that
the shipped benchmark appears to pass is no longer met — the prior headline was
inflated by the shortcut and a split bug. We further test a natural fix — a
gradient-reversal type-adversary intended to make RCA type-invariant — and show
it **fails in a mechanistically informative way**: attribution quality is
inherently type-predictive, so the adversary raises type leakage (TCRL 0.623→0.855)
rather than lowering it. Finally, we show attribution-stability-based **deferral**
(precision-at-coverage) improves selective RCA, and that a lightweight model
transfers cross-vendor to SpotLight (F1 0.919).

This document defines the paper's core contributions, the mechanisms that
realize them, the new metrics, and the evaluation design. It is the companion to
`docs/data_acquisition_plan.md`; the data tracks exist to enable the
experiments here.

The contribution is **benchmark audit + corrected benchmark + honest evaluation**
(plus a clean negative result and a deferral result), not "we built a GAT".
Architecture components (temporal convolution, graph attention, GRU) are treated
as prior art.

---

## 0. Problem statement

5G RAN observability data (TelecomTS, SpotLight) is zero-inflated, bursty, and
recorded in long scenario-contaminated recordings. Current practice reports
anomaly detection and root-cause analysis (RCA) scores that are inflated by two
artifacts:

1. **Pseudo-replication** — overlapping windows are treated as independent
   samples, so confidence intervals are optimistic.
2. **Template shortcut** — KPI-level RCA ground truth is a constant template per
   anomaly type, so "RCA" largely reduces to fault-type lookup.

The field has measured these problems (Cross-Layer O-RAN, arXiv:2606.22450;
GNN-TSAD critical evaluation, arXiv:2603.09675) but has not removed them. This
paper removes them and reports what remains.

---

## 1. Contributions

### C1 — Template-free, deferral-capable RCA (method)

> **Status (exp_012, seed 42): NEGATIVE on type-invariance — confirmed on
> template (v1), instance-varying-in-train (v2), and the corrected split with
> instance-varying anomalies in test (v3).** GRL did not reduce type leakage
> (v1 0.654 → 0.736; v2 0.666 → 0.711; v3 0.623 → 0.855) because a well-localized
> attribution is inherently type-predictive; an adversary on the attribution
> logits fights the supervised RCA objective and loses. Deferral helps the
> baseline (v3 precision-at-coverage RCA 0.291 → 0.33 at 10%; v2 0.295 → 0.477).
> See `docs/exp_012_typeinvariant_rca.md`.
>
> **Consequence for the paper.** C1's *mechanism* failed, but the underlying
> measurement problem is real and now quantified: on shipped TelecomTS data the
> KPI-set ground truth is a constant template per type, so an **input-free
> type-oracle** reaches 0.971 top-k Jaccard — above any trained model. The
> headline contribution therefore shifts to the **audit + corrected benchmark +
> honest (subset-split) evaluation** (`docs/benchmark_shortcut_audit.md`,
> `outputs/results/shortcut_audit.json`), with deferral kept as a positive
> secondary result.

**Claim.** A diagnosis head that cannot recover the KPI set from the predicted
type, and that abstains when RCA is not identifiable in the input.

**Mechanism.**
- Type-independence constraint on the RCA representation:
  - Primary: **gradient-reversal adversary** — a type classifier is trained on
    RCA features through a gradient-reversal layer, forcing the features to be
    type-invariant.
  - Ablation: **conditional mutual-information penalty** between RCA features
    and predicted type.
- **Deferral / selective abstention**: an identifiability score is computed from
  attribution stability under input perturbation; low-stability inputs return
  "no reliable root cause" instead of a template.

**Why it is novel.**
- CAAD (KDD 2026), GCAD (AAAI 2025), AERCA (ICLR 2025), and KAC (ICDM 2026) all
  attribute root causes, but every one is scored against the type-derived
  template. None removes the shortcut or formalizes abstention.
- Deferral for diagnosis (as opposed to classification confidence) is not
  established in RAN observability.

**Enabling data.** Track 2 injection provides non-template KPI ground truth;
without it C1 collapses back to template Jaccard.

### C2 — Burst-cluster statistics (measurement rigor)

**Claim.** In bursty overlapping-window data, the burst cluster — not the window
— is the independent unit; ignoring this systematically overstates performance.

**Mechanism.**
- Cluster-aware training loss weighting by burst group.
- **Cluster-bootstrap confidence intervals** over burst groups.
- A correction study quantifying how much reported 5G AD/RCA scores drop when
  pseudo-replication is removed.

**Why it is novel.** Pseudo-replication is unaddressed in the 5G observability
AD/RCA literature; existing work either aggregates at recording level or uses
naive chunk-level i.i.d. assumptions (see `docs/limitations.md` §3).

### C3 — KPI-role abstraction (cross-vendor transfer)

**Claim.** Heterogeneous KPI channels from different vendors can be projected
into a shared **semantic role space**, enabling zero-shot transfer between RAN
observability datasets.

**Mechanism.**
- Define roles, e.g.: `error`, `load`, `quality`, `resource`, `mobility`.
- Map each dataset's channels to roles (partly automated, partly manual),
  producing a role matrix per window.
- Share the encoder and heads across datasets via a per-dataset input projection
  (see `docs/data_acquisition_plan.md` §1.3).
- Evaluate **zero-shot TelecomTS→SpotLight** and per-dataset fine-tune.

**Why it is novel.** No cross-vendor KPI-role transfer for RAN diagnosis exists;
scale-aware foundation-model work does not abstract KPI semantics.

### Supporting (non-headline)

- Shortcut/leakage audit framework (recording-prior, template-overlap, burst,
  cross-recording).
- Multi-task TAGN (detection, type, RCA, early warning) as the vehicle.
- Interactive Streamlit demonstration.

---

## 2. New metrics

| Metric | Definition | Direction |
|---|---|---|
| **Type-Conditional RCA Leakage (TCRL)** | How much of the RCA score is explained by predicted type alone; e.g. the R² of predicting the RCA score from the type label (or mutual information between RCA output and type). | Lower is better |
| **RCA Identifiability (RCAI)** | Stability of the KPI attribution under input perturbations; e.g. mean rank correlation between attributions of a window and its perturbed copies. | Higher is better |
| **Coverage–Risk curve** | Accuracy of RCA vs fraction of windows where the model does not defer. | Reported as a curve, not a point |
| **Cluster-corrected F1 / PR-AUC** | Point metrics recomputed with cluster-bootstrap CIs over burst groups. | Higher + tighter CI is better |

TCRL replaces the implicit template shortcut with a number; RCAI operationalizes
deferral; the coverage–risk curve prevents deferral from being used to game the
point score.

---

## 3. Evaluation design

**Datasets.** TelecomTS (internal), SpotLight (external), generated synthetic
recordings (Track 2).

**Protocols.**
1. Chronological held-out (internal, as today).
2. LORO grouped by recording (generalization).
3. Burst-cluster split (pseudo-replication-corrected).
4. External zero-shot and fine-tune (SpotLight).

**Metrics.** F1 (val-tuned + max), PR-AUC, ROC-AUC, TPR@FPR 1/5%, type accuracy
+ macro-F1, TCRL, RCAI, coverage–risk, all with 5-seed (42/123/456/789/1337)
cluster-bootstrap CIs.

**Baselines to beat / compare.**
- Corrected logistic regression (PR-AUC 0.9523 / ROC 0.9880).
- Template-RCA baseline (type→template) as the thing C1 removes.
- KAC (TelecomTS F1 0.960; SpotLight F1 0.950) — the SOTA bar.
- A graph reconstruction baseline (e.g. GDN-style) for context.

**Ablations.**
- RCA head with vs without the type-independence constraint.
- Deferral on vs off.
- Burst-cluster weighting on vs off.
- Role abstraction on vs off (for transfer).
- Graph on/off, static/dynamic (existing exp_003 variants).

---

## 4. Positioning vs prior art

| Work | What it does | What it does not do (our gap) |
|---|---|---|
| GDN (AAAI 2021) | Learned sensor graph for MTS AD | No RCA shortcut audit, no cross-vendor transfer |
| GCAD (AAAI 2025) | Granger-causal MTS AD | RCA scored on causal graphs, not template-free; no deferral |
| AERCA (ICLR 2025) | Granger causal discovery + RCA | No burst-cluster correction, no cross-vendor roles |
| CAAD (KDD 2026) | Causality-aware MTS AD, directed causal matrix | Leans on causal graph; no template-independence or abstention |
| Simba (2024) | GNN + Transformer 5G RAN RCA | Simulator data; template-style RCA; no leakage rigor |
| SpotLight (MobiCom 2024) | Generative AD + KPI minimization in Open RAN | Minimizes KPI set but not type-independent; no deferral |
| KAC (ICDM 2026) | KPI-aware multimodal residuals on TelecomTS/SpotLight | Strong F1 but template-scored RCA; no pseudo-replication correction |
| TelecomTS (ICML 2026) | Dataset + foundation-model benchmark | Scale-aware baselines; no shortcut removal |

Our combination — template-free RCA + deferral + burst-cluster rigor +
cross-vendor roles — is not occupied by any single prior work.

---

## 5. Success criteria (what makes this a paper)

1. **Beats the SOTA bar** — TelecomTS F1 > 0.960 under a documented comparable
   protocol, plus a positive SpotLight external result.
2. **Shows the correction matters** — cluster-corrected metrics lower than naive
   metrics by a reportable margin, consistently.
3. **Shows the shortcut is real and removable** — TCRL high for template-RCA
   baselines and low for C1; input-driven RCA improves.
4. **Shows transfer** — non-trivial zero-shot TelecomTS→SpotLight over a naive
   baseline.
5. **Honest negative results are kept** — if a contribution fails, it is
   reported, not hidden.

If criteria 1–2 fail but 3–4 hold, the paper is still viable as an
evaluation/benchmark paper (contribution shifts to C2 + C3).

---

## 6. Dependencies and sequencing

```
Track 1 (SpotLight)  ──► C3 zero-shot + external benchmark
Track 2 (injection)  ──► C1 non-template ground truth + C2 new groups
        │
        └─► C1 method experiments ─► C2 correction study ─► C3 transfer
                                            │
                                            └─► SOTA comparison vs KAC
```

C2 can proceed with existing data alone; C1 requires Track 2; C3 requires
Track 1. Track 3 (SDR) is deferred and would strengthen C1/C3 later.

---

## 7. Risks

| Risk | Mitigation |
|---|---|
| Deferral games the metric | Report coverage–risk curve, never a bare point score |
| Type-independence removes useful signal | Report detection/type metrics alongside RCA to show no degradation |
| Synthetic ground truth is circular | Track 2 validation gate + real-jammer cross-check |
| Zero-shot transfer fails | Fall back to per-dataset retrain; report both |
| KAC protocol not identical | State comparability flag explicitly; avoid a single mixed table |

---

## 8. Non-claims

- Not claiming we invented temporal GNNs, graph attention, or transfer entropy.
- Not claiming the learned graphs recover a physical causal DAG.
- Not claiming universal SOTA; the claim is scoped to the documented protocol.
