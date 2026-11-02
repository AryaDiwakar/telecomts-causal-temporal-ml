# Data Acquisition Plan (Track 1 + Track 2)

Purpose: define exactly where additional data comes from, how it is normalized
onto the existing pipeline, and what counts as success. This plan exists to
support one outcome: **TelecomTS F1 > 0.960 (KAC, ICDM 2026) under a documented
comparable protocol, plus a positive SpotLight external result.**

Two tracks are in scope:

- **Track 1 — External public datasets** (cross-vendor transfer and external validation).
- **Track 2 — Synthetic injection** (non-template KPI-level ground truth and new held-out groups).

Track 3 (own SDR testbed collection) is deliberately out of scope here and is
recorded only as a pointer at the end.

---

## 0. Why more data is needed

The audited chronological split already clears the minimum detection gate
(hybrid F1 0.956, PR-AUC 0.9774, ROC-AUC 0.9926), but:

- **Generalization fails** — LORO PR-AUC 0.606 / ROC 0.893.
- **Type macro-F1 is weak** — 0.310 (below XGB 0.347).
- **Input-driven RCA is weak** — 0.287, because the only KPI ground truth is the
  per-type template.
- **The SOTA bar moved** — KAC reports TelecomTS F1 0.960 and SpotLight F1 0.950.

More *independent recordings* (not more overlapping windows) is the binding
constraint. Both tracks below add independent units.

---

## 1. Track 1 — External public datasets

### 1.1 Source inventory

| # | Source | Access | Contents | Labels | Value | License / redistribution |
|---|---|---|---|---|---|---|
| 1 | **SpotLight** | `github.com/netsys-edinburgh/SpotLight` → Google Drive | Open RAN radio + platform + network KPIs, 1/5/7 UE, enterprise indoor 5G testbed (Microsoft Cambridge), ~100M+ points | anomaly class (platform/network/radio/mixed) | **Primary** — same external set KAC uses | Public dataset; cite MobiCom 2024 |
| 2 | ColO-RAN | `github.com/wineslab/colosseum-oran-coloran-dataset`; OpenRAN Gym | Large-scale O-RAN emulator (Colosseum), 7 BS / 42 UEs, slicing + scheduling KPIs | mostly control-task, weak fault labels | Secondary (transfer robustness) | Public; cite IEEE TMC 2023 |
| 3 | Cross-Layer O-RAN | arXiv:2606.22450, public code | Live 5G O-RAN CU flow records + DU radio telemetry, run-disjoint | intrusion/run labels | Secondary (leakage-robust transfer) | Public code; check data terms |
| 4 | O-RAN SC dataset | O-RAN Software Community GitHub | 10k KPI reports, 20 UEs, mobility/handover | weak/derived | Tertiary | Public |
| 5 | O-RAN OOM | IEEE DataPort `10.21227/mngc-dp65` | Control-plane logs + out-of-order-message annotations | OOM events | Tertiary | IEEE DataPort terms |
| 6 | Simba / Simu5G | Simu5G simulator; Simba repo | Synthetic 5G RAN fault KPIs (antenna/interference/handover) | fault types | Tertiary | Open source |

Only SpotLight is a full build in this pass. Items 2–6 are **contract stubs**:
recorded here with access, schema, and mapping notes, but not ingested yet.

### 1.2 SpotLight ingestion workflow

```
download ─► data/external/SpotLight/raw/
   └─ schema audit vs the TelecomTS 16-KPI space
        └─ configs/kpi_crosswalk.yaml   (channel mapping + explicit gaps)
             └─ src/data/adapters/spotlight.py   → emits chunked.jsonl
                  └─ configs/data_spotlight.yaml  (128-step window, robust-zscore,
                       60/20/20 grouped-by-run, leakage-safe)
                       └─ src/data/loader.py:35 auto-discovers (RAW.rglob)
                            └─ src/preprocessing/prepare.py
                                 └─ eval: (a) zero-shot TelecomTS→SpotLight
                                          (b) fine-tune-on-SpotLight
                                          (c) F1 + PR-AUC/ROC-AUC, 5 seeds
                                               └─ compare vs KAC SpotLight F1 0.950
```

Concrete artifacts to add:

- `data/external/SpotLight/raw/` — downloaded, hashed, provenance recorded.
- `configs/kpi_crosswalk.yaml` — TelecomTS ↔ SpotLight channel mapping.
- `src/data/adapters/spotlight.py` — reads SpotLight schema/labels, writes
  records in the same JSON shape as TelecomTS `chunked.jsonl` so
  `src/data/loader.py:35` discovers it with no loader change.
- `configs/data_spotlight.yaml` — window/split/normalization config.
- `data/processed/spotlight/` — loader/prepare outputs.

### 1.3 KPI harmonization (the hard part)

SpotLight channels (radio / platform / network) do not equal the TelecomTS 16
KPIs. Three options, with a recommendation:

1. **Common-subset transfer** — keep only semantically shared KPIs; safest but
   thin.
2. **Shared encoder + per-dataset input projection (recommended)** — a small
   per-dataset linear projection maps each vendor's channels into a shared
   latent space; the encoder and heads are shared. Enables zero-shot transfer.
3. **Per-dataset retrain** — train separately on SpotLight; the fair fine-tune
   baseline against which zero-shot is judged.

The crosswalk must be explicit: for every SpotLight channel, either map it to a
TelecomTS KPI, map it to a **semantic role** (see `docs/novelty_plan.md` C3), or
mark it `unmapped`. Unmapped channels are dropped, and the drop is reported.

### 1.4 Window / time alignment

- SpotLight window ~6.4 s vs TelecomTS 12.8 s (128 steps @ 10 Hz).
- Resample to a common 128-step window before adapter output.
- Record the resampling method in `configs/data_spotlight.yaml`.

### 1.5 Acceptance criteria (Track 1)

1. SpotLight fully ingested; `loader` + `prepare` run clean on it.
2. A documented **zero-shot** result and a **fine-tune** result, both with
   5-seed confidence intervals (42/123/456/789/1337).
3. A positive external signal over the naive baseline; reported honestly against
   KAC's 0.950 and **never** mixed into the same column as TelecomTS F1.

---

## 2. Track 2 — Synthetic injection

Track 2 is load-bearing: it is the only source of **KPI-level ground truth that
is not the per-type template**, which the template-free RCA contribution
requires.

### 2.1 Method

`src/data/synthetic_inject.py` re-implements the TelecomTS synthetic anomaly
pipeline from arXiv:2510.06063 Appendix C. Important honesty note: **the paper
publishes the 8 function classes and the injection procedure but not the
numeric parameter ranges** (GPT-4.1 proposed them and they are not printed).
We therefore:

1. **Recover the per-type affected-KPI templates from the shipped dataset
   itself** — each of the 10 `configs/synthetic.yaml` type definitions uses the
   exact `affected_kpis` list present in the shipped
   `anomalous/synthetic/**/processed/chunked.jsonl`.
2. **Publish our own parameter ranges** (physical/symptom-motivated) in
   `configs/synthetic.yaml`, uniformly sampled per event exactly as the paper's
   procedure, with soft bounds + saturation noise (Appendix C) applied per KPI.
3. **Reuse the paper's windowing** — length 128, stride 32, keep only
   single-type anomalous windows.

Each generated anomalous recording is written as
`.../processed/chunked.jsonl` with the same fields the loader expects, so it is
auto-discovered by `src/data/loader.py:35`.

### 2.2 Generation policy (anti-circularity)

- Inject into the **21 normal recordings** under `data/raw/TelecomTS/normal/`.
- Use **parameter settings and seeds disjoint** from the original injected
  anomalies.
- Write to `data/raw/TelecomTS/anomalous/synthetic_gen/...` to keep generated
  data separate from the source dataset.

### 2.3 Validation gate (mandatory before use)

Bit-level reproduction of the original synthetic anomalies is **not possible**
(the paper does not publish the ranges), so the gate validates semantics. Before
any generated data is used, `scripts/generate_synthetic.py` must pass, per
recording and in aggregate:

1. **Direction consistency** — for every affected KPI in every kept window, the
   transformed value deviates from the pre-injection baseline in the direction
   implied by the sampled parameters (accuracy recorded below; windows whose
   base is all-zero or already saturated are reported as `unobservable`, not
   failures).
2. **Boundary / integrity** — all KPI values within the config soft bounds; zero
   NaN/Inf.
3. **Schema parity** — written chunks are auto-discovered by `src/data/loader.py`
   and mirror the shipped record shape.
4. **Single-type windows only** — all-normal and multi-type windows are removed
   (paper rule); the count is reported.

Current status (seed 2026, 12 recordings): mean direction accuracy **1.000**,
bounds respected in all, NaN/Inf = 0, multi-type windows dropped and reported.

### 2.4 Integration

- `configs/synthetic.yaml` — recording count, amplitude/onset ranges, seeds.
- `scripts/` runner — emits new recordings.
- `src/preprocessing/prepare.py` / `src/preprocessing/prepare_loro.py` — admit
  generated recordings as new hold-out groups for LORO and cluster statistics.

### 2.5 Acceptance criteria (Track 2)

1. Injector passes the §2.3 gate (semantic, not bit-exact reproduction).
2. Generated recordings are parameter-disjoint and seed-disjoint from originals.
3. New groups improve LORO PR-AUC without introducing leakage.
4. KPI-level ground truth from injection is available for template-free RCA.

---

## 3. Comparability & reporting rules

- Report F1 **and** PR-AUC/ROC-AUC; val-tuned **and** max thresholds.
- Cluster-bootstrap confidence intervals over burst groups (see
  `docs/novelty_plan.md` C2).
- Every external number carries a comparability flag:
  `same-task-same-dataset`, `related-metric`, or `not-comparable`.
- Never place TelecomTS and SpotLight F1 in the same unlabeled column.

---

## 4. Licensing / redistribution audit

- **SpotLight** — public; cite MobiCom 2024. Safe to redistribute pointers, not
  necessarily raw copies; verify dataset terms.
- **ColO-RAN / O-RAN SC / Simba** — public; check per-repo license.
- **O-RAN OOM** — IEEE DataPort terms.
- KAC's `ProdTrace-SA` (Nokia production) is **not redistributable** — do not
  depend on it.
- Generated synthetic data is ours to redistribute.

---

## 5. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Cross-vendor KPI mismatch blocks zero-shot | C3 transfer fails | Shared encoder + per-dataset projection; per-dataset retrain fallback |
| Synthetic-on-synthetic circularity | C1 ground truth invalid | Disjoint parameters; validate against the real jammer anomaly |
| KAC "balanced" TelecomTS split ≠ ours | Unfair SOTA comparison | Document protocol; report both raw and balanced where possible |
| SpotLight window/label granularity differs | Misaligned eval | Explicit resampling + label mapping in the adapter |
| Redistribution limits | Cannot ship data | Ship adapters + instructions, not raw data |

---

## 6. Out of scope (deferred)

- **Track 3 — own SDR testbed collection.** Local SDR testbed (srsRAN/OAI +
  Open5GS + USRP) fault-injection spec is deferred; it will be the strongest
  independent-data source when executed.
- Literature review / related-work comparison is a separate deliverable.

---

## 7. Immediate next actions (when implementation starts)

1. Download SpotLight; record provenance and hash.
2. Write `configs/kpi_crosswalk.yaml` from the schema audit.
3. Write `src/data/adapters/spotlight.py` and `configs/data_spotlight.yaml`.
4. Run `loader` + `prepare` on SpotLight; confirm clean outputs.
5. Implement `src/data/synthetic_inject.py` + validation gate.
6. Re-run LORO and 5-seed evaluation; feed results into the novelty experiments.
