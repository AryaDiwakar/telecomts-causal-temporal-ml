# RANTransfer: TelecomTS → SpotLight cross-vendor evaluation (Track 1)

Status: detection results finalized (seed 42); the integrated TelecomTS 5-seed
CI is complete and reported in `docs/integrated_results_ci.md`.

## Objective
Demonstrate that a method built on the TelecomTS 5G observability corpus
transfers to a *second, independently collected* 5G Open RAN testbed
(SpotLight, MobiCom 2024) using only a shared KPI space, with no per-step
labels from the target vendor — evidence for cross-vendor generality behind
novelty claims C1–C3.

## Data
- TelecomTS: 32,714 windows (128 steps @10 Hz) after Track 2 injection fold-in.
- SpotLight: public Drive tree (`Anomaly/…`, 16 anomalous runs), `radio.csv`
  (~6.7 Hz, 208 cols) + `platform.csv` (~3 Hz, 249 cols). Baselines (1UE/5UE/
  7UE) are not used for detection; they are reserved for future normal-load
  sensitivity checks.
- **No per-step labels are published.** The authors' merged-label release used
  by external evaluations is offline (404). Labels here are **self-derived**
  per the injection mechanism documented in the paper and README, from the
  KPI(s) directly stressed:

| class  | injected mechanism            | label monitors |
|--------|-------------------------------|----------------|
| PDCP   | `stress-ng` on `pdcp_worker_0` (CU contention) | pdcp_worker, f1_cu_worker (platform max) |
| MAC    | busy-loop on `TxCtrl_DU1_C0` (scheduler cont.) | txctrl, txctrl_runtime, f1_du_worker |
| RADIO  | USRP interference on carrier   | sinr, crc_loss, nack_rate, mcs (radio) |
| NETWORK| FH link contention             | fh_bytes, sinr, crc_loss |
| MIXED  | any combination                | union (PDCP/MAC/RADIO monitors) |

Step geometry: 1 s resampling (radio mean, platform max); a step deviates if it
belongs to a contiguous segment of length ≥ 1 s with |robust z| ≥ 5 against the
run's own median+MAD (floor 1.0); a 128 s non-overlapping window is positive if
≥ 2% of steps deviate. This matches the paper's ~1 s burst / ~10 s cadence.

## Shared observation space (K=10)
TelecomTS KPIs are mapped to SpotLight channels (configs/kpi_crosswalk.yaml):
9 direct + 4 proxy + 3 gap. Two channels (`ul_fapi_*_count`) are constant on
SpotLight and are dropped from the shared space at build time → **K=10**
(8 direct, 2 proxy). Windows are summarized per channel (mean/std/p10/p50/p90/
min/max/slope/energy/zero-frac/last/Δ) exactly as the TelecomTS classical
baseline, then standardized on TelecomTS *train* statistics only.

Adapter + provenance: `src/data/adapters/spotlight.py` →
`data/processed/spotlight/` (475 windows, 16 runs, positive rate 38.9%,
per-run `summary.json` + `provenance.json` with input sha256).

## Detectors (seed 42, threshold-free)
| detector | ROC-AUC | PR-AUC | F1-max | TPR@FPR1% |
|---|---|---|---|---|
| naive z (per-run robust z, no training) | 0.925 | 0.837 | 0.893 | 0.124 |
| Gaussian zero-shot (TelecomTS normal only) | 0.281 | 0.333 | 0.561 | 0.065 |
| LogisticRegression, TelecomTS train only | 0.705 | 0.623 | 0.754 | 0.103 |
| LogisticRegression + LOO target adaptation | **0.944** | **0.886** | **0.919** | 0.103 |

Results: `outputs/results/exp_010_rantransfer.json` (per-run breakdown inside).

## vs KAC SpotLight (F1 0.950 reported)
Best transfer configuration reaches **F1 0.919 / PR-AUC 0.886 / ROC 0.944**
(run-grouped leave-one-run-out with in-vendor weak labels). This is close to —
but does not yet clear — the KAC-reported SpotLight F1 of 0.950. Comparative
reading is **not apples-to-apples**; see flags below.

## Comparability flags (mandatory when quoting these numbers)
1. **Weak labels.** KAC experiments use the paper's (unpublished) per-step
   labels; ours are self-derived from the documented injection KPIs. A detector
   that reads those same KPIs shares its source signal with the label function,
   so `naive z` is an *upper-bound-ish control* rather than a clean baseline.
2. **Windowing.** T=128 s non-overlap (TelecomTS budget) vs KAC's T=64.
   Non-overlap halves label density vs KAC's any-exposure windowing.
3. **Shared K=10, not full SpotLight schema.** Only cross-vendor channels. The
   reference pipeline was never shown these exact channels; a direct rerun of
   the reference model class on SpotLight is therefore not reproduced here.
4. **Zero-shot failure is informative.** The Gaussian zero-shot underperforms
   chance (ROC 0.281): TelecomTS-only scale statistics do not transfer to a
   different vendor's absolute magnitudes (BHTX bytes, MCS, etc.). This mirrors
   the paper's "scale matters" finding and argues for per-vendor normalization
   (which the LOO adapter gets from in-vendor windows).
5. **Sparse classes.** MAC/MAC2's anomalous interval is genuinely short (~last
   350 s of a ~3,300 s run) → few positive windows (2/25). Disclosed per class.

## Reproduce
```
python3 -m src.data.adapters.spotlight        # rebuild SpotLight windows + labels
python3 -m src.experiments.exp_010_rantransfer --seeds 42
python3 -m pytest tests -q                    # regression suite
```

## References (BibTeX)
```bibtex
@inproceedings{sun2024spotlight,
  title     = {SpotLight: Accurate, Explainable and Efficient Anomaly Detection for Open {RAN}},
  author    = {Sun, Chuanhao and Pawar, Ujjwal and Khoja, Molham and
               Foukas, Xenofon and Marina, Mahesh K. and Radunovic, Bozidar},
  booktitle = {Proc. 30th Annual Int. Conf. on Mobile Computing and Networking (MobiCom)},
  pages     = {923--937},
  year      = {2024},
  publisher = {ACM},
  doi       = {10.1145/3636534.364938}
}

@misc{feng2025telecomts,
  title         = {Telecom{TS}: A Multi-Modal Observability Dataset for Time Series and Language Analysis},
  author        = {Feng, Austin and Varvarigos, Andreas and Panitsas, Ioannis and Fernandez, Daniela and
                   Wei, Jinbiao and Guo, Yuwei and Chen, Jialin and Maatouk, Ali and Tassiulas, Leandros and Ying, Rex},
  year          = {2025},
  eprint        = {2510.06063},
  archiveprefix = {arXiv},
  primaryclass  = {cs.AI},
  url           = {https://arxiv.org/abs/2510.06063}
}
```