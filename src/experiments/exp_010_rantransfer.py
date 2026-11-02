"""exp_010: RANTransfer - TelecomTS -> SpotLight cross-vendor evaluation.

Setup
-----
* Shared K=10 observation space (configs/kpi_crosswalk.yaml, direct+proxy only).
* Each window -> per-channel stats via ``chunk_summary`` (mean/std/p10..p90/
  slope/energy/zero-frac/iqr), standardised with statistics fit on the
  TelecomTS *train* split only (cross-vendor honest normalisation).
* SpotLight windows/labels come from src.data.adapters.spotlight (475 windows,
  16 runs; weak labels self-derived from documented injection KPIs).

Detectors
---------
1. naive_z       per-run robust z of the 10 channels, score = mean |z|.
                 (no training; shares signal source with the labels -> lower bound)
2. gaussian_zs   zero-shot: diagonal Gaussian fit on TelecomTS train *normal*
                 windows only; -log-loss on SpotLight.
3. logreg_zs     supervised on TelecomTS train labels, NO target adaptivity.
4. logreg_transf leave-one-run-out on SpotLight: train logreg on TelecomTS +
                 the other 15 runs, score the held-out run (pooled).

Every detector is evaluated threshold-free (ROC-AUC, PR-AUC, F1_max,
TPR@FPR 1%/5%) with the shared ``src.evaluation.metrics``. Results land in
outputs/results/exp_010_rantransfer.json with explicit comparability flags
(weak self-derived labels, T=128 vs KAC T=64, K=10 shared space).

Usage: python -m src.experiments.exp_010_rantransfer [--seeds 42]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import summarize  # noqa: E402
from src.experiments.common import load_splits  # noqa: E402
from src.models.tagn import chunk_summary  # noqa: E402

SPOT_DIR = ROOT / "data" / "processed" / "spotlight"
OUT = ROOT / "outputs" / "results" / "exp_010_rantransfer.json"
CROSSWALK = yaml.safe_load((ROOT / "configs" / "kpi_crosswalk.yaml").read_text())


def build_channel_to_kpi(feature_cols: list[str]) -> list[str]:
    """Map shared SpotLight channels back to TelecomTS KPI names (crosswalk order)."""
    kpi_for = {}
    for kpi, m in CROSSWALK["telecomts_to_spotlight"].items():
        if m["quality"] == "gap":
            continue
        kpi_for.setdefault(m["channel"], kpi)
    kpis = [kpi_for[c] for c in feature_cols]
    return kpis


def load_telecomts(kpis: list[str], target_dim: int):
    frames, _, _ = load_splits()
    tj = json.load((ROOT / "data" / "interim" / "kpi_names.json").open())
    numeric = tj["numeric_kpis"]
    keep = [numeric.index(k) for k in kpis]
    Xs = {}
    ys = {}
    for name in ("train", "test"):
        X = frames[name]["X"][:, :, keep]
        with torch.no_grad():
            Xs[name] = chunk_summary(X, target_dim=target_dim).cpu().numpy().astype(np.float64)
        ys[name] = frames[name]["ano"].cpu().numpy().astype(int)
    return Xs, ys


def load_spotlight():
    d = np.load(SPOT_DIR / "spotlight_windows.npz", allow_pickle=True)
    X = torch.tensor(d["X"], dtype=torch.float32)
    target_dim = 13 * len(d["feature_cols"])
    with torch.no_grad():
        S = chunk_summary(X, target_dim=target_dim).cpu().numpy().astype(np.float64)
    meta = pd.read_parquet(SPOT_DIR / "spotlight_meta.parquet").set_index("window_id")
    return (
        S,
        d["y"].astype(int),
        np.asarray([r for r in d["run_ids"]]),
        list(d["feature_cols"]),
        meta,
    )


def fit_scaler(train_stats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = train_stats.mean(0)
    sd = train_stats.std(0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return mu, sd


def score_baseline_z(S: np.ndarray, runs: np.ndarray) -> np.ndarray:
    """Per-run robust z (MAD) of summary stats, mean abs z across the window."""
    out = np.zeros(len(S))
    for r in np.unique(runs):
        i = np.where(runs == r)[0]
        med = np.median(S[i], axis=0)
        mad = np.median(np.abs(S[i] - med), axis=0)
        sd = 1.4826 * np.where(mad <= 0, 1.0, mad)
        out[i] = np.abs((S[i] - med) / sd).mean(axis=1)
    return out


def gaussian_scores(S, mu, sd, logdet):
    z = (S - mu) / sd
    return -(0.5 * ((z * z).sum(1) + logdet))


def logreg_scores(clf: LogisticRegression, S) -> np.ndarray:
    return clf.decision_function(S)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    args = ap.parse_args()

    Ss, ys, runs, feature_cols, meta = load_spotlight()
    kpis = build_channel_to_kpi(feature_cols)
    print("shared channels:", list(zip(feature_cols, kpis)))
    target_dim = 13 * len(feature_cols)
    Xt, yt = load_telecomts(kpis, target_dim)
    print(f"TelecomTS train {Xt['train'].shape} test-meta n={len(Xt['test'])}; "
          f"SpotLight {Ss.shape} pos={ys.sum()}")

    norm_mu, norm_sd = fit_scaler(Xt["train"])
    S_n = (Ss - norm_mu) / norm_sd
    Tr_n = (Xt["train"] - norm_mu) / norm_sd

    results: dict = {
        "analysis": "RANTransfer TelecomTS->SpotLight (cross-vendor, K=10 shared space)",
        "config_hash": CROSSWALK["config_sha256"] if "config_sha256" in CROSSWALK else None,
        "comparability": {
            "spotlight_labels": "weak self-derived robust-z (no public per-step labels); see configs/kpi_crosswalk.yaml",
            "window_t_s": 128,
            "kac_window_t_s": 64,
            "shared_k": 10,
            "telecomts_train_windows": int(len(Tr_n)),
            "spotlight_windows": int(len(Ss)),
            "spotlight_runs": int(len(np.unique(runs))),
            "positive_rate": float(ys.mean()),
        },
        "per_run": [],
        "models": {},
    }
    results["comparability"]["config_sha256"] = hashlib.sha256(
        (ROOT / "configs" / "kpi_crosswalk.yaml").read_bytes()
    ).hexdigest()
    results["comparability"]["crosswalk_quality_counts"] = {
        q: sum(1 for m in CROSSWALK["telecomts_to_spotlight"].values() if m["quality"] == q)
        for q in ("direct", "proxy", "gap")
    }

    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        scores: dict[str, np.ndarray] = {}
        labels = ys

        # 1) naive per-run robust z baseline
        scores["naive_z"] = score_baseline_z(S_n, runs)

        # 2) zero-shot gaussian (TelecomTS normal windows only)
        tr_normal = Tr_n[yt["train"] == 0]
        mu = tr_normal.mean(0)
        sd = tr_normal.std(0)
        sd = np.where(sd < 1e-9, 1.0, sd)
        logdet = np.log(sd).sum()
        scores["gaussian_zs"] = gaussian_scores(S_n, mu, sd, logdet)

        # 3) LogisticRegression trained on TelecomTS train labels
        clf_zs = LogisticRegression(max_iter=4000, C=1.0)
        clf_zs.fit(Tr_n, yt["train"])
        scores["logreg_zs"] = logreg_scores(clf_zs, S_n)

        # 4) leave-one-run-out transfer (TelecomTS + other 15 runs per fold)
        run_ids = np.unique(runs)
        loo = np.zeros(len(Ss))
        for hold in run_ids:
            i = np.where(runs != hold)[0]
            clf = LogisticRegression(max_iter=4000, C=1.0)
            clf.fit(np.concatenate([Tr_n, S_n[i]]), np.concatenate([yt["train"], ys[i]]))
            loo[np.where(runs == hold)[0]] = clf.decision_function(S_n[runs == hold])
        scores["logreg_transfer_loo"] = loo

        for name, sc in scores.items():
            m = summarize(labels, sc)
            results["models"][name] = m
            results["models"][name]["seed"] = seed
            print(f"  {name:<22} ROC {m['roc_auc']:.4f} PR {m['pr_auc']:.4f} "
                  f"F1max {m['f1_max']:.4f} TPR@1% {m['tpr_at_fpr_0.01']:.3f}")

    # per-run breakdown for the strongest transfer detector
    for r in np.unique(runs):
        i = np.where(runs == r)[0]
        row = {"run": r, "class": meta.loc[i[0], "cls"], "n": int(len(i)),
               "pos": int(ys[i].sum())}
        for name, sc in scores.items():
            row[name] = summarize(ys[i], sc[i])
        results["per_run"].append(row)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, sort_keys=True))
    print("saved ->", OUT)


if __name__ == "__main__":
    main()