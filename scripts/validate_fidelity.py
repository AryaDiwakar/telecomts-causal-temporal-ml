"""Validation of interventional RCA fidelity with explicit null controls.

The proposed metric compares the anomaly-score drop obtained by intervening on
a candidate root-KPI set against the drop obtained by intervening on a random
non-root set of the same size.  A number near 1 is only meaningful if it beats
a calibrated null, so this script reports:

  * predicted-root vs random-non-root (the proposed metric),
  * true-root vs random-non-root (positive control),
  * full true-root set vs a random set of the same size (untruncated control),
  * random-root vs random-non-root (negative control, expectation 0.5),
  * a recording-level cluster-bootstrap 95% CI and p-value,
  * sensitivity to the root-set size k,
  * per-seed and pooled-over-seeds results.

Every intervention is a model-internal do-style perturbation: the selected
normalized KPI channels are set to zero (their training mean) and the anomaly
head is re-scored.  This is *not* an identified physical intervention.

Usage:
    python scripts/validate_fidelity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from src.models.tagn import TagnModel, load_graphs, model_config

INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
CKPT = ROOT / "checkpoints" / "exp_002"
RES = ROOT / "outputs" / "results"
SEEDS = (42, 123, 456, 789, 1337)
KS = (3, 5, 7)


def load_test():
    X = np.load(INTERIM / "chunks_X.npy")
    scaler = json.loads((PROCESSED / "scaler.json").read_text())
    loc = np.asarray(scaler["loc"])[:16]
    scale = np.asarray(scaler["scale"])[:16]
    Xs = X.copy().astype(np.float32)
    Xs[:, :, :16] = (Xs[:, :, :16] - loc) / scale
    idx = np.load(PROCESSED / "split_indices.npz")["test"]
    tar = dict(np.load(PROCESSED / "targets.npz"))
    proto = np.load(PROCESSED / "graph_prototypes.npz")["chunk_prototypes"][idx]
    meta = pd.read_parquet(INTERIM / "chunks_meta.parquet").iloc[idx]
    return (
        torch.tensor(Xs[idx], dtype=torch.float32),
        torch.tensor(proto, dtype=torch.long),
        torch.tensor(tar["y_anomaly"][idx], dtype=torch.long),
        torch.tensor(tar["y_affected"][idx], dtype=torch.float32),
        meta["recording"].to_numpy(),
    )


def load_model(seed):
    cfg = model_config()
    m = TagnModel(cfg)
    m.set_prototypes(load_graphs()[0])
    state = torch.load(CKPT / f"tagn_seed{seed}.pt", map_location="cpu")
    m.load_state_dict(state, strict=True)
    m.eval()
    return m


@torch.no_grad()
def forward_scores(model, X, proto, batch=128):
    base, kpi = [], []
    for i in range(0, len(X), batch):
        lg, _, _ = model(X[i:i + batch], proto[i:i + batch])
        base.append(lg["anomaly"])
        kpi.append(lg["kpi"])
    return torch.cat(base).numpy(), torch.cat(kpi).numpy()


@torch.no_grad()
def intervene_scores(model, X, proto, sets, batch=64):
    """Anomaly score after zeroing the KPI indices in each set."""
    out = []
    for i in range(0, len(X), batch):
        xb = X[i:i + batch].clone()
        for j, cols in enumerate(sets[i:i + batch]):
            if cols:
                xb[j, :, list(cols)] = 0.0
        lg, _, _ = model(xb, proto[i:i + batch])
        out.append(lg["anomaly"])
    return torch.cat(out).numpy()


def win_outcome(drop_root, drop_other):
    """1 if root intervention helps, 0.5 on a tie, 0 otherwise."""
    return (drop_root > drop_other).astype(float) + 0.5 * (drop_root == drop_other)


def cluster_bootstrap(values, groups, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    means = []
    for _ in range(n_boot):
        pick = rng.choice(unique, size=len(unique), replace=True)
        sel = np.concatenate([np.where(groups == g)[0] for g in pick])
        if len(sel):
            means.append(float(np.mean(values[sel])))
    means = np.asarray(means)
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = 2.0 * min(np.mean(means <= 0.5), np.mean(means >= 0.5))
    return [float(lo), float(hi)], float(min(1.0, p))


def make_sets(order, affected, rng, k):
    """Candidate root sets and their matched random non-root controls."""
    pred = [order[i, :k].tolist() for i in range(len(order))]
    true = []
    for row in affected:
        cols = np.where(row > 0)[0].tolist()
        if len(cols) > k:
            cols = rng.choice(cols, size=k, replace=False).tolist()
        true.append(cols)
    rand = [rng.choice(16, size=k, replace=False).tolist() for _ in range(len(order))]
    return {"predicted": pred, "true": true, "random": rand}


def main():
    X, proto, y, affected, groups = load_test()
    anom = y.numpy() == 1
    Xa, pa, ga = X[anom], proto[anom], groups[anom]
    affected = affected[anom].numpy()
    n = len(Xa)

    # Same control draws across seeds: only the model changes.
    rng = np.random.default_rng(2026)
    per_seed = []
    pooled = {}
    for seed in SEEDS:
        rng = np.random.default_rng(2026)  # same control draws for every seed
        model = load_model(seed)
        base, kpi = forward_scores(model, Xa, pa)
        order = np.argsort(-kpi, axis=1)
        print(f"seed {seed}: scored {len(Xa)} anomalous test windows")
        for k in KS:
            sets = make_sets(order, affected, rng, k)
            scores = {}
            for name in ("predicted", "true", "random"):
                sr = intervene_scores(model, Xa, pa, sets[name])
                pool = [[c for c in range(16) if c not in s] for s in sets[name]]
                # matched non-root set, drawn from the complement
                nonroot = [rng.choice(p, size=len(s), replace=False).tolist()
                           for p, s in zip(pool, sets[name])]
                sn = intervene_scores(model, Xa, pa, nonroot)
                outcome = win_outcome(base - sr, base - sn)
                scores[name] = outcome
                pooled.setdefault((name, k), []).append(outcome)
            per_seed.append({"seed": seed, "k": k, "scores": scores})

        # Untruncated positive control: use every ground-truth root and compare
        # against a random set of the same size (which may overlap the roots).
        true_full = [np.where(row > 0)[0].tolist() for row in affected]
        rand_full = [rng.choice(16, size=len(s), replace=False).tolist()
                     for s in true_full]
        sr = intervene_scores(model, Xa, pa, true_full)
        sn = intervene_scores(model, Xa, pa, rand_full)
        pooled.setdefault(("true_full", "full"), []).append(
            win_outcome(base - sr, base - sn)
        )

    results = []
    for (name, k), outcomes in pooled.items():
        stacked = np.vstack(outcomes)  # [n_seeds, n_windows]
        mean_per_window = stacked.mean(axis=0)
        fidelity = float(mean_per_window.mean())
        ci, p = cluster_bootstrap(mean_per_window, ga)
        per_seed_means = [float(o.mean()) for o in outcomes]
        results.append({
            "control": name,
            "k": k,
            "n_windows": int(n),
            "n_seeds": len(SEEDS),
            "fidelity": fidelity,
            "ci95_cluster_bootstrap": ci,
            "p_two_sided_vs_0.5": p,
            "per_seed_fidelity": per_seed_means,
            "per_seed_sd": float(np.std(per_seed_means)),
        })

    out = {
        "checkpoints": [f"checkpoints/exp_002/tagn_seed{s}.pt" for s in SEEDS],
        "n_seeds": len(SEEDS),
        "n_anomalous_test_windows": int(n),
        "n_recordings": int(len(np.unique(ga))),
        "intervention": "set selected normalized KPI channels to zero (training mean)",
        "results": results,
    }
    RES.mkdir(parents=True, exist_ok=True)
    path = RES / "fidelity_validation.json"
    path.write_text(json.dumps(out, indent=2))
    for r in results:
        print(f"{r['control']:10s} k={r['k']} fid={r['fidelity']:.3f} "
              f"CI={r['ci95_cluster_bootstrap']} p={r['p_two_sided_vs_0.5']:.4g} "
              f"per-seed={r['per_seed_fidelity']}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
