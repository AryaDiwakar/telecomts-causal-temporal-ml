"""Audit whether the learned KPI graphs are stable and directionally useful.

This is deliberately a *predictive* validation, not a causal-identification
claim. It measures (a) within-prototype edge-mask stability and (b) whether a
stored source->target direction agrees with lag-1 cross-correlation on held-out
chunks. A positive result does not prove causality; a negative result limits
the graph's interpretation.

Usage:
    python -m src.evaluation.graph_validation
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
INTERIM = ROOT / "data" / "interim"
OUT = ROOT / "outputs"


def mask_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 1.0


def corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    graphs = np.load(PROCESSED / "graph_bins.npz", allow_pickle=True)
    adj = graphs["adjacencies"]
    uids = graphs["uids"].astype(str)
    prototypes = np.load(PROCESSED / "graph_prototypes.npz")["prototypes"]
    chunk_proto = np.load(PROCESSED / "graph_prototypes.npz")["chunk_prototypes"]
    X = np.load(INTERIM / "chunks_X.npy", mmap_mode="r")
    idx = np.load(PROCESSED / "split_indices.npz")
    names = json.load((INTERIM / "kpi_names.json").open())["numeric_kpis"]

    train_bins = np.array([u.startswith("train|") for u in uids])
    rows = []
    for p in range(len(prototypes)):
        # Prototype membership is recovered by nearest-centre assignment.
        d = ((adj[train_bins, None] - prototypes[None]) ** 2).sum(axis=(2, 3))
        own = adj[train_bins][d.argmin(axis=1) == p]
        masks = own != 0
        pair = []
        for i in range(len(masks)):
            for j in range(i):
                pair.append(mask_jaccard(masks[i], masks[j]))
        mask = prototypes[p] != 0
        directed = np.logical_and(mask, np.logical_not(mask.T)).sum()
        total = mask.sum()
        rows.append({
            "prototype": p,
            "n_train_bins": int(len(own)),
            "within_prototype_mask_jaccard": float(np.mean(pair)) if pair else None,
            "prototype_edges": int(total),
            "prototype_directed_edge_fraction": float(directed / total) if total else 0.0,
        })

    # Directional predictivity on test chunks. This is only a lagged
    # association audit, not a causal test.
    edge_rows = []
    test_ids = idx["test"][:2000]
    for p in range(len(prototypes)):
        edges = np.argwhere(prototypes[p] != 0)
        ids = test_ids[chunk_proto[test_ids] == p]
        for source, target in edges:
            forward, reverse = [], []
            for c in ids[:300]:
                z = np.asarray(X[c, :, :16], dtype=np.float64)
                forward.append(corr(z[:-1, source], z[1:, target]))
                reverse.append(corr(z[:-1, target], z[1:, source]))
            if forward:
                delta = np.asarray(forward) - np.asarray(reverse)
                edge_rows.append({
                    "prototype": p,
                    "source": names[source],
                    "target": names[target],
                    "n_chunks": len(forward),
                    "forward_lag_corr": float(np.mean(forward)),
                    "reverse_lag_corr": float(np.mean(reverse)),
                    "forward_minus_reverse": float(np.mean(delta)),
                    "forward_wins_fraction": float(np.mean(delta > 0)),
                })

    edge_df = pd.DataFrame(edge_rows)
    proto_df = pd.DataFrame(rows)
    edge_df.to_csv(OUT / "tables" / "graph_validation_edges.csv", index=False)
    proto_df.to_csv(OUT / "tables" / "graph_validation_prototypes.csv", index=False)
    summary = {
        "interpretation": "predictive lagged-association audit; not causal identification",
        "n_graph_bins": int(len(adj)),
        "n_train_bins": int(train_bins.sum()),
        "prototype_stability_mean": float(proto_df.within_prototype_mask_jaccard.mean()),
        "prototype_directed_fraction_mean": float(proto_df.prototype_directed_edge_fraction.mean()),
        "n_test_chunks_scanned": int(len(test_ids)),
        "direction_forward_wins_mean": float(edge_df.forward_wins_fraction.mean()) if len(edge_df) else None,
        "direction_delta_mean": float(edge_df.forward_minus_reverse.mean()) if len(edge_df) else None,
        "prototype_rows": rows,
    }
    (PROCESSED / "graph_validation.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "prototype_rows"}, indent=2))


if __name__ == "__main__":
    main()
