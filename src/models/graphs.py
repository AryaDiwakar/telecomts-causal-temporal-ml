"""Dynamic dependency graphs over KPI channels.

Pipeline
--------
1. Chunks of each recording (split-specific) are binned along time
   (``make_bins``); every bin pools its chunk-steps into a multivariate
   sample [~steps, 16].
2. Per bin: a sparse undirected skeleton is inferred with GraphicalLasso
   (partial correlations) thresholded and kept to top-K edges/node; edge
   direction is set with discretized transfer entropy between the endpoint
   channels (TE_ij vs TE_ji).
3. Bin adjacencies are clustered into ``N_PROTOTYPES`` (KMeans fit on train
   bins only); every chunk maps to its bin's prototype -> the temporal
   evolution of topology, summarized by a discrete graph-id per chunk.
4. Statistical audit (edge frequencies, densities, stability) exported.

Outputs (data/processed/):
  graph_bins.npz        bins x 16 x 16 adjacency, uids
  graph_prototypes.npz  K x 16 x 16 prototypes, chunk->prototype, chunk->bin
  graph_audit.json      edge stats

Usage:
    python -m src.models.graphs
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.covariance import GraphicalLasso

ROOT = Path(__file__).resolve().parents[2]
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"

TOP_K = 4
THRESHOLD = 0.15
BIN_CHUNKS = 90
BIN_HOP = 45
N_PROTOTYPES = 6
N_NODES = 16
E_TOL = 1.15
MAX_POOL = 20000
GLASSO_ALPHA = 0.05

RNG = np.random.default_rng(0)


def load() -> tuple[np.ndarray, pd.DataFrame]:
    X = np.load(INTERIM / "chunks_X.npy", mmap_mode="r")
    meta = pd.read_parquet(INTERIM / "chunks_meta.parquet")
    idx = np.load(PROCESSED / "split_indices.npz")
    split_map = {}
    for name in ("train", "val", "test"):
        for c in idx[name]:
            split_map[int(c)] = name
    meta["split"] = meta.chunk_id.map(split_map)
    return X, meta


def make_bins(meta: pd.DataFrame, split: str) -> pd.DataFrame:
    """Bins = sliding time blocks of consecutive chunks within one recording."""
    sub = meta[meta.split == split].sort_values(["recording", "start_time"])
    bins = []
    for rec, g in sub.groupby("recording"):
        ids = g.index.to_numpy()
        for i in range(0, len(ids), BIN_HOP):
            sel = ids[i:i + BIN_CHUNKS]
            if len(sel) < 12:
                continue
            bins.append(
                {
                    "bins_uid": f"{split}|{rec}|{i}",
                    "recording": rec,
                    "split": split,
                    "chunk_ids": sel,
                }
            )
    return pd.DataFrame(bins).reset_index(drop=True)


def _joint_entropy(c: np.ndarray) -> float:
    p = c[c > 0]
    p = p / p.sum()
    return float(-np.sum(p * np.log2(p)))


def transfer_entropy(x: np.ndarray, y: np.ndarray, bins: int = 4, history: int = 1) -> float:
    """TE x(t-history) -> y(t) via 3D joint histogram: H|..., exporter-level."""
    qx = np.minimum(bins - 1, ((x - x.min()) / max(1e-12, (x.max() - x.min()) / bins)).astype(int))
    qy = np.minimum(bins - 1, ((y - y.min()) / max(1e-12, (y.max() - y.min()) / bins)).astype(int))
    H = history
    h3, _ = np.histogramdd(
        np.stack([qx[:-H], qy[:-H], qy[H:]], axis=1),
        bins=(bins, bins, bins),
        range=((0, bins), (0, bins), (0, bins)),
    )
    h2_xy = h3.sum(axis=2)   # p(x_t1, y_t1)
    h2_yy = h3.sum(axis=0)   # p(y_t1, y_t)
    h_y = h2_yy.sum(axis=1)  # p(y_t1)
    te = 0.0
    for i in range(bins):
        for j in range(bins):
            for k in range(bins):
                joint = h3[i, j, k]
                if joint <= 0:
                    continue
                te += joint * np.log2(
                    (h3[i, j, k] * h_y[j]) / max(1e-12, (h2_xy[i, j] * h2_yy[j, k]))
                )
    return max(0.0, te)


def bin_graph(pool: np.ndarray) -> np.ndarray:
    """pool: [steps, 16] -> directed, weighted adjacency [16,16].

    Undirected skeleton = thresholded/skeletonized partial correlations from
    GraphicalLasso precision; direction via transfer entropy.
    """
    if pool.shape[0] > MAX_POOL:
        pool = pool[RNG.choice(pool.shape[0], MAX_POOL, replace=False)]
    # normalize columns to unit variance so GraphicalLasso's alpha is on a stable scale
    mu = pool.mean(axis=0, keepdims=True)
    sd = pool.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Z = (pool - mu) / sd
    try:
        gl = GraphicalLasso(alpha=GLASSO_ALPHA, max_iter=100).fit(Z)
    except Exception as exc:  # ill-conditioned -> empty graph
        print("  graph fit failed; empty adjacency:", exc)
        return np.zeros((N_NODES, N_NODES), dtype=np.float32)
    if not np.all(np.isfinite(gl.precision_)):
        return np.zeros((N_NODES, N_NODES), dtype=np.float32)
    d = np.sqrt(np.clip(np.diag(gl.precision_), 1e-12, None))
    pcorr = -gl.precision_ / np.outer(d, d)
    np.fill_diagonal(pcorr, 0.0)
    strength = np.abs(pcorr)

    A = np.zeros((N_NODES, N_NODES), dtype=np.float32)
    for i in range(N_NODES):
        order = np.argsort(strength[i])[::-1]
        kept = 0
        for j in order:
            if kept >= TOP_K:
                break
            if j == i or A[j, i] > 0:
                continue
            if strength[i, j] >= THRESHOLD:
                A[i, j] = strength[i, j]
                kept += 1
    A = A + A.T  # symmetric skeleton with strengths

    rows, cols = np.nonzero(A)
    for i, j in zip(rows, cols):
        if i >= j:
            continue
        te_ij = transfer_entropy(pool[:, i], pool[:, j])
        te_ji = transfer_entropy(pool[:, j], pool[:, i])
        strength_ij = max(float(A[i, j]), float(A[j, i]))
        if te_ij > E_TOL * te_ji:
            # Store source -> target as A[source, target].
            A[i, j] = strength_ij
            A[j, i] = 0.0
        elif te_ji > E_TOL * te_ij:
            A[j, i] = strength_ij
            A[i, j] = 0.0
        else:
            pass  # keep symmetric (undirected edge)
    return A


def main() -> None:
    X, meta = load()
    bins_list = []
    chunk2bin = np.full(len(meta), -1, dtype=np.int64)
    chunk2proto = np.full(len(meta), -1, dtype=np.int64)

    all_adj: list[np.ndarray] = []
    uids: list[str] = []
    stat_rows = []

    for split in ("train", "val", "test"):
        bins = make_bins(meta, split)
        for b in bins.itertuples():
            ids = np.asarray(b.chunk_ids)
            pool = np.concatenate([X[c][:, :N_NODES] for c in ids], axis=0)
            A = bin_graph(pool)
            bins_list.append(b.bins_uid)
            uids.append(b.bins_uid)
            all_adj.append(A)
            for c in ids:
                chunk2bin[c] = len(uids) - 1
            stat_rows.append(
                {
                    "split": split,
                    "recording": b.recording,
                    "n_edges": int((A != 0).sum() // 2),
                    "density": float((A != 0).sum() / (N_NODES * N_NODES)),
                }
            )
            print(".", end="", flush=True)
        print(f" '{split}' bins={len(bins)}")

    adj_stack = np.stack(all_adj, axis=0)  # [B,16,16] float32
    train_mask = np.array([u.startswith("train|") for u in uids])

    # prototype clustering fit on train bins only
    km = KMeans(n_clusters=N_PROTOTYPES, random_state=0, n_init=10)
    km.fit(adj_stack[train_mask].reshape(train_mask.sum(), -1))
    full_proto = np.zeros(len(uids), dtype=np.int64)
    if train_mask.sum() > 0:
        full_proto[train_mask] = km.predict(adj_stack[train_mask].reshape(train_mask.sum(), -1))
    if (~train_mask).sum() > 0:
        full_proto[~train_mask] = km.predict(adj_stack[~train_mask].reshape((~train_mask).sum(), -1))

    for c in range(len(meta)):
        b = chunk2bin[c]
        if b >= 0:
            chunk2proto[c] = full_proto[b]
        else:
            # tiny recording/bin pool -> graph its own steps and query a prototype
            A = bin_graph(X[c][:, :N_NODES])
            chunk2proto[c] = int(km.predict(A.reshape(1, -1))[0])
    if (chunk2proto < 0).any():
        raise RuntimeError("chunks unassigned to a graph bin")

    prototypes = km.cluster_centers_.reshape(N_PROTOTYPES, N_NODES, N_NODES).astype(np.float32)

    edge_freq = (adj_stack != 0).mean(axis=0)

    np.savez_compressed(
        PROCESSED / "graph_bins.npz",
        adjacencies=adj_stack,
        uids=np.array(uids),
    )
    np.savez_compressed(
        PROCESSED / "graph_prototypes.npz",
        prototypes=prototypes,
        chunk_prototypes=chunk2proto,
        chunk_bins=chunk2bin,
    )
    with open(PROCESSED / "graph_audit.json", "w") as fh:
        json.dump(
            {
                "n_bins": int(len(uids)),
                "n_train_bins": int(train_mask.sum()),
                "n_val_test_bins": int((~train_mask).sum()),
                "mean_edges": float(np.mean([r["n_edges"] for r in stat_rows])),
                "mean_density": float(np.mean([r["density"] for r in stat_rows])),
                "edge_frequency": edge_freq.tolist(),
                "n_prototypes": N_PROTOTYPES,
                "bin_chunks": BIN_CHUNKS,
                "algorithm": "GraphicalLasso(alpha=0.05)+topk(4)+TE-direction",
            },
            fh,
            indent=2,
        )
    pd.DataFrame(stat_rows).to_csv(ROOT / "outputs" / "tables" / "graph_bin_stats.csv", index=False)
    print("\nmean edges:", round(float(np.mean([r["n_edges"] for r in stat_rows])), 2))
    print("prototype usage:", np.bincount(chunk2proto).tolist())
    print("edge_frequency (mask of top-k):\n", np.round(edge_freq, 2))
    print("saved graph artifacts ->", PROCESSED)


if __name__ == "__main__":
    main()
