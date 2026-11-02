"""Leakage-free graph prototypes for grouped (LORO) evaluation.

The global ``data/processed/graph_prototypes.npz`` centroids are fit on
Setting-I *train* bins (``graphs.py``), which overlap the recordings held out by
each LORO fold. For the honest recording-held-out setting, re-fit KMeans on the
bins whose member chunks are **all** in the fold's train set, and re-assign every
chunk through the fold-train centroids.

Bins with a single anomalous recording's chunks are used as-is (their adjacency
was estimated from that bin's own chunks); we only change *which* bins
contribute to the clustering and *which* centroids are used for assignment.
Chunks in impure bins are assigned by querying the fold-train centroids with
their own bin adjacency -- no fold-test data enters the fit. Chunks that belong
to no bin (``chunk_bins == -1``; tiny recording/bin pools) are handled exactly as
in ``graphs.py``: graph their own steps and query the fold-train centroids.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans

N_PROTOTYPES = 6
N_NODES = 16


def fold_prototypes(
    chunk_bins: np.ndarray,
    adjacencies: np.ndarray,
    fold_train_ids: np.ndarray,
    X: np.ndarray | None = None,
    n_prototypes: int = N_PROTOTYPES,
    n_nodes: int = N_NODES,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Refit graph prototypes for one fold.

    Returns ``(prototypes [K,n_nodes,n_nodes] float32, chunk_prototypes [n] int64)``.
    ``X`` (raw chunks, ``[n, T, C]``) is required only when some chunks belong to
    no graph bin.
    """
    train = set(int(i) for i in fold_train_ids)
    members: dict[int, list[int]] = defaultdict(list)
    for c, b in enumerate(np.asarray(chunk_bins).tolist()):
        if b >= 0:
            members[int(b)].append(int(c))

    pure = sorted(b for b, cs in members.items() if all(c in train for c in cs))
    if len(pure) < n_prototypes:
        raise RuntimeError(
            f"only {len(pure)} fold-train-pure graph bins; need >= {n_prototypes} "
            "to refit prototypes (regenerate the LORO folds)"
        )
    flat = adjacencies[pure].reshape(len(pure), -1)
    km = KMeans(n_clusters=n_prototypes, random_state=seed, n_init=10).fit(flat)
    prototypes = km.cluster_centers_.reshape(
        n_prototypes, *adjacencies.shape[1:]
    ).astype(np.float32)

    assignment = np.full(len(chunk_bins), -1, dtype=np.int64)
    for b, cs in members.items():
        lab = int(km.predict(adjacencies[b].reshape(1, -1))[0])
        for c in cs:
            assignment[c] = lab

    unassigned = np.where(np.asarray(chunk_bins) < 0)[0]
    if len(unassigned):
        if X is None:
            raise RuntimeError(
                f"{len(unassigned)} chunks belong to no graph bin; pass X to "
                "graph them and query a prototype"
            )
        import contextlib
        import io

        from src.models.graphs import bin_graph

        with contextlib.redirect_stdout(io.StringIO()):  # bin_graph warns on tiny pools
            for c in unassigned:
                a = bin_graph(np.asarray(X[c])[:, :n_nodes])
                assignment[c] = int(km.predict(np.asarray(a).reshape(1, -1))[0])

    if (assignment < 0).any():
        raise RuntimeError("chunks unassigned to a graph bin")
    return prototypes, assignment


def fold_scaler(X: np.ndarray, fold_train_ids: np.ndarray, method: str) -> dict:
    """Scaler fit on this fold's train chunks only (delegates to prepare)."""
    from src.preprocessing.prepare import fit_scaler

    return fit_scaler(X, np.asarray(fold_train_ids), method)
