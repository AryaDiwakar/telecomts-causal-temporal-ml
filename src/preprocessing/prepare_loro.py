"""Setting II: leave-one-out style cross-fold splits grouped by recording.

Anomaly label is constant within a recording, so grouping by recording is the
only way to force test chunks to come from recordings never seen in training.
A recording acts as one unit -> GroupKFold(5).

Outputs: data/processed/loro/fold{K}_{role}.npy

Usage:
    python -m src.preprocessing.prepare_loro
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
INTERIM = ROOT / "data" / "interim"
OUT = ROOT / "data" / "processed" / "loro"


def main() -> None:
    meta = pd.read_parquet(INTERIM / "chunks_meta.parquet")
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    # moderate class imbalance: drop the very largest normal recordings rarely
    fold = GroupKFold(n_splits=5)
    recs = meta.recording.unique()
    splits = [(idx, t_idx) for idx, t_idx in fold.split(recs, recs, recs)]
    for k, (tr_idx, te_idx) in enumerate(splits):
        tr_recs = recs[tr_idx]
        te_recs = recs[te_idx]
        # within-train, hold out one more recording-forming val from train side
        val_recs = tr_recs[rng.choice(len(tr_recs), size=max(1, len(tr_recs) // 5), replace=False)]
        train_recs = np.setdiff1d(tr_recs, val_recs)
        tr_ids = meta.loc[meta.recording.isin(train_recs), "chunk_id"].to_numpy()
        va_ids = meta.loc[meta.recording.isin(val_recs), "chunk_id"].to_numpy()
        te_ids = meta.loc[meta.recording.isin(te_recs), "chunk_id"].to_numpy()
        for role, ids in (("train", tr_ids), ("val", va_ids), ("test", te_ids)):
            np.save(OUT / f"fold{k}_{role}.npy", np.sort(ids))
        sub = meta[meta.chunk_id.isin(te_ids)]
        print(
            f"fold {k}: train_recs={len(train_recs)} val={len(va_ids)} "
            f"test_chunks={len(te_ids)} (anom {int(sub.anomaly.sum())}, "
            f"n_recs={sub.recording.nunique()})"
        )


if __name__ == "__main__":
    main()