"""Shortcut audit: is KPI-level RCA solvable without looking at the input?

Compares the shipped TelecomTS RCA ground truth (per-type-constant KPI set,
"template" windows) against the Track 2 injected ground truth (instance-varying
KPI subsets, "gen" windows) in the same test split.

For each subset we report:
  * distinct affected-KPI sets per anomaly type (within-type variability)
  * input-free "type-oracle" RCA: predict the train-majority KPI set of the
    window's TRUE type, score Jaccard vs y_affected
  * input-free "global-majority" RCA: predict the single most common set overall

A type-oracle Jaccard near 1.0 means RCA is fully determined by the type label
(the benchmark admits a type->template shortcut); lower means the KPI set must
be inferred from the input.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

INTERIM = Path("data/interim")
PROCESSED = Path("data/processed")


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def main() -> None:
    meta = pd.read_parquet(INTERIM / "chunks_meta.parquet")
    tg = np.load(PROCESSED / "targets.npz")
    sp = np.load(PROCESSED / "split_indices.npz")
    y_aff = tg["y_affected"]
    kpis = json.load(open(INTERIM / "kpi_names.json"))["numeric_kpis"]
    anom = meta.anomaly.to_numpy()
    meta = meta.copy()
    meta["is_gen"] = meta.recording.str.contains("synthetic_gen")

    def gt(i):
        return frozenset(kpis[j] for j in np.where(y_aff[i] > 0)[0])

    tr = sp["train"]
    canon = {}
    for t in meta.anomaly_type.unique():
        if t is None:
            continue
        cnt = {}
        for i in tr:
            if anom[i] and meta.anomaly_type.iloc[i] == t:
                cnt[gt(i)] = cnt.get(gt(i), 0) + 1
        if cnt:
            canon[t] = max(cnt, key=cnt.get)
    gcnt = {}
    for i in tr:
        if anom[i]:
            gcnt[gt(i)] = gcnt.get(gt(i), 0) + 1
    gset = max(gcnt, key=gcnt.get)

    def subset_stats(idx, name):
        idx = [i for i in idx if anom[i]]
        per_type = {}
        for i in idx:
            per_type.setdefault(meta.anomaly_type.iloc[i], set()).add(gt(i))
        oracle = [jaccard(canon.get(meta.anomaly_type.iloc[i], frozenset()), gt(i)) for i in idx]
        gmaj = [jaccard(gset, gt(i)) for i in idx]
        return {
            "subset": name,
            "n_windows": len(idx),
            "n_types": len(per_type),
            "max_distinct_sets_per_type": max((len(v) for v in per_type.values()), default=0),
            "mean_distinct_sets_per_type": round(
                float(np.mean([len(v) for v in per_type.values()])) if per_type else 0.0, 3),
            "test_type_oracle_jaccard": round(float(np.mean(oracle)) if oracle else 0.0, 4),
            "test_global_majority_jaccard": round(float(np.mean(gmaj)) if gmaj else 0.0, 4),
        }

    te = sp["test"]
    rows = [
        subset_stats(te, "all_test"),
        subset_stats([i for i in te if not meta.is_gen.iloc[i]], "template_test(shipped)"),
        subset_stats([i for i in te if meta.is_gen.iloc[i]], "instance_varying_test(track2)"),
    ]
    out = {"test_subsets": rows}
    print(json.dumps(out, indent=2))
    Path("outputs/results/shortcut_audit.json").write_text(json.dumps(out, indent=2))
    print("\nsaved outputs/results/shortcut_audit.json")


if __name__ == "__main__":
    main()
