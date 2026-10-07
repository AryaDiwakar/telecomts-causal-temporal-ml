"""Exploratory data analysis for TelecomTS.

Generates figures into outputs/figures/eda/, tables into outputs/tables/eda/,
and a machine-readable eda_summary.json.

Usage:
    python -m src.eda.run_eda
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
FIG = ROOT / "outputs" / "figures" / "eda"
TAB = ROOT / "outputs" / "tables" / "eda"

RNG = np.random.default_rng(42)


def ensure_dirs() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)


def split_of(meta: pd.DataFrame) -> pd.Series:
    idx = np.load(PROCESSED / "split_indices.npz")
    mapping = {}
    for name in ("train", "val", "test"):
        for cid in idx[name]:
            mapping[int(cid)] = name
    return meta.chunk_id.map(mapping).rename("split")


def fig_class_balance(meta: pd.DataFrame) -> None:
    counts = (
        meta.groupby(["split", "anomaly"]).size().unstack(fill_value=0)
    )
    counts.columns = ["normal", "anomalous"]
    types = (
        meta[meta.anomaly]
        .groupby(["split", "anomaly_type"])
        .size()
        .unstack(fill_value=0)
        .reindex(["train", "val", "test"])
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    counts.reindex(["train", "val", "test"]).plot.bar(ax=axes[0], color=["#4c72b0", "#c44e52"])
    axes[0].set_title("Chunks per split")
    axes[0].set_ylabel("chunks")
    axes[0].tick_params(axis="x", rotation=0)

    types.plot.bar(ax=axes[1], stacked=True)
    axes[1].set_title("Anomaly types per split")
    axes[1].set_ylabel("chunks")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "fig_class_balance.png", dpi=150)
    plt.close(fig)


def fig_kpi_distributions(X: np.ndarray, meta: pd.DataFrame, kpis: list[str]) -> None:
    train = meta[meta.split == "train"]
    nrm = train[~train.anomaly].chunk_id.to_numpy()
    ano = train[train.anomaly].chunk_id.to_numpy()
    nrm = RNG.choice(nrm, size=min(4000, len(nrm)), replace=False)
    ano = RNG.choice(ano, size=min(1000, len(ano)), replace=False)

    nrm_means = X[nrm].mean(axis=1)  # [n, 18]
    ano_means = X[ano].mean(axis=1)

    # robust-z per channel for visualization only (fit on train normals)
    loc = np.median(nrm_means, axis=0)
    scale = np.percentile(nrm_means, 75, axis=0) - np.percentile(nrm_means, 25, axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale) / 1.349

    n_show = len(kpis)
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    for j in range(n_show):
        ax = axes[j // 4, j % 4]
        ax.boxplot(
            [
                (nrm_means[:, j] - loc[j]) / scale[j],
                (ano_means[:, j] - loc[j]) / scale[j],
            ],
            showfliers=False,
            labels=["normal", "anomalous"],
            patch_artist=True,
            boxprops=dict(facecolor="#dddddd"),
        )
        ax.set_title(kpis[j], fontsize=9)
    fig.suptitle("Per-chunk KPI means (train split, robust-z)", y=1.01)
    fig.tight_layout()
    fig.savefig(FIG / "fig_kpi_distributions.png", dpi=150)
    plt.close(fig)


def fig_kpi_dynamics_example(X: np.ndarray, meta: pd.DataFrame, kpis: list[str]) -> None:
    key = ["RSRP", "DL_BLER", "UL_BLER", "PRB_Utilization_DL", "UL_SNR", "Estimated_UL_Buffer"]
    cols = [kpis.index(k) for k in key]
    train = meta[meta.split == "train"]
    ex_n = train[~train.anomaly].iloc[0].chunk_id
    ex_a = train[train.anomaly].iloc[len(train[train.anomaly]) // 2].chunk_id

    fig, axes = plt.subplots(len(key), 2, figsize=(12, 11), sharex=True)
    for j, (c, name) in enumerate(zip(cols, key)):
        axes[j, 0].plot(X[ex_n, :, c], color="#4c72b0")
        axes[j, 0].set_ylabel(name, fontsize=9)
        axes[j, 1].plot(X[ex_a, :, c], color="#c44e52")
        row = meta.iloc[ex_a]
        axes[j, 1].set_title(
            f"anomalous example ({row.anomaly_type}), onset={row.onset_step}",
            fontsize=8,
        )
    axes[0, 0].set_title("normal example chunk")
    axes[-1, 0].set_xlabel("step (0.1 s)")
    axes[-1, 1].set_xlabel("step (0.1 s)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_kpi_dynamics_example.png", dpi=150)
    plt.close(fig)


def fig_corr_matrix(X: np.ndarray, meta: pd.DataFrame, kpis: list[str]) -> np.ndarray:
    train = meta[meta.split == "train"]
    sample = RNG.choice(train.chunk_id.to_numpy(), size=min(5000, len(train)), replace=False)
    means = X[sample].mean(axis=1)[:, : len(kpis)]
    corr = np.corrcoef(means.T)

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(kpis)))
    ax.set_yticks(range(len(kpis)))
    ax.set_xticklabels(kpis, rotation=90, fontsize=7)
    ax.set_yticklabels(kpis, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlation of per-chunk KPI means (train sample)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_corr_matrix.png", dpi=150)
    plt.close(fig)
    return corr


def fig_recording_timeline(meta: pd.DataFrame) -> None:
    recs = sorted(meta.recording.unique())
    color = {"train": "#4c72b0", "val": "#55a868", "test": "#c44e52", None: "#999999"}
    fig, ax = plt.subplots(figsize=(14, 9))
    for i, rec in enumerate(recs):
        sub = meta[meta.recording == rec]
        for row in sub.itertuples():
            c = color.get(row.split, "#999999")
            ax.plot(
                [row.start_time, row.end_time],
                [i, i],
                color=c,
                lw=2,
                alpha=0.9 if row.anomaly else 0.5,
            )
    ax.set_yticks(range(len(recs)))
    ax.set_yticklabels(recs, fontsize=6)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax.set_title("Chunk coverage per recording (blue=train, green=val, red=test)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG / "fig_recording_timeline.png", dpi=150)
    plt.close(fig)


def fig_burst_structure(meta: pd.DataFrame) -> None:
    rec = "anomalous/synthetic/Zone_A/File"
    sub = meta[meta.recording == rec].sort_values("start_time").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.scatter(sub.start_time, sub.onset_step, c=sub.anomaly, cmap="coolwarm", s=14)
    ax2 = ax.twinx()
    ax2.bar(sub.start_time, [12.8] * len(sub), width=0.01, color="gray", alpha=0.3)
    ax.set_ylabel("onset step within chunk (anomaly starts)")
    ax2.set_ylabel("chunk duration (s)")
    ax.set_title(f"Burst structure: {rec} (start times sorted; onset>0 means chunk begins pre-fault)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG / "fig_burst_structure.png", dpi=150)
    plt.close(fig)


def fig_protocol_dist(meta: pd.DataFrame) -> None:
    # protocols recovered from categorical channel via codebook
    with open(INTERIM / "kpi_names.json") as fh:
        info = json.load(fh)
    inv_ul = {i: v for v, i in info["categorical_maps"]["UL_Protocol"].items()}
    X = np.load(INTERIM / "chunks_X.npy", mmap_mode="r")
    idx = np.load(PROCESSED / "split_indices.npz")
    train_ids = idx["train"]
    ul_codes = sorted({int(c) for c_id in train_ids[:3000] for c in np.unique(X[c_id, :, 16])})
    labels = [inv_ul.get(int(c), str(c)) for c in ul_codes]
    counts = []
    for c in ul_codes:
        counts.append(int(np.mean([np.any(X[c_id, :, 16] == c) for c_id in train_ids[:3000]])))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, counts, color="#4c72b0")
    ax.set_title("UL_Protocol share of chunks (train sample)")
    ax.set_ylabel("# chunks containing protocol")
    fig.tight_layout()
    fig.savefig(FIG / "fig_protocol_dist.png", dpi=150)
    plt.close(fig)


def tables(meta: pd.DataFrame, kpis: list[str], X: np.ndarray, corr: np.ndarray) -> dict:
    overview = {
        "n_chunks": int(len(meta)),
        "n_anomalous": int(meta.anomaly.sum()),
        "prevalence": float(meta.anomaly.mean()),
        "n_recordings": int(meta.recording.nunique()),
        "split_counts": meta.split.value_counts().to_dict(),
        "split_anomalous": meta.groupby("split").anomaly.sum().to_dict(),
    }
    pd.DataFrame([overview]).to_csv(TAB / "table_dataset_overview.csv", index=False)

    type_counts = (
        meta[meta.anomaly]
        .groupby(["anomaly_type", "split"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["train", "val", "test"])
        .sort_values("train", ascending=False)
    )
    type_counts.to_csv(TAB / "table_type_counts.csv")

    train = meta[meta.split == "train"]
    nrm = RNG.choice(train[~train.anomaly].chunk_id.to_numpy(), 3000, replace=False)
    ano = RNG.choice(train[train.anomaly].chunk_id.to_numpy(), min(800, int(train.anomaly.sum())), replace=False)
    rows = []
    for j, k in enumerate(kpis):
        n = X[nrm, :, j].ravel()
        a = X[ano, :, j].ravel()
        pooled_std = float(np.std(np.concatenate([n, a])) or 1.0)
        rows.append(
            {
                "kpi": k,
                "normal_mean": float(np.mean(n)),
                "normal_std": float(np.std(n)),
                "anomalous_mean": float(np.mean(a)),
                "anomalous_std": float(np.std(a)),
                "std_effect_size": float(abs(np.mean(a) - np.mean(n)) / pooled_std),
            }
        )
    pd.DataFrame(rows).sort_values("std_effect_size", ascending=False).to_csv(
        TAB / "table_kpi_stats.csv", index=False
    )

    pd.DataFrame(corr, index=kpis, columns=kpis).to_csv(TAB / "table_corr_matrix.csv")

    inv = (
        meta.groupby("recording")
        .agg(
            n_chunks=("chunk_id", "count"),
            n_anomalous=("anomaly", "sum"),
            start=("start_time", "min"),
            end=("end_time", "max"),
            types=("anomaly_type", lambda s: "; ".join(sorted({x for x in s if x}))),
        )
    )
    inv["split_of_chunks"] = [
        sorted(set(meta.loc[meta.recording == r, "split"].dropna())) for r in inv.index
    ]
    inv.to_csv(TAB / "table_recording_inventory.csv")
    return overview


def main() -> None:
    ensure_dirs()
    meta = pd.read_parquet(INTERIM / "chunks_meta.parquet")
    meta["split"] = split_of(meta)
    with open(INTERIM / "kpi_names.json") as fh:
        info = json.load(fh)
    kpis = info["numeric_kpis"]
    X = np.load(INTERIM / "chunks_X.npy", mmap_mode="r")

    fig_class_balance(meta)
    fig_kpi_distributions(X, meta, kpis)
    fig_kpi_dynamics_example(X, meta, kpis)
    corr = fig_corr_matrix(X, meta, kpis)
    fig_recording_timeline(meta)
    fig_burst_structure(meta)
    fig_protocol_dist(meta)
    overview = tables(meta, kpis, X, corr)

    # temporal autocorrelation of chunk anomalies within recordings
    ac_rows = []
    for rec, sub in meta.sort_values("start_time").groupby("recording"):
        flags = sub.anomaly.to_numpy().astype(float)
        if len(flags) > 10 and flags.std() > 0:
            for lag in (1, 5, 20):
                ac_rows.append(
                    {
                        "recording": rec,
                        "lag": lag,
                        "acf": float(np.corrcoef(flags[:-lag], flags[lag:])[0, 1]),
                    }
                )
    ac = pd.DataFrame(ac_rows)
    ac.to_csv(TAB / "table_anomaly_autocorr.csv", index=False)

    mean_acf_lag1 = float(ac[ac.lag == 1].acf.mean()) if len(ac) else None
    summary = {
        "overview": overview,
        "anomaly_type_total": meta.loc[meta.anomaly, "anomaly_type"].value_counts().to_dict(),
        "onset_gt0_total": int((meta.anomaly & (meta.onset_step > 0)).sum()),
        "mean_anomaly_autocorr_lag1": mean_acf_lag1,
        "note": "No recording mixes normal and anomalous chunks; anomaly flags are constant within every recording.",
    }
    with open(FIG.parent / "eda_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
