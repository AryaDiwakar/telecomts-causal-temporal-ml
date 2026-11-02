"""SpotLight (MobiCom 2024) -> 1 s grid adapter with self-derived labels.

The public SpotLight Drive tree ships only ``radio.csv`` (radio KPIs, ~6.7 Hz)
and ``platform.csv`` (per-thread RAN/PTP CPU KPIs, ~3 Hz) per anomalous run; no
per-step label files are published. The authors' merged-label mirror used by
external evaluations is offline. We therefore derive transparent per-run labels
from the documented injection mechanism: stress-ng on ``pdcp_worker_0`` (CU
contention), a busy-loop on ``TxCtrl_DU`` (scheduler contention), USRP
interference on the carrier (radio), and fronthaul contention (network).

Labeling rule (configs/kpi_crosswalk.yaml, section label_rules/monitor_sets):
  * steps are aggregated to a 1 s grid (radio: mean level/rate, platform: max);
  * a step deviates if |robust z| >= z_thr against the *run's own* median+MAD;
  * isolated deviations are dropped: a segment must last >= min_sustained steps;
  * a 128 s window is positive if >= window_label_fraction of its steps deviate.

Loading/splitting is deliberately decoupled from the TelecomTS pipeline
(``src.data.loader``) so cross-vendor leakage is impossible: this adapter writes
``data/processed/spotlight/`` (windows NPZ + meta parquet + provenance), and the
RANTransfer evaluation groups windows by run.

Usage: python -m src.data.adapters.spotlight
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "external" / "SpotLight" / "raw" / "Anomaly"
OUT = ROOT / "data" / "processed" / "spotlight"
CONFIG_PATH = ROOT / "configs" / "kpi_crosswalk.yaml"

NOMINAL_DT = 1.0  # seconds per grid step


def _load_config() -> dict:
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    xw = cfg["telecomts_to_spotlight"]
    # shared feature channels = direct + proxy only, keep TelecomTS order
    channel_list = []
    quality = {}
    for kpi, m in xw.items():
        if m["quality"] == "gap":
            continue
        ch = m["channel"]
        if ch not in channel_list:
            channel_list.append(ch)
        quality[ch] = m["quality"]
    window = cfg["window"]
    lr = cfg["label_rules"]
    return {
        "channels": channel_list,
        "quality": quality,
        "monitors": cfg["label_monitors"],
        "monitor_sets": cfg["monitor_sets"],
        "window": window,
        "label_rules": lr,
        "agg": cfg["agg"],
    }


def _read_ts(df: pd.DataFrame) -> np.ndarray:
    ts = pd.to_datetime(df["timestamp"], format="mixed").astype("int64").to_numpy() / 1e9
    return ts


def _choose_column(df: pd.DataFrame, name: str) -> str | None:
    if name in df.columns:
        return name
    for cand in (name, name + "_size", name + "_mean"):
        if cand in df.columns:
            return cand
    return None


def _grid(run_df: dict[str, pd.DataFrame]) -> np.ndarray:
    # anchor on the longest telemetry surface (radio); other series are
    # reindexed onto it and NaN-padded
    base = run_df.get("radio", next(iter(run_df.values())))
    ts = _read_ts(base)
    return np.arange(int(np.floor(np.min(ts))), int(np.floor(np.max(ts))) + 1)


def _aggregate(
    df: pd.DataFrame, ts: np.ndarray, col: str, method: str, grid: np.ndarray
) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(grid), np.nan)
    x = pd.to_numeric(df[col], errors="coerce").to_numpy()
    bins = np.floor(ts).astype(np.int64)
    g = pd.Series(x).groupby(pd.Series(bins)).agg(method)
    return g.reindex(grid).to_numpy(dtype=np.float64)


def _robust_z(x: np.ndarray, floor_sd: float) -> np.ndarray:
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    sd = 1.4826 * mad if mad > 0 else 0.0
    sd = max(sd, floor_sd)
    return (x - med) / sd


def _sustained(flag: np.ndarray, min_len: int) -> np.ndarray:
    out = np.zeros_like(flag)
    if min_len <= 1:
        return flag
    n = 0
    for i, v in enumerate(np.concatenate([flag, [False]])):
        if v:
            n += 1
        else:
            if n >= min_len:
                out[i - n:i] = True
            n = 0
    return out


def _step_labels(monitor_series: dict[str, np.ndarray], cfg: dict) -> np.ndarray:
    lr = cfg["label_rules"]
    flags = []
    for s in monitor_series.values():
        if s is None:
            continue
        z = np.abs(_robust_z(s, lr["z_floor_sd"]))
        dev = np.isfinite(z) & (z >= lr["robust_z_threshold"])
        flags.append(_sustained(dev, int(lr["min_sustained_steps"])))
    if not flags:
        return np.zeros(len(next(iter(monitor_series.values()))), dtype=bool)
    return np.logical_or.reduce(flags)


def _class_of(run_dir: Path) -> str:
    p = run_dir.as_posix().upper()
    for cls in ("RADIO", "NETWORK", "MAC", "PDCP"):
        if f"/{cls}/" in p or p.endswith(f"/{cls}"):
            return cls
    return "MIXED"


def _windows(X: np.ndarray, y_step: np.ndarray, T: int, frac: float):
    n = X.shape[0]
    n_win = n // T
    Xw = X[: n_win * T].reshape(n_win, T, X.shape[1])
    yfrac = y_step[: n_win * T].reshape(n_win, T).mean(axis=1)
    yw = (yfrac >= frac).astype(np.int64)
    return Xw, yw, yfrac


def _run_id(run_dir: Path) -> str:
    return run_dir.relative_to(RAW).as_posix().replace("/", "__")


def process_run(run_dir: Path, cfg: dict) -> dict | None:
    rp = run_dir / "radio.csv"
    pp = run_dir / "platform.csv"
    if not (rp.exists() and pp.exists()):
        return None
    radio = pd.read_csv(rp, low_memory=False)
    platform = pd.read_csv(pp, low_memory=False)
    radio_ts = _read_ts(radio)
    plat_ts = _read_ts(platform)
    grid = _grid({"radio": radio, "platform": platform})

    features = {}
    for ch in cfg["channels"]:
        src = radio if ch in radio.columns else platform
        agg_m = cfg["agg"]["radio"] if ch in radio.columns else cfg["agg"]["platform"]
        ts_src = radio_ts if ch in radio.columns else plat_ts
        col = _choose_column(src, ch)
        if col is None:
            features[ch] = np.full(len(grid), np.nan)
            continue
        features[ch] = _aggregate(src, ts_src, col, agg_m, grid)
    X = np.column_stack([features[c] for c in cfg["channels"]]).astype(np.float32)
    # per-run median imputation (channels missing in a run)
    for j in range(X.shape[1]):
        col = X[:, j]
        med = np.nanmedian(col) if np.isfinite(col).sum() > 0 else 0.0
        col[~np.isfinite(col)] = med
    X = X.astype(np.float32)
    monitors = {}
    mset = cfg["monitor_sets"][_class_of(run_dir)]
    for name in mset:
        spec = cfg["monitors"][name]
        src = radio if spec["table"] == "radio" else platform
        ts_src = radio_ts if spec["table"] == "radio" else plat_ts
        col = _choose_column(src, spec["channel"])
        if col is not None:
            monitors[name] = _aggregate(src, ts_src, col, spec["agg"], grid)
    y_step = _step_labels(monitors, cfg)
    y_step = y_step[: len(grid)]

    T = int(cfg["window"]["t_seconds"])
    Xw, yw, yfrac = _windows(X, y_step.astype(float), T, float(cfg["label_rules"]["window_label_fraction"]))

    keep = np.isfinite(Xw.reshape(Xw.shape[0], -1)).mean(axis=1) >= 0.5
    Xw = Xw[keep]
    yw = yw[keep]
    yfrac = yfrac[keep]
    if len(Xw) == 0:
        return None

    return {
        "run_dir": run_dir,
        "run_id": _run_id(run_dir),
        "cls": _class_of(run_dir),
        "Xw": Xw,
        "yw": yw,
        "yfrac": yfrac,
        "y_step_frac": float(y_step.mean()),
        "grid": grid,
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    cfg = _load_config()
    run_dirs = sorted({p.parent for p in RAW.rglob("radio.csv")})
    rows = []
    for rd in run_dirs:
        try:
            res = process_run(rd, cfg)
        except Exception as exc:  # noqa: BLE001
            print(f"  !! {rd.relative_to(RAW)} failed: {exc}")
            continue
        if res is None:
            continue
        rows.append(res)
        print(
            f"  {res['run_id']:<40} cls={res['cls']:<7} "
            f"win={len(res['Xw']):4d} pos={int(res['yw'].sum()):4d} "
            f"(+{100 * res['yw'].mean():.0f}%) step_frac={100 * res['y_step_frac']:.0f}%"
        )

    if not rows:
        raise RuntimeError("no usable SpotLight runs found")

    # global constant-channel filter: drop channels that are flat everywhere
    Xall = np.concatenate([r["Xw"] for r in rows], axis=0).astype(np.float32)
    var = Xall.reshape(-1, Xall.shape[2]).var(axis=0)
    keep = var > 0.0
    Xall = Xall[:, :, keep]
    chan = [c for c, k in zip(cfg["channels"], keep) if k]
    qual = [cfg["quality"][c] for c in chan]

    X = Xall
    y = np.concatenate([r["yw"] for r in rows], axis=0).astype(np.int64)
    runs = np.array([r["run_id"] for r in np.repeat(rows, [len(r["Xw"]) for r in rows])])
    print(f"\n  shared KPI space: {len(cfg['channels'])} -> {len(chan)} ({[c.split('_')[0] for c in chan]})")

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / "spotlight_windows.npz",
        X=X, y=y,
        run_ids=runs.astype(object),
        feature_cols=np.asarray(chan, dtype=object),
        quality=np.asarray(qual, dtype=object),
    )
    meta = pd.DataFrame(
        {
            "window_id": (np.arange(len(X))),
            "run_id": runs,
            "cls": [r["cls"] for r in np.repeat(rows, [len(r["Xw"]) for r in rows])],
            "anomaly": y,
            "yfrac": np.concatenate([r["yfrac"] for r in rows]),
        }
    )
    meta.to_parquet(OUT / "spotlight_meta.parquet")

    summary = {
        "n_windows": int(len(X)),
        "n_positive": int(y.sum()),
        "positive_rate": float(y.mean()),
        "n_runs": len(rows),
        "feature_kpis": chan,
        "window_t": int(cfg["window"]["t_seconds"]),
        "classes": sorted(set(r["cls"] for r in rows)),
        "per_run": [
            {"run": r["run_id"], "cls": r["cls"], "n_windows": len(r["Xw"]), "n_positive": int(r["yw"].sum())}
            for r in rows
        ],
    }
    src_files = sorted({str(p) for rd in run_dirs for p in (rd / "radio.csv", rd / "platform.csv")})
    provenance = {
        "download": {
            "url": "https://drive.google.com/drive/folders/1x7WnU6q9EodacUdh3iXRmi9FWrMD3rME",
            "date": "2026-10-12",
            "note": "public Drive tree, gdown folder download; baseline zips partially fetched, not used here",
        },
        "label_policy": "self-derived robust-z from documented injection mechanism "
                       "(no public per-step labels); see configs/kpi_crosswalk.yaml",
        "config_sha256": hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
        "source_files": {f.replace(str(ROOT) + "/", ""): _sha256(Path(f)) for f in src_files},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True))

    print("\nSaved ->", OUT)
    print(json.dumps(summary, indent=2)[:1200])


if __name__ == "__main__":
    sys.exit(main())