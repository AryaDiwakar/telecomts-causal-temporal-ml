"""Load raw TelecomTS chunks into arrays + metadata frame.

Outputs (data/interim/):
  chunks_X.npy       float32 [N, 128, 18]  (16 numeric + 2 protocol-coded)
  chunks_meta.parquet
  kpi_names.json

Usage:
    python -m src.data.loader
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "TelecomTS"
INTERIM = ROOT / "data" / "interim"

NUMERIC_KPIS = [
    "RSRP", "DL_BLER", "DL_MCS", "UL_BLER", "UL_MCS", "UL_NPRB", "UL_SNR",
    "TX_Bytes", "RX_Bytes", "Estimated_UL_Buffer", "PRBs_DL_Current",
    "PRBs_UL_Current", "PRB_Utilization_DL", "PRB_Utilization_UL",
    "UL_NumberOfPackets", "DL_NumberOfPackets",
]
CATEGORICAL_KPIS = ["UL_Protocol", "DL_Protocol"]
ALL_KPIS = NUMERIC_KPIS + CATEGORICAL_KPIS


def load_chunks() -> tuple[np.ndarray, pd.DataFrame, dict]:
    files = sorted(RAW.rglob("chunked.jsonl"))
    if not files:
        raise FileNotFoundError(
            "Dataset not found.\nPlease run:\npython -m src.data.download"
        )

    numeric: list[np.ndarray] = []
    cat_str: dict[str, list[list[str]]] = {k: [] for k in CATEGORICAL_KPIS}
    rows: list[dict] = []

    for f in files:
        rel = f.relative_to(RAW)
        parts = list(rel.parts)
        recording = "/".join(parts[:-2])
        source = parts[0]
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                kpis = rec["KPIs"]
                n = len(kpis[NUMERIC_KPIS[0]])
                arr = np.empty((n, len(NUMERIC_KPIS)), dtype=np.float32)
                for j, name in enumerate(NUMERIC_KPIS):
                    arr[:, j] = kpis[name]
                numeric.append(arr)
                for name in CATEGORICAL_KPIS:
                    cat_str[name].append(kpis[name])

                anom = rec["anomalies"]
                dur = anom.get("anomaly_duration") or {}
                labels = rec.get("labels") or {}

                def _as_int(v, default: int = -1) -> int:
                    return default if v is None else int(v)

                rows.append(
                    {
                        "recording": recording,
                        "source": source,
                        "start_time": pd.Timestamp(rec["start_time"]),
                        "end_time": pd.Timestamp(rec["end_time"]),
                        "anomaly": bool(anom.get("exists", False)),
                        "anomaly_type": anom.get("type") if anom.get("exists") else None,
                        "onset_step": _as_int(dur.get("start")),
                        "offset_step": _as_int(dur.get("end")),
                        "affected_kpis": json.dumps(sorted(anom.get("affected_kpis") or [])),
                        "zone": labels.get("zone"),
                        "application": labels.get("application"),
                        "mobility": labels.get("mobility"),
                        "congestion": labels.get("congestion"),
                    }
                )

    meta = pd.DataFrame(rows)
    meta.insert(0, "chunk_id", np.arange(len(meta), dtype=np.int64))

    X_num = np.stack(numeric)  # [N, 128, 16]
    X_cat = np.zeros((len(meta), 128, len(CATEGORICAL_KPIS)), dtype=np.float32)
    codebooks: dict[str, dict[str, int]] = {}
    for k, name in enumerate(CATEGORICAL_KPIS):
        values = sorted({v for col in cat_str[name] for v in col})
        codebooks[name] = {v: i for i, v in enumerate(values)}
        cb = codebooks[name]
        for i, col in enumerate(cat_str[name]):
            X_cat[i, :, k] = [cb[v] for v in col]

    X = np.concatenate([X_num, X_cat], axis=2)

    info = {
        "numeric_kpis": NUMERIC_KPIS,
        "categorical_kpis": CATEGORICAL_KPIS,
        "categorical_maps": codebooks,
        "n_chunks": int(X.shape[0]),
        "chunk_length": int(X.shape[1]),
        "n_channels": int(X.shape[2]),
    }
    return X, meta, info


def save() -> None:
    INTERIM.mkdir(parents=True, exist_ok=True)
    X, meta, info = load_chunks()
    np.save(INTERIM / "chunks_X.npy", X.astype(np.float32))
    meta.to_parquet(INTERIM / "chunks_meta.parquet", index=False)
    with open(INTERIM / "kpi_names.json", "w") as fh:
        json.dump(info, fh, indent=2)
    print(f"saved X={X.shape} meta={meta.shape} -> {INTERIM}")


if __name__ == "__main__":
    save()
