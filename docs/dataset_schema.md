# TelecomTS Dataset Schema (verified)

> All facts below are verified against the downloaded data
> (`data/raw/TelecomTS`, Hugging Face `AliMaatouk/TelecomTS`, 33 recordings) and
> the artifacts produced by `src/data/loader.py` and `src/preprocessing/prepare.py`.

## Layout

```
TelecomTS/
  normal/stationary/{Zone_A,Zone_B,Zone_C}/{congestion,no_congestion}/
      {File,Twitch,YouTube}/{processed/chunked.jsonl, raw/metrics.csv}
  normal/mobile/{File,Twitch,YouTube}/...
  anomalous/jammer/{File,Twitch,YouTube}/...
  anomalous/synthetic/{Zone_A,Zone_B,Zone_C}/{File,Twitch,YouTube}/...
```

- 21 normal recordings, 3 jammer recordings, 9 synthetic recordings (33 total).
- `chunked.jsonl`: one JSON object (record) per 128-step window (12.8 s @ 10 Hz).
- `raw/metrics.csv`: raw per-step rows (9 999 rows for synthetic bursts,
  99 999 rows for normal/mobile, etc.) used only for the early-warning analysis.

## Record schema (`chunked.jsonl`)

| key                        | type               | notes |
|----------------------------|--------------------|-------|
| `start_time` / `end_time`  | ISO-8601 timestamp | chunk extent |
| `sampling_rate`            | int                | 10 Hz (verified) |
| `KPIs`                     | dict[str, list]    | 18 channels x 128 steps |
| `description`              | str                | NOT used as model input |
| `anomalies`                | dict               | `exists, type, anomaly_duration{start,end}, affected_kpis[], troubleshooting_tickets` |
| `statistics`               | dict               | baseline feature aggregations — NOT used as model input (only reference) |
| `labels`                   | dict               | zone / application / mobility / congestion |
| `QnA`                      | list               | conversational texts — NOT model input |

## KPI channels (18)

16 numeric/continuous + 2 categorical protocol codes:

`RSRP, DL_BLER, DL_MCS, UL_BLER, UL_MCS, UL_NPRB, UL_SNR, TX_Bytes,
RX_Bytes, Estimated_UL_Buffer, PRBs_DL_Current, PRBs_UL_Current,
PRB_Utilization_DL, PRB_Utilization_UL, UL_NumberOfPackets, DL_NumberOfPackets,
UL_Protocol (code), DL_Protocol (code)`.

- No missing KPI values across all 32 000 chunks (73 728 000 values).
- Categorical channels are re-encoded by a global codebook
  (`data/interim/kpi_names.json`) in `src/data/loader.py`.

## Anomaly labels

- 32 000 chunks total; **1 235 anomalous (3.86 %)**, 30 765 normal.
- Each anomalous chunk has exactly one `anomaly_type`; `anomaly_duration` is
  chunk-relative: 873 chunks start at step 0, **362 chunks have onset step > 0**
  (usable for early-warning evaluation); jammer chunks always start at step 0.
- 11 types (by chunk count):

```
Jamming                                    279
Co-Channel Interference (Mild)             146
Buffer Overflow (Gradual Buildup)          138
Co-Channel Interference (Severe)           114
Antenna Failure                            107
Faulty RF Filters (Temporal)               100
High Network Congestion (Gradual Buildup)   93
Doppler Shift (Severe)                      86
Faulty Handover Algorithm (Too Frequent)    64
Resource Allocation Bugs                    60
High Network Congestion (Sudden Spike)      48
```

- **`affected_kpis` is a per-type constant template** (exactly one variant per
  type across the whole dataset). KPI-level root-cause "ground truth" is
  therefore type-derived *by construction*. This is a documented limitation
  (see `docs/data_leakage.md`, `docs/limitations.md`) and motivates: (re-1) the
  type-as-target formulation, and (re-2) input-driven RCA evaluation that does
  not use the template.

## Structural facts (verified)

- Sampling rate 10 Hz; chunk = 128 steps = 12.8 s.
- Anomalous chunks form overlapping **burst clusters** within synthetic
  recordings (stride 32-101 steps); e.g. `anomalous/synthetic/Zone_A/File`
  has 66 chunks across 6 clusters. Normal recordings are densely overlapping
  (stride ~3.3 s).
- Recording purity: **no recording mixes normal and anomalous chunks** and
  **no chunk duplicates** exist (`(recording, start_time)` unique) — the
  anomaly flag is constant within every recording. This makes recording origin
  a potential confound (see `docs/data_leakage.md`).
- Each anomaly type occurs in 3-8 distinct recordings.
- Exactly one cross-file time overlap exists (anomalous/jammer/File vs
  normal/.../YouTube, both 2025-01-01); recordings are treated as independent.

## Processed artifacts

| artifact                     | content |
|------------------------------|---------|
| `data/interim/chunks_X.npy`  | float32 `[32000, 128, 18]` |
| `data/interim/chunks_meta.parquet` | 14-column metadata frame |
| `data/processed/split_indices.npz` | train (22 278) / val (4 683) / test (4 768) chunk ids |
| `data/processed/scaler.json` | per-channel loc/scale fit on train only |
| `data/processed/targets.npz` | anomaly/type/affected/onset/early-warning targets |
| `data/processed/rca_templates.json` | type -> affected-KPI template mapping |
| `data/processed/split_stats.json` | split audit incl. dropped chunks/clusters |

271 chunks are dropped at split boundaries (straddling chunks/clusters).