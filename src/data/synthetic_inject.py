"""Track 2 - synthetic fault injection (src/data/synthetic_inject.py).

Re-implements the TelecomTS synthetic anomaly pipeline from arXiv:2510.06063
Appendix C:
  * function classes   -> the paper's 8 function types (Table 9),
  * affected KPIs      -> templates recovered from the shipped dataset itself,
  * parameter ranges   -> our own (uniformly sampled per event, disjoint from
                          the original data; the paper never publishes ranges),
  * boundaries         -> soft bounds + saturation noise (Appendix C),
  * windowing          -> length 128, stride 32, drop all-normal and
                          multi-type windows (paragraph "Anomaly Dataset").

Output files are written in the exact JSON shape of the TelecomTS
``chunked.jsonl`` so ``src/data/loader.py`` auto-discovers them.

Note: we do NOT claim bit-level reproduction of the original data (the paper
publishes neither the per-KPI mappings nor the numeric ranges). ``payload()``
returns per-event direction targets so the validation gate in
``scripts/generate_synthetic.py`` can confirm the generated anomalies shift
their affected KPIs in the intended direction.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "synthetic.yaml"

# Canonical channel order (= src/data/loader.py NUMERIC_KPIS).
KPI_ORDER = [
    "RSRP", "DL_BLER", "DL_MCS", "UL_BLER", "UL_MCS", "UL_NPRB", "UL_SNR",
    "TX_Bytes", "RX_Bytes", "Estimated_UL_Buffer", "PRBs_DL_Current",
    "PRBs_UL_Current", "PRB_Utilization_DL", "PRB_Utilization_UL",
    "UL_NumberOfPackets", "DL_NumberOfPackets",
]
PROTOCOL_COLS = ["UL_Protocol", "DL_Protocol"]

TWO_PI = 2.0 * math.pi

# Tolerance for deciding whether an injected effect observably changed a KPI
# (G2 fix): |post - base| must exceed this (relative to the KPI magnitude)
# for the KPI to count as affected in a window.
_OBS_TOL = 1e-9


# ---------------------------------------------------------------------------
# Function classes (paper Table 9)
# ---------------------------------------------------------------------------

def _apply_func(func: str, x: np.ndarray, u: np.ndarray, p: dict) -> np.ndarray:
    if func == "constant_shift":
        return x + p["delta"]
    if func == "constant_scale":
        return x * p["factor"]
    if func == "linear_growth":
        return x + p["delta"] * u
    if func == "exponential_growth":
        return x * np.exp(p["rate"] * u)
    if func == "logistic_growth":
        return x + p["L"] / (1.0 + np.exp(-p["k"] * (u - 0.5)))
    if func == "sinusoidal_add":
        return x + p["amp"] * np.sin(TWO_PI * p["freq"] * u + p["phase"])
    if func == "sinusoidal_scale":
        return x * (1.0 + p["amp"] * np.sin(TWO_PI * p["freq"] * u + p["phase"]))
    raise ValueError(func)


# Expected post-vs-pre mean change direction from SAMPLED parameters.
# sinusoidal types have no monotone guarantee -> returned as 0 (variance-only).
def _direction_from_sampled(func: str, p: dict) -> int:
    if func == "constant_shift":
        return int(np.sign(p["delta"]))
    if func == "linear_growth":
        return int(np.sign(p["delta"]))
    if func == "exponential_growth":
        return int(np.sign(p["rate"]))
    if func == "constant_scale":
        return int(np.sign(p["factor"] - 1.0))
    if func == "logistic_growth":
        return int(np.sign(p["L"]))
    return 0


def _params_zeroed_to_one(effect: dict, cut: dict) -> bool:
    """True when the sampled params collapse to the identity transform."""
    f = effect["func"]
    if f == "constant_shift":
        return abs(cut["delta"]) < 1e-12
    if f == "constant_scale":
        return abs(cut["factor"] - 1.0) < 1e-12
    if f == "linear_growth":
        return abs(cut["delta"]) < 1e-12
    if f == "exponential_growth":
        return abs(cut["rate"]) < 1e-12
    if f == "logistic_growth":
        return abs(cut["L"]) < 1e-12
    return False


# ---------------------------------------------------------------------------
# Event schedule
# ---------------------------------------------------------------------------

@dataclass
class Effect:
    kpi: str
    func: str
    params: dict                      # name -> sampled value
    expected_dir: int                 # -1, 0, +1
    base_inert: bool = False          # all-zero base -> scale/exp is invisible


@dataclass
class Event:
    anomaly_type: str
    start: int                        # raw step, inclusive
    end: int                          # raw step, inclusive
    effects: list[Effect] = field(default_factory=list)


def sample_params_for(effect: dict, rng: np.random.Generator,
                      shared_keys: set[str],
                      shared: dict | None) -> tuple[dict, dict | None]:
    """Uniformly sample parameter values for one effect occurrence.

    ``shared_keys`` are param names drawn once per event and reused across all
    effects of that event (config ``functions.<f>.shared``, e.g. sinusoid freq
    so oscillatory KPIs share a common cycle). Returns (sampled, shared).
    """
    spec = effect["params"]
    out: dict = {}
    for name, (lo, hi) in spec.items():
        if name in shared_keys and shared is not None and name in shared:
            out[name] = shared[name]
        else:
            out[name] = float(rng.uniform(lo, hi))
    if shared is not None:
        for name in shared_keys:
            shared[name] = out[name]
    return out, shared


def make_schedule(tr: dict, n_steps: int, rng: np.random.Generator,
                  types: list[dict]) -> list[Event]:
    sched = tr["schedule"]
    mean_arrival = sched["inter_arrival_mean_steps"]
    mean_duration = sched["duration_mean_steps"]
    min_dur = sched.get("min_duration_steps", 48)
    max_events = sched.get("max_events", 80)

    # event types are drawn WITHOUT replacement from a shuffled pool that
    # repeats the full type list, so each recording gets distinct types and a
    # batch of recordings covers every anomaly type at least once.
    n_types = len(types)
    repeats = int(np.ceil(max_events / n_types))
    pool = np.repeat(np.arange(n_types, dtype=int), repeats)
    rng.shuffle(pool)
    pool_i = 0

    events: list[Event] = []
    t = float(rng.exponential(mean_arrival))
    if t >= 0.7 * n_steps:           # first arrival beyond the trace -> dead recording
        t = float(rng.uniform(0.05, 0.2 * n_steps))
    while len(events) < max_events:
        dur = max(min_dur, float(rng.exponential(mean_duration)))
        if t + dur > n_steps:
            break
        start = int(round(t))
        events.append(
            Event(
                anomaly_type=str(int(pool[pool_i])),
                start=start,
                end=start + int(round(dur)),
            )
        )
        pool_i += 1
        t = start + dur + float(rng.exponential(mean_arrival))
    return events


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------

@dataclass
class InjectedRecording:
    chunks: list[dict]                # chunked.jsonl records
    events: list[Event]               # resolved events (with effects)
    raw_x: np.ndarray                 # transformed numeric matrix [T,16]
    raw_num: np.ndarray               # pre-transform numeric matrix [T,16]
    raw_t0: str
    labels: dict
    per_event_targets: list[dict]     # validation targets (window index, kpi, dir)
    windows_dropped_all_normal: int
    windows_dropped_multi_type: int


def _parse_labels(source_rel: Path) -> dict:
    parts = [p for p in source_rel.parts if p not in ("raw", "metrics.csv")]
    labels = {
        "zone": None, "application": None, "mobility": None, "congestion": None,
    }
    for p in parts:
        if p.startswith("Zone_"):
            labels["zone"] = p[-1]
        elif p == "mobile":
            labels["mobility"] = "Yes"
        elif p == "stationary":
            labels["mobility"] = "No"
        elif p == "congestion":
            labels["congestion"] = "Yes"
        elif p == "no_congestion":
            labels["congestion"] = "No"
        else:
            labels["application"] = p
    labels["anomaly_present"] = "Yes"
    return labels


def _statistics_for(values: np.ndarray) -> dict:
    mean = float(values.mean())
    var = float(values.var()) if values.size else 0.0
    if var < 1e-12:
        return {"mean": mean, "variance": 0.0, "trend": 0, "periodicity": 1}
    x = np.arange(values.size)
    slope = np.polyfit(x, values, 1)[0]
    trend = int(np.sign(slope))
    spec = np.abs(np.fft.rfft(values - mean))
    spec[0] = 0.0
    k = int(np.argmax(spec))
    period = max(1, int(round(values.size / k))) if k > 0 else 1
    return {
        "mean": mean,
        "variance": var,
        "trend": trend,
        "periodicity": int(period),
    }


def _description(rec: dict, anom_type: str, ev_start: int, ev_end: int,
                 effect_kpis: list[str]) -> str:
    d = rec["anomalies"]["anomaly_duration"]
    return (
        f"Injected {anom_type}: anomaly interval [{ev_start}, {ev_end}] raw steps; "
        f"window onset_step={d['start']}, offset_step={d['end']}; "
        f"affected KPIs: {', '.join(effect_kpis)}."
    )


def _observable_kpis(
    num: np.ndarray, transformed: np.ndarray, events: list[Event],
    w0: int, w_end: int,
) -> list[str]:
    """KPIs whose value *actually* changed inside the window (G2/G3 fix).

    A KPI is listed as affected only when the post-clip transformed segment
    differs from the pre-injection base. This excludes identity parameter draws,
    effects on an all-zero base (scale/exp is invisible), and effects fully
    clipped away by the boundary saturation. Effects from *every* overlapping
    event are pooled (multi-event windows were previously under-labelled by
    looking only at ``overlap[0]``).
    """
    out: set[str] = set()
    for ev in events:
        s = max(ev.start, w0)
        e = min(ev.end, w_end)
        if e < s:
            continue
        for fx in ev.effects:
            col = KPI_ORDER.index(fx.kpi)
            base = num[s : e + 1, col]
            post = transformed[s : e + 1, col]
            tol = _OBS_TOL * max(1.0, float(np.abs(base).max()))
            if float(np.max(np.abs(post - base))) > tol:
                out.add(fx.kpi)
    return sorted(out, key=KPI_ORDER.index)


def inject_recording(
    metrics_csv: Path,
    cfg: dict,
    rng: np.random.Generator,
    source_rel: Path,
) -> InjectedRecording:
    """Inject an anomaly schedule into one normal raw metrics.csv."""
    tr = cfg["track2"]
    boundaries = tr["boundaries"]
    functions = tr["functions"]
    type_specs = tr["anomalies"]

    df = pd.read_csv(metrics_csv, dtype={c: str for c in PROTOCOL_COLS})
    t0 = str(df["timestamp"].iloc[0])
    num = df[KPI_ORDER].astype(float).to_numpy(dtype=np.float64)
    # missing protocol cells -> the original dataset uses the string "None"
    protos = {
        c: df[c].fillna("None").astype(str).tolist() for c in PROTOCOL_COLS
    }
    n_steps, n_k = num.shape

    # ---- construct events with sampled per-type effects ----
    events = make_schedule(tr, n_steps, rng, type_specs)
    for ev in events:
        spec = type_specs[int(ev.anomaly_type)]
        ev.anomaly_type = spec["name"]
        # instance-varying affected-KPI subset: a real fault hits a coherent but
        # not type-constant subset of the type's symptom KPIs. Sampling a random
        # subset per event is what breaks the per-type-template shortcut in the
        # original data and yields non-template KPI-level ground truth.
        n_effects = len(spec["effects"])
        if n_effects <= 1:
            active = [0]
        else:
            lo = max(1, int(n_effects * 0.5))
            keep = int(rng.integers(lo, n_effects + 1))
            active = sorted(rng.choice(n_effects, keep, replace=False).tolist())

        shared: dict = {}
        for i_eff in active:
            effect = spec["effects"][int(i_eff)]
            fname = effect["func"]
            shared_keys = set(functions[fname].get("shared", []))
            # merge function-level defaults (e.g. sinusoid phase) into the
            # effect's explicit parameter ranges
            merged = dict(effect["params"])
            for k, v in functions[fname].get("params", {}).items():
                if k not in merged:
                    merged[k] = v
            params, shared = sample_params_for(
                {**effect, "params": merged}, rng, shared_keys,
                shared if shared_keys else None,
            )
            base_seg = num[ev.start : ev.end + 1, KPI_ORDER.index(effect["kpi"])]
            base_inert = (
                base_seg.std() < 1e-12
                and fname in ("constant_scale", "exponential_growth",
                              "sinusoidal_scale")
            )
            ev.effects.append(
                Effect(
                    kpi=effect["kpi"],
                    func=fname,
                    params=params,
                    expected_dir=(
                        _direction_from_sampled(fname, params)
                        if not _params_zeroed_to_one(effect, params)
                        else 0
                    ),
                    base_inert=base_inert,
                )
            )

    # ---- apply transforms ----
    transformed = num.copy()
    for ev in events:
        s, e = ev.start, ev.end
        u = np.linspace(0.0, 1.0, e - s + 1)
        for fx in ev.effects:
            col = KPI_ORDER.index(fx.kpi)
            transformed[s : e + 1, col] = _apply_func(
                fx.func, transformed[s : e + 1, col], u, fx.params
            )

    # ---- boundary clipping + saturation noise ----
    hi_v = np.array([boundaries[k][1] for k in KPI_ORDER], dtype=float)
    lo_v = np.array([boundaries[k][0] for k in KPI_ORDER], dtype=float)
    sat_before = (num > hi_v[None, :]) | (num < lo_v[None, :])   # clipped points
    transformed = np.clip(transformed, lo_v[None, :], hi_v[None, :])
    sat_mask = sat_before & (
        (np.abs(transformed - hi_v[None, :]) < 1e-9)
        | (np.abs(transformed - lo_v[None, :]) < 1e-9)
    )
    n_sat = int(sat_mask.sum())
    if n_sat:
        transformed[sat_mask] += rng.normal(0.0, 0.05, n_sat)
        transformed = np.clip(transformed, lo_v[None, :], hi_v[None, :])

    # ---- windows (length 128, stride 32); keep single-type anomalous only ----
    wlen, stride = tr["window_length"], tr["stride"]
    n_windows = (n_steps - wlen) // stride + 1

    def window_events(w0: int) -> list[Event]:
        w_end = w0 + wlen - 1
        return [ev for ev in events if not (ev.end < w0 or ev.start > w_end)]

    chunks: list[dict] = []
    targets: list[dict] = []
    dropped_normal = 0
    dropped_multi = 0

    for w_i in range(n_windows):
        w0 = w_i * stride
        w_end = w0 + wlen - 1
        overlap = window_events(w0)
        if not overlap:
            dropped_normal += 1
            continue
        types_in = {ev.anomaly_type for ev in overlap}
        if len(types_in) > 1:
            dropped_multi += 1
            continue
        ev = overlap[0]  # representative; all overlap events share one type
        # G3: union the interval and affected KPIs across *all* overlapping
        # (same-type) events, not just the first one.
        ev_start = min(e.start for e in overlap)
        ev_end = max(e.end for e in overlap)
        start = max(0, ev_start - w0)
        end = min(wlen - 1, ev_end - w0)
        effect_kpis = _observable_kpis(num, transformed, overlap, w0, w_end)

        rec = {
            "start_time": _shift_time(t0, w0),
            "end_time": _shift_time(t0, w0 + (wlen - 1)),  # G4: was 10x too large
            "sampling_rate": tr["sampling_rate"],
            "KPIs": {},
            "description": "",
            "anomalies": {
                "exists": True,
                "type": ev.anomaly_type,
                "anomaly_duration": {"start": int(start), "end": int(end)},
                "affected_kpis": list(effect_kpis),
                "troubleshooting_tickets": "",
            },
            "statistics": {},
            "labels": _parse_labels(source_rel),
            "QnA": [],
        }
        rec["KPIs"] = {
            k: [float(v) for v in transformed[w0 : w0 + wlen, j]]
            for j, k in enumerate(KPI_ORDER)
        }
        rec["KPIs"].update(
            {c: protos[c][w0 : w0 + wlen] for c in PROTOCOL_COLS}
        )
        rec["description"] = _description(rec, ev.anomaly_type, ev_start, ev_end,
                                         effect_kpis)
        rec["statistics"] = {
            k: _statistics_for(transformed[w0 : w0 + wlen, j])
            for j, k in enumerate(KPI_ORDER)
        }

        chunks.append(rec)
        # G3: emit validation targets for every overlapping event's effects.
        for e in overlap:
            for fx in e.effects:
                targets.append(
                    {
                        "window": w_i,
                        "kpi": fx.kpi,
                        "expected_dir": fx.expected_dir,
                        "base_inert": fx.base_inert,
                        "event": e.anomaly_type,
                        "ev_start": ev.start,
                        "ev_end": ev.end,
                    }
                )

    return InjectedRecording(
        chunks=chunks,
        events=events,
        raw_x=transformed,
        raw_num=num,
        raw_t0=t0,
        labels=_parse_labels(source_rel),
        per_event_targets=targets,
        windows_dropped_all_normal=dropped_normal,
        windows_dropped_multi_type=dropped_multi,
    )


def _shift_time(t0: str, step: int) -> str:
    # t0 like "2025-06-18 00:40:09.600"; +step steps @ 10 Hz = CS X 100 ms
    dt = pd.Timestamp(t0) + pd.Timedelta(milliseconds=100 * step)
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def write_recording(out_dir: Path, rec: InjectedRecording, metrics_csv: Path) -> None:
    """Write the same layout as the shipped dataset:
    <out>/processed/chunked.jsonl, <out>/raw/metrics.csv, <out>/raw/description.txt.
    """
    proc = out_dir / "processed"
    raw = out_dir / "raw"
    proc.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)

    with open(proc / "chunked.jsonl", "w") as fh:
        for c in rec.chunks:
            fh.write(json.dumps(c) + "\n")

    src = pd.read_csv(metrics_csv)
    out_df = src.copy()
    for j, k in enumerate(KPI_ORDER):
        out_df[k] = rec.raw_x[:, j]
    out_df.to_csv(raw / "metrics.csv", index=False)

    desc = "\n".join(
        [f"Generated synthetic anomaly recording (Track 2).",
         f"Source normal recording: {metrics_csv}",
         f"Anomalies: " + "; ".join(
             f"{e.anomaly_type}[{e.start}:{e.end}]" for e in rec.events
         ),
         f"Anomalous windows kept: {len(rec.chunks)}",
         f"Dropped (all-normal): {rec.windows_dropped_all_normal}",
         f"Dropped (multi-type): {rec.windows_dropped_multi_type}",
         f"Parameters/seed disjoint from original TelecomTS synthetic data."]
    )
    (raw / "description.txt").write_text(desc + "\n")


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)