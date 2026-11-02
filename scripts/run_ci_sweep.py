"""5-seed confidence-interval sweep for the integrated TAGN + hybrid numbers.

For each seed in {42, 123, 456, 789, 1337}:
  1. train TAGN (exp_002) from scratch  -> exp_002_tagn_seed{S}.json + checkpoint
  2. hybrid TAGN+logreg (exp_009) reads that checkpoint -> exp_009_hybrid_seed{S}.json
  3. (exp_005 LORO is ~50 min/run; run separately when a GPU/idle window allows)

Writes outputs/results/ci_sweep_5seed.json aggregating mean/std (or min/max CI)
across seeds for the headline numbers (F1, PR-AUC, ROC-AUC).

Usage: python scripts/run_ci_sweep.py [--seeds 42 123 456 789 1337] [--loro]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats as _st

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "outputs" / "results"
PY = sys.executable


def run(cmd: list[str], tag: str, log) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {tag}: {' '.join(cmd)}\n"
    print(line, end="")
    log.write(line)
    log.flush()
    rc = subprocess.call(cmd, cwd=str(ROOT))
    line = f"[{time.strftime('%H:%M:%S')}] {tag} rc={rc}\n"
    print(line, end="")
    log.write(line)
    log.flush()
    if rc != 0:
        raise RuntimeError(f"{tag} failed with rc={rc}")


def stats_agg(key: str, rows: list[dict]) -> dict:
    """Aggregate a metric across seeds with a Student-t interval.

    Finding M6 (docs/benchmark_audit_v3.md): the previous code used a fixed
    z-quantile (1.645) for a 90% interval on n=5 seeds. With so few degrees of
    freedom that interval is far too narrow; we now use the exact t-quantile
    ``t_{0.95, n-1}`` and expose both 90% and 95% intervals.
    """
    vals = np.asarray([r[key] for r in rows if r.get(key) is not None])
    if len(vals) == 0:
        return {}
    n = len(vals)
    mean = float(vals.mean())
    sd = float(vals.std(ddof=1)) if n > 1 else 0.0
    sem = sd / n ** 0.5 if n > 1 else 0.0

    def _t(level: float) -> float:
        return float(_st.t.ppf(0.5 + level / 2.0, df=n - 1)) if n > 1 else 0.0

    return {
        "mean": round(mean, 4),
        "std": round(sd, 4),
        "min": round(float(vals.min()), 4),
        "max": round(float(vals.max()), 4),
        "ci90_low": round(mean - _t(0.90) * sem, 4),
        "ci90_high": round(mean + _t(0.90) * sem, 4),
        "ci95_low": round(mean - _t(0.95) * sem, 4),
        "ci95_high": round(mean + _t(0.95) * sem, 4),
        "t_quantile_90": round(_t(0.90), 4),
        "t_quantile_95": round(_t(0.95), 4),
        "n": int(n),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1337])
    ap.add_argument("--loro", action="store_true", help="also run the expensive LORO (exp_005) sweep")
    args = ap.parse_args()

    log_path = RES / "ci_sweep_log.txt"
    log = open(log_path, "a")
    log.write(f"\n=== sweep started {time.strftime('%Y-%m-%d %H:%M:%S')} seeds={args.seeds} loro={args.loro}\n")

    collected = []
    for seed in args.seeds:
        t0 = time.time()
        run([PY, "-m", "src.experiments.exp_002_tagn", "--seed", str(seed)],
            f"exp_002 seed {seed}", log)
        run([PY, "-m", "src.experiments.exp_009_hybrid", "--seed", str(seed)],
            f"exp_009 seed {seed}", log)
        if args.loro:
            run([PY, "-m", "src.experiments.exp_005_loro", "--seed", str(seed)],
                f"exp_005 seed {seed}", log)
        d2 = json.loads((RES / f"exp_002_tagn_seed{seed}.json").read_text())
        d9 = json.loads((RES / f"exp_009_hybrid_seed{seed}.json").read_text())
        row = {
            "seed": seed,
            "elapsed_s": round(time.time() - t0, 1),
            "tagn_test": {k: d2["test"]["anomaly"][k] for k in ("f1_max", "pr_auc", "roc_auc", "threshold_val_tuned")},
            "hybrid_test": {k: d9["test"]["hybrid"][k] for k in ("f1_max", "pr_auc", "roc_auc")},
        }
        collected.append(row)
        print(f"  seed {seed}: TAGN {row['tagn_test']} hybrid {row['hybrid_test']}", flush=True)
        log.write(json.dumps(row) + "\n")
        log.flush()

    tags = {
        "tagn_f1_max": ("tagn_test", "f1_max"),
        "tagn_pr_auc": ("tagn_test", "pr_auc"),
        "tagn_roc_auc": ("tagn_test", "roc_auc"),
        "hybrid_f1_max": ("hybrid_test", "f1_max"),
        "hybrid_pr_auc": ("hybrid_test", "pr_auc"),
        "hybrid_roc_auc": ("hybrid_test", "roc_auc"),
    }
    aggregate = {}
    for tag, (grp, metric) in tags.items():
        aggregate[tag] = stats_agg(metric, [r[grp] for r in collected])
    summary = {"seeds": args.seeds, "per_seed": collected, "aggregate": aggregate}
    (RES / "ci_sweep_5seed.json").write_text(json.dumps(summary, indent=2))
    log.write(f"=== sweep done {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log.close()
    print("\nsummary ->", RES / "ci_sweep_5seed.json")
    for k, g in summary["aggregate"].items():
        print(f"  {k:18s} {g}")


if __name__ == "__main__":
    main()