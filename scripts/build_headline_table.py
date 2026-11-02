"""Headline comparison table: Setting-I detection + recording-held-out (LORO).

Merges the leak-free per-model test metrics with their cluster-bootstrap 95%
intervals (finding M5) into a single paper-ready table.

Sources:
  outputs/results/bootstrap_ci.json             (TAGN, per seed)
  outputs/results/bootstrap_ci_classical.json   (exp_001 classical baselines)
  outputs/results/exp_013_{model}_seed*.json    (TCN / GRU)
  outputs/results/exp_005_loro_seed*.json       (LORO, per fold)

Outputs:
  outputs/tables/headline_v3.md
  outputs/tables/headline_v3.csv

Usage:
    python -m scripts.build_headline_table
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "outputs" / "results"
TABLES = ROOT / "outputs" / "tables"


def _load(name: str) -> dict | None:
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def _sm(vals: list[float]) -> str:
    if not vals:
        return "MISSING"
    if len(vals) == 1:
        return f"{vals[0]:.4f}"
    return f"{np.mean(vals):.4f} ± {np.std(vals, ddof=1):.4f}"


def _envelope(boots: list[dict]) -> tuple[float, float]:
    return (min(b["lo"] for b in boots), max(b["hi"] for b in boots))


def setting1_rows() -> list[dict]:
    rows: list[dict] = []

    bc = _load("bootstrap_ci.json")
    if bc:
        seeds = sorted(bc, key=int)
        pr = [bc[s]["bootstrap"]["pr_auc"]["point"] for s in seeds]
        roc = [bc[s]["bootstrap"]["roc_auc"]["point"] for s in seeds]
        pr_lo, pr_hi = _envelope([bc[s]["bootstrap"]["pr_auc"] for s in seeds])
        roc_lo, roc_hi = _envelope([bc[s]["bootstrap"]["roc_auc"] for s in seeds])
        rows.append({
            "model": "TAGN (graph + summary)", "seeds": len(seeds),
            "pr": _sm(pr), "pr_ci": f"[{pr_lo:.4f}, {pr_hi:.4f}]",
            "roc": _sm(roc), "roc_ci": f"[{roc_lo:.4f}, {roc_hi:.4f}]",
        })

    cl = _load("bootstrap_ci_classical.json")
    order = ["logreg", "lof", "iforest", "ocsvm", "xgb"]
    labels = {
        "logreg": "LogReg (hand features)", "lof": "LOF (unsupervised)",
        "iforest": "IsolationForest (unsupervised)",
        "ocsvm": "OneClassSVM (unsupervised)", "xgb": "XGBoost (hand features)",
    }
    if cl:
        for name in order:
            if name not in cl["models"]:
                continue
            m = cl["models"][name]
            rows.append({
                "model": labels[name], "seeds": 1,
                "pr": f"{m['pr_auc']['point']:.4f}",
                "pr_ci": f"[{m['pr_auc']['lo']:.4f}, {m['pr_auc']['hi']:.4f}]",
                "roc": f"{m['roc_auc']['point']:.4f}",
                "roc_ci": f"[{m['roc_auc']['lo']:.4f}, {m['roc_auc']['hi']:.4f}]",
            })

    for model in ("tcn", "gru"):
        files = sorted(RES.glob(f"exp_013_{model}_seed*.json"))
        if not files:
            continue
        ds = [json.loads(f.read_text()) for f in files]
        pr = [d["test"]["pr_auc"] for d in ds]
        roc = [d["test"]["roc_auc"] for d in ds]
        pr_lo, pr_hi = _envelope([d["bootstrap"]["pr_auc"] for d in ds])
        roc_lo, roc_hi = _envelope([d["bootstrap"]["roc_auc"] for d in ds])
        rows.append({
            "model": f"{model.upper()} (temporal deep)", "seeds": len(ds),
            "pr": _sm(pr), "pr_ci": f"[{pr_lo:.4f}, {pr_hi:.4f}]",
            "roc": _sm(roc), "roc_ci": f"[{roc_lo:.4f}, {roc_hi:.4f}]",
        })
    return rows


def setting2_rows() -> list[dict]:
    rows = []
    for f in sorted(RES.glob("exp_005_loro_seed*.json")):
        d = json.loads(f.read_text())
        folds = d["folds"]
        pr = [x["test_anomaly"]["pr_auc"] for x in folds]
        roc = [x["test_anomaly"]["roc_auc"] for x in folds]
        rows.append({
            "model": f"TAGN (seed {f.stem.split('seed')[-1]})",
            "protocol": "leave-recordings-out (5 folds)",
            "pr": _sm(pr), "pr_range": f"[{min(pr):.4f}, {max(pr):.4f}]",
            "roc": _sm(roc), "roc_range": f"[{min(roc):.4f}, {max(roc):.4f}]",
        })
    return rows


def ablation_rows() -> list[dict]:
    rows = []
    for f in sorted(RES.glob("exp_003_ablation_*.json")):
        d = json.loads(f.read_text())
        if "test" not in d or "variant" not in d:
            continue
        anchor = d.get("anchor", {}).get("test", {}).get("pr_auc")
        rows.append({
            "variant": d["variant"],
            "anchor_pr": f"{anchor:.4f}" if anchor is not None else "-",
            "pr": f"{d['test'].get('anomaly', {}).get('pr_auc', float('nan')):.4f}",
            "roc": f"{d['test'].get('anomaly', {}).get('roc_auc', float('nan')):.4f}",
            "type_acc": f"{d['test'].get('type', {}).get('accuracy', float('nan')):.4f}",
            "rca_jac": f"{d['test'].get('rca', {}).get('type_template_jaccard', float('nan')):.4f}",
        })
    order = {v: i for i, v in enumerate(
        ["full", "no_graph", "static_graph", "undirected", "no_nodeconv", "hazard_only"])}
    rows.sort(key=lambda r: order.get(r["variant"], 99))
    return rows


def paired_rows() -> list[dict]:
    p = RES / "paired_test.json"
    if not p.exists():
        return []
    d = json.loads(p.read_text())
    rows = []
    for seed in sorted(d["seeds"], key=int):
        s = d["seeds"][seed]
        dpr, droc = s["dPR"], d["seeds"][seed]["dROC"]
        rows.append({
            "baseline": d["baseline"], "seed": seed,
            "dpr": f"{dpr['point']:+.4f} [{dpr['lo']:+.4f}, {dpr['hi']:+.4f}]",
            "p": f"{dpr['frac_le_0']:.4f}",
            "droc": f"{droc['point']:+.4f} [{droc['lo']:+.4f}, {droc['hi']:+.4f}]",
        })
    return rows


def main() -> None:
    s1, s2 = setting1_rows(), setting2_rows()
    lines = [
        "# Headline benchmark — v3 (leak-free)", "",
        "Setting I — pooled window, validation-tuned threshold. CIs are 95%",
        "*cluster-bootstrap* intervals (recordings resampled with replacement);",
        "for multi-seed models the interval shown is the union of per-seed intervals.",
        "",
        "| model | seeds | PR-AUC (mean ± sd) | PR 95% CI | ROC-AUC (mean ± sd) | ROC 95% CI |",
        "|---|---|---|---|---|---|",
    ]
    for r in setting1_rows():
        lines.append(f"| {r['model']} | {r['seeds']} | {r['pr']} | {r['pr_ci']} | "
                     f"{r['roc']} | {r['roc_ci']} |")

    lines += ["", "## Setting II — leave-recordings-out (honest generalization)", "",
              "| model | protocol | PR-AUC (mean ± sd over folds) | fold range | "
              "ROC-AUC (mean ± sd) | fold range |",
              "|---|---|---|---|---|---|"]
    for r in setting2_rows():
        lines.append(f"| {r['model']} | {r['protocol']} | {r['pr']} | {r['pr_range']} | "
                     f"{r['roc']} | {r['roc_range']} |")

    abl = ablation_rows()
    if abl:
        lines += ["", "## Component ablation (exp_003, seed 42, shared stage-A anchor)",
                  "", "All variants share the summary anchor, split, seed and optimiser.",
                  "`anchor PR` is the frozen summary readout before stage-B trunk training.",
                  "",
                  "| variant | anchor PR | test PR-AUC | test ROC-AUC | type acc | RCA jaccard |",
                  "|---|---|---|---|---|---|"]
        for r in abl:
            lines.append(f"| {r['variant']} | {r['anchor_pr']} | {r['pr']} | {r['roc']} | "
                         f"{r['type_acc']} | {r['rca_jac']} |")

    pr = paired_rows()
    if pr:
        import json as _json

        base = json.loads((RES / "paired_test.json").read_text())["baseline"]
        lines += ["", f"## Paired cluster bootstrap vs `{base}` (same test chunks)", "",
                  "dPR = PR(TAGN) − PR(baseline) on shared resamples; `p` is the fraction",
                  "of resamples with dPR <= 0 (one-sided evidence for TAGN).", "",
                  "| seed | dPR (95% CI) | p | dROC |", "|---|---|---|---|"]
        for r in paired_rows():
            lines.append(f"| {r['seed']} | {r['dpr']} | {r['p']} | {r['droc']} |")

    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "headline_v3.md").write_text("\n".join(lines) + "\n")

    with open(TABLES / "headline_v3.csv", "w", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["setting", "model", "protocol", "seeds", "pr", "pr_ci",
                            "pr_range", "roc", "roc_ci", "roc_range"],
            extrasaction="ignore",
        )
        w.writeheader()
        for r in setting1_rows():
            w.writerow({"setting": "I", **r})
        for r in setting2_rows():
            w.writerow({"setting": "II", **r})

    print("\n".join(lines))
    print(f"\nwrote {TABLES / 'headline_v3.md'} and .csv")


if __name__ == "__main__":
    main()
