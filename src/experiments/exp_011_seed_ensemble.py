"""exp_011: seed-ensemble + hybrid for the anomaly head.

Mean-votes the TAGN anomaly logits across the 5 CI-sweep checkpoints
(exp_002 seeds 42/123/456/789/1337) and blends the result with the classical
logistic reference (exp_009 reference path, deterministic solver). Inference
only; threshold-free metrics from the shared summarize().

Usage: python -m src.experiments.exp_011_seed_ensemble --seeds 42 123 \
        456 789 1337 --weight 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import summarize  # noqa: E402
from src.experiments.exp_002_tagn import load_splits  # noqa: E402
from src.experiments.exp_009_hybrid import model_scores  # noqa: E402
from src.models.tagn import TagnModel, chunk_summary, load_graphs, model_config  # noqa: E402

RES = ROOT / "outputs" / "results"
CKPT = ROOT / "checkpoints" / "exp_002"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1337])
    ap.add_argument("--weight", type=float, default=0.5)
    args = ap.parse_args()

    cfg = model_config()
    frames, _, _ = load_splits(cfg)
    device = cfg["device"]
    y = {name: frames[name]["ano"].numpy() for name in ("val", "test")}

    per_seed = {s: {"tagn": {}, "ref": {}} for s in args.seeds}
    ens = {"val": [], "test": []}
    for seed in args.seeds:
        model = TagnModel(cfg).to(device)
        model.set_prototypes(load_graphs()[0].to(device))
        ckpt = CKPT / f"tagn_seed{seed}.pt"
        if not ckpt.exists():
            raise SystemExit(f"missing checkpoint: {ckpt}")
        model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
        model.eval()
        with torch.no_grad():
            ss = {
                name: ((chunk_summary(frames[name]["X"].to(device)) - model.summary_loc)
                       / model.summary_scale).cpu().numpy()
                for name in ("train", "val", "test")
            }
        clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000, n_jobs=-1)
        clf.fit(ss["train"], frames["train"]["ano"].numpy())
        lrn = {name: model_scores(model, frames[name]) for name in ("val", "test")}
        ref = {name: clf.decision_function(ss[name]) for name in ("val", "test")}
        for k in ("val", "test"):
            per_seed[seed]["tagn"][k] = summarize(y[k], lrn[k])
            per_seed[seed]["ref"][k] = summarize(y[k], ref[k])
            ens[k].append(lrn[k])
        print(f"  seed {seed}: TAGN test F1max {per_seed[seed]['tagn']['test']['f1_max']:.4f}")

    core = args.seeds[0]
    ens_blend = {k: np.mean(ens[k], axis=0) for k in ("val", "test")}

    # re-fit the reference deterministically on the core seed's stats so the
    # blend uses the same reference as exp_009
    model0 = TagnModel(cfg).to(device)
    model0.set_prototypes(load_graphs()[0].to(device))
    model0.load_state_dict(torch.load(CKPT / f"tagn_seed{core}.pt", map_location=device), strict=False)
    model0.eval()
    with torch.no_grad():
        ss0 = {
            name: ((chunk_summary(frames[name]["X"].to(device)) - model0.summary_loc)
                   / model0.summary_scale).cpu().numpy()
            for name in ("train", "val", "test")
        }
    ref0 = LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000, n_jobs=-1)
    ref0.fit(ss0["train"], frames["train"]["ano"].numpy())

    w = float(np.clip(args.weight, 0.0, 1.0))
    out = {
        "seeds": args.seeds,
        "weight": w,
        "per_seed": {str(s): {k: {"tagn": per_seed[s]["tagn"][k], "ref": per_seed[s]["ref"][k]}
                              for k in ("val", "test")} for s in args.seeds},
    }
    for k in ("val", "test"):
        out[k] = {
            "ensemble": summarize(y[k], ens_blend[k]),
            "ref": summarize(y[k], ref0.decision_function(ss0[k])),
            "hybrid": summarize(y[k], w * ens_blend[k] + (1.0 - w) * ref0.decision_function(ss0[k])),
        }

    # val-selected fusion weight (no test peeking): choose w maximising val
    # max-F1, then report the test operating point at that w. Also record the
    # test-optimal w as an upper bound, explicitly flagged as not deployable.
    ref_val = ref0.decision_function(ss0["val"])
    ref_test = ref0.decision_function(ss0["test"])
    grid = np.round(np.arange(0.0, 1.0001, 0.1), 2)
    val_f1 = {float(g): summarize(y["val"], g * ens_blend["val"] + (1 - g) * ref_val)["f1_max"] for g in grid}
    w_star = max(val_f1, key=val_f1.get)
    test_at_wstar = summarize(y["test"], w_star * ens_blend["test"] + (1 - w_star) * ref_test)
    test_f1 = {float(g): summarize(y["test"], g * ens_blend["test"] + (1 - g) * ref_test)["f1_max"] for g in grid}
    w_test = max(test_f1, key=test_f1.get)
    out["weight_sweep"] = {
        "grid": [float(g) for g in grid],
        "val_f1_max": val_f1,
        "test_f1_max": test_f1,
        "w_selected_on_val": w_star,
        "test_at_w_selected_on_val": test_at_wstar,
        "w_optimal_on_test_LEAKY": w_test,
        "test_f1_at_w_optimal_on_test_LEAKY": test_f1[w_test],
    }
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "exp_011_seed_ensemble.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["test"], indent=2))


if __name__ == "__main__":
    main()