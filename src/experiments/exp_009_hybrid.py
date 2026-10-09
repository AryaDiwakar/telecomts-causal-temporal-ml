"""exp_009 - reproducible TAGN + classical expert ensemble.

The two experts see complementary views of the same window:

* TAGN's learned summary/residual anomaly score;
* a deterministic logistic expert fit on the train-only normalized summary.

The default 50/50 logit blend is fixed before test evaluation. This is an
ensemble experiment, not a replacement for the standalone TAGN result.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import summarize  # noqa: E402
from src.experiments.exp_002_tagn import load_splits  # noqa: E402
from src.models.tagn import TagnModel, chunk_summary, load_graphs, model_config  # noqa: E402

RES = ROOT / "outputs" / "results"


@torch.no_grad()
def model_scores(model: TagnModel, frame: dict, batch: int = 32) -> np.ndarray:
    model.eval()
    x = frame["X"].to(next(model.parameters()).device)
    proto = frame["proto"].to(x.device)
    scores = []
    for i in range(0, len(x), batch):
        scores.append(model(x[i:i + batch], proto[i:i + batch])[0]["anomaly"].cpu())
    return torch.cat(scores).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weight", type=float, default=0.5,
                    help="fixed TAGN logit weight; default is an equal expert blend")
    ap.add_argument("--checkpoint", type=str, default=None)
    args = ap.parse_args()

    cfg = model_config()
    frames, _, _ = load_splits(cfg)
    device = cfg["device"]
    model = TagnModel(cfg).to(device)
    model.set_prototypes(load_graphs()[0].to(device))
    checkpoint = Path(args.checkpoint) if args.checkpoint else ROOT / "checkpoints" / "exp_002" / f"tagn_seed{args.seed}.pt"
    if not checkpoint.exists():
        raise SystemExit(f"missing checkpoint: {checkpoint}")
    model.load_state_dict(torch.load(checkpoint, map_location=device), strict=False)

    with torch.no_grad():
        stats = {
            name: ((chunk_summary(frames[name]["X"].to(device)) - model.summary_loc) / model.summary_scale)
            .cpu().numpy()
            for name in ("train", "val", "test")
        }
    y = {name: frames[name]["ano"].numpy() for name in stats}

    reference = LogisticRegression(
        C=0.1, class_weight="balanced", max_iter=5000, n_jobs=-1,
    )
    reference.fit(stats["train"], y["train"])
    reference_scores = {
        name: reference.decision_function(stats[name]) for name in ("val", "test")
    }
    learned_scores = {
        name: model_scores(model, frames[name]) for name in ("val", "test")
    }
    w = float(np.clip(args.weight, 0.0, 1.0))
    hybrid_scores = {
        name: w * learned_scores[name] + (1.0 - w) * reference_scores[name]
        for name in ("val", "test")
    }

    result = {
        "seed": args.seed,
        "checkpoint": str(checkpoint),
        "tagn_weight": w,
        "validation": {
            "tagn": summarize(y["val"], learned_scores["val"]),
            "reference_logreg": summarize(y["val"], reference_scores["val"]),
            "hybrid": summarize(y["val"], hybrid_scores["val"]),
        },
        "test": {
            "tagn": summarize(y["test"], learned_scores["test"]),
            "reference_logreg": summarize(y["test"], reference_scores["test"]),
            "hybrid": summarize(y["test"], hybrid_scores["test"]),
        },
    }
    RES.mkdir(parents=True, exist_ok=True)
    with open(RES / f"exp_009_hybrid_seed{args.seed}.json", "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result["test"], indent=2))


if __name__ == "__main__":
    main()
