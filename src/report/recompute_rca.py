"""Recompute test RCA metrics for exp_002 seed checkpoints with the FIXED
prediction-based type-template jaccard (older sweeps wrote ground-truth-based
metric = trivially 1.0). Rewrites each exp_002_tagn_seed*.json in place.

Usage:
    python -m src.report.recompute_rca
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

import src.experiments.common as C  # noqa: E402
from src.models.tagn import TagnModel, load_graphs, model_config  # noqa: E402

RES = ROOT / "outputs" / "results"
CKPT = ROOT / "checkpoints" / "exp_002"


@torch.no_grad()
def recompute_seed(seed: int) -> dict:
    frames, meta, rca = C.load_splits()
    cfg = model_config()
    model = TagnModel(cfg)
    prototypes, _ = load_graphs()
    model.set_prototypes(prototypes.to(cfg["device"]))
    model.load_state_dict(torch.load(CKPT / f"tagn_seed{seed}.pt"), strict=False)
    model.eval()
    f = frames["test"]
    logits_list = {k: [] for k in ("anomaly", "type", "kpi", "hazard")}
    for i in range(0, len(f["X"]), 128):
        lg, _, _ = model(f["X"][i:i + 128], f["proto"][i:i + 128])
        for k in logits_list:
            logits_list[k].append(lg[k])
    cat = {k: torch.cat(v).detach().cpu().numpy() for k, v in logits_list.items()}
    ano = f["ano"].detach().cpu().numpy()
    anom_i = ano == 1
    yp = cat["type"].argmax(1)[anom_i]
    yt = f["typ"].detach().cpu().numpy()[anom_i]
    gt = f["aff"].detach().cpu().numpy()[anom_i]
    inv = {v: k for k, v in rca["type_map"].items()}
    import json as _json
    kpi_names = _json.load((ROOT / "data" / "interim" / "kpi_names.json").open())["numeric_kpis"]
    kpi_idx = {k: i for i, k in enumerate(kpi_names)}
    jac_t, jac_k = [], []
    for j in range(len(gt)):
        gi = set(np.where(gt[j])[0])
        ti = inv[int(yp[j])]
        pt = {kpi_idx[k] for k in rca["rca_templates"][ti] if k in kpi_idx}
        jac_t.append(len(gi & pt) / max(1, len(gi | pt)))
        ks = len(gi)
        topk = set(np.argsort(cat["kpi"][np.where(anom_i)[0][j]])[::-1][:max(1, ks)])
        jac_k.append(len(gi & topk) / max(1, len(gi | topk)))
    return {
        "type_template_jaccard": round(float(np.mean(jac_t)), 4),
        "topk_shift_jaccard": round(float(np.mean(jac_k)), 4),
        "type_accuracy": round(float(np.mean(yt == yp)), 4),
    }


def main() -> None:
    reg_fp = ROOT / "outputs" / "results_registry.json"
    registry = json.loads(reg_fp.read_text()) if reg_fp.exists() else []
    for fp in sorted(glob.glob(str(RES / "exp_002_tagn_seed*.json"))):
        seed = int(Path(fp).stem.split("_")[-1].removeprefix("seed"))
        if not (CKPT / f"tagn_seed{seed}.pt").exists():
            print(f"seed {seed}: no checkpoint, skipping")
            continue
        r = recompute_seed(seed)
        d = json.load(open(fp))
        d["test"]["rca"] = {
            "type_template_jaccard": r["type_template_jaccard"],
            "topk_shift_jaccard": r["topk_shift_jaccard"],
        }
        d["test"]["type"]["accuracy"] = r["type_accuracy"]
        json.dump(d, open(fp, "w"), indent=2)
        # patch matching registry rows (exp_002, task root_cause)
        for row in registry:
            if row.get("experiment") == "exp_002" and row.get("seed") == seed \
                    and row.get("task") == "root_cause":
                row["metrics"] = {"type_template_jaccard": r["type_template_jaccard"],
                                  "topk_shift_jaccard": r["topk_shift_jaccard"]}
        print(f"seed {seed}: {r}")
    if registry:
        reg_fp.write_text(json.dumps(registry, indent=2))


if __name__ == "__main__":
    import numpy as np
    main()
