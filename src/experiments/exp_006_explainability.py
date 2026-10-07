"""exp_006 - explainability / attribution on the KPI head.

For the trained TAGN (seed 42) we report:
  1. per-KPI attribution scores (input-driven kpi head logits) on test
     anomalous chunks;
  2. |gradient x input| attribution of the anomaly score per KPI channel;
  3. how well these match the affected-KPI template (rank-aware comparison).

Usage:
    python -m src.experiments.exp_006_explainability
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import src.experiments.common as C  # noqa: E402
from src.models.tagn import TagnModel, load_graphs, model_config  # noqa: E402
from src.utils.registry import append  # noqa: E402

RES = ROOT / "outputs" / "results"
TAB = ROOT / "outputs" / "tables"
FIG = ROOT / "outputs" / "figures" / "exp"
CKPT = ROOT / "checkpoints" / "exp_002"
PROCESSED = ROOT / "data" / "processed"


def main() -> None:
    frames, meta, rca = C.load_splits()
    cfg = model_config()
    model = TagnModel(cfg)
    prototypes, _ = load_graphs()
    model.set_prototypes(prototypes.to(cfg["device"]))
    model.load_state_dict(torch.load(CKPT / "tagn_seed42.pt"), strict=False)
    model.eval()

    f = frames["test"]
    idx = np.where(f["ano"].detach().cpu().numpy() == 1)[0]
    test_ids = np.load(PROCESSED / "split_indices.npz")["test"]
    anom_meta = meta.iloc[test_ids[idx]].reset_index(drop=True)
    kpi_names = json.load((ROOT / "data" / "interim" / "kpi_names.json").open())["numeric_kpis"]

    # 1) input-driven per-KPI attention of the kpi head
    attn_rows = []
    kpi_logits_all = []
    with torch.no_grad():
        for i in range(0, len(idx), 64):
            ids = idx[i:i + 64]
            lg, _, _ = model(f["X"][ids], f["proto"][ids])
            kpi_logits_all.append(lg["kpi"].detach().cpu().numpy())
    K = np.concatenate(kpi_logits_all)  # [n_anom, 16]
    sigK = 1 / (1 + np.exp(-K))
    for j, r in enumerate(anom_meta.itertuples()):
        aff = json.loads(r.affected_kpis)
        set_idx = {kpi_names.index(k) for k in aff if k in kpi_names}
        topk = set(np.argsort(K[j])[::-1][:len(set_idx)])
        attn_rows.append({
            "type": r.anomaly_type, "recording": r.recording,
            "jaccard_topk_vs_template": len(set_idx & topk) / max(1, len(set_idx | topk)),
            "mean_attn_in_template": float(sigK[j, list(set_idx)].mean()) if set_idx else 0.0,
        })
    attn_df = pd.DataFrame(attn_rows)
    attn_df.to_csv(TAB / "exp_006_attribution_chunks.csv", index=False)

    # gradient x input attribution of the anomaly score
    def gxinput(ids: np.ndarray) -> np.ndarray:
        x = f["X"][ids].clone().requires_grad_(True)
        proto = f["proto"][ids]
        model.zero_grad()
        lg, _, _ = model(x, proto)
        (lg["anomaly"].sigmoid().sum()).backward()
        g = x.grad.abs().mean(dim=(0, 1))[:16].detach().cpu().numpy()
        gi = x.detach().abs().mean(dim=(0, 1))[:16].detach().cpu().numpy()
        return g * gi / max(1e-12, (g * gi).max())

    grad_rows = []
    for i in range(0, len(idx), 32):
        gi = gxinput(idx[i:i + 32])
        grad_rows.append(gi)
    Grad = np.mean(grad_rows, axis=0)

    # 2) per-type summary
    per_type = {}
    for t in anom_meta.anomaly_type.unique():
        mask = (anom_meta.anomaly_type == t)
        per_type[t] = {
            "n": int(mask.sum()),
            "top3_attn_kpis": [kpi_names[i] for i in np.argsort(sigK[mask].mean(0))[::-1][:3]],
            "template_kpis": rca["rca_templates"][t],
        }

    # figure: mean attribution per KPI per type
    types = sorted(per_type)
    M = np.array([sigK[anom_meta.anomaly_type.to_numpy() == t].mean(0) for t in types])
    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(M, cmap="viridis", aspect="auto")
    ax.set_xticks(range(16)); ax.set_xticklabels(kpi_names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(types))); ax.set_yticklabels([t[:34] for t in types], fontsize=7)
    fig.colorbar(im, ax=ax)
    ax.set_title("TAGN input-driven per-KPI attribution (test anomalous windows)")
    fig.tight_layout(); fig.savefig(FIG / "exp_006_attribution.png", dpi=150); plt.close(fig)

    # 3) comparison vs template members (rank-biserial-like)
    template_flags = np.array([
        [1 if kpi_names[k] in per_type[t]["template_kpis"] else 0 for k in range(16)]
        for t in types
    ])
    in_t, out_t = [], []
    for i in range(len(types)):
        in_t += list(M[i, template_flags[i] == 1])
        out_t += list(M[i, template_flags[i] == 0])
    from scipy.stats import rankdata
    both = np.concatenate([in_t, out_t])
    ranks = rankdata(both)
    ri = ranks[:len(in_t)].mean()
    n1, n2 = len(in_t), len(out_t)
    auc = (ri * n1 - n1 * (n1 + 1) / 2) / (n1 * n2)

    summary = {
        "agreement_with_template": {
            "attn_template_vs_non_auc": float(auc),
            "n_template_kpis_total": int((template_flags == 1).sum()),
            "n_non_template_kpis_total": int((template_flags == 0).sum()),
            "jaccard_topk_vs_template_mean_chunklevel": float(attn_df.jaccard_topk_vs_template.mean()),
            "mean_attn_in_template": float(attn_df.mean_attn_in_template.mean()),
        },
        "top_grad_kpis": [kpi_names[i] for i in np.argsort(Grad)[::-1][:5]],
        "per_type": per_type,
    }
    with open(RES / "exp_006_explainability.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    pd.DataFrame(M, index=types, columns=kpi_names).to_csv(TAB / "exp_006_attribution_by_type.csv")
    append({"experiment": "exp_006", "seed": 42, "task": "explainability", "model": "tagn",
            "metrics": {"attn_template_vs_non_auc": float(auc),
                        "jaccard_topk_vs_template_mean": float(attn_df.jaccard_topk_vs_template.mean())}})
    print(json.dumps(summary["agreement_with_template"], indent=2))
    print("top-grad KPIs:", summary["top_grad_kpis"])


if __name__ == "__main__":
    main()
