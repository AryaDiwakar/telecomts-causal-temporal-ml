"""Shared train/eval utilities used by exp_002..exp_007.

The exp_002 script historically own a copy; new experiments import from here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]

INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"


def load_splits():
    X = np.load(INTERIM / "chunks_X.npy")
    meta = pd.read_parquet(INTERIM / "chunks_meta.parquet")
    scaler = json.loads((PROCESSED / "scaler.json").read_text())
    loc = np.asarray(scaler["loc"])[:16]
    scale = np.asarray(scaler["scale"])[:16]
    Xs = X.copy()
    Xs[:, :, :16] = (Xs[:, :, :16] - loc) / scale
    idx = np.load(PROCESSED / "split_indices.npz")
    tar = dict(np.load(PROCESSED / "targets.npz"))
    rca = json.loads((PROCESSED / "rca_templates.json").read_text())
    proto = np.load(PROCESSED / "graph_prototypes.npz")["chunk_prototypes"]
    frames = {
        name: {
            "X": torch.tensor(Xs[c], dtype=torch.float32),
            "ano": torch.tensor(tar["y_anomaly"][c], dtype=torch.long),
            "typ": torch.tensor(tar["y_type"][c], dtype=torch.long),
            "aff": torch.tensor(tar["y_affected"][c], dtype=torch.float32),
            "ewy": torch.tensor(tar["y_ew"][c], dtype=torch.float32),
            "ewm": torch.tensor(tar["ew_mask"][c], dtype=torch.float32),
            "ons": torch.tensor(tar["onset"][c], dtype=torch.long),
            "proto": torch.tensor(proto[c], dtype=torch.long),
        }
        for name, c in [("train", idx["train"]), ("val", idx["val"]), ("test", idx["test"])]
    }
    return frames, meta, rca


def epoch_data(frames, name: str, ratio: float, rng):
    f = frames[name]
    ano = f["ano"]
    pos = (ano == 1).nonzero(as_tuple=True)[0]
    neg = (ano == 0).nonzero(as_tuple=True)[0]
    if name == "train":
        k = int(ratio * len(pos))
        if k < len(neg):
            neg = neg[rng.choice(len(neg), k, replace=False)]
    order = torch.cat([pos, neg])[rng.permutation(len(pos) + len(neg))]
    return {k_: f[k_].index_select(0, order) for k_ in f}


def train_one_epoch(model, opt, frames, cfg, rng, active_heads=None):
    model.train()
    active = active_heads or ["anomaly", "type", "root_cause", "early_warning"]
    w = cfg["loss_weights"]
    bce = nn.BCEWithLogitsLoss(reduction="mean")
    cce = nn.CrossEntropyLoss()
    batch = epoch_data(frames, "train", cfg.get("normal_to_anomalous_ratio", 4.0), rng)
    Xb, ano = batch["X"], batch["ano"]
    B = cfg.get("train_batch", 64)
    total_loss = 0.0
    n = 0
    for i in range(0, len(Xb), B):
        sl = slice(i, i + B)
        logits, _, _ = model(Xb[sl], batch["proto"][sl])
        loss = logits["anomaly"].new_zeros(())
        if "anomaly" in active:
            loss = loss + w["anomaly"] * bce(logits["anomaly"], ano[sl].float())
        anom_b = ano[sl] == 1
        if anom_b.any():
            if "type" in active:
                loss = loss + w["type"] * cce(logits["type"][anom_b], batch["typ"][sl][anom_b])
            if "root_cause" in active:
                aff = batch["aff"][sl][anom_b]
                loss = loss + w["root_cause"] * bce(logits["kpi"][anom_b], aff)
        if "early_warning" in active:
            mask = batch["ewm"][sl]
            if mask.sum() > 0:
                ew = bce(logits["hazard"], batch["ewy"][sl]) * mask
                loss = loss + w["early_warning"] * (ew.sum() / mask.sum())
        if not loss.requires_grad:
            continue
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        total_loss += float(loss.item()) * len(ano[sl])
        n += len(ano[sl])
    return total_loss / max(1, n)


@torch.no_grad()
def evaluate(model, frames, rca, cfg, split="test", threshold=None):
    model.eval()
    f = frames[split]
    logits_list = {k: [] for k in ("anomaly", "type", "kpi", "hazard")}
    B = 128
    for i in range(0, len(f["X"]), B):
        lg, _, _ = model(f["X"][i:i + B], f["proto"][i:i + B])
        for k in logits_list:
            logits_list[k].append(lg[k])
    cat = {k: torch.cat(v).detach().cpu().numpy() for k, v in logits_list.items()}
    ano = f["ano"].detach().cpu().numpy()
    from src.evaluation.metrics import summarize

    out = {"anomaly": summarize(ano, cat["anomaly"], threshold=threshold)}
    out["anomaly"]["threshold_val_tuned"] = out["anomaly"]["threshold"]

    anom_i = ano == 1
    if anom_i.any():
        yt, yp = f["typ"].detach().cpu().numpy()[anom_i], cat["type"].argmax(1)[anom_i]
        out["type"] = {
            "accuracy": float(np.mean(yt == yp)),
            "macro_f1_seen": float(f1_score(yt, yp, average="macro", zero_division=0)),
            "micro_f1": float(f1_score(yt, yp, average="micro", zero_division=0)),
        }
        types = rca["type_map"]
        inv = {v: k for k, v in types.items()}
        kpi_names = json.load((INTERIM / "kpi_names.json").open())["numeric_kpis"]
        kpi_idx = {k: i for i, k in enumerate(kpi_names)}
        gt = f["aff"].detach().cpu().numpy()[anom_i]
        yp = cat["type"].argmax(1)[anom_i]
        jac_t, jac_k = [], []
        for j in range(len(gt)):
            gt_idx = set(np.where(gt[j])[0])
            tname = inv[int(yp[j])]
            pred_t = {kpi_idx[k] for k in rca["rca_templates"][tname] if k in kpi_idx}
            jac_t.append(len(gt_idx & pred_t) / max(1, len(gt_idx | pred_t)))
            topk = set(np.argsort(cat["kpi"][np.where(anom_i)[0][j]])[::-1][:max(1, len(gt_idx))])
            jac_k.append(len(gt_idx & topk) / max(1, len(gt_idx | topk)))
        out["rca"] = {
            "type_template_jaccard": float(np.mean(jac_t)),
            "topk_shift_jaccard": float(np.mean(jac_k)),
        }
    ons = f["ons"].detach().cpu().numpy()
    oi = np.where((ano == 1) & (ons > 0))[0]
    if len(oi):
        sig = 1 / (1 + np.exp(-cat["hazard"]))
        pred_step = np.argmax(sig[oi], axis=1)
        onsets = ons[oi]
        err = pred_step - onsets
        out["early_warning"] = {
            "n_chunks": int(len(oi)),
            "median_lag_steps": float(np.median(err)),
            "mean_abs_lead_err_steps": float(np.mean(np.abs(err))),
            "warning_before_onset_frac": float(np.mean(err < 0)),
        }
    return out


def fit(model, frames, rca, cfg, seed, active_heads=None, val_metric="pr_auc"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    best = -1.0
    best_state = None
    best_threshold = None
    history = []
    for ep in range(cfg["epochs"]):
        loss = train_one_epoch(model, opt, frames, cfg, rng, active_heads)
        ev = evaluate(model, frames, rca, cfg, split="val")
        key = "anomaly" if val_metric == "pr_auc" else "rca"
        val_score = ev.get("anomaly", {}).get("pr_auc", 0.0) if val_metric == "pr_auc" else ev.get("rca", {}).get("type_template_jaccard", 0.0)
        history.append({"epoch": ep, "loss": loss, "val_pr_auc": ev.get("anomaly", {}).get("pr_auc", 0.0),
                        "val_roc": ev.get("anomaly", {}).get("roc_auc", 0.0)})
        print(f"  ep {ep:02d} loss {loss:7.3f} val PR {val_score:.4f}", flush=True)
        if val_score > best:
            best = val_score
            best_threshold = ev.get("anomaly", {}).get("threshold")
            best_state = {k: v.detach().cpu().clone().contiguous() for k, v in model.state_dict().items()}
        prs = [h["val_pr_auc"] for h in history]
        best_epoch = max(i for i, _ in enumerate(prs) if prs[i] == max(prs))
        if ep - best_epoch > cfg["patience"]:
            print("  early stop at", ep)
            break
    model.load_state_dict(best_state, strict=False)
    test = evaluate(model, frames, rca, cfg, split="test", threshold=best_threshold)
    return {"test": test, "history": history, "best_val_pr_auc": float(best), "best_state": best_state}
