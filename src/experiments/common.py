"""Shared train/eval utilities used by exp_002..exp_007.

The exp_002 script historically own a copy; new experiments import from here.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]

INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"


def masked_bce_with_logits(
    logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Compute BCE only over valid time steps, preserving gradient flow."""
    loss = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


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
    if name == "train" and ratio and ratio > 0:
        k = int(ratio * len(pos))
        if k < len(neg):
            neg = neg[rng.choice(len(neg), k, replace=False)]
    order = torch.cat([pos, neg])[rng.permutation(len(pos) + len(neg))]
    return {k_: f[k_].index_select(0, order) for k_ in f}


def schedule_lr(opt, ep: int, cfg: dict) -> float:
    """Warmup + cosine decay on a per-epoch basis.

    A constant 1e-3 lets the heads keep walking after the shared trunk has
    found a good basin, which is what produced the val PR-AUC collapse at
    epoch 13 in the original seed-123 run.
    """
    base = cfg["lr"]
    warmup = int(cfg.get("warmup_epochs", 2))
    total = max(warmup + 1, int(cfg["epochs"]))
    floor = float(cfg.get("lr_floor", 0.05))
    if ep < warmup:
        lr = base * (ep + 1) / warmup
    else:
        t = (ep - warmup) / max(1, total - warmup)
        lr = base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * t)))
    for group in opt.param_groups:
        group["lr"] = lr
    return lr


def negative_pos_weight(frames) -> float:
    """pos_weight for BCE when training on the full (unbalanced) train split."""
    ano = frames["train"]["ano"]
    n_pos = int((ano == 1).sum().item())
    n_neg = int((ano == 0).sum().item())
    return n_neg / max(1, n_pos)


def fit_summary_head(model, frames, cfg, rng, epochs=None, lr=None, batch=256) -> dict:
    """Stage A: fit the linear summary readout before any trunk training.

    ``chunk_summary`` depends only on the input window, so the normalized
    statistics are cached once and the readout is fit directly. That is
    exactly what training ``summary_head`` through ``forward`` with the trunk
    frozen would do, but it costs seconds instead of ~70s per epoch.

    It also removes the gradient competition that caused the original
    failures: a default-init trunk could fit the training split immediately,
    collapsing ``sigmoid(logit) - y`` and leaving the linear path with almost
    no gradient. Starting stage B from this fit means the temporal/graph
    branches learn a *residual* on top of a solution that already clears the
    classical baseline.
    """
    from sklearn.metrics import average_precision_score

    from src.models.tagn import chunk_summary

    epochs = int(epochs if epochs is not None else cfg.get("summary_epochs", 60))
    lr = float(lr if lr is not None else cfg.get("summary_lr", cfg["lr"]))
    ratio = cfg.get("normal_to_anomalous_ratio", 4.0)
    warmup = min(5, max(1, epochs // 10))

    with torch.no_grad():
        stats = {
            name: (chunk_summary(frames[name]["X"]) - model.summary_loc) / model.summary_scale
            for name in ("train", "val")
        }
    y_tr = frames["train"]["ano"].float()
    y_va = frames["val"]["ano"].cpu().numpy()
    pos = (y_tr == 1).nonzero(as_tuple=True)[0]
    neg = (y_tr == 0).nonzero(as_tuple=True)[0]

    opt = torch.optim.AdamW(
        [{"params": model.summary_head.parameters(), "weight_decay": 0.0},
         {"params": model.summary_type_head.parameters(), "weight_decay": 0.0}],
        lr=lr,
    )
    if ratio and ratio > 0:
        bce = nn.BCEWithLogitsLoss()
    else:
        bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                float(len(neg)) / max(1, len(pos)),
                device=y_tr.device,
                dtype=torch.float32,
            )
        )
    typ = frames["train"]["typ"]
    valid = (typ >= 0).nonzero(as_tuple=True)[0]
    cce = nn.CrossEntropyLoss(label_smoothing=0.02)

    def val_pr() -> float:
        with torch.no_grad():
            return float(average_precision_score(y_va, model.summary_head(stats["val"]).squeeze(-1).cpu().numpy()))

    best = -1.0
    best_state = {k: v.detach().cpu().clone() for k, v in model.summary_head.state_dict().items()}
    history = []
    for ep in range(epochs):
        # per-epoch warmup + cosine, mirroring schedule_lr
        if ep < warmup:
            cur = lr * (ep + 1) / warmup
        else:
            t = (ep - warmup) / max(1, epochs - warmup)
            cur = lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * t)))
        for g in opt.param_groups:
            g["lr"] = cur

        if ratio and ratio > 0:
            k = int(ratio * len(pos))
            sel_neg = neg[torch.as_tensor(rng.choice(len(neg), k, replace=False))] if k < len(neg) else neg
            order = torch.cat([pos, sel_neg])
            order = order[torch.as_tensor(rng.permutation(len(order)))]
        else:
            order = torch.arange(len(y_tr))
        loss_a = 0.0
        n_a = 0
        for i in range(0, len(order), batch):
            b = order[i:i + batch].to(stats["train"].device)
            logits = model.summary_head(stats["train"][b]).squeeze(-1)
            loss = bce(logits, y_tr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_a += float(loss.item()) * len(b)
            n_a += len(b)
        # type readout on anomalous chunks only
        loss_t = 0.0
        n_t = 0
        for i in range(0, len(valid), batch):
            b = valid[i:i + batch].to(stats["train"].device)
            tl = model.summary_type_head(stats["train"][b])
            loss = cce(tl, typ[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_t += float(loss.item()) * len(b)
            n_t += len(b)

        pr = val_pr()
        history.append({"epoch": ep, "train_anomaly_loss": loss_a / max(1, n_a),
                        "train_type_loss": loss_t / max(1, n_t), "val_pr_auc": pr, "lr": cur})
        if pr > best:
            best = pr
            best_state = {k: v.detach().cpu().clone() for k, v in model.summary_head.state_dict().items()}
        if ep % max(1, epochs // 10) == 0 or ep == epochs - 1:
            print(f"  stageA ep{ep:03d} lr {cur:.1e} anom {loss_a/max(1,n_a):.4f} "
                  f"type {loss_t/max(1,n_t):.4f} val PR {pr:.4f}", flush=True)

    model.summary_head.load_state_dict(best_state)
    return {"best_val_pr_auc": float(best), "history": history, "epochs": epochs, "lr": lr}


def fit_summary_residual(model, frames, cfg, rng, epochs=40, lr=3e-3, batch=256) -> dict:
    """Fit a compact nonlinear residual without running the expensive GAT.

    The linear summary readout remains fixed. This stage can therefore only
    add nonlinear interactions between validated chunk statistics; it cannot
    regress to a weaker detector while the temporal/graph trunk is warming up.
    """
    from sklearn.metrics import average_precision_score
    from src.models.tagn import chunk_summary

    with torch.no_grad():
        stats = {
            name: (chunk_summary(frames[name]["X"]) - model.summary_loc) / model.summary_scale
            for name in ("train", "val")
        }
    y_tr = frames["train"]["ano"].float()
    y_va = frames["val"]["ano"].cpu().numpy()
    pos = (y_tr == 1).nonzero(as_tuple=True)[0]
    neg = (y_tr == 0).nonzero(as_tuple=True)[0]
    for p in model.summary_head.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW(model.summary_residual.parameters(), lr=lr, weight_decay=1e-4)
    ratio = cfg.get("normal_to_anomalous_ratio", 4.0)
    if ratio and ratio > 0:
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                float(len(neg)) / max(1, len(pos)),
                device=y_tr.device,
                dtype=torch.float32,
            )
        )

    def val_pr() -> float:
        was_training = model.training
        model.eval()
        with torch.no_grad():
            logits = model.summary_head(stats["val"]).squeeze(-1) + model.summary_residual(stats["val"]).squeeze(-1)
            score = float(average_precision_score(y_va, logits.cpu().numpy()))
        if was_training:
            model.train()
        return score

    best = val_pr()
    best_state = {k: v.detach().cpu().clone() for k, v in model.summary_residual.state_dict().items()}
    for ep in range(int(epochs)):
        if ratio and ratio > 0:
            k = int(ratio * len(pos))
            selected = neg[torch.as_tensor(rng.choice(len(neg), k, replace=False))] if k < len(neg) else neg
            order = torch.cat([pos, selected])
            order = order[torch.as_tensor(rng.permutation(len(order)))]
        else:
            order = torch.arange(len(y_tr))
        for i in range(0, len(order), batch):
            b = order[i:i + batch]
            logits = model.summary_head(stats["train"][b]).squeeze(-1) + model.summary_residual(stats["train"][b]).squeeze(-1)
            loss = loss_fn(logits, y_tr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        pr = val_pr()
        if pr > best:
            best = pr
            best_state = {k: v.detach().cpu().clone() for k, v in model.summary_residual.state_dict().items()}
    model.summary_residual.load_state_dict(best_state)
    return {"best_val_pr_auc": float(best), "epochs": int(epochs), "lr": float(lr)}


def build_optimizer(model, cfg) -> torch.optim.Optimizer:
    """AdamW with weight decay disabled on the summary readout heads.

    ``summary_head`` has to reproduce the classical logistic-regression fit,
    which in the baseline is essentially unregularized. Applying the trunk's
    weight decay there shrinks precisely the path that must clear the
    baseline's PR-AUC, so those heads get their own group.
    """
    summary_params = set()
    for name in ("summary_head", "summary_type_head"):
        mod = getattr(model, name, None)
        if mod is not None:
            summary_params.update(id(p) for p in mod.parameters())
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (no_decay if id(p) in summary_params else decay).append(p)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg["lr"],
    )


def train_one_epoch(model, opt, frames, cfg, rng, active_heads=None):
    model.train()
    active = active_heads or ["anomaly", "type", "root_cause", "early_warning"]
    w = cfg["loss_weights"]
    ratio = cfg.get("normal_to_anomalous_ratio", 4.0)
    # undersampled batches are already rebalanced; the full split needs
    # pos_weight so the anomaly head is not swamped by normal chunks.
    if ratio and ratio > 0:
        pos_weight = None
    else:
        pos_weight = negative_pos_weight(frames)
    type_target = frames["train"]["typ"]
    bce = nn.BCEWithLogitsLoss(reduction="mean")
    if pos_weight is None:
        anom_bce = bce
    else:
        anom_bce = nn.BCEWithLogitsLoss(
            reduction="mean",
            pos_weight=torch.tensor(pos_weight, device=type_target.device, dtype=torch.float32),
        )
    counts = torch.bincount(type_target[type_target >= 0], minlength=model.type_head.out_features).float()
    type_weights = counts.clamp_min(1.0).rsqrt()
    type_weights = type_weights / type_weights.mean()
    cce = nn.CrossEntropyLoss(weight=type_weights.to(type_target.device), label_smoothing=0.02)
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
            loss = loss + w["anomaly"] * anom_bce(logits["anomaly"], ano[sl].float())
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
                loss = loss + w["early_warning"] * masked_bce_with_logits(
                    logits["hazard"], batch["ewy"][sl], mask
                )
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
    if not getattr(model, "_summary_calibrated", False):
        model.calibrate_summary(frames["train"]["X"])
        model._summary_calibrated = True
    opt = build_optimizer(model, cfg)
    best = -1.0
    best_state = None
    best_threshold = None
    history = []
    for ep in range(cfg["epochs"]):
        lr = schedule_lr(opt, ep, cfg)
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
