"""exp_002 - train + evaluate the proposed temporal-GNN (TAGN).

Jointly optimizes anomaly detection, type classification, per-KPI
root-cause localization and the causal early-warning hazard head.

Outputs:
  outputs/results/exp_002_tagn.json, checkpoints/exp_002/*.pt,
  outputs/figures/exp/exp_002_*.png, tables, registry entries.

Usage:
    python -m src.experiments.exp_002_tagn --seed 42
"""

from __future__ import annotations

import argparse
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
import torch.nn as nn
import yaml

from src.evaluation.metrics import summarize  # noqa: E402
from src.experiments.common import (  # noqa: E402
    build_optimizer,
    fit_summary_head,
    fit_summary_residual,
    masked_bce_with_logits,
    negative_pos_weight,
    schedule_lr,
)
from src.models.tagn import TagnModel, load_graphs, model_config  # noqa: E402
from src.utils.registry import append  # noqa: E402

INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
RES = ROOT / "outputs" / "results"
TAB = ROOT / "outputs" / "tables"
FIG = ROOT / "outputs" / "figures" / "exp"
CKPT = ROOT / "checkpoints" / "exp_002"


def load_splits(cfg: dict):
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
    frames = {
        name: {
            "X": torch.tensor(Xs[c], dtype=torch.float32),
            "ano": torch.tensor(tar["y_anomaly"][c], dtype=torch.long),
            "typ": torch.tensor(tar["y_type"][c], dtype=torch.long),
            "aff": torch.tensor(tar["y_affected"][c], dtype=torch.float32),
            "ewy": torch.tensor(tar["y_ew"][c], dtype=torch.float32),
            "ewm": torch.tensor(tar["ew_mask"][c], dtype=torch.float32),
            "ons": torch.tensor(tar["onset"][c], dtype=torch.long),
            "proto": torch.tensor(np.load(PROCESSED / "graph_prototypes.npz")["chunk_prototypes"][c], dtype=torch.long),
        }
        for name, c in [("train", idx["train"]), ("val", idx["val"]), ("test", idx["test"])]
    }
    return frames, meta, rca


def epoch_data(frames, name: str, ratio: float, rng):
    f = frames[name]
    X, ano = f["X"], f["ano"]
    pos = (ano == 1).nonzero(as_tuple=True)[0]
    neg = (ano == 0).nonzero(as_tuple=True)[0]
    if name == "train" and ratio and ratio > 0:
        k = int(ratio * len(pos))
        neg = neg[rng.choice(len(neg), k, replace=False)] if k < len(neg) else neg
    order = torch.cat([pos, neg])[rng.permutation(len(pos) + len(neg))]
    batch = {k_: f[k_].index_select(0, order) for k_ in f}
    return batch


def train_one_epoch(model, opt, frames, cfg, rng):
    model.train()
    w = cfg["loss_weights"]
    ratio = cfg.get("normal_to_anomalous_ratio", 4.0)
    # undersampled batches are already rebalanced; the full split needs
    # pos_weight so the anomaly head is not swamped by normal chunks.
    if ratio and ratio > 0:
        anom_bce = nn.BCEWithLogitsLoss(reduction="mean")
    else:
        anom_bce = nn.BCEWithLogitsLoss(
            reduction="mean",
            pos_weight=torch.tensor(
                negative_pos_weight(frames), device=frames["train"]["ano"].device,
                dtype=torch.float32,
            ),
        )
    type_target = frames["train"]["typ"]
    bce = nn.BCEWithLogitsLoss(reduction="mean")
    counts = torch.bincount(type_target[type_target >= 0], minlength=model.type_head.out_features).float()
    type_weights = counts.clamp_min(1.0).rsqrt()
    type_weights = type_weights / type_weights.mean()
    cce = nn.CrossEntropyLoss(weight=type_weights.to(type_target.device), label_smoothing=0.02)
    batch = epoch_data(frames, "train", cfg.get("normal_to_anomalous_ratio", 4.0), rng)
    Xb, ano = batch["X"], batch["ano"]
    B = cfg.get("train_batch", 64)
    total_loss = 0.0
    n_chunks = 0
    for i in range(0, len(Xb), B):
        sl = slice(i, i + B)
        logits, _, _ = model(Xb[sl], batch["proto"][sl])
        loss = w["anomaly"] * anom_bce(logits["anomaly"], ano[sl].float())
        anom_b = ano[sl] == 1
        if anom_b.any():
            loss = loss + w["type"] * cce(logits["type"][anom_b], batch["typ"][sl][anom_b])
            aff = batch["aff"][sl][anom_b]
            loss = loss + w["root_cause"] * bce(logits["kpi"][anom_b], aff)
        mask = batch["ewm"][sl]
        if mask.sum() > 0:
            loss = loss + w["early_warning"] * masked_bce_with_logits(
                logits["hazard"], batch["ewy"][sl], mask
            )
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        total_loss += float(loss.item()) * (len(ano[sl]))
        n_chunks += len(ano[sl])
    return total_loss / max(1, n_chunks)


@torch.no_grad()
def evaluate(model, frames, rca, cfg, threshold=None):
    model.eval()
    out = {}
    for name in ("val", "test"):
        f = frames[name]
        logits_list = {k: [] for k in ("anomaly", "type", "kpi", "hazard")}
        B = 128
        for i in range(0, len(f["X"]), B):
            sl = slice(i, i + B)
            lg, _, _ = model(f["X"][sl], f["proto"][sl])
            for k in logits_list:
                logits_list[k].append(lg[k])
        cat = {k: torch.cat(v).detach().cpu().numpy() for k, v in logits_list.items()}
        ano = f["ano"].detach().cpu().numpy()
        res = summarize(ano, cat["anomaly"], threshold=threshold)
        res["threshold_val_tuned"] = res["threshold"]
        out[name] = {"anomaly": res}

        # type prediction (anomalous chunks only)
        anom_i = ano == 1
        if anom_i.any():
            pred_type = cat["type"].argmax(1)
            tru_type = f["typ"].detach().cpu().numpy()
            seen_i = (tru_type >= 0)  # all anomalous are typed
            from sklearn.metrics import f1_score

            yt, yp = tru_type[anom_i], pred_type[anom_i]
            out[name]["type"] = {
                "accuracy": float(np.mean(yt == yp)),
                "macro_f1_seen": float(f1_score(yt, yp, average="macro", zero_division=0)),
                "micro_f1": float(f1_score(yt, yp, average="micro", zero_division=0)),
            }
            # root cause via type -> template
            templates = rca["rca_templates"]
            types = json.loads((PROCESSED / "rca_templates.json").read_text())["type_map"]
            inv = {v: k for k, v in types.items()}
            kpi_idx = {k: i for i, k in enumerate(json.load((INTERIM / "kpi_names.json").open())["numeric_kpis"])}
            gt = f["aff"].detach().cpu().numpy()[anom_i]
            jac_t, jac_k = [], []
            for j in range(len(gt)):
                gt_idx = set(np.where(gt[j])[0])
                tname = inv[int(yp[j])]
                pred_t = {kpi_idx[k] for k in templates[tname] if k in kpi_idx}
                jac_t.append(len(gt_idx & pred_t) / max(1, len(gt_idx | pred_t)))
                ks = len(gt_idx)
                topk = set(np.argsort(cat["kpi"][np.where(anom_i)[0][j]])[::-1][:max(1, ks)])
                jac_k.append(len(gt_idx & topk) / max(1, len(gt_idx | topk)))
            out[name]["rca"] = {
                "type_template_jaccard": float(np.mean(jac_t)),
                "topk_shift_jaccard": float(np.mean(jac_k)),
            }
        # early warning on onset>0 chunks
        ons = f["ons"].detach().cpu().numpy()
        oi = np.where((ano == 1) & (ons > 0))[0]
        if len(oi):
            sig = 1 / (1 + np.exp(-cat["hazard"]))  # [n,128]
            sig_oi = sig[oi]
            pred_step = np.argmax(sig_oi, axis=1)
            onsets = ons[oi]
            err = pred_step - onsets
            out[name]["early_warning"] = {
                "n_chunks": int(len(oi)),
                "median_lag_steps": float(np.median(err)),
                "mean_abs_lead_err_steps": float(np.mean(np.abs(err))),
                "warning_before_onset_frac": float(np.mean(err < 0)),
            }
    return out


def set_encoder_trainable(model, flag: bool) -> int:
    """Freeze/unfreeze the shared trunk during the head-warmup phase."""
    n = 0
    for name in model.ENCODER_MODULES:
        mod = getattr(model, name, None)
        if mod is None:
            continue
        for p in mod.parameters():
            p.requires_grad = flag
            n += p.numel()
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--pretrained", action="store_true",
                    help="init the trunk from exp_008 self-supervised weights")
    ap.add_argument("--pretrained-path", type=str, default=None)
    ap.add_argument("--freeze-epochs", type=int, default=3,
                    help="head warmup: keep the pretrained trunk frozen this long")
    ap.add_argument("--ratio", type=float, default=4.0,
                    help="normal:anomalous undersampling; 0 = train on the full split")
    ap.add_argument("--lr", type=float, default=None, help="override base learning rate")
    ap.add_argument("--summary-epochs", type=int, default=200,
                    help="stage A: epochs to fit the linear summary readout alone")
    ap.add_argument("--summary-lr", type=float, default=1e-2,
                    help="stage A: learning rate for the linear summary readout")
    ap.add_argument("--skip-stage-a", action="store_true",
                    help="skip the linear-readout warmup (reproduces the old behaviour)")
    ap.add_argument("--summary-residual-epochs", type=int, default=40,
                    help="epochs for the fast nonlinear summary residual")
    ap.add_argument("--anchor-only", action="store_true",
                    help="fit and benchmark the anomaly anchor without the slow GAT loop")
    ap.add_argument("--trunk-anomaly-weight", type=float, default=0.0,
                    help="validation-gated weight for the graph anomaly residual; 0 keeps the anchor fixed")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    cfg = model_config()
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.patience:
        cfg["patience"] = args.patience
    if args.lr:
        cfg["lr"] = args.lr
    cfg["normal_to_anomalous_ratio"] = args.ratio
    cfg["anomaly_trunk_weight"] = args.trunk_anomaly_weight
    cfg["train_batch"] = 64
    frames, meta, rca = load_splits(cfg)

    model = TagnModel(cfg)
    model.to(cfg["device"])
    prototypes, _ = load_graphs()
    model.set_prototypes(prototypes.to(cfg["device"]))
    # fit the summary normalization on the train split before any training
    model.calibrate_summary(frames["train"]["X"])
    model._summary_calibrated = True
    print(f"summary calibrated: {model.summary_scale.shape[0]} dims", flush=True)

    pretrained_used = False
    if args.pretrained:
        from src.models.pretrain import pretrained_encoder_path

        p = Path(args.pretrained_path) if args.pretrained_path else pretrained_encoder_path()
        if not p.exists():
            raise SystemExit(
                f"missing {p}\nrun: python -m src.experiments.exp_008_pretrain"
            )
        payload = torch.load(p, map_location="cpu")
        n = model.load_encoder_state(payload["encoder_state"])
        pretrained_used = True
        print(f"loaded {n} pretrained trunk tensors from {p}", flush=True)

    opt = build_optimizer(model, cfg)
    for split in frames:
        frames[split] = {k: v.to(cfg["device"]) if isinstance(v, torch.Tensor) else v for k, v in frames[split].items()}

    stage_a = None
    if not args.skip_stage_a:
        # Stage A: fit the linear summary readout on cached statistics.
        # Reaches the classical baseline in seconds; stage B then learns a
        # residual on top of it with the trunk.
        print(f"stage A: {args.summary_epochs} epochs on the linear summary readout",
              flush=True)
        stage_a = fit_summary_head(model, frames, cfg, rng,
                                   epochs=args.summary_epochs, lr=args.summary_lr)
        residual = fit_summary_residual(
            model, frames, cfg, rng, epochs=args.summary_residual_epochs
        )
        # The summary readout is the validated anomaly anchor. Keeping it
        # fixed makes the trunk a residual learner instead of allowing the
        # multi-task objective to destroy the strongest detector.
        for parameter in model.summary_head.parameters():
            parameter.requires_grad = False

    best_pr = -1.0
    best_state = None
    history = []
    if stage_a is not None:
        # Anchor the checkpoint at the stage-A solution. Stage B may only
        # replace it if the full model beats the readout on val.
        from sklearn.metrics import average_precision_score
        from src.models.tagn import chunk_summary
        with torch.no_grad():
            val_stats = (chunk_summary(frames["val"]["X"]) - model.summary_loc) / model.summary_scale
            val_logits = model.summary_head(val_stats).squeeze(-1) + model.summary_residual(val_stats).squeeze(-1)
        val_pr = float(average_precision_score(frames["val"]["ano"].cpu().numpy(), val_logits.cpu().numpy()))
        best_pr = val_pr
        val_anomaly = summarize(frames["val"]["ano"].cpu().numpy(), val_logits.cpu().numpy())
        best_state = {k: v.detach().cpu().clone().contiguous() for k, v in model.state_dict().items()}
        print(f"stage A anchored: val PR-AUC {best_pr:.4f}", flush=True)
        if args.anchor_only:
            with torch.no_grad():
                test_stats = (chunk_summary(frames["test"]["X"]) - model.summary_loc) / model.summary_scale
                test_logits = model.summary_head(test_stats).squeeze(-1) + model.summary_residual(test_stats).squeeze(-1)
            # Operating point uses the validation-tuned threshold, not a
            # test-tuned one (docs/benchmark_audit_v3.md, finding R3).
            test_anomaly = summarize(
                frames["test"]["ano"].cpu().numpy(),
                test_logits.cpu().numpy(),
                threshold=val_anomaly["threshold"],
            )
            out = {
                "seed": args.seed,
                "model": "tagn_summary_anchor_residual",
                "val": {"anomaly": val_anomaly},
                "test": {"anomaly": test_anomaly},
                "stage_a": stage_a,
                "summary_residual": residual,
            }
            with open(RES / f"exp_002_tagn_seed{args.seed}_anchor.json", "w") as fh:
                json.dump(out, fh, indent=2)
            print(json.dumps(out["test"], indent=2))
            return
    progress_fp = CKPT / f"tagn_seed{args.seed}{'_pt' if pretrained_used else ''}.progress.pt"
    start_epoch = 0
    if args.resume and progress_fp.exists():
        progress = torch.load(progress_fp, map_location="cpu")
        model.load_state_dict(progress["model_state"], strict=False)
        opt.load_state_dict(progress["optimizer_state"])
        for state in opt.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(cfg["device"])
        best_pr = progress["best_pr"]
        best_state = progress["best_state"]
        history = progress["history"]
        rng.bit_generator.state = progress["rng_state"]
        start_epoch = progress["next_epoch"]
        print(f"resuming at epoch {start_epoch}", flush=True)

    suffix = "_pt" if pretrained_used else ""
    for ep in range(start_epoch, cfg["epochs"]):
        if pretrained_used:
            want_trainable = ep >= args.freeze_epochs
            if ep == args.freeze_epochs or (ep == start_epoch and not want_trainable):
                n_par = set_encoder_trainable(model, want_trainable)
                if ep in (start_epoch, args.freeze_epochs):
                    print(f"epoch {ep}: trunk {'unfrozen' if want_trainable else 'frozen'} "
                          f"({n_par:,} params)", flush=True)
        schedule_lr(opt, ep, cfg)
        loss = train_one_epoch(model, opt, frames, cfg, rng)
        ev = evaluate(model, frames, rca, cfg)["val"]["anomaly"]
        pr = ev["pr_auc"]
        history.append({"epoch": ep, "loss": loss, "val_pr_auc": pr, "val_roc_auc": ev["roc_auc"],
                        "lr": opt.param_groups[0]["lr"]})
        print(f"ep {ep:02d} loss {loss:7.3f} val PR-AUC {pr:.4f} ROC {ev['roc_auc']:.4f}")
        if pr > best_pr:
            best_pr = pr
            best_state = {k: v.detach().cpu().clone().contiguous() for k, v in model.state_dict().items()}
        if ep - last_improve(history) > cfg["patience"]:
            print("early stop")
            break

        torch.save(
            {
                "next_epoch": ep + 1,
                "model_state": {k: v.detach().cpu().clone().contiguous() for k, v in model.state_dict().items()},
                "optimizer_state": opt.state_dict(),
                "best_pr": best_pr,
                "best_state": best_state,
                "history": history,
                "rng_state": rng.bit_generator.state,
            },
            progress_fp,
        )

    tag = f"tagn_seed{args.seed}{suffix}"
    model.load_state_dict(best_state, strict=False)
    val_threshold = evaluate(model, frames, rca, cfg)["val"]["anomaly"]["threshold"]
    out = evaluate(model, frames, rca, cfg, threshold=val_threshold)
    torch.save(best_state, CKPT / f"{tag}.pt")
    if progress_fp.exists():
        progress_fp.unlink()

    res = {"seed": args.seed, "test": out["test"], "val": out["val"],
           "history": history, "best_val_pr_auc": best_pr,
           "pretrained": pretrained_used, "freeze_epochs": args.freeze_epochs,
           "summary_residual": residual if stage_a is not None else None}
    with open(RES / f"exp_002_tagn_seed{args.seed}{suffix}.json", "w") as fh:
        json.dump(res, fh, indent=2)

    # figures
    test_m = out["test"]["anomaly"]
    pd.DataFrame(history).to_csv(TAB / f"exp_002_history{suffix}.csv", index=False)
    fig, ax = plt.subplots()
    ax.plot([h["epoch"] for h in history], [h["val_pr_auc"] for h in history], marker="o")
    ax.set_xlabel("epoch"); ax.set_ylabel("val PR-AUC")
    ax.set_title(f"TAGN training (seed {args.seed}{', pretrained' if pretrained_used else ''})")
    fig.tight_layout(); fig.savefig(FIG / f"exp_002_train_seed{args.seed}{suffix}.png", dpi=150); plt.close(fig)

    model_name = "tagn_pretrained" if pretrained_used else "tagn"
    append({
        "experiment": "exp_002",
        "seed": args.seed,
        "task": "anomaly_detection", "model": model_name,
        "metrics": {k: v for k, v in test_m.items() if isinstance(v, (int, float))},
    })
    append({
        "experiment": "exp_002", "seed": args.seed, "task": "type_classification",
        "model": model_name, "metrics": out["test"]["type"],
    })
    append({
        "experiment": "exp_002", "seed": args.seed, "task": "root_cause",
        "model": model_name, "metrics": out["test"]["rca"],
    })
    append({
        "experiment": "exp_002", "seed": args.seed, "task": "early_warning",
        "model": model_name, "metrics": out["test"].get("early_warning", {"n_chunks": 0}),
    })
    print(json.dumps(out["test"], indent=2))


def last_improve(history):
    prs = [h["val_pr_auc"] for h in history]
    return max(i for i, p in enumerate(prs) if p == max(prs))


if __name__ == "__main__":
    main()
