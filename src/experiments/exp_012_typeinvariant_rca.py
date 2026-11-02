#!/usr/bin/env python3
"""exp_012 - type-invariant, deferral-capable RCA (C1, on data-as-is).

Goal: demonstrate type-invariant attribution via gradient reversal (GRL) adversary
predicting type from kpi attribution logits, reducing type leakage. Add deferral
based on attribution stability + identifiability (post-hoc calibrator).
Measure: TCRL (type leakage), RCAI (stability), coverage-risk.

Variants: baseline (reuse exp_002 ckpt), grl (train with adversary), defer (post-hoc).

"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "outputs" / "results"
CKPT = ROOT / "checkpoints" / "exp_002"

if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn

from src.experiments.common import (
    build_optimizer,
    epoch_data,
    evaluate,
    fit_summary_head,
    fit_summary_residual,
    load_splits,
)
from src.models.tagn import TagnModel, load_graphs, model_config
from src.utils.registry import append


def train_one_epoch_grl(model, opt, frames, cfg, rng):
    model.train()
    model.adv_lambda = float(cfg.get("grl_lambda_cur", cfg.get("grl_lambda", 0.0)))
    w = cfg["loss_weights"]
    ratio = cfg.get("normal_to_anomalous_ratio", 4.0)
    from src.experiments.common import negative_pos_weight

    pos_weight = None
    if ratio == 0:
        pos_weight = negative_pos_weight(frames)
    bce = nn.BCEWithLogitsLoss(reduction="mean")
    if pos_weight is None:
        anom_bce = bce
    else:
        anom_bce = nn.BCEWithLogitsLoss(
            reduction="mean",
            pos_weight=torch.tensor(pos_weight, device=frames["train"]["ano"].device, dtype=torch.float32),
        )
    type_target = frames["train"]["typ"]
    counts = torch.bincount(type_target[type_target >= 0], minlength=model.type_head.out_features).float()
    type_weights = counts.clamp_min(1.0).rsqrt()
    type_weights = type_weights / type_weights.mean()
    cce = nn.CrossEntropyLoss(weight=type_weights.to(type_target.device), label_smoothing=0.02)
    batch = epoch_data(frames, "train", ratio, rng)
    Xb, ano = batch["X"], batch["ano"]
    B = cfg.get("train_batch", 64)
    total_loss = 0.0
    n = 0
    for i in range(0, len(Xb), B):
        sl = slice(i, i + B)
        logits, _, _ = model(Xb[sl], batch["proto"][sl])
        loss = logits["anomaly"].new_zeros(())
        loss = loss + w["anomaly"] * anom_bce(logits["anomaly"], ano[sl].float())
        anom_b = ano[sl] == 1
        if anom_b.any():
            loss = loss + w["type"] * cce(logits["type"][anom_b], batch["typ"][sl][anom_b])
            aff = batch["aff"][sl][anom_b]
            loss = loss + w["root_cause"] * bce(logits["kpi"][anom_b], aff)
            if "rca_adv" in logits:
                w_adv = float(cfg.get("adv_weight", 1.0))
                if w_adv > 0:
                    loss = loss + w_adv * cce(logits["rca_adv"][anom_b], batch["typ"][sl][anom_b])
        mask = batch["ewm"][sl]
        if mask.sum() > 0:
            from src.experiments.common import masked_bce_with_logits

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


def attribution_scores(model, frames, split="test", batch=128):
    model.eval()
    f = frames[split]
    atts = []
    for i in range(0, len(f["X"]), batch):
        sl = slice(i, i + batch)
        lg, _, _ = model(f["X"][sl], f["proto"][sl])
        atts.append(lg["kpi"].detach().cpu().numpy())
    return np.concatenate(atts) if atts else np.array([])


def stability(model, frames, split="test", K=6, sigma=0.1, batch=128, seed=42):
    rng = np.random.default_rng(seed)
    model.eval()
    f = frames[split]
    X = f["X"]
    proto = f["proto"]
    s_list = []
    ent_list = []
    for i in range(0, len(X), batch):
        sl = slice(i, i + batch)
        xb = X[sl]
        pb = proto[sl]
        with torch.no_grad():
            a0 = model(xb, pb)[0]["kpi"]
        a0n = a0.detach().cpu().numpy()
        sims = []
        ents = []
        std = xb.std(dim=(0, 1), keepdim=True).clamp_min(1e-4)
        for _ in range(K):
            xn = xb + sigma * std * torch.tensor(rng.normal(0, 1, xb.shape), dtype=xb.dtype, device=xb.device)
            with torch.no_grad():
                ak = model(xn, pb)[0]["kpi"]
            akn = ak.detach().cpu().numpy()
            for j in range(len(ak)):
                sp = np.corrcoef(a0n[j], akn[j])[0, 1]
                if np.isnan(sp):
                    sp = 0.0
                sims.append(sp)
            a = akn - akn.max(axis=1, keepdims=True)
            p = np.exp(a) / (np.exp(a).sum(axis=1, keepdims=True) + 1e-10)
            ents.append(-np.sum(p * np.log(p + 1e-10), axis=1))
        s_batch = np.array(sims).reshape(K, len(xb)).mean(0)
        s_list.append(s_batch)
        ent_batch = np.array(ents).reshape(K, len(xb)).mean(0)
        ent_list.append(ent_batch)
    return np.concatenate(s_list), np.concatenate(ent_list)


def compute_tcrl(A_val, ytyp_val, A_test, ytyp_test):
    """Type-conditional RCA leakage.

    Fit a type probe on held-out attributions and report how well it recovers
    the anomaly type. Val only covers a subset of types, so we use repeated
    within-test splits (train half / eval half) to get a comparable leakage
    number for every variant on the same windows.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, accuracy_score

    mask = ytyp_test >= 0
    A = A_test[mask]
    y = ytyp_test[mask]
    if len(y) < 4:
        return {"probe_acc": 0.0, "macro_f1": 0.0, "majority": 0.0, "tcrl": 0.0, "n": int(len(y))}
    rng = np.random.default_rng(42)
    accs, mf1s = [], []
    for _ in range(5):
        perm = rng.permutation(len(y))
        cut = len(y) // 2
        tr, ev = perm[:cut], perm[cut:]
        if len(np.unique(y[tr])) < 2:
            continue
        lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
        lr.fit(A[tr], y[tr])
        pred = lr.predict(A[ev])
        accs.append(accuracy_score(y[ev], pred))
        mf1s.append(f1_score(y[ev], pred, average="macro", zero_division=0))
    if not accs:
        return {"probe_acc": 0.0, "macro_f1": 0.0, "majority": 0.0, "tcrl": 0.0, "n": int(len(y))}
    acc = float(np.mean(accs))
    mf1 = float(np.mean(mf1s))
    counts = np.bincount(y, minlength=y.max() + 1).astype(float)
    maj = float(counts.max() / counts.sum())
    tcrl = (acc - maj) / max(1e-6, 1 - maj)
    return {"probe_acc": acc, "macro_f1": mf1, "majority": maj, "tcrl": float(tcrl), "n": int(len(y))}


def _topk_jaccard(A_a, aff_a):
    j = []
    for k in range(len(A_a)):
        gt = set(np.where(aff_a[k] > 0)[0])
        ks = len(gt)
        if ks < 1:
            j.append(0.0)
            continue
        topk = set(np.argsort(A_a[k])[::-1][:ks])
        j.append(len(gt & topk) / max(1, len(gt | topk)))
    return np.array(j)


def subset_mean_jaccard(A, aff, mask):
    """Mean top-k RCA Jaccard restricted to anomalous windows in ``mask``."""
    ano = (aff.sum(axis=1) >= 1) & mask
    if int(ano.sum()) < 1:
        return None
    return round(float(_topk_jaccard(A[ano], aff[ano]).mean()), 4)


def _coverage_curve(j, score, points=(0.1, 0.25, 0.5, 0.75, 0.9, 1.0)):
    order = np.argsort(score)[::-1]
    out = []
    for c in points:
        m = max(1, int(c * len(order)))
        out.append({"coverage": c, "jaccard": float(j[order[:m]].mean())})
    return out


def compute_coverage_risk(A, aff, s, ent, val=None):
    """Attribution precision-at-coverage under two deferral signals.

    * raw stability ``s`` (unsupervised) and negative entropy.
    * a learned calibrator g([s, ent, max_prob]) -> P(top-k correct), fit on the
      validation split, ranking test windows for deferral.
    """
    ano = aff.sum(axis=1) >= 1
    N = int(ano.sum())
    if N < 1:
        return {"best_jaccard": 0.0, "s_mean": 0.0}
    A_a = A[ano]
    aff_a = aff[ano]
    s_a = s[ano] if s is not None else np.ones(N)
    ent_a = ent[ano] if ent is not None else np.zeros(N)
    maxp_a = (1.0 / (1.0 + np.exp(-A_a))).max(axis=1)
    j = _topk_jaccard(A_a, aff_a)
    best = float(j.mean())
    cov_s = _coverage_curve(j, s_a)
    cov_e = _coverage_curve(j, -ent_a)
    learned = None
    if val is not None:
        from sklearn.linear_model import LogisticRegression

        Av, affv, sv, ev = val
        anov = affv.sum(axis=1) >= 1
        if anov.sum() >= 8:
            Av_a = Av[anov]
            affv_a = affv[anov]
            sv_a = sv[anov]
            ev_a = ev[anov]
            maxp_v = (1.0 / (1.0 + np.exp(-Av_a))).max(axis=1)
            jv = _topk_jaccard(Av_a, affv_a)
            yv = (jv >= 0.5).astype(int)
            if len(np.unique(yv)) == 2:
                Xv = np.stack([sv_a, ev_a, maxp_v], axis=1)
                Xt = np.stack([s_a, ent_a, maxp_a], axis=1)
                clf = LogisticRegression(max_iter=2000, class_weight="balanced")
                clf.fit(Xv, yv)
                pv = clf.predict_proba(Xt)[:, 1]
                learned = _coverage_curve(j, pv)
        cov_s = cov_s
    return {
        "best_jaccard": best,
        "s_mean": float(s_a.mean()),
        "cov_stability": cov_s,
        "cov_neg_entropy": cov_e,
        "cov_learned": learned,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--grl-lambda", type=float, default=0.5)
    ap.add_argument("--adv-weight", type=float, default=1.0)
    ap.add_argument("--lam-ramp", type=float, default=0.2)
    ap.add_argument("--no-train-grl", action="store_true")
    args = ap.parse_args()

    cfg = model_config()
    if args.epochs:
        cfg["epochs"] = args.epochs
    cfg["grl_lambda"] = args.grl_lambda
    cfg["adv_weight"] = args.adv_weight

    frames, meta, rca = load_splits()
    te_idx = np.load(ROOT / "data" / "processed" / "split_indices.npz")["test"]
    is_gen = meta.recording.str.contains("synthetic_gen").to_numpy()[te_idx]

    # GRL variant
    res_out = {}
    if not args.no_train_grl:
        model = TagnModel(cfg, n_types=11)
        from src.models.tagn import load_graphs

        p, cp = load_graphs()
        model.to(cfg["device"])
        model.set_prototypes(p.to(cfg["device"]))
        for s in frames:
            frames[s] = {k: v.to(cfg["device"]) if hasattr(v, "device") else v for k, v in frames[s].items()}
        model.calibrate_summary(frames["train"]["X"].cpu())
        model._summary_calibrated = True

        from src.experiments.common import build_optimizer, fit_summary_head, fit_summary_residual

        fit_summary_head(model, frames, cfg, __import__("numpy").random.default_rng(args.seed), epochs=60, lr=1e-2)
        fit_summary_residual(model, frames, cfg, __import__("numpy").random.default_rng(args.seed), epochs=40)
        for parameter in model.summary_head.parameters():
            parameter.requires_grad = False

        opt = build_optimizer(model, cfg)

        best = -1
        best_state = None
        rng = __import__("numpy").random.default_rng(args.seed)
        for ep in range(cfg["epochs"]):
            lam = args.grl_lambda
            if args.lam_ramp > 0 and ep < int(args.lam_ramp * cfg["epochs"]):
                lam = args.grl_lambda * (ep + 1) / max(1, int(args.lam_ramp * cfg["epochs"]))
            model.adv_lambda = float(lam)
            cfg["grl_lambda_cur"] = float(lam)
            loss = train_one_epoch_grl(model, opt, frames, cfg, rng)
            ev = evaluate(model, frames, rca, cfg, split="val")
            pr = ev["anomaly"]["pr_auc"]
            if pr > best:
                best = pr
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if best_state:
            model.load_state_dict(best_state, strict=False)
        ev_test = evaluate(model, frames, rca, cfg, split="test")
        torch.save(best_state or model.state_dict(), CKPT / f"exp_012_grl_seed{args.seed}.pt")

        # analysis
        A_val = attribution_scores(model, frames, split="val")
        A_test = attribution_scores(model, frames, split="test")
        s_val, ent_val = stability(model, frames, split="val", K=4, seed=args.seed)
        s_test, ent_test = stability(model, frames, split="test", K=4, seed=args.seed)
        fval = frames["val"]
        ftest = frames["test"]
        tcrl = compute_tcrl(A_val, fval["typ"].cpu().numpy(), A_test, ftest["typ"].cpu().numpy())
        cov = compute_coverage_risk(
            A_test, ftest["aff"].cpu().numpy(), s_test, ent_test,
            val=(A_val, fval["aff"].cpu().numpy(), s_val, ent_val),
        )
        res_grl = {
            "model": "tagn_grl",
            "seed": args.seed,
            "test": {k: v for k, v in ev_test["anomaly"].items() if isinstance(v, (int, float))},
            "test_rca": ev_test.get("rca", {}),
            "rca_by_subset": {
                "instance_varying": subset_mean_jaccard(A_test, ftest["aff"].cpu().numpy(), is_gen),
                "template": subset_mean_jaccard(A_test, ftest["aff"].cpu().numpy(), ~is_gen),
            },
            "tcrl": tcrl,
            "rcai": {"stability_mean": float(cov["s_mean"])},
            "coverage_risk": cov,
            "best_jaccard": cov["best_jaccard"],
        }
        res_out["grl"] = res_grl
        with open(RES / f"exp_012_grl_seed{args.seed}.json", "w") as fh:
            json.dump(res_grl, fh, indent=2)

    # baseline from exp_002
    try:
        ckpt = CKPT / f"tagn_seed{args.seed}.pt"
        if ckpt.exists():
            model_b = TagnModel(cfg, n_types=11)
            from src.models.tagn import load_graphs

            p, cp = load_graphs()
            model_b.to(cfg["device"])
            model_b.set_prototypes(p.to(cfg["device"]))
            model_b.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
            for s in frames:
                frames[s] = {k: v.to(cfg["device"]) if hasattr(v, "device") else v for k, v in frames[s].items()}
            model_b.calibrate_summary(frames["train"]["X"])
            model_b._summary_calibrated = True
            A_val_b = attribution_scores(model_b, frames, split="val")
            A_test_b = attribution_scores(model_b, frames, split="test")
            s_val_b, ent_val_b = stability(model_b, frames, split="val", K=4, seed=args.seed)
            s_test_b, ent_test_b = stability(model_b, frames, split="test", K=4, seed=args.seed)
            fval = frames["val"]
            ftest = frames["test"]
            tcrl_b = compute_tcrl(A_val_b, fval["typ"].cpu().numpy(), A_test_b, ftest["typ"].cpu().numpy())
            cov_b = compute_coverage_risk(
                A_test_b, ftest["aff"].cpu().numpy(), s_test_b, ent_test_b,
                val=(A_val_b, fval["aff"].cpu().numpy(), s_val_b, ent_val_b),
            )
            ev_test_b = evaluate(model_b, frames, rca, cfg, split="test")
            res_b = {
                "model": "tagn_baseline",
                "seed": args.seed,
                "test": {k: v for k, v in ev_test_b["anomaly"].items() if isinstance(v, (int, float))},
                "test_rca": ev_test_b.get("rca", {}),
                "rca_by_subset": {
                    "instance_varying": subset_mean_jaccard(A_test_b, ftest["aff"].cpu().numpy(), is_gen),
                    "template": subset_mean_jaccard(A_test_b, ftest["aff"].cpu().numpy(), ~is_gen),
                },
                "tcrl": tcrl_b,
                "rcai": {"stability_mean": float(cov_b["s_mean"])},
                "coverage_risk": cov_b,
                "best_jaccard": cov_b["best_jaccard"],
            }
            res_out["baseline"] = res_b
            with open(RES / f"exp_012_baseline_seed{args.seed}.json", "w") as fh:
                json.dump(res_b, fh, indent=2)
    except Exception as e:
        print("baseline failed:", e)

    with open(RES / f"exp_012_typeinvariant_rca_seed{args.seed}.json", "w") as fh:
        json.dump(res_out, fh, indent=2)

    append(
        {
            "experiment": "exp_012",
            "seed": args.seed,
            "task": "type_invariant_rca",
            "model": "tagn_grl" if "grl" in res_out else "tagn_baseline",
            "metrics": {"tcrl": res_out.get("grl", res_out.get("baseline", {})).get("tcrl", {})},
        }
    )
    append(
        {
            "experiment": "exp_012",
            "seed": args.seed,
            "task": "deferral_rca",
            "model": "tagn_grl" if "grl" in res_out else "tagn_baseline",
            "metrics": {"rcai": res_out.get("grl", res_out.get("baseline", {})).get("rcai", {})},
        }
    )
    print(json.dumps(res_out, indent=2))


if __name__ == "__main__":
    main()
