"""Phase 1 - error anatomy on the v3 test split (no new method, decision tool).

Answers *where the headroom is*, so Phase 2 can target it:

  1. overall per-model PR/ROC (sanity vs the headline table);
  2. subgroup split by anomaly family (shipped/template vs Track-2 instance-varying);
  3. **seen vs unseen session** - the G5 diagnostic: does a model collapse when the
     test recording's session was never in train, and does that differ across models?
  4. per-anomaly-type detection difficulty (threshold-free recall@FPR1% + AP);
  5. where the false positives come from (normal chunks from seen vs unseen sessions).

All score vectors are per-chunk and aligned to ``split_indices['test']`` order.

Usage:
    python -m scripts.error_anatomy --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

RES = ROOT / "outputs" / "results"
PREDS = RES / "preds"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"

GEN_PREFIX = "anomalous/synthetic_gen/"


def session_key(rec: str) -> str:
    """Map a recording to the physical session it came from.

    Track-2 injected recordings are derived from a ``normal/<rel>`` source, so
    ``anomalous/synthetic_gen/<rel>`` and ``normal/<rel>`` share a session.
    """
    if rec.startswith(GEN_PREFIX):
        return "normal/" + rec[len(GEN_PREFIX):]
    return rec


def load_all(seed: int):
    """Return (y, groups, scores) with every score vector aligned to test order."""
    split = np.load(PROCESSED / "split_indices.npz")
    test_ids = np.asarray(split["test"])
    meta = pd.read_parquet(INTERIM / "chunks_meta.parquet")
    mt = meta.iloc[test_ids].reset_index(drop=True)
    groups = mt["recording"].to_numpy()

    t = np.load(PREDS / f"tagn_seed{seed}_preds.npz", allow_pickle=True)
    y = t["y_test"].astype(int)
    assert np.array_equal(np.asarray(t["chunk_ids"]), test_ids), "TAGN chunk order mismatch"
    assert np.array_equal(np.asarray(t["groups"]), groups), "TAGN group mismatch"

    scores: dict[str, np.ndarray] = {"TAGN": t["s_test"].astype(float)}

    for m in ("tcn", "gru"):
        p = RES / f"exp_013_{m}_seed{seed}_preds.npz"
        if not p.exists():
            continue
        d = np.load(p, allow_pickle=True)
        assert np.array_equal(np.asarray(d["y_test"]).astype(int), y), f"label mismatch {p}"
        assert np.array_equal(np.asarray(d["groups"]), groups), f"group mismatch {p}"
        scores[m.upper()] = d["s_test"].astype(float)

    cp = PREDS / "exp_001_test_preds.npz"
    if cp.exists():
        d = np.load(cp, allow_pickle=True)
        assert np.array_equal(np.asarray(d["y"]).astype(int), y), "classical label mismatch"
        assert np.array_equal(np.asarray(d["groups"]), groups), "classical group mismatch"
        for k in ("logreg", "xgb", "lof", "ocsvm", "iforest"):
            if k in d.files:
                scores[k] = d[k].astype(float)

    return y, groups, mt, scores


def pair_metrics(y, s, pos_mask):
    """PR/ROC of one positive family against the common normal pool."""
    sel = (y == 0) | pos_mask
    yy, ss = y[sel], s[sel]
    if yy.sum() == 0 or yy.sum() == len(yy):
        return None, None, int(yy.sum()), int(len(yy))
    return float(average_precision_score(yy, ss)), float(roc_auc_score(yy, ss)), int(yy.sum()), int(len(yy))


def within_metrics(y, s, mask):
    """PR/ROC computed entirely inside ``mask`` (positives vs normals in the subset)."""
    yy, ss = y[mask], s[mask]
    if yy.sum() == 0 or yy.sum() == len(yy):
        return None, None, int(yy.sum()), int(len(yy))
    return float(average_precision_score(yy, ss)), float(roc_auc_score(yy, ss)), int(yy.sum()), int(len(yy))


def recall_at_fpr(y, s, pos_mask, fpr=0.01):
    neg = s[y == 0]
    if len(neg) == 0 or pos_mask.sum() == 0:
        return float("nan")
    thr = np.quantile(neg, 1 - fpr)
    return float((s[pos_mask] >= thr).mean())


def fmt(v):
    return "  n/a  " if v is None else f"{v:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42, help="like-for-like nn seed")
    ap.add_argument("--out", type=str, default=str(RES / "error_anatomy.json"))
    args = ap.parse_args()

    y, groups, mt, scores = load_all(args.seed)
    split = np.load(PROCESSED / "split_indices.npz")
    test_ids = np.asarray(split["test"])
    meta = pd.read_parquet(INTERIM / "chunks_meta.parquet")

    train_ids = np.asarray(split["train"])
    train_sessions = {session_key(r) for r in meta.iloc[train_ids]["recording"].unique()}

    sk = np.array([session_key(r) for r in groups])
    unseen_mask = np.array([k not in train_sessions for k in sk])
    seen_mask = ~unseen_mask

    is_gen = np.array([r.startswith("anomalous/synthetic_gen") for r in groups])
    types = np.array([t if isinstance(t, str) else "normal" for t in mt["anomaly_type"]])

    pos = y == 1
    print("=" * 78)
    print(f"ERROR ANATOMY - v3 test  n={len(y)}  pos={int(pos.sum())}  "
          f"recordings={len(np.unique(groups))}  seed={args.seed}")
    print(f"  test sessions={len(np.unique(sk))}   "
          f"unseen-session chunks={int(unseen_mask.sum())} "
          f"(pos {int((unseen_mask & pos).sum())})   "
          f"seen-session chunks={int(seen_mask.sum())} "
          f"(pos {int((seen_mask & pos).sum())})")
    print("=" * 78)

    results: dict = {"seed": args.seed, "n": int(len(y)), "n_pos": int(pos.sum())}

    # --- 1. overall ---------------------------------------------------------
    print("\n=== 1. overall (positives = all anomalies, negatives = all normals) ===")
    print(f"{'model':>9s} {'PR':>8s} {'ROC':>8s}")
    overall = {}
    for name, s in scores.items():
        pr, roc = float(average_precision_score(y, s)), float(roc_auc_score(y, s))
        overall[name] = {"pr": pr, "roc": roc}
        print(f"{name:>9s} {pr:8.4f} {roc:8.4f}")
    results["overall"] = overall

    # --- 2. anomaly family --------------------------------------------------
    print("\n=== 2. anomaly family (family positives vs common normal pool) ===")
    print(f"{'model':>9s} | {'template PR':>11s} {'ROC':>7s} | {'inst-var PR':>11s} {'ROC':>7s}")
    fam = {}
    for name, s in scores.items():
        tp, tr, tpn, _ = pair_metrics(y, s, ~is_gen & pos)
        ip, ir, ipn, _ = pair_metrics(y, s, is_gen & pos)
        fam[name] = {"template_pr": tp, "template_roc": tr, "template_n": tpn,
                     "instvar_pr": ip, "instvar_roc": ir, "instvar_n": ipn}
        print(f"{name:>9s} | {fmt(tp):>11s} {fmt(tr):>7s} | {fmt(ip):>11s} {fmt(ir):>7s}")
    results["family"] = fam

    # --- 3. seen vs unseen session -----------------------------------------
    print("\n=== 3. seen vs unseen session (within-subset PR/ROC) ===")
    print("    session = Track-2 twin + its normal source; 'unseen' = session absent from train")
    print(f"{'model':>9s} | {'unseen PR':>9s} {'ROC':>7s} | {'seen PR':>9s} {'ROC':>7s}")
    sess = {}
    for name, s in scores.items():
        up, ur, upn, _ = within_metrics(y, s, unseen_mask)
        sp, sr, spn, _ = within_metrics(y, s, seen_mask)
        sess[name] = {"unseen_pr": up, "unseen_roc": ur, "unseen_pos": upn,
                      "seen_pr": sp, "seen_roc": sr, "seen_pos": spn}
        print(f"{name:>9s} | {fmt(up):>9s} {fmt(ur):>7s} | {fmt(sp):>9s} {fmt(sr):>7s}")
    results["session"] = sess

    # --- 5. false-positive source ------------------------------------------
    print("\n=== 5. normal chunks: mean z-score by session provenance ===")
    print("    (higher unseen than seen => unseen-session normals inflate the FP tail)")
    print(f"{'model':>9s} {'seen-normal':>12s} {'unseen-normal':>14s} {'delta':>8s}")
    fpz = {}
    for name, s in scores.items():
        z = (s - s.mean()) / (s.std() + 1e-12)
        a = float(z[seen_mask & (y == 0)].mean())
        b = float(z[unseen_mask & (y == 0)].mean())
        fpz[name] = {"seen_normal_z": a, "unseen_normal_z": b, "delta": b - a}
        print(f"{name:>9s} {a:12.3f} {b:14.3f} {b - a:8.3f}")
    results["normal_z"] = fpz

    # --- 4. per anomaly type ------------------------------------------------
    print("\n=== 4. per-anomaly-type difficulty (recall@FPR1% and per-type AP vs normals) ===")
    type_list = sorted({t for t in types if t != "normal"})
    per_type: dict = {}
    for name, s in scores.items():
        per_type[name] = {}
        for t in type_list:
            m = types == t
            if m.sum() == 0:
                continue
            ap_t, _, _, _ = pair_metrics(y, s, m)
            per_type[name][t] = {"n": int(m.sum()),
                                 "recall_fpr1": recall_at_fpr(y, s, m),
                                 "ap": ap_t}
    results["per_type"] = per_type
    for name in ("TAGN", "TCN", "logreg"):
        if name not in per_type:
            continue
        print(f"\n  {name}")
        for t in type_list:
            d = per_type[name][t]
            print(f"    n={d['n']:4d}  r@1%={d['recall_fpr1']:.3f}  AP={fmt(d['ap'])}  {t}")
    results["types"] = type_list

    # --- 6. class sensitivity: does the ranking hinge on the degenerate class? ----
    jam = np.array([str(t) == "Jamming" for t in types])
    template = ~is_gen & pos
    gen = is_gen & pos
    print("\n=== 6. pooled PR with classes removed (pooled positives vs all normals) ===")
    print(f"{'model':>9s} {'ALL':>8s} {'-Jamming':>9s} {'template':>9s} {'inst-var':>9s} "
          f"{'rsrp_probe':>11s}")
    sub = {"all": {}, "no_jam": {}, "template": {}, "gen": {}}
    for name, s in scores.items():
        pr_all = float(average_precision_score(y, s))
        pr_nj = float(average_precision_score(y[~jam], s[~jam]))
        # positives = family, negatives = all normals (pair_metrics convention)
        pr_t = pair_metrics(y, s, template)[0]
        pr_g = pair_metrics(y, s, gen)[0]
        sub["all"][name] = pr_all
        sub["no_jam"][name] = pr_nj
        sub["template"][name] = pr_t
        sub["gen"][name] = pr_g
        print(f"{name:>9s} {pr_all:8.4f} {pr_nj:9.4f} {fmt(pr_t):>9s} {fmt(pr_g):>9s}")
    results["class_sensitivity"] = sub

    # --- 7. recording-level shortcut probes ------------------------------------
    X = np.load(INTERIM / "chunks_X.npy")
    rsrp_te = X[test_ids][:, :, 0].mean(1)          # RSRP level (per-chunk mean)
    prot_te = X[test_ids][:, :, 16].mean(1)          # UL_Protocol composition
    print("\n=== 7. recording-level shortcut probes (single raw covariate, no model) ===")
    print(f"{'covariate':>16s} {'Jamming':>9s} {'template':>9s} {'inst-var':>9s} {'ALL':>9s}")
    probes = {}
    for nm, v in (("RSRP_mean", rsrp_te), ("UL_Protocol_mean", prot_te)):
        row = {}
        for key, mask in (("jamming", jam), ("template", template), ("inst-var", gen), ("all", pos)):
            sel = (y == 0) | mask
            row[key] = float(average_precision_score(y[sel], v[sel]))
        probes[nm] = row
        print(f"{nm:>16s} {row['jamming']:9.4f} {row['template']:9.4f} "
              f"{row['inst-var']:9.4f} {row['all']:9.4f}")
    results["shortcut_probes"] = probes

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
