"""exp_001 - classical / non-temporal baselines.

High-level anomaly detection (unsupervised + supervised), anomaly-type
classification, and root-cause baselines for the TelecomTS benchmark.

Outputs:
  outputs/results/exp_001_*.json, outputs/tables/exp_001_*.csv,
  outputs/figures/exp/exp_001_*.png, results_registry entries (seed 42).

Usage:
    python -m src.experiments.exp_001_classical_baselines
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, average_precision_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
RES = ROOT / "outputs" / "results"
TAB = ROOT / "outputs" / "tables"
FIG = ROOT / "outputs" / "figures" / "exp"

import sys

sys.path.insert(0, str(ROOT))
from src.evaluation.metrics import pr_f1_at_threshold, summarize  # noqa: E402
from src.features.classical import extract_features, make_feature_names  # noqa: E402
from src.utils.registry import append  # noqa: E402

SEED = 42


def load_scaled() -> tuple[np.ndarray, pd.DataFrame, dict, dict, dict, list[str]]:
    X = np.load(INTERIM / "chunks_X.npy")
    meta = pd.read_parquet(INTERIM / "chunks_meta.parquet")
    with open(INTERIM / "kpi_names.json") as fh:
        info = json.load(fh)
    kpis = info["numeric_kpis"]
    scaler = json.loads((PROCESSED / "scaler.json").read_text())
    # apply train-fit standardization to the 16 numeric channels
    loc = np.asarray(scaler["loc"])[:16]
    scale = np.asarray(scaler["scale"])[:16]
    Xs = X.copy()
    Xs[:, :, :16] = (X[:, :, :16] - loc) / scale
    idx = np.load(PROCESSED / "split_indices.npz")
    split_map = {}
    for name in ("train", "val", "test"):
        for c in idx[name]:
            split_map[int(c)] = name
    meta["split"] = meta.chunk_id.map(split_map)
    targets = dict(np.load(PROCESSED / "targets.npz"))
    return Xs, meta, targets, info, idx, kpis


def val_tuned_threshold(y_val: np.ndarray, s_val: np.ndarray) -> float:
    return float(pr_f1_at_threshold(y_val, s_val)["threshold"])


def main() -> None:
    RES.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    X, meta, tar, info, idx, kpis = load_scaled()
    num = len(kpis)
    y = tar["y_anomaly"]

    train_mask = meta.split == "train"
    val_mask = meta.split == "val"
    test_mask = meta.split == "test"

    F = extract_features(X, num)
    feature_names = make_feature_names(kpis, num)
    Ftr, Fva, Fte = F[train_mask], F[val_mask], F[test_mask]
    ytr, yva, yte = y[train_mask], y[val_mask], y[test_mask]
    meta_tr, meta_va, meta_te = meta[train_mask], meta[val_mask], meta[test_mask]
    Xte = X[test_mask]

    nrm_tr = (ytr == 0)
    rng = np.random.default_rng(SEED)

    results: dict[str, dict] = {}

    # ---------------- unsupervised anomaly detectors ----------------
    fitter_subsample = 6000
    norm_tr_fit = Ftr[nrm_tr]
    if len(norm_tr_fit) > fitter_subsample:
        sel = rng.choice(len(norm_tr_fit), fitter_subsample, replace=False)
        norm_tr_fit = norm_tr_fit[sel]

    from sklearn.ensemble import IsolationForest

    unsup = {
        "isolation_forest": IsolationForest(
            n_estimators=300, contamination="auto", random_state=SEED, n_jobs=-1
        ),
        "one_class_svm": OneClassSVM(nu=0.05, kernel="rbf", gamma="scale"),
    }
    lof = LocalOutlierFactor(
        n_neighbors=20, contamination="auto", novelty=True, n_jobs=-1
    )

    for name, model in unsup.items():
        model.fit(norm_tr_fit)
        s_test = model.score_samples(Fte)
        s_val = model.score_samples(Fva)
        res = summarize(yte, s_test)
        res["threshold_val_tuned"] = val_tuned_threshold(yva, s_val)
        res.update(pr_f1_at_threshold(yte, s_test))
        results[f"unsup_{name}"] = res
        print(name, res)

    lof.fit(norm_tr_fit)
    s_test = lof.score_samples(Fte)
    s_val = lof.score_samples(Fva)
    res = summarize(yte, s_test)
    res["threshold_val_tuned"] = val_tuned_threshold(yva, s_val)
    res.update(pr_f1_at_threshold(yte, s_test))
    results["unsup_lof"] = res
    print("lof", res)

    # ---------------- supervised anomaly classifiers ----------------
    sup = {
        "xgb": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="aucpr",
            tree_method="hist",
            random_state=SEED,
            n_jobs=-1,
        ),
        "logreg": LogisticRegression(max_iter=5000, class_weight="balanced", n_jobs=-1),
    }
    for name, model in sup.items():
        model.fit(Ftr, ytr)
        s_val = model.predict_proba(Fva)[:, 1]
        s_test = model.predict_proba(Fte)[:, 1]
        res = summarize(yte, s_test)
        res["threshold_val_tuned"] = val_tuned_threshold(yva, s_val)
        res.update(pr_f1_at_threshold(yte, s_test))
        results[f"sup_{name}"] = res
        print(name, res)

    # ---------------- type classification (anomalous only) ----------------
    ytype_present = tar["y_type"] >= 0
    tr_pos = np.where(train_mask)[0]
    te_pos = np.where(test_mask)[0]
    tr_a = tr_pos[ytype_present[tr_pos]]
    te_a = te_pos[ytype_present[te_pos]]
    type_names = sorted(meta.loc[ytype_present, "anomaly_type"].unique())
    type_map = {t: i for i, t in enumerate(type_names)}
    ytype_tr = np.array([type_map[meta.anomaly_type.iloc[i]] for i in tr_a])
    ytype_te = np.array([type_map[meta.anomaly_type.iloc[i]] for i in te_a])

    clf_type = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.1, eval_metric="mlogloss",
        tree_method="hist", random_state=SEED, n_jobs=-1,
    )
    clf_type.fit(F[tr_a], ytype_tr)
    from sklearn.metrics import f1_score

    pte = clf_type.predict(F[te_a])
    type_acc = float(np.mean(pte == ytype_te))
    macro_f1_seen = float(f1_score(ytype_te, pte, average="macro", zero_division=0))
    micro_f1 = float(f1_score(ytype_te, pte, average="micro", zero_division=0))
    # per-class recall over all classes present in train (unseen test classes -> 0)
    recalls = []
    for c in range(len(type_names)):
        gt_c = (ytype_te == c)
        if gt_c.sum() > 0:
            recalls.append((pte[gt_c] == c).mean())
    macro_recall_over_seen = float(np.mean(recalls)) if recalls else float("nan")
    type_res = {
        "type_accuracy": type_acc,
        "type_macro_f1_seen": macro_f1_seen,
        "type_micro_f1": micro_f1,
        "type_macro_recall_over_seen": macro_recall_over_seen,
        "test_types_present": len(recalls),
        "test_types_total": len(type_names),
    }
    te_anom = te_a  # positional (global) indices of anomalous test rows
    te_meta_rows = meta.iloc[te_anom]
    yte_kpi = tar["y_affected"][te_anom]  # [n, 16]
    results["type_ml"] = type_res
    print("type xgb acc", type_acc, "macro-f1-seen", macro_f1_seen, "micro-f1", micro_f1)
    pd.DataFrame(
        {"true": [type_names[i] for i in ytype_te], "pred": [type_names[i] for i in pte]}
    ).to_csv(TAB / "exp_001_type_predictions.csv", index=False)

    # ---------------- root-cause baselines ----------------
    templates = json.loads((PROCESSED / "rca_templates.json").read_text())["rca_templates"]
    tpl_idx = {k: i for i, k in enumerate(kpis)}

    def template_of(pred_types: np.ndarray, name_map) -> list[set]:
        return [
            {tpl_idx[k] for k in templates[name_map[i]] if k in tpl_idx}
            for i in pred_types
        ]

    # (a) type-prior template via ML type model
    gt_sets = [set(np.where(r)[0]) for r in yte_kpi]
    seen_types_te = list(te_meta_rows.anomaly_type)

    pred_sets_ml = template_of(pte, type_names)  # uses pte aligned to te_anom order
    jac_ml = [
        len(a & b) / max(1, len(a | b)) for a, b in zip(pred_sets_ml, gt_sets)
    ]

    # (b) recording-prior: most frequent type in the recording (train stats)
    rec_prior = (
        meta_tr.loc[(ytr == 1), ["recording", "anomaly_type"]]
        .groupby("recording")["anomaly_type"]
        .agg(lambda s: s.value_counts().index[0])
        .to_dict()
    )
    pred_types_rec = [rec_prior.get(r) or "Jamming" for r in te_meta_rows.recording]
    pred_sets_rec = [{tpl_idx[k] for k in templates[tn] if k in tpl_idx} for tn in pred_types_rec]
    jac_rec = [len(a & b) / max(1, len(a | b)) for a, b in zip(pred_sets_rec, gt_sets)]

    # (c) input-driven: top-K largest |z-mean shift| KPIs per chunk
    with open(PROCESSED / "scaler.json") as fh:
        scaler = json.load(fh)
    loc = np.asarray(scaler["loc"])[:num]
    scale = np.asarray(scaler["scale"])[:num]
    norm_mean = loc
    norm_std = scale
    zshift = np.abs((X[te_anom][:, :, :num].mean(axis=1) - norm_mean) / norm_std)
    k_sizes = [len(s) for s in gt_sets]
    pred_sets_shift = [
        set(np.argsort(zshift[i])[::-1][:k_sizes[i]].tolist()) for i in range(len(zshift))
    ]
    jac_shift = [
        len(a & b) / max(1, len(a | b))
        for a, b in zip(pred_sets_shift, gt_sets)
    ]

    # micro-F1 treating each KPI as an independent localizer
    def micro_f1(pred_sets):
        tp = sum(len(a & b) for a, b in zip(pred_sets, gt_sets))
        fp = sum(len(a - b) for a, b in zip(pred_sets, gt_sets))
        fn = sum(len(b - a) for a, b in zip(pred_sets, gt_sets))
        return 2 * tp / max(1, 2 * tp + fp + fn)

    rca = {
        "type_prior_ml": {"jaccard_mean": float(np.mean(jac_ml)), "micro_f1": micro_f1(pred_sets_ml)},
        "recording_prior": {"jaccard_mean": float(np.mean(jac_rec)), "micro_f1": micro_f1(pred_sets_rec)},
        "input_shift_topK": {"jaccard_mean": float(np.mean(jac_shift)), "micro_f1": micro_f1(pred_sets_shift)},
    }
    results["rca"] = rca
    pd.DataFrame(
        {
            "anomaly_type": seen_types_te,
            "recording": te_meta_rows.recording,
            "jaccard_type_prior": jac_ml,
            "jaccard_rec_prior": jac_rec,
            "jaccard_input_shift": jac_shift,
        }
    ).to_csv(TAB / "exp_001_rca_chunk_level.csv", index=False)
    print("RCA:", json.dumps(rca, indent=2))

    # ---------------- ROC / PR curves ----------------
    from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

    scorer_funs = {
        "XGBoost (sup.)": ("sup_xgb", "prob"),
        "LogReg (sup.)": ("sup_logreg", "prob"),
        "IsolationForest": ("unsup_isolation_forest", "score"),
        "One-Class SVM": ("unsup_one_class_svm", "score"),
        "LOF": ("unsup_lof", "score"),
    }
    score_cache: dict[str, np.ndarray] = {}
    for name, (key, kind) in scorer_funs.items():
        if kind == "prob":
            model_key = key.replace("sup_", "")  # sup dict keyed by "xgb"/"logreg"
            score_cache[name] = sup[model_key].predict_proba(Fte)[:, 1]
        else:
            score_cache[name] = unsup[key].score_samples(Fte) if key in unsup else lof.score_samples(Fte)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for name in score_cache:
        RocCurveDisplay.from_predictions(yte, score_cache[name], ax=axes[0], name=name)
        PrecisionRecallDisplay.from_predictions(yte, score_cache[name], ax=axes[1], name=name)
    axes[0].set_title("Anomaly detection ROC (test split)")
    axes[1].set_title("Anomaly detection PR (test split)")
    axes[1].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "exp_001_ad_curves.png", dpi=150)
    plt.close(fig)

    # ---------------- save outputs ----------------
    all_res = {"meta": {"seed": SEED, "dataset": "TelecomTS", "config": "exp_001"}, **results}
    with open(RES / "exp_001_classical_baselines.json", "w") as fh:
        json.dump(all_res, fh, indent=2)

    rows = [
        {"model": k, **v}
        for k, v in results.items()
        if k not in ("rca", "type_ml")
    ]
    pd.DataFrame(rows).to_csv(TAB / "exp_001_ad_results.csv", index=False)

    for name, res in results.items():
        if "roc_auc" in res:
            append(
                {
                    "experiment": "exp_001",
                    "seed": SEED,
                    "task": "anomaly_detection",
                    "model": name,
                    "metrics": {k: v for k, v in res.items() if isinstance(v, (int, float))},
                }
            )
    append({"experiment": "exp_001", "seed": SEED, "task": "type_classification", "model": "xgb", "metrics": type_res})
    append({"experiment": "exp_001", "seed": SEED, "task": "root_cause_baselines", "model": "rca", "metrics": rca})


if __name__ == "__main__":
    main()