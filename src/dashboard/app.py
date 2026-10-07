"""Streamlit dashboard for the TelecomTS project.

Run:
    streamlit run src/dashboard/app.py

Sections: Dataset, Splits, Graphs, Results (exp_001..007), Early warning.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "outputs" / "results"
TAB = ROOT / "outputs" / "tables"
FIG = ROOT / "outputs" / "figures"

st.set_page_config(page_title="TelecomTS RCA Dashboard", layout="wide")
st.title("5G TelecomTS — Causal-Temporal Root-Cause Localization")


@st.cache_data
def load_meta():
    return pd.read_parquet(ROOT / "data" / "interim" / "chunks_meta.parquet")


@st.cache_data
def load_split_stats():
    p = ROOT / "data" / "processed" / "split_stats.json"
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data
def load_registry():
    p = ROOT / "outputs" / "results_registry.json"
    if not p.exists():
        return pd.DataFrame()
    return pd.json_normalize(json.loads(p.read_text()))


@st.cache_data
def load_multi_seed():
    p = TAB / "exp_002_multi_seed_raw.csv"
    return pd.read_csv(p) if p.exists() else None


section = st.sidebar.radio(
    "Section",
    ["Overview", "Dataset", "Graphs", "Anomaly Detection", "Type & RCA",
     "Early Warning", "Robustness / LORO", "Explainability"],
)
meta = load_meta()

if section == "Overview":
    st.subheader("What this is")
    st.markdown(
        "A study measuring how far anomaly detection, anomaly-type "
        "classification and KPI-level root-cause localization can go on "
        "TelecomTS under leakage-controlled splits, plus the **recording-"
        "memorization** honest-baseline analysis. Numbers live in "
        "`outputs/results_registry.json`."
    )
    st.dataframe(load_registry(), use_container_width=True)
    if (stats := load_split_stats()):
        st.subheader("Split audit")
        st.json(stats.get("counts", {}))

elif section == "Dataset":
    st.subheader("Dataset overview")
    st.write(meta.shape, "chunks")
    c1, c2, c3 = st.columns(3)
    c1.metric("Chunks", len(meta))
    c2.metric("Anomalous", int(meta.anomaly.sum()))
    c3.metric("Prevalence", f"{meta.anomaly.mean():.2%}")
    split = pd.read_parquet(ROOT / "data" / "interim" / "chunks_meta.parquet")
    st.subheader("Anomaly types (test)")
    f = glob.glob(str(ROOT / "outputs" / "tables" / "eda" / "table_type_counts.csv"))
    if f:
        st.dataframe(pd.read_csv(f[0], index_col=0), use_container_width=True)
    fig = FIG / "eda" / "fig_class_balance.png"
    if fig.exists():
        st.image(str(fig))

elif section == "Graphs":
    st.subheader("Learned dynamic KPI graph")
    audit_path = ROOT / "data" / "processed" / "graph_audit.json"
    if audit_path.exists():
        st.json(json.loads(audit_path.read_text()))
    p = ROOT / "data" / "processed" / "graph_prototypes.npz"
    if p.exists():
        z = np.load(p)
        st.write("prototype usage:", np.bincount(z["chunk_prototypes"]).tolist())
    st.image(str(FIG / "eda" / "fig_corr_matrix.png"))

elif section == "Anomaly Detection":
    st.subheader("exp_001 classical baselines vs exp_002 TAGN (test)")
    f1 = RESULTS / "exp_001_classical_baselines.json"
    if f1.exists():
        d1 = json.loads(f1.read_text())
        st.dataframe(
            pd.DataFrame(
                [{"model": k, "roc_auc": v.get("roc_auc"), "pr_auc": v.get("pr_auc"),
                  "f1_max": v.get("f1_max"), "tpr_fpr_0.01": v.get("tpr_at_fpr_0.01")}
                 for k, v in d1.items() if isinstance(v, dict) and "roc_auc" in v]
            ),
            use_container_width=True,
        )
    ms = load_multi_seed()
    if ms is not None:
        st.subheader("TAGN multi-seed (exp_002)")
        st.dataframe(ms, use_container_width=True)
    fig = FIG / "exp" / "exp_001_ad_curves.png"
    if fig.exists():
        st.image(str(fig))

elif section == "Type & RCA":
    st.subheader("Type classification & root-cause baselines")
    f1 = RESULTS / "exp_001_classical_baselines.json"
    if f1.exists():
        d1 = json.loads(f1.read_text())
        st.json(d1.get("type_ml"))
        st.json(d1.get("rca"))
    f2 = RESULTS / "exp_002_tagn_seed42.json"
    if f2.exists():
        d2 = json.loads(f2.read_text())
        st.json({k: v for k, v in d2["test"].items() if k in ("type", "rca")})

elif section == "Early Warning":
    st.subheader("Early-warning hazard head")
    f = RESULTS / "exp_007_early_warning.json"
    if f.exists():
        d = json.loads(f.read_text())
        st.json(d)
    fig = FIG / "exp" / "exp_007_hazard_curves.png"
    if fig.exists():
        st.image(str(fig))

elif section == "Robustness / LORO":
    st.subheader("Robustness (exp_004)")
    for fp in sorted(RESULTS.glob("exp_004_robustness_seed*.json")):
        st.json(json.loads(fp.read_text()))
    st.subheader("Leave-recording-out generalization (exp_005)")
    for fp in sorted(RESULTS.glob("exp_005_loro_seed*.json")):
        st.json(json.loads(fp.read_text())["aggregate"])

elif section == "Explainability":
    st.subheader("Model attribution (exp_006)")
    f = RESULTS / "exp_006_explainability.json"
    if f.exists():
        st.json(json.loads(f.read_text())["agreement_with_template"])
        st.json(json.loads(f.read_text())["per_type"])
    fig = FIG / "exp" / "exp_006_attribution.png"
    if fig.exists():
        st.image(str(fig))