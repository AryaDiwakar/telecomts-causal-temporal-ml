"""Streamlit dashboard for the TelecomTS project.

Run:
    streamlit run src/dashboard/app.py

Sections: Dataset, Splits, Graphs, Results (exp_001..007), Early warning.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.models.tagn import TagnModel, load_graphs, model_config

RESULTS = ROOT / "outputs" / "results"
TAB = ROOT / "outputs" / "tables"
FIG = ROOT / "outputs" / "figures"

st.set_page_config(page_title="TelecomTS RCA Dashboard", layout="wide")
st.markdown(
    """
    <style>
.block-container {max-width: 1380px; padding-top: 2rem;}
h1, h2, h3 {font-family: Georgia, serif;}
[data-testid="stMetricValue"] {color: #087f8c;}
.op-card {background: #f3f7f5; border: 1px solid #d9e5df; border-radius: 12px; padding: 14px 16px; min-height: 92px;}
.op-kicker {color: #087f8c; font-size: 0.72rem; letter-spacing: .12em; text-transform: uppercase; font-weight: 700;}
.op-card strong {display: block; font-size: 1.05rem; margin-top: 8px;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("5G TelecomTS | Causal-Temporal Root-Cause Localization")
st.caption("Evidence-first research demonstration. Open `outputs/scopus_demo.html` for the polished static version.")


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


@st.cache_data
def load_operation_data():
    X = np.load(ROOT / "data" / "interim" / "chunks_X.npy", mmap_mode="r")
    info = json.loads((ROOT / "data" / "interim" / "kpi_names.json").read_text())
    split = np.load(ROOT / "data" / "processed" / "split_indices.npz")
    return X, info["numeric_kpis"], split["test"]


@st.cache_resource
def load_operation_model():
    """Load the verified checkpoint once for the interactive operation view."""
    checkpoint = ROOT / "checkpoints" / "exp_002" / "tagn_seed42.pt"
    if not checkpoint.exists():
        return None
    cfg = model_config()
    cfg["anomaly_trunk_weight"] = 0.0
    model = TagnModel(cfg).to(cfg["device"])
    model.set_prototypes(load_graphs()[0].to(cfg["device"]))
    model.load_state_dict(torch.load(checkpoint, map_location=cfg["device"]), strict=False)
    model.eval()
    return model


def operation_inference(raw: np.ndarray, proto: int, model: TagnModel) -> dict:
    scaler = json.loads((ROOT / "data" / "processed" / "scaler.json").read_text())
    loc = np.asarray(scaler["loc"], dtype=np.float32)[:16]
    scale = np.asarray(scaler["scale"], dtype=np.float32)[:16]
    scaled = raw.copy().astype(np.float32)
    scaled[:, :16] = (scaled[:, :16] - loc) / scale
    device = next(model.parameters()).device
    x = torch.from_numpy(scaled[None]).to(device)
    p = torch.tensor([proto], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, node_states, hidden = model(x, p)
    return {
        "anomaly_logit": float(logits["anomaly"].cpu().item()),
        "type_logits": logits["type"].cpu().numpy()[0],
        "kpi_logits": logits["kpi"].cpu().numpy()[0],
        "hazard": logits["hazard"].cpu().numpy()[0],
        "node_states": node_states.cpu().numpy()[0],
        "hidden": hidden.cpu().numpy()[0],
    }


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30, 30)))


section = st.sidebar.radio(
    "Section",
    ["Live Operation & Simulation", "Overview", "Dataset", "Graphs", "Anomaly Detection", "Type & RCA",
     "Early Warning", "Robustness / LORO", "Explainability"],
)
meta = load_meta()

if section == "Live Operation & Simulation":
    st.header("Live Operation | Detect, Localize, Simulate")
    st.caption("Replay a held-out test window end to end. The what-if panel is predictive, not an identified causal intervention.")
    operation_model = load_operation_model()
    if operation_model is None:
        st.error("Missing checkpoints/exp_002/tagn_seed42.pt. Run exp_002 before opening this view.")
    else:
        X_op, kpi_names, test_ids = load_operation_data()
        test_meta = meta.iloc[test_ids].copy()
        test_meta["row_id"] = test_ids
        choices = test_meta["chunk_id"].tolist()
        anomaly_choices = test_meta.loc[test_meta["anomaly"], "chunk_id"].tolist()
        default_chunk = anomaly_choices[0] if anomaly_choices else choices[0]
        selected = st.selectbox(
            "Replay test chunk",
            choices,
            index=choices.index(default_chunk),
            format_func=lambda cid: (
                f"chunk {cid} | "
                f"{'ANOMALY' if bool(meta.loc[meta.chunk_id == cid, 'anomaly'].iloc[0]) else 'normal'} | "
                f"{meta.loc[meta.chunk_id == cid, 'anomaly_type'].iloc[0] or 'normal'}"
            ),
        )
        row = meta.loc[meta.chunk_id == selected].iloc[0]
        raw = np.asarray(X_op[int(selected)])
        proto_data = np.load(ROOT / "data" / "processed" / "graph_prototypes.npz")
        proto = int(proto_data["chunk_prototypes"][int(selected)])
        result = operation_inference(raw, proto, operation_model)
        type_map = json.loads((ROOT / "data" / "processed" / "rca_templates.json").read_text())["type_map"]
        inverse_types = {v: k for k, v in type_map.items()}
        type_id = int(result["type_logits"].argmax())
        predicted_name = inverse_types.get(type_id, "unknown")
        top_nodes = np.argsort(result["kpi_logits"])[::-1][:5]

        st.markdown(
            """
            <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin:14px 0 20px">
              <div class="op-card"><span class="op-kicker">01 / ingest</span><strong>128-step telemetry</strong><span>16 KPI + 2 protocol channels</span></div>
              <div class="op-card"><span class="op-kicker">02 / graph</span><strong>Prototype %d</strong><span>dynamic KPI dependency context</span></div>
              <div class="op-card"><span class="op-kicker">03 / detect</span><strong>TAGN hybrid path</strong><span>anchor-preserved anomaly score</span></div>
              <div class="op-card"><span class="op-kicker">04 / explain</span><strong>Type + KPI RCA</strong><span>joint operational diagnosis</span></div>
              <div class="op-card"><span class="op-kicker">05 / act</span><strong>What-if replay</strong><span>predictive intervention simulator</span></div>
            </div>
            """ % proto,
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Anomaly score", f"{sigmoid(result['anomaly_logit']):.3f}", "flagged" if row.anomaly else "normal")
        c2.metric("Predicted type", predicted_name.split("(")[0].strip())
        c3.metric("Top KPI", kpi_names[int(top_nodes[0])])
        c4.metric("Observed label", str(row.anomaly_type or "normal"))

        left, right = st.columns([1.55, 1])
        with left:
            st.subheader("Telemetry replay")
            change = np.abs(raw[:, :16][-1] - raw[:, :16].mean(axis=0))
            visible = np.argsort(change)[::-1][:6]
            chart = pd.DataFrame(raw[:, visible], columns=[kpi_names[i] for i in visible])
            st.line_chart(chart, height=330)
            st.caption("Top six channels by endpoint deviation. The full window remains available in the selected chunk.")
        with right:
            st.subheader("Root-cause ranking")
            rca_table = pd.DataFrame({
                "KPI": [kpi_names[i] for i in top_nodes],
                "model score": [round(float(result["kpi_logits"][i]), 3) for i in top_nodes],
            })
            st.dataframe(rca_table, hide_index=True, use_container_width=True)
            st.subheader("Graph activity")
            adjacency = proto_data["prototypes"][proto]
            edges = [
                {"source": kpi_names[int(source)], "target": kpi_names[int(target)], "strength": round(float(adjacency[source, target]), 3)}
                for source, target in zip(*np.where(adjacency > 0))
            ]
            st.dataframe(pd.DataFrame(edges).sort_values("strength", ascending=False).head(8), hide_index=True, use_container_width=True)

        st.subheader("Early-warning hazard")
        st.line_chart(pd.DataFrame({"hazard": sigmoid(result["hazard"])}), height=180)
        st.caption("Per-step predictive hazard. It is not a causal probability of onset.")

        st.subheader("Predictive intervention simulator")
        with st.form("intervention_form"):
            s1, s2, s3 = st.columns(3)
            intervention_kpi = s1.selectbox("Perturb KPI", kpi_names, index=int(top_nodes[0]))
            reduction = s2.slider("Reduce selected KPI (%)", 0, 80, 25, 5)
            horizon = s3.slider("Apply over final steps", 8, 128, 32, 8)
            simulate = st.form_submit_button("Run what-if simulation", type="primary")
        if simulate:
            k = kpi_names.index(intervention_kpi)
            counterfactual = raw.copy()
            counterfactual[-horizon:, k] *= 1.0 - reduction / 100.0
            counterfactual_result = operation_inference(counterfactual, proto, operation_model)
            before = float(sigmoid(result["anomaly_logit"]))
            after = float(sigmoid(counterfactual_result["anomaly_logit"]))
            s1, s2, s3 = st.columns(3)
            s1.metric("Baseline model score", f"{before:.3f}")
            s2.metric("Counterfactual score", f"{after:.3f}", f"{after - before:+.3f}")
            s3.metric("Score movement", "improved" if after < before else "worsened")
            st.info(
                f"Reducing {intervention_kpi} by {reduction}% over the final {horizon} steps changed the predictive score. "
                "This is a model what-if, not proof that changing the KPI causes the fault to disappear."
            )

elif section == "Overview":
    baseline_path = RESULTS / "exp_001_classical_baselines.json"
    loro_path = RESULTS / "exp_005_loro_seed42.json"
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())
        loro = json.loads(loro_path.read_text()) if loro_path.exists() else {}
        hybrid_path = RESULTS / "exp_009_hybrid_seed42.json"
        hybrid = json.loads(hybrid_path.read_text()) if hybrid_path.exists() else {}
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best baseline PR-AUC", f"{baseline['sup_logreg']['pr_auc']:.3f}")
        c2.metric("TAGN hybrid PR-AUC", f"{hybrid['test']['hybrid']['pr_auc']:.3f}" if hybrid else "Pending")
        c3.metric("LORO PR-AUC", f"{loro.get('aggregate', {}).get('anomaly_pr_auc', 0):.3f}" if loro else "Pending")
        c4.metric("Recording-prior Jaccard", f"{baseline['rca']['recording_prior']['jaccard_mean']:.3f}")
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
    st.subheader("exp_001 classical baselines vs TAGN hybrid (test)")
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
    hybrid_path = RESULTS / "exp_009_hybrid_seed42.json"
    if hybrid_path.exists():
        hybrid = json.loads(hybrid_path.read_text())
        st.subheader("Hybrid anomaly expert")
        st.dataframe(
            pd.DataFrame(
                [
                    {"model": name, "roc_auc": values.get("roc_auc"),
                     "pr_auc": values.get("pr_auc"),
                     "tpr_fpr_0.01": values.get("tpr_at_fpr_0.01")}
                    for name, values in hybrid["test"].items()
                ]
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
