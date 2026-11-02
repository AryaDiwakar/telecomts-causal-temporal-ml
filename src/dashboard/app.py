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
V3_DATA_FINGERPRINT = "f089ca31f9e6d100"

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
st.caption("Live replay of held-out 5G telemetry with anomaly scoring, candidate KPI ranking, and predictive what-if simulation.")


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
def load_registry_records():
    p = ROOT / "outputs" / "results_registry.json"
    return json.loads(p.read_text()) if p.exists() else []


@st.cache_data
def load_multi_seed():
    p = TAB / "exp_002_multi_seed_raw.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def load_v3_headline():
    p = TAB / "headline_v3.csv"
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


def show_kpi_matrix(matrix, names, *, cmap="Blues", vmin=None, vmax=None, precision=2):
    """Render a labeled KPI-by-KPI matrix with a consistent color scale."""
    frame = pd.DataFrame(matrix, index=names, columns=names)
    styled = frame.style.background_gradient(cmap=cmap, axis=None, vmin=vmin, vmax=vmax)
    styled = styled.format(f"{{:.{precision}f}}")
    st.dataframe(styled, use_container_width=True, height=560)


section = st.sidebar.radio(
    "Section",
    ["Live Operation & Simulation", "Results & Evidence", "Walkthru", "Overview", "Dataset", "Graphs", "Anomaly Detection", "Type & RCA",
     "Early Warning", "Robustness / LORO", "Explainability"],
)
meta = load_meta()

if section == "Walkthru":
    st.header("Walkthru | Page-by-page demonstration")
    st.caption("Use the step selector as your presentation remote. The final two steps hand off to the live replay and simulation pages.")
    steps = [
        "1. Problem",
        "2. Data Reality",
        "3. Novelty",
        "4. Pipeline",
        "5. Evidence",
        "6. Live Operation",
        "7. Simulation",
        "8. Close",
    ]
    step = st.radio("Demonstration page", steps, horizontal=True)
    st.progress((steps.index(step) + 1) / len(steps))
    split_stats = load_split_stats() or {}

    if step == "1. Problem":
        st.title("Can a 5G operator go from alarm to diagnosis?")
        st.markdown("""
        **Opening sentence:** 5G telemetry produces many KPI alarms, but an alarm
        does not tell an operator what failed, which KPI to inspect, or whether
        the signal began before the incident.

        This project connects anomaly detection, fault-type recognition, KPI
        root-cause ranking, early warning, and predictive what-if replay.
        """)
        c1, c2, c3 = st.columns(3)
        c1.metric("Operational outputs", "4")
        c2.metric("Telemetry channels", "18")
        c3.metric("Window length", "12.8 s")
        st.info("Faculty takeaway: this is a network decision-support workflow, not only a binary classifier.")
    elif step == "2. Data Reality":
        st.title("What the data really represents")
        counts = split_stats.get("counts", {})
        test = counts.get("test", {})
        test_n = int(test.get("n", 0))
        test_pos = int(test.get("n_anomalous", 0))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Held-out windows", f"{test_n:,}")
        c2.metric("Held-out anomalies", f"{test_pos:,}")
        c3.metric("Test recordings", str(test.get("n_recordings", "—")))
        c4.metric("Test prevalence", f"{test_pos / test_n:.2%}" if test_n else "—")
        st.markdown("""
        This is **aggregated 5G RAN telemetry**, not per-phone data. There are no
        UE identifiers, tower IDs, GPS traces, packet captures, or subscriber
        trajectories. The model detects network/scenario behavior. These counts
        describe the corrected held-out test partition, not the full corpus.

        The split is chronological within recordings. Synthetic recordings and
        their clean source backgrounds stay in the same partition; preprocessing
        is fit on training data only.
        """)
        st.json({"train": counts.get("train", {}), "validation": counts.get("val", {}), "test": test})
    elif step == "3. Novelty":
        st.title("The important idea: make sure the model is not cheating")
        st.markdown("""
        The new part is not that we used GAT or GRU layers. Other researchers
        already use those tools. Our new part is that we test whether the model
        learned real fault patterns or used easy clues in the dataset.

        We check four possible shortcuts:

        - **File recognition:** did the model recognize which recording or
          scenario it came from?
        - **Label lookup:** did it simply learn that fault type A always maps to
          the same KPI list?
        - **Repeated windows:** did overlapping windows show the model the same
          event many times?
        - **Unstable connections:** do the KPI relationships still appear on
          new recordings?

        The graph shows which KPIs move together in the data. It does not prove
        that one KPI physically caused another.
        """)
        st.success("Say this: the neural network is the tool. Our contribution is the set of checks that show whether its answers can be trusted.")
    elif step == "4. Pipeline":
        st.title("From telemetry to operator action")
        st.markdown("""
        `telemetry -> leakage-safe split -> train-only scaling -> statistical + temporal features -> dynamic KPI graph -> TAGN -> diagnosis -> what-if replay`
        """)
        st.markdown("""
        **Model path:** causal temporal convolution, KPI-specific node encoding,
        graph attention, GRU sequence pooling, and four task heads.

        **Output path:** anomaly score, predicted type, ranked KPI causes, and
        pre-onset hazard curve.
        """)
        st.image(str(FIG / "eda" / "fig_corr_matrix.png"), caption="KPI dependency structure used to motivate candidate graph construction")
    elif step == "5. Evidence":
        st.title("Corrected v3 benchmark: show the result, including the negative one")
        headline = load_v3_headline()
        if headline is not None:
            pooled = headline.loc[headline["setting"] == "I", ["model", "seeds", "pr", "pr_ci", "roc", "roc_ci"]]
            st.dataframe(pooled, hide_index=True, use_container_width=True)
        st.markdown("""
        **Takeaway:** TCN has the highest mean PR-AUC (0.8606); TAGN's mean is
        0.7589. TAGN has the highest mean ROC-AUC (0.9543), but the paired
        recording bootstrap detects no PR-AUC advantage over logistic regression.
        The earlier hybrid headline used an invalid/mixed-vintage evaluation and
        is not a result to present.
        """)
        st.warning("PR-AUC and ROC-AUC use continuous scores. Validation data selects thresholds only for threshold-dependent operating metrics.")
    elif step == "6. Live Operation":
        st.title("Now switch to the actual operation replay")
        st.markdown("""
        Use the sidebar and click **Live Operation & Simulation**.

        Then demonstrate this exact order:

        1. Select an anomalous held-out chunk.
        2. Show the telemetry replay.
        3. Show anomaly score and predicted type.
        4. Show top KPI root causes.
        5. Show graph activity.
        6. Show the early-warning hazard curve.
        """)
        st.info("Do not explain every chart. Keep asking: what would the operator inspect next?")
    elif step == "7. Simulation":
        st.title("Then run the what-if simulation")
        st.markdown("""
        On **Live Operation & Simulation**:

        1. Select the top-ranked KPI.
        2. Reduce it over the final 32 steps.
        3. Click **Run what-if simulation**.
        4. Compare baseline and counterfactual scores.

        Say explicitly: this is a predictive replay, not proof that changing the
        KPI physically causes the fault to disappear.
        """)
        st.info("The simulation demonstrates operational usefulness without making an invalid causal claim.")
    else:
        st.title("Close with the research contribution")
        st.markdown("""
        **Closing sentence:** This project makes the complete path from
        aggregated 5G telemetry to anomaly detection, candidate root cause,
        early warning, and simulation inspectable, while measuring the shortcuts
        that can make RCA results look stronger than they really are.

        On the corrected split, TCN leads mean PR-AUC and TAGN has no detectable
        PR-AUC advantage over logistic regression. The main contribution is the
        leakage-aware evaluation and the evidence about where graph-based
        detection and template-based RCA do not yet earn their complexity.
        """)
        st.success("If faculty remember one thing: the project does not hide its shortcuts; it turns them into measurable experiments.")

elif section == "Live Operation & Simulation":
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
        c1.metric("Model anomaly score", f"{sigmoid(result['anomaly_logit']):.3f}", "ground truth: anomaly" if row.anomaly else "ground truth: normal")
        c2.metric("Predicted type", predicted_name.split("(")[0].strip())
        c3.metric("Top KPI", kpi_names[int(top_nodes[0])])
        c4.metric("Observed label", str(row.anomaly_type or "normal"))

        st.subheader("Explore this replay")
        st.caption("Choose a visualization. Each view describes a different part of the same 12.8-second test window.")
        graph_view = st.radio(
            "Visualization",
            ["KPI traces", "KPI heatmap", "Dependency map", "Root-cause ranking", "Early-warning curve"],
            horizontal=True,
            label_visibility="collapsed",
        )

        scaler = json.loads((ROOT / "data" / "processed" / "scaler.json").read_text())
        loc = np.asarray(scaler["loc"], dtype=np.float32)[:16]
        scale = np.asarray(scaler["scale"], dtype=np.float32)[:16]
        scale = np.where(scale == 0, 1.0, scale)
        scaled_kpis = (raw[:, :16].astype(np.float32) - loc) / scale
        seconds = np.arange(raw.shape[0]) / 10.0

        if graph_view == "KPI traces":
            change = np.abs(scaled_kpis[-1] - scaled_kpis.mean(axis=0))
            visible = np.argsort(change)[::-1][:6]
            default_names = [kpi_names[int(i)] for i in visible]
            selected_kpis = st.multiselect("KPI channels to plot", kpi_names, default=default_names)
            value_mode = st.radio("Signal values", ["Train-scaled", "Raw"], horizontal=True)
            indices = [kpi_names.index(name) for name in selected_kpis]
            values = scaled_kpis[:, indices] if value_mode == "Train-scaled" else raw[:, indices]
            if indices:
                trace = pd.DataFrame(values, columns=selected_kpis, index=pd.Index(seconds, name="Time (s)"))
                st.line_chart(trace, height=390)
            st.info(
                "What this is: observed KPI values at each of the 128 samples (10 samples/second). "
                "Train-scaled values use the training-set center and scale, so unlike raw values, "
                "different KPIs can be compared on a common scale. This is input telemetry, not a model explanation."
            )

        elif graph_view == "KPI heatmap":
            heat = pd.DataFrame(scaled_kpis, columns=kpi_names, index=pd.Index(np.round(seconds, 1), name="Time (s)"))
            st.dataframe(
                heat.style.background_gradient(cmap="RdYlBu_r", axis=None, vmin=-4, vmax=4).format("{:.1f}"),
                use_container_width=True,
                height=520,
            )
            st.info(
                "What this is: a color-coded view of all 16 numeric KPIs across the window. "
                "Rows are time; columns are KPIs. Red/blue intensity shows deviation from training-set "
                "location in robust-scaled units (color range clipped at ±4 for readability). It does not show importance."
            )

        elif graph_view == "Dependency map":
            adjacency = np.maximum(np.asarray(proto_data["prototypes"][proto], dtype=float), 0.0)
            show_kpi_matrix(adjacency, kpi_names, cmap="YlGnBu", vmin=0, vmax=float(adjacency.max()) or 1.0, precision=3)
            st.info(
                f"What this is: the 16×16 estimated adjacency-weight matrix for prototype {proto}, "
                "the graph pattern assigned to this chunk. Brighter cells mean stronger estimated "
                "KPI dependencies in this prototype. It is a learned statistical structure—not a "
                "physical network map, causal arrow, or proof that one KPI causes another."
            )

        elif graph_view == "Root-cause ranking":
            ranking = pd.DataFrame({
                "KPI": kpi_names,
                "Model ranking score": result["kpi_logits"].astype(float),
            }).sort_values("Model ranking score", ascending=False)
            st.bar_chart(ranking.head(8).set_index("KPI"), height=340)
            st.dataframe(ranking, hide_index=True, use_container_width=True)
            st.info(
                "What this is: the KPI head's relative scores, sorted from highest to lowest. "
                "These are ranking scores, not probabilities. They nominate signals for inspection; "
                "they do not establish causal root causes. Dataset affected-KPI labels can also be type templates."
            )

        else:
            hazard_frame = pd.DataFrame(
                {"Predicted hazard score": sigmoid(result["hazard"])},
                index=pd.Index(seconds, name="Time (s)"),
            )
            st.line_chart(hazard_frame, height=340)
            st.info(
                "What this is: the model's per-time-step early-warning head output over this window. "
                "A rise indicates the model is responding more strongly to pre-onset patterns. "
                "It is not a calibrated probability that a fault will occur, and it is not causal evidence."
            )

        with st.expander("How to explain the dashboard outputs", expanded=False):
            st.markdown(
                "- **Anomaly score:** sigmoid-transformed detector output for this window; it is a model score, not a calibrated risk probability.\n"
                "- **Predicted type:** highest-scoring fault category from the type head.\n"
                "- **Top KPI / RCA:** highest-ranked KPI candidates from the KPI head, intended to guide inspection.\n"
                "- **Graph prototype:** one of six training-derived KPI dependency patterns assigned to the window.\n"
                "- **Observed label:** dataset annotation for comparison; it is not the model prediction.\n"
                "- **What-if:** reruns the model after changing input values; it does not simulate a real network intervention."
            )

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

elif section == "Results & Evidence":
    st.header("Validated v3 results and evidence")
    st.caption("This page separates the corrected headline benchmark from task-specific diagnostics. Pre-repair and mixed-vintage results are excluded.")
    headline = load_v3_headline()
    if headline is not None:
        pooled = headline.loc[headline["setting"] == "I"].copy()
        st.subheader("1 · Pooled-window anomaly detection")
        st.dataframe(pooled[["model", "seeds", "pr", "pr_ci", "roc", "roc_ci"]], hide_index=True, use_container_width=True)
        st.caption("PR-AUC and ROC-AUC are computed from continuous scores. Reported confidence intervals resample recordings; model-versus-model claims use the paired bootstrap below.")
        curves = ROOT / "outputs" / "doc" / "review3_assets" / "review3_roc_pr_curves.png"
        if curves.exists():
            st.image(str(curves), caption="Seed-42 diagnostic curves on the same corrected 5,689-window test set; multi-seed summary is in the table above.")

    with st.expander("2 · Paired TAGN vs logistic-regression test", expanded=True):
        paired_path = RESULTS / "paired_test.json"
        if paired_path.exists():
            paired = json.loads(paired_path.read_text())
            rows = []
            for seed, values in paired["seeds"].items():
                pr = values["dPR"]
                roc = values["dROC"]
                rows.append({
                    "Seed": int(seed), "Δ PR-AUC": pr["point"],
                    "PR 95% CI": f"[{pr['lo']:+.4f}, {pr['hi']:+.4f}]",
                    "P(ΔPR ≤ 0)": pr["frac_le_0"],
                    "Δ ROC-AUC": roc["point"],
                    "ROC 95% CI": f"[{roc['lo']:+.4f}, {roc['hi']:+.4f}]",
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            st.info("Interpretation: TAGN's PR-AUC difference is negative on 4/5 seeds; every paired 95% interval includes zero. This supports ‘no detectable advantage,’ not ‘significantly worse.’")
        else:
            st.warning("Paired-bootstrap artifact is missing.")

    with st.expander("3 · Recording-held-out generalization and graph ablation"):
        if headline is not None:
            loro = headline.loc[headline["setting"] == "II"]
            st.markdown("**Leave-recordings-out (five folds)**")
            st.dataframe(loro[["model", "protocol", "pr", "pr_range", "roc", "roc_range"]], hide_index=True, use_container_width=True)
        ablation_rows = []
        for path in sorted(RESULTS.glob("exp_003_ablation_*.json")):
            artifact = json.loads(path.read_text())
            test = artifact.get("test", {}).get("anomaly", {})
            rca = artifact.get("test", {}).get("rca", {})
            if test.get("n") == 5689:
                ablation_rows.append({
                    "Variant": artifact.get("variant", path.stem.removeprefix("exp_003_ablation_")),
                    "Seed": artifact.get("seed"), "PR-AUC": test.get("pr_auc"),
                    "ROC-AUC": test.get("roc_auc"),
                    "RCA Jaccard": rca.get("type_template_jaccard"),
                })
        if ablation_rows:
            st.markdown("**Component ablation (single seed; indicative, not a significance test)**")
            st.dataframe(pd.DataFrame(ablation_rows), hide_index=True, use_container_width=True)
            st.warning("The no-graph variant is at least as strong on detection in this single-seed ablation. The current graph design has no demonstrated detection gain here.")

    with st.expander("4 · Root-cause ranking and label-shortcut audit"):
        rca_path = RESULTS / "rca_ranking_exp_002.json"
        if rca_path.exists():
            rca = json.loads(rca_path.read_text())
            rca_rows = [{"Seed": int(seed), **values} for seed, values in rca.items()]
            rca_frame = pd.DataFrame(rca_rows).sort_values("Seed")
            st.markdown("**TAGN input-driven KPI ranking across five seeds**")
            st.dataframe(rca_frame[["Seed", "recall@1", "recall@3", "recall@5", "mrr", "map", "type_accuracy"]], hide_index=True, use_container_width=True)
            mean_row = rca_frame.drop(columns="Seed").mean(numeric_only=True).to_frame("Five-seed mean").T
            st.dataframe(mean_row[["recall@1", "recall@3", "recall@5", "mrr", "map", "type_accuracy"]], hide_index=True, use_container_width=True)
        shortcut_path = RESULTS / "shortcut_audit.json"
        if shortcut_path.exists():
            shortcut = json.loads(shortcut_path.read_text())
            shortcut_frame = pd.DataFrame(shortcut["test_subsets"])
            st.markdown("**Privileged type-template oracle versus global-majority reference**")
            st.dataframe(shortcut_frame, hide_index=True, use_container_width=True)
            st.info("The type oracle is given the true fault type, so it is not deployable. Its high score on shipped faults measures how much the labels permit type lookup to explain.")

    with st.expander("5 · Early-warning results (corrected test set)"):
        ew_rows = [
            r for r in load_registry_records()
            if r.get("task") == "early_warning"
            and r.get("experiment") == "exp_002"
            and r.get("data_fingerprint") == V3_DATA_FINGERPRINT
            and r.get("status") == "COMPLETE"
        ]
        if ew_rows:
            ew_frame = pd.DataFrame([
                {"Seed": r["seed"], **r.get("metrics", {})} for r in ew_rows
            ]).drop_duplicates(subset=["Seed"]).sort_values("Seed")
            st.dataframe(ew_frame, hide_index=True, use_container_width=True)
            st.caption("Computed over 93 test windows with non-zero onset. Seed variation is visible; treat these as descriptive, not as a calibrated operational guarantee.")
        else:
            st.warning("No current-fingerprint early-warning result is registered.")

    with st.expander("6 · Benchmark repairs and scope"):
        st.markdown("""
        - The repaired test partition has **5,689 windows / 424 anomalies / 26 recordings**.
        - Clean backgrounds and their Track-2 injected twins are assigned to the same split; the identified train/test background leak is removed.
        - Synthetic affected-KPI labels were corrected; 280 spurious labels were removed across 237 of 714 windows.
        - The estimated KPI graph is an exploratory statistical dependency structure, not an identified causal graph.
        - Older pre-repair and mixed-vintage hybrid results are intentionally not shown as headline evidence.
        """)

    with st.expander("Additional registered research tracks (separate from the headline benchmark)"):
        extra_rows = [
            r for r in load_registry_records()
            if r.get("data_fingerprint") == V3_DATA_FINGERPRINT
            and r.get("status") == "COMPLETE"
            and r.get("task") in {"phase1_backbone", "phase2_fusion", "phase3_causal"}
        ]
        if extra_rows:
            extra_frame = pd.DataFrame([
                {"Task": r.get("task"), "Experiment": r.get("experiment"), "Seed": r.get("seed"),
                 "Metrics": json.dumps(r.get("metrics", {}), sort_keys=True)}
                for r in extra_rows
            ]).drop_duplicates()
            st.dataframe(extra_frame, hide_index=True, use_container_width=True)
            st.caption("These are exploratory extensions with their own objectives/configurations. Do not rank them directly against the pooled-window headline table without a matched protocol and model identity.")
        else:
            st.caption("No additional current-fingerprint research runs are registered.")

elif section == "Overview":
    st.header("Corrected v3 results at a glance")
    headline = load_v3_headline()
    split_stats = load_split_stats() or {}
    test = split_stats.get("counts", {}).get("test", {})
    if headline is not None:
        pooled = headline.loc[headline["setting"] == "I"].set_index("model")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test windows", f"{int(test.get('n', 0)):,}")
        c2.metric("Test anomalies", f"{int(test.get('n_anomalous', 0)):,}")
        c3.metric("TCN mean PR-AUC", str(pooled.loc["TCN (temporal deep)", "pr"]))
        c4.metric("TAGN mean PR-AUC", str(pooled.loc["TAGN (graph + summary)", "pr"]))
        st.dataframe(
            pooled.reset_index()[["model", "seeds", "pr", "pr_ci", "roc", "roc_ci"]],
            hide_index=True,
            use_container_width=True,
        )
        st.info("TAGN's paired recording-bootstrap comparison finds no detectable PR-AUC advantage over logistic regression. See docs/results_v3.md for the full analysis.")
    st.subheader("What this is")
    st.markdown(
        "A study measuring how far anomaly detection, anomaly-type "
        "classification and KPI-level root-cause localization can go on "
        "TelecomTS under leakage-controlled splits, plus the **recording-"
        "memorization** honest-baseline analysis. Numbers live in "
        "`outputs/results_registry.json`."
    )
    if split_stats:
        st.subheader("Leakage-controlled split inventory")
        st.json(split_stats.get("counts", {}))

elif section == "Dataset":
    st.subheader("Dataset overview")
    st.write(f"{len(meta):,} processed windows in the current metadata artifact")
    c1, c2, c3 = st.columns(3)
    c1.metric("Chunks", len(meta))
    c2.metric("Anomalous", int(meta.anomaly.sum()))
    c3.metric("Prevalence", f"{meta.anomaly.mean():.2%}")
    st.caption("These are whole-artifact counts. The leakage-controlled test-set counts are shown separately below.")
    _, _, test_ids = load_operation_data()
    test_meta = meta.iloc[test_ids]
    st.subheader("Corrected held-out test partition")
    st.dataframe(
        test_meta.loc[test_meta.anomaly, "anomaly_type"].value_counts().rename_axis("Fault type").reset_index(name="Test anomalies"),
        hide_index=True,
        use_container_width=True,
    )
    fig = FIG / "eda" / "fig_class_balance.png"
    if fig.exists():
        st.image(str(fig))

elif section == "Graphs":
    st.header("KPI dependency graph explorer")
    st.markdown(
        "The model uses a small set of graph prototypes to represent estimated relationships among the 16 numeric KPIs. "
        "These are data-derived statistical dependencies—not verified physical links or causal structure."
    )
    audit_path = ROOT / "data" / "processed" / "graph_audit.json"
    audit = {}
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
    p = ROOT / "data" / "processed" / "graph_prototypes.npz"
    kpi_names = json.loads((ROOT / "data" / "interim" / "kpi_names.json").read_text())["numeric_kpis"]
    if p.exists():
        z = np.load(p)
        prototypes = z["prototypes"]
        usage = np.bincount(z["chunk_prototypes"], minlength=len(prototypes))
        c1, c2, c3 = st.columns(3)
        c1.metric("Graph prototypes", len(prototypes))
        c2.metric("Graph-estimation bins", audit.get("n_bins", "—"))
        c3.metric("Mean edges per bin", f"{audit.get('mean_edges', 0):.1f}" if audit else "—")
        choice = st.selectbox("Choose graph view", ["Assigned prototype", "Edge frequency across bins", "Raw KPI correlation"],
                              help="The prototype view shows a model input pattern; edge frequency summarizes how often a relationship appears across graph-estimation bins; correlation is a simple pairwise baseline.")
        if choice == "Assigned prototype":
            proto = st.select_slider("Prototype", options=list(range(len(prototypes))), value=0)
            matrix = np.maximum(np.asarray(prototypes[proto], dtype=float), 0.0)
            show_kpi_matrix(matrix, kpi_names, cmap="YlGnBu", vmin=0, vmax=float(matrix.max()) or 1.0, precision=3)
            st.caption(f"Prototype {proto}: model's estimated adjacency weights for one recurring KPI-dependency pattern. Chunk usage counts by prototype: {usage.tolist()}.")
        elif choice == "Edge frequency across bins":
            freq = np.asarray(audit.get("edge_frequency", []), dtype=float)
            if freq.shape == (len(kpi_names), len(kpi_names)):
                show_kpi_matrix(freq, kpi_names, cmap="PuBu", vmin=0, vmax=1, precision=2)
                st.caption("Cell value = fraction of graph-estimation bins in which the pair appears. This measures recurrence, not model importance or causality.")
            else:
                st.warning("The edge-frequency audit matrix is unavailable.")
        else:
            fig = FIG / "eda" / "fig_corr_matrix.png"
            if fig.exists():
                st.image(str(fig), caption="Pairwise KPI correlation snapshot. Correlation shows co-movement; it cannot identify cause and effect.")
            else:
                st.warning("The correlation figure is unavailable.")
    else:
        st.warning("Graph prototype artifact is missing. Regenerate the graph artifacts before presenting this view.")

elif section == "Anomaly Detection":
    st.subheader("Corrected v3 pooled-window benchmark")
    headline = load_v3_headline()
    if headline is not None:
        st.dataframe(
            headline.loc[headline["setting"] == "I", ["model", "seeds", "pr", "pr_ci", "roc", "roc_ci"]],
            hide_index=True,
            use_container_width=True,
        )
    st.caption("PR-AUC/ROC-AUC are score-ranking metrics; intervals resample recordings. Historical hybrid results from the pre-repair split are excluded.")
    fig = ROOT / "outputs" / "doc" / "review3_assets" / "review3_roc_pr_curves.png"
    if fig.exists():
        st.image(str(fig))

elif section == "Type & RCA":
    st.subheader("Type classification and candidate KPI localization")
    st.caption("Current corrected-v3 checkpoint results. KPI rankings are candidate explanations, not verified physical causes.")
    rca_path = RESULTS / "rca_ranking_exp_002.json"
    if rca_path.exists():
        rca = json.loads(rca_path.read_text())
        frame = pd.DataFrame([{"seed": int(seed), **metrics} for seed, metrics in rca.items()]).sort_values("seed")
        st.dataframe(frame[["seed", "type_accuracy", "mrr", "map", "recall@1", "recall@3", "recall@5"]], hide_index=True, use_container_width=True)
        st.info("Mean type accuracy is about 0.224; mean MRR is 0.671 and MAP is 0.533. Ground-truth root sets have a median of 9/16 KPIs, so interpret Recall@1 alongside ranking metrics.")
    shortcut_path = RESULTS / "shortcut_audit.json"
    if shortcut_path.exists():
        st.subheader("How much can a fault-type lookup explain?")
        shortcut = pd.DataFrame(json.loads(shortcut_path.read_text())["test_subsets"])
        st.dataframe(shortcut, hide_index=True, use_container_width=True)
        st.caption("The true-type oracle is deliberately privileged and not deployable. The instance-varying subset is more demanding, though its labels are synthetic.")

elif section == "Early Warning":
    st.subheader("Early-warning results on corrected v3")
    st.caption("Per-window hazard is shown in Live Operation; this page summarizes registered test results across TAGN seeds.")
    ew_rows = [
        r for r in load_registry_records()
        if r.get("task") == "early_warning" and r.get("experiment") == "exp_002"
        and r.get("data_fingerprint") == V3_DATA_FINGERPRINT and r.get("status") == "COMPLETE"
    ]
    if ew_rows:
        ew_frame = pd.DataFrame([{"seed": r["seed"], **r["metrics"]} for r in ew_rows]).drop_duplicates("seed").sort_values("seed")
        st.dataframe(ew_frame, hide_index=True, use_container_width=True)
        st.info("The early-warning evaluation uses 93 test windows with non-zero onset. Results vary by seed and should be treated as descriptive, not a deployment guarantee.")
    else:
        st.warning("No corrected-v3 registered early-warning results are available.")

elif section == "Robustness / LORO":
    st.subheader("Generalization and robustness status")
    headline = load_v3_headline()
    if headline is not None:
        loro = headline.loc[headline["setting"] == "II"]
        st.markdown("**Corrected leave-recordings-out result**")
        st.dataframe(loro[["model", "protocol", "pr", "pr_range", "roc", "roc_range"]], hide_index=True, use_container_width=True)
    robust_rows = [r for r in load_registry_records() if r.get("task") == "robustness" and r.get("data_fingerprint") == V3_DATA_FINGERPRINT and r.get("status") == "COMPLETE"]
    if robust_rows:
        st.dataframe(pd.json_normalize(robust_rows), hide_index=True, use_container_width=True)
    else:
        st.warning("No robustness result is registered against the corrected v3 data fingerprint. Historical robustness files are not presented as current evidence.")

elif section == "Explainability":
    st.subheader("Explainability evidence status")
    current = [r for r in load_registry_records() if r.get("task") == "explainability" and r.get("data_fingerprint") == V3_DATA_FINGERPRINT and r.get("status") == "COMPLETE"]
    if current:
        st.dataframe(pd.json_normalize(current), hide_index=True, use_container_width=True)
    else:
        st.warning("No explainability run is registered against corrected v3 data. Older attribution plots are not shown as validated results.")
    st.info("The live RCA ranking is an inspectable model output, but should not be mistaken for a validated causal explanation.")
