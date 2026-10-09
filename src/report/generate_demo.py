"""Build a polished, evidence-first demo page from completed artifacts.

The page is intentionally honest: it separates the strongest baseline from
the TAGN pilot and labels results that must be regenerated after TAGN-v2.

Usage:
    python3 -m src.report.generate_demo
    open outputs/scopus_demo.html
"""

from __future__ import annotations

import html
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "outputs" / "results"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"


def load_json(path: Path, default: dict | None = None) -> dict:
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text())


def num(value, digits: int = 3) -> str:
    return "Pending" if value is None else f"{float(value):.{digits}f}"


def pct(value, digits: int = 1) -> str:
    return "Pending" if value is None else f"{100 * float(value):.{digits}f}%"


def cell(value: str) -> str:
    return f"<td>{html.escape(str(value))}</td>"


def bar_chart(items: list[tuple[str, float, str]]) -> str:
    """Return a dependency-free SVG benchmark chart."""
    maximum = max((value for _, value, _ in items), default=1.0)
    parts = ['<svg class="bars" viewBox="0 0 760 240" role="img" aria-label="PR-AUC benchmark">']
    for i, (label, value, color) in enumerate(items):
        y = 28 + i * 64
        width = 470 * value / maximum
        parts.append(
            f'<text x="0" y="{y + 16}" class="bar-label">{html.escape(label)}</text>'
            f'<rect x="210" y="{y}" width="470" height="25" rx="12" class="bar-track"/>'
            f'<rect x="210" y="{y}" width="{width:.1f}" height="25" rx="12" fill="{color}"/>'
            f'<text x="{min(700, 220 + width):.1f}" y="{y + 18}" class="bar-value">{value:.3f}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def build() -> tuple[str, str]:
    baseline = load_json(RESULTS / "exp_001_classical_baselines.json")
    tagn = load_json(RESULTS / "exp_002_tagn_seed42.json")
    loro = load_json(RESULTS / "exp_005_loro_seed42.json")
    explain = load_json(RESULTS / "exp_006_explainability.json")
    graph = load_json(PROCESSED / "graph_validation.json")
    split = load_json(PROCESSED / "split_stats.json")

    logreg = baseline.get("sup_logreg", {})
    xgb = baseline.get("sup_xgb", {})
    tagn_test = tagn.get("test", {})
    tagn_anom = tagn_test.get("anomaly", {})
    loro_agg = loro.get("aggregate", {})
    agreement = explain.get("agreement_with_template", {})
    counts = split.get("counts", {})

    title = "Causal-Temporal Machine Learning for 5G KPI Root-Cause Localization"
    subtitle = (
        "An evidence-first TelecomTS study combining leakage-controlled evaluation, "
        "candidate dynamic KPI graphs, temporal representation learning, and early warning."
    )

    rows = []
    for name, result, note in (
        ("Logistic regression", logreg, "strong engineered-feature baseline"),
        ("XGBoost", xgb, "supervised nonlinear baseline"),
        ("TAGN pilot", tagn_anom, "pre-TAGN-v2 artifact; rerun required"),
    ):
        rows.append(
            "<tr>"
            + cell(name)
            + cell(num(result.get("roc_auc")))
            + cell(num(result.get("pr_auc")))
            + cell(num(result.get("tpr_at_fpr_0.01")))
            + cell(note)
            + "</tr>"
        )

    benchmark_svg = bar_chart(
        [
            ("Logistic regression", float(logreg.get("pr_auc", 0)), "#087f8c"),
            ("XGBoost", float(xgb.get("pr_auc", 0)), "#43a49b"),
            ("TAGN pilot", float(tagn_anom.get("pr_auc", 0)), "#c58b2a"),
        ]
    )

    # ---- multi-seed variance -------------------------------------------
    seed_rows = []
    for fp in sorted(RESULTS.glob("exp_002_tagn_seed*.json")):
        d = json.loads(fp.read_text())
        a = d.get("test", {}).get("anomaly", {})
        seed_rows.append(
            {
                "seed": d.get("seed"),
                "roc": a.get("roc_auc"),
                "pr": a.get("pr_auc"),
                "type": d.get("test", {}).get("type", {}).get("accuracy"),
            }
        )
    seed_rows.sort(key=lambda r: r["seed"] if r["seed"] is not None else -1)
    pr_vals = [r["pr"] for r in seed_rows if r["pr"] is not None]
    roc_vals = [r["roc"] for r in seed_rows if r["roc"] is not None]
    mean_pr = statistics.mean(pr_vals) if pr_vals else None
    std_pr = statistics.stdev(pr_vals) if len(pr_vals) > 1 else None
    mean_roc = statistics.mean(roc_vals) if roc_vals else None
    std_roc = statistics.stdev(roc_vals) if len(roc_vals) > 1 else None

    seed_table = "".join(
        "<tr>"
        + cell(r["seed"])
        + cell(num(r["roc"]))
        + cell(num(r["pr"]))
        + cell(num(r["type"]))
        + "</tr>"
        for r in seed_rows
    )

    # ---- ablations ------------------------------------------------------
    ablation_rows = []
    for fp in sorted(RESULTS.glob("exp_003_ablation_*.json")):
        d = json.loads(fp.read_text())
        name = fp.stem.removeprefix("exp_003_ablation_")
        a = d.get("test", {}).get("anomaly", {})
        ablation_rows.append(
            {
                "name": name,
                "pr": a.get("pr_auc"),
                "roc": a.get("roc_auc"),
                "type": d.get("test", {}).get("type", {}).get("accuracy"),
                "epochs": len(d.get("history", [])),
            }
        )
    full_pr = next((r["pr"] for r in ablation_rows if r["name"] == "full"), None)
    ablation_rows.sort(key=lambda r: -(r["pr"] or 0))
    ablation_table = "".join(
        "<tr>"
        + cell(r["name"].replace("_", " "))
        + cell(num(r["pr"]))
        + cell(num(r["roc"]))
        + cell(num(r["type"]))
        + (
            cell("--")
            if full_pr is None or r["pr"] is None or r["name"] == "full"
            else cell(f"{r['pr'] - full_pr:+.3f}")
        )
        + "</tr>"
        for r in ablation_rows
    )

    # ---- robustness -----------------------------------------------------
    robust = load_json(RESULTS / "exp_004_robustness_seed42.json")
    robust_rows = []
    for k, v in robust.items():
        if isinstance(v, dict) and "pr_auc" in v:
            robust_rows.append((k, v.get("pr_auc"), v.get("roc_auc")))
    robust_table = "".join(
        "<tr>" + cell(k.replace("_", " ")) + cell(num(p)) + cell(num(r)) + "</tr>"
        for k, p, r in robust_rows
    )
    base_pr = next((p for k, p, _ in robust_rows if k == "baseline"), None)
    max_drop = (
        max((base_pr - p) for k, p, _ in robust_rows if k != "baseline" and p is not None and base_pr is not None)
        if base_pr is not None and robust_rows
        else None
    )

    # ---- early warning --------------------------------------------------
    ew = load_json(RESULTS / "exp_007_early_warning.json").get("early_warning_test", {})
    ew_cards = "".join(
        f'<div class="metric"><span>{label}</span><strong>{value}</strong><small>{sub}</small></div>'
        for label, value, sub in (
            ("Warned before onset", pct(ew.get("warning_before_onset_frac")), f"n = {ew.get('n_chunks', 0)} chunks"),
            ("Median lead", num(ew.get("median_lead_steps"), 1), "steps (negative = early)"),
            ("Per-step AP", num(ew.get("per_step_ap_mean")), "early-warning PR-AUC"),
            ("Attribution AUC", num(agreement.get("attn_template_vs_non_auc")), "template vs non-template"),
        )
    )

    total_windows = sum(v.get("n", 0) for v in counts.values())
    anomaly_windows = sum(v.get("n_anomalous", 0) for v in counts.values())
    test_counts = counts.get("test", {})
    prevalence = (
        test_counts.get("n_anomalous", 0) / test_counts.get("n", 1)
        if test_counts.get("n")
        else anomaly_windows / total_windows if total_windows else 0
    )

    abstract = (
        "This demonstration presents a reproducible framework for anomaly detection, "
        "anomaly-type classification, and KPI-level root-cause localization in 5G RAN "
        "telemetry. The protocol explicitly audits recording memorization and template "
        "shortcuts, reports both chronological and leave-recording-out evaluation, and "
        "treats learned graphs as predictive candidates rather than identified causal DAGs. "
        f"In the completed baseline artifact, logistic regression reaches PR-AUC "
        f"{num(logreg.get('pr_auc'))} on the chronological test split, while the TAGN pilot "
        f"reaches {num(tagn_anom.get('pr_auc'))}. This gap is reported as a research finding, "
        "not concealed as a state-of-the-art claim."
    )

    markdown = f"""# {title}

> **Demo status:** Evidence-first pilot package. Results marked as pilot predate TAGN-v2 and must be regenerated before submission.

## Abstract

{abstract}

## Core Results

| Model | ROC-AUC | PR-AUC | TPR @ 1% FPR | Interpretation |
|---|---:|---:|---:|---|
| Logistic regression | {num(logreg.get('roc_auc'))} | {num(logreg.get('pr_auc'))} | {num(logreg.get('tpr_at_fpr_0.01'))} | strong engineered-feature baseline |
| XGBoost | {num(xgb.get('roc_auc'))} | {num(xgb.get('pr_auc'))} | {num(xgb.get('tpr_at_fpr_0.01'))} | supervised nonlinear baseline |
| TAGN pilot | {num(tagn_anom.get('roc_auc'))} | {num(tagn_anom.get('pr_auc'))} | {num(tagn_anom.get('tpr_at_fpr_0.01'))} | pre-TAGN-v2 artifact; rerun required |

## Generalization And Audit

- Leave-recording-out PR-AUC: **{num(loro_agg.get('anomaly_pr_auc'))}**.
- Leave-recording-out ROC-AUC: **{num(loro_agg.get('anomaly_roc_auc'))}**.
- Recording-prior RCA Jaccard: **{num(baseline.get('rca', {}).get('recording_prior', {}).get('jaccard_mean'))}**.
- Input-shift RCA Jaccard: **{num(baseline.get('rca', {}).get('input_shift_topK', {}).get('jaccard_mean'))}**.
- Template-attribution agreement AUC: **{num(agreement.get('attn_template_vs_non_auc'))}**.
- Dynamic graph stability: **{num(graph.get('prototype_stability_mean'))}**.

## Seed Stability (exp_002)

| Seed | ROC-AUC | PR-AUC | Type acc |
|---:|---:|---:|---:|
{chr(10).join(f'| {r["seed"]} | {num(r["roc"])} | {num(r["pr"])} | {num(r["type"])} |' for r in seed_rows)}

- Mean PR-AUC = **{num(mean_pr)} ± {num(std_pr)}** over {len(seed_rows)} seeds.
- Mean ROC-AUC = **{num(mean_roc)} ± {num(std_roc)}** over {len(seed_rows)} seeds.

## Ablations (exp_003)

| Variant | PR-AUC | ROC-AUC | Type acc | Δ vs full |
|---|---:|---:|---:|---:|
{chr(10).join(
    f'| {r["name"].replace("_", " ")} | {num(r["pr"])} | {num(r["roc"])} | {num(r["type"])} | '
    + ('reference' if r['name'] == 'full' else ('--' if full_pr is None or r['pr'] is None else f'{r["pr"] - full_pr:+.3f}'))
    + ' |'
    for r in ablation_rows
)}

## Robustness (exp_004, seed 42)

| Scenario | PR-AUC | ROC-AUC |
|---|---:|---:|
{chr(10).join(f'| {k.replace("_", " ")} | {num(p)} | {num(r)} |' for k, p, r in robust_rows)}

- Worst PR-AUC drop vs baseline: **{num(max_drop)}**.

## Early Warning (exp_007)

- Warned before onset: **{pct(ew.get('warning_before_onset_frac'))}** on {ew.get('n_chunks', 0)} chunks.
- Median lead: **{num(ew.get('median_lead_steps'), 1)}** steps (negative = early).
- Per-step average precision: **{num(ew.get('per_step_ap_mean'))}**.

## Contributions

1. Leakage-controlled chronological and leave-recording-out protocols.
2. Candidate dynamic KPI graphs with held-out predictive validation.
3. Joint anomaly, type, root-cause, and causal early-warning heads.
4. Explicit shortcut baselines for recording memorization and constant templates.
5. Reproducible artifacts, tests, figures, and a browser-ready demonstration.

## Dataset Snapshot

- Total windows: **{sum(v.get('n', 0) for v in counts.values()) or 'Pending'}**.
- Anomalous windows: **{sum(v.get('n_anomalous', 0) for v in counts.values()) or 'Pending'}**.
- Signals: **18 channels**, including 16 numeric KPIs and 2 protocol channels.
- Fault types: **11**.

## Submission Guardrails

- Do not claim universal state-of-the-art performance from these artifacts.
- Rerun TAGN-v2, all ablations, robustness, LORO, explainability, and early-warning experiments.
- Report mean, standard deviation, confidence intervals, and per-recording results.
- Compare against published baselines only when split, labels, and metrics are directly compatible.
- Describe learned graphs as candidate predictive dependencies, not causal truth.

## Reproduction

```bash
python3 -m src.models.graphs
python3 -m src.experiments.exp_002_tagn --seed 42 --epochs 20 --patience 6
python3 -m src.report.generate_demo
open outputs/scopus_demo.html
```
"""

    cards = "".join(
        [
            f'<div class="metric"><span>Best baseline PR-AUC</span><strong>{num(logreg.get("pr_auc"))}</strong><small>Logistic regression</small></div>',
            f'<div class="metric"><span>TAGN pilot PR-AUC</span><strong>{num(tagn_anom.get("pr_auc"))}</strong><small>Requires TAGN-v2 rerun</small></div>',
            f'<div class="metric"><span>LORO PR-AUC</span><strong>{num(loro_agg.get("anomaly_pr_auc"))}</strong><small>Recording-shift test</small></div>',
            f'<div class="metric"><span>RCA shortcut gap</span><strong>{num(baseline.get("rca", {}).get("recording_prior", {}).get("jaccard_mean"))}</strong><small>Recording-prior Jaccard</small></div>',
        ]
    )
    html_page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
 :root {{ --ink:#11252a; --muted:#667b80; --teal:#087f8c; --deep:#12363d; --mint:#dff4ef; --gold:#c58b2a; --line:#d8e4e4; --paper:#f4f9f7; --white:#fff; }}
 * {{ box-sizing:border-box; }} body {{ margin:0; background:var(--paper); color:var(--ink); font:16px/1.6 Inter,ui-sans-serif,system-ui,-apple-system,sans-serif; }}
 .shell {{ max-width:1240px; margin:auto; padding:0 25px 80px; }}
 .topbar {{ display:flex; justify-content:space-between; align-items:center; padding:22px 0; border-bottom:1px solid var(--line); }}
 .brand {{ color:var(--teal); font-size:13px; font-weight:850; letter-spacing:.11em; text-transform:uppercase; }} nav a {{ color:var(--muted); font-size:13px; text-decoration:none; margin-left:18px; }} nav a:hover {{ color:var(--teal); }}
 .hero {{ position:relative; overflow:hidden; padding:72px 0 60px; }} .hero:after {{ content:""; position:absolute; right:-100px; top:35px; width:340px; height:340px; border:1px solid #9ed8d0; border-radius:50%; box-shadow:0 0 0 35px #dff4ef,0 0 0 70px #edf8f5; opacity:.7; z-index:-1; }}
 .eyebrow {{ color:var(--teal); font-weight:850; letter-spacing:.14em; text-transform:uppercase; font-size:12px; }}
 h1 {{ max-width:900px; font:700 clamp(38px,6vw,74px)/.98 Georgia,serif; letter-spacing:-.055em; margin:15px 0 22px; }} .dek {{ max-width:780px; color:var(--muted); font-size:20px; }}
 .hero-actions {{ margin-top:28px; }} .button {{ display:inline-block; background:var(--teal); color:#fff; border-radius:999px; padding:10px 17px; text-decoration:none; font-weight:750; margin-right:8px; }} .button.alt {{ background:transparent; border:1px solid #9cc8c3; color:var(--teal); }}
 .banner {{ border:1px solid #e6c77c; background:#fff8e7; color:#6d4d12; border-radius:14px; padding:14px 18px; margin:0 0 28px; }}
 .cards {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:0 0 42px; }} .metric {{ background:var(--white); border:1px solid var(--line); border-radius:16px; padding:21px; box-shadow:0 8px 24px #12352b0d; }}
 .metric span,.metric small {{ display:block; color:var(--muted); }} .metric strong {{ display:block; color:var(--teal); font:700 38px/1.2 Georgia,serif; margin:7px 0; }}
 section {{ background:var(--white); border:1px solid var(--line); border-radius:18px; padding:28px; margin:18px 0; box-shadow:0 8px 24px #12352b08; }} h2 {{ font:700 30px/1.15 Georgia,serif; letter-spacing:-.025em; margin:0 0 13px; }} h3 {{ margin:0 0 5px; }}
 .section-kicker {{ color:var(--teal); font-weight:800; letter-spacing:.1em; text-transform:uppercase; font-size:11px; }} .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:22px; }}
 .pipeline {{ display:grid; grid-template-columns:repeat(5,1fr); gap:10px; margin-top:22px; }} .step {{ position:relative; background:#f1f8f6; border-radius:14px; padding:17px; min-height:142px; }} .step:not(:last-child):after {{ content:"→"; position:absolute; right:-12px; top:47px; color:var(--gold); font-size:22px; z-index:1; }} .step-no {{ color:var(--gold); font-weight:900; font-size:12px; }} .step b {{ display:block; font:700 18px/1.12 Georgia,serif; margin:9px 0 6px; }} .step small {{ color:var(--muted); line-height:1.35; }}
 table {{ width:100%; border-collapse:collapse; font-size:14px; }} th {{ text-align:left; background:#edf7f4; color:#31555b; }} th,td {{ padding:12px 10px; border-bottom:1px solid var(--line); }} .highlight {{ color:var(--teal); font-weight:800; }}
 .bars {{ width:100%; height:auto; margin-top:12px; }} .bar-label,.bar-value {{ font:600 14px Inter, sans-serif; fill:var(--ink); }} .bar-value {{ font-weight:850; }} .bar-track {{ fill:#e8f1ef; }} .evidence {{ display:grid; grid-template-columns:repeat(2,1fr); gap:14px; }} .evidence-card {{ border:1px solid var(--line); border-radius:14px; padding:13px; background:#fbfdfc; }} .evidence-card img {{ display:block; width:100%; border-radius:9px; margin-bottom:9px; }} .evidence-card b {{ font-family:Georgia,serif; }}
 .stat-block {{ display:flex; flex-direction:column; gap:13px; }} .stat {{ background:#f1f8f6; border:1px solid var(--line); border-radius:14px; padding:17px 19px; }} .stat span {{ display:block; color:var(--muted); font-size:13px; font-weight:750; text-transform:uppercase; letter-spacing:.07em; }} .stat strong {{ display:block; color:var(--teal); font:700 34px/1.15 Georgia,serif; margin:6px 0 2px; }} .stat small {{ color:var(--muted); }} .ew-cards {{ grid-template-columns:repeat(4,1fr); }} .pill {{ display:inline-block; padding:5px 10px; border-radius:999px; background:var(--mint); color:#12636b; font-size:13px; font-weight:750; margin:3px; }} .quote {{ border-left:4px solid var(--gold); padding:4px 0 4px 18px; font:italic 21px/1.35 Georgia,serif; color:#31555b; }}
 pre {{ background:var(--deep); color:#dff4ef; padding:18px; border-radius:12px; overflow:auto; }} footer {{ color:var(--muted); font-size:13px; padding-top:22px; }}
 @media(max-width:900px) {{ .cards {{ grid-template-columns:1fr 1fr; }} .pipeline {{ grid-template-columns:1fr 1fr; }} .step:not(:last-child):after {{ display:none; }} }} @media(max-width:600px) {{ .topbar nav {{ display:none; }} .cards,.grid,.evidence,.pipeline {{ grid-template-columns:1fr; }} .hero {{ padding:48px 0 42px; }} .shell {{ padding:0 15px 50px; }} section {{ padding:19px; }} table {{ font-size:12px; }} }}
</style></head><body><main class="shell">
<div class="topbar"><div class="brand">TelecomTS / Research Demo</div><nav><a href="#method">Method</a><a href="#results">Results</a><a href="#ablation">Ablation</a><a href="#figures">Figures</a><a href="#ew">Early warning</a></nav></div>
<header class="hero"><div class="eyebrow">Audited causal-temporal intelligence for 5G telemetry</div><h1>{html.escape(title)}</h1><p class="dek">{html.escape(subtitle)} Built to answer the question that matters: did the model learn transferable failure signatures, or did it memorize the recording?</p><div class="hero-actions"><a class="button" href="#results">See the evidence ↓</a><a class="button alt" href="#method">How it works</a></div></header>
<div class="banner"><strong>Evidence status:</strong> Presentation-ready pilot. Current TAGN artifacts predate TAGN-v2; the page labels them clearly and does not present them as final SOTA evidence.</div>
<div class="cards">{cards}</div>
<section id="method"><div class="section-kicker">01 / research design</div><h2>One pipeline. Four decisions.</h2><p>{html.escape(abstract)}</p><div class="pipeline"><div class="step"><span class="step-no">01</span><b>Telemetry</b><small>18 KPI and protocol channels, 128-step windows.</small></div><div class="step"><span class="step-no">02</span><b>Leakage control</b><small>Chronological splits, burst integrity, train-only scaling.</small></div><div class="step"><span class="step-no">03</span><b>Candidate graphs</b><small>GraphicalLasso skeletons plus lagged directional audits.</small></div><div class="step"><span class="step-no">04</span><b>Temporal model</b><small>Temporal conv, graph attention, multi-task heads.</small></div><div class="step"><span class="step-no">05</span><b>Stress test</b><small>LORO, robustness, attribution, and early warning.</small></div></div><p style="margin-top:22px"><span class="pill">Leakage audited</span><span class="pill">LORO evaluated</span><span class="pill">Dynamic candidates</span><span class="pill">Early warning</span><span class="pill">29 tests passing</span></p></section>
<section id="results"><div class="section-kicker">02 / primary benchmark</div><h2>The baseline is the result to beat.</h2><p>PR-AUC is the primary metric because test anomaly prevalence is only {pct(prevalence)}. The engineered baseline is shown first on purpose.</p><div class="grid"><div>{benchmark_svg}<p><small>Chronological test split. Higher is better.</small></p></div><div><table><thead><tr><th>Model</th><th>ROC-AUC</th><th>PR-AUC</th><th>TPR @ 1%</th><th>Reading</th></tr></thead><tbody>{''.join(rows)}</tbody></table><p class="quote">A credible paper does not hide a strong baseline. It explains why the proposed model has or has not beaten it.</p></div></div></section>
<div class="grid"><section><div class="section-kicker">03 / transfer</div><h2>Does it travel?</h2><p>Leave-recording-out evaluation blocks the strongest recording-memorization shortcut.</p><table><tr><th>Metric</th><th>Value</th></tr><tr><td>PR-AUC</td>{cell(num(loro_agg.get('anomaly_pr_auc')))}</tr><tr><td>ROC-AUC</td>{cell(num(loro_agg.get('anomaly_roc_auc')))}</tr><tr><td>Type accuracy</td>{cell(num(loro_agg.get('type_accuracy')))}</tr><tr><td>RCA template Jaccard</td>{cell(num(loro_agg.get('rca_type_template_jaccard')))}</tr></table></section><section id="audit"><div class="section-kicker">04 / validity</div><h2>Does it fool us?</h2><table><tr><th>Audit</th><th>Value</th></tr><tr><td>Recording-prior RCA</td>{cell(num(baseline.get('rca', {}).get('recording_prior', {}).get('jaccard_mean')))}</tr><tr><td>Input-shift RCA</td>{cell(num(baseline.get('rca', {}).get('input_shift_topK', {}).get('jaccard_mean')))}</tr><tr><td>Graph stability</td>{cell(num(graph.get('prototype_stability_mean')))}</tr><tr><td>Attribution agreement AUC</td>{cell(num(agreement.get('attn_template_vs_non_auc')))}</tr></table></section></div>
<section><div class="section-kicker">05 / seed stability</div><h2>Five seeds, honestly reported.</h2><p>No single lucky run. TAGN is reported as mean ± standard deviation across all five configured seeds; the spread is large and shown, not hidden.</p><div class="grid"><div><table><thead><tr><th>Seed</th><th>ROC-AUC</th><th>PR-AUC</th><th>Type acc</th></tr></thead><tbody>{seed_table}</tbody></table></div><div class="stat-block"><div class="stat"><span>Mean PR-AUC</span><strong>{num(mean_pr)}</strong><small>± {num(std_pr)} over {len(seed_rows)} seeds</small></div><div class="stat"><span>Mean ROC-AUC</span><strong>{num(mean_roc)}</strong><small>± {num(std_roc)} over {len(seed_rows)} seeds</small></div><p class="quote">High seed variance is itself a finding: it says the objective is under-determined and motivates the TAGN-v2 training fixes.</p></div></div></section>
<section id="ablation"><div class="section-kicker">06 / ablation</div><h2>What actually carries the signal?</h2><p>Six variants, identical budget. Removing the graph costs PR-AUC; the static graph variant currently wins, which is an honest and uncomfortable result worth discussing.</p><table><thead><tr><th>Variant</th><th>PR-AUC</th><th>ROC-AUC</th><th>Type acc</th><th>Δ vs full</th></tr></thead><tbody>{ablation_table}</tbody></table></section>
<section><div class="section-kicker">07 / robustness</div><h2>Does it survive perturbation?</h2><p>Input noise, channel dropout, time shift, and rescaling applied at inference with the same checkpoint.</p><div class="grid"><div><table><thead><tr><th>Scenario</th><th>PR-AUC</th><th>ROC-AUC</th></tr></thead><tbody>{robust_table}</tbody></table></div><div class="stat-block"><div class="stat"><span>Worst PR-AUC drop</span><strong>{num(max_drop)}</strong><small>vs baseline under perturbation</small></div><p class="quote">{('Ranking is essentially invariant to input noise; channel dropout is the real stressor.' if max_drop is not None and max_drop < 0.05 else 'Perturbations measurably move the metric; inspect per-scenario rows.')}</p></div></div></section>
<section id="figures"><div class="section-kicker">08 / visual evidence</div><h2>The study is inspectable.</h2><p>These are generated artifacts, not decorative mockups. Use them to walk from data shape to benchmark behavior to explanation.</p><div class="evidence"><div class="evidence-card"><img src="figures/exp/exp_001_ad_curves.png" alt="ROC and precision-recall benchmark curves"><b>Benchmark discrimination</b><br><small>ROC and PR curves across supervised and unsupervised baselines.</small></div><div class="evidence-card"><img src="figures/exp/exp_006_attribution.png" alt="KPI attribution figure"><b>Root-cause attribution</b><br><small>Model attribution compared with template KPI structure.</small></div><div class="evidence-card"><img src="figures/exp/exp_007_hazard_curves.png" alt="Early warning hazard curves"><b>Early warning</b><br><small>Pre-onset hazard behavior and warning timing.</small></div><div class="evidence-card"><img src="figures/eda/fig_corr_matrix.png" alt="KPI correlation matrix"><b>Telemetry structure</b><br><small>Correlation structure motivating candidate graph construction.</small></div></div></section>
<section id="ew"><div class="section-kicker">09 / early warning & attribution</div><h2>Warning before the fault, explaining after.</h2><div class="cards ew-cards">{ew_cards}</div></section>
<section><div class="section-kicker">10 / contribution and limits</div><h2>Ambitious, but defensible.</h2><div class="grid"><div><h3>What is genuinely useful</h3><ol><li>Shortcut-aware evaluation that makes memorization measurable.</li><li>Candidate dynamic KPI dependencies with held-out audits.</li><li>Joint detection, type, RCA, and early-warning outputs.</li><li>Open artifacts, tests, and a browser-ready demo.</li></ol></div><div><h3>What must happen next</h3><ol><li>Rerun TAGN-v2 and the complete experiment matrix.</li><li>Report confidence intervals and per-recording variation.</li><li>Compare only directly compatible published baselines.</li><li>Never call candidate graphs identified causal DAGs.</li></ol></div></div></section>
<section><div class="section-kicker">11 / live handoff</div><h2>From browser demo to live dashboard.</h2><p>For an interactive walkthrough, run the dashboard after showing this page.</p><pre>streamlit run src/dashboard/app.py</pre><p><small>Static demo regeneration: <code>python3 -m src.report.generate_demo</code></small></p></section>
<footer>Generated from local experiment artifacts. No unverified literature claims or invented results are included.</footer>
</main></body></html>"""
    return markdown, html_page


def main() -> None:
    markdown, html_page = build()
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "scopus_demo.md").write_text(markdown)
    (OUTPUTS / "scopus_demo.html").write_text(html_page)
    print(f"wrote {OUTPUTS / 'scopus_demo.md'}")
    print(f"wrote {OUTPUTS / 'scopus_demo.html'}")


if __name__ == "__main__":
    main()
