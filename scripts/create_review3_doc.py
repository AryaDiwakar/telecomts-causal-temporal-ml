"""Create the four-page Review-3 validation/results/discussion PDF."""

from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fpdf import FPDF
from scripts.create_review2_doc import COLORS, ReviewPDF, add_box, arrow, section


OUT = ROOT / "outputs" / "doc" / "review3_validation_results_discussion.pdf"
DESKTOP = Path.home() / "Desktop" / "review3_validation_results_discussion.pdf"


class Review3PDF(ReviewPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("TNR", "", 8)
            self.set_text_color(35, 55, 78)
            self.cell(0, 5, "REVIEW 3  /  VALIDATION, RESULTS AND DISCUSSION", align="L")
            self.cell(0, 5, "TELECOMTS  •  V3 EVALUATION", align="R", new_x="LMARGIN", new_y="NEXT")
            self.set_draw_color(35, 55, 78)
            self.set_line_width(0.45)
            self.line(15, 20, 195, 20)
            self.set_y(23)

    def footer(self):
        self.set_y(-13)
        self.set_draw_color(35, 55, 78)
        self.set_line_width(0.35)
        self.line(15, self.get_y(), 195, self.get_y())
        self.set_y(-11)
        self.set_font("TNR", "", 8)
        self.set_text_color(35, 55, 78)
        self.cell(140, 5, "TELECOMTS  •  REVIEW 3  •  24 OCTOBER 2026", align="L")
        self.cell(40, 5, f"{self.page_no()} / 9", align="R")


def heading(pdf, text, size=15):
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("TNR", "B", size)
    pdf.multi_cell(180, 7, text)
    pdf.ln(1)


def draw_table(pdf, x, y, widths, headers, rows, row_h=7.2, font_size=8.5, header_fill=None):
    """Compact table with black text, black rules, and an optional pastel header."""
    pdf.set_font("TNR", "B", font_size)
    pdf.set_text_color(0, 0, 0)
    if header_fill:
        pdf.set_fill_color(*header_fill)
    pdf.set_draw_color(0, 0, 0)
    xx = x
    for i, (head, w) in enumerate(zip(headers, widths)):
        pdf.rect(xx, y, w, row_h, style="DF" if header_fill else "D")
        pdf.set_xy(xx + 1, y + 1)
        pdf.multi_cell(w - 2, 4, head, align="C")
        xx += w
    yy = y + row_h
    pdf.set_font("TNR", "", font_size)
    for row in rows:
        xx = x
        for value, w in zip(row, widths):
            pdf.rect(xx, yy, w, row_h, style="D")
            pdf.set_xy(xx + 1.2, yy + 1)
            pdf.multi_cell(w - 2.4, 4, str(value), align="C")
            xx += w
        yy += row_h
    return yy


def bar_chart(pdf, x, y, width=177, chart_h=58):
    """PR-AUC bars from the authoritative v3 headline table."""
    items = [
        ("TCN", 0.8606, COLORS["blue"]),
        ("LogReg", 0.8196, COLORS["teal"]),
        ("TAGN", 0.7589, COLORS["gold"]),
        ("GRU", 0.7635, COLORS["lavender"]),
    ]
    left = x + 25
    base = y + chart_h - 10
    top = y + 3
    max_h = base - top
    pdf.set_draw_color(0, 0, 0)
    pdf.line(left, base, x + width - 3, base)
    pdf.set_font("TNR", "", 8)
    pdf.set_text_color(0, 0, 0)
    pdf.text(x + 1, y + 7, "PR-AUC")
    bar_w = 24
    gap = 16
    for i, (name, value, fill) in enumerate(items):
        bx = left + 9 + i * (bar_w + gap)
        bh = max_h * value
        pdf.set_fill_color(*fill)
        pdf.set_draw_color(0, 0, 0)
        pdf.rect(bx, base - bh, bar_w, bh, style="DF")
        pdf.set_xy(bx - 4, base + 1)
        pdf.cell(bar_w + 8, 4, name, align="C")
        pdf.set_font("TNR", "B", 8)
        pdf.set_xy(bx - 4, base - bh - 6)
        pdf.cell(bar_w + 8, 4, f"{value:.4f}", align="C")
        pdf.set_font("TNR", "", 8)
    pdf.set_xy(x, y + chart_h + 1)
    pdf.set_font("TNR", "I", 8)
    pdf.multi_cell(width, 4, "Figure 1. Mean PR-AUC on the corrected v3 pooled-window test set. TCN is the strongest PR-AUC result; TAGN has the highest mean ROC-AUC in the headline table.")
    return pdf.get_y()


def validation_flow(pdf, y=39):
    labels = ["Train\nfit", "Validation\ntune", "Freeze\nthreshold", "Held-out\ntest", "Recording\nbootstrap"]
    fills = [COLORS["blue"], COLORS["teal"], COLORS["gold"], COLORS["rose"], COLORS["lavender"]]
    x0, w, gap, h = 16, 32, 4.5, 19
    for i, (text, fill) in enumerate(zip(labels, fills)):
        x = x0 + i * (w + gap)
        add_box(pdf, x, y, w, h, text, fill, 7.7)
        if i < len(labels) - 1:
            arrow(pdf, x + w, y + h / 2, x + w + gap, y + h / 2)
    pdf.set_xy(16, y + 22)
    pdf.set_font("TNR", "I", 8)
    pdf.multi_cell(180, 4, "Figure 2. Operating thresholds are selected on validation data; test data is reserved for final scoring and recording-level uncertainty estimates.")


def make_result_figures():
    """Regenerate figures directly from saved v3 prediction/result artifacts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
    import numpy as np

    assets = OUT.parent / "review3_assets"
    assets.mkdir(parents=True, exist_ok=True)
    font_file = "/System/Library/Fonts/Supplemental/Times New Roman.ttf"
    prop = FontProperties(fname=font_file)
    plt.rcParams["font.family"] = prop.get_name()
    plt.rcParams["text.color"] = "black"
    plt.rcParams["axes.labelcolor"] = "black"
    plt.rcParams["xtick.color"] = "black"
    plt.rcParams["ytick.color"] = "black"
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    palette = {"LogReg": "#285f8f", "TCN": "#20806d", "GRU": "#8a5d9f", "TAGN": "#c27a24"}

    # Same-test-window curves, single seed 42; headline summary remains multi-seed.
    pred_files = {
        "LogReg": ROOT / "outputs/results/preds/exp_001_test_preds.npz",
        "TAGN": ROOT / "outputs/results/preds/tagn_seed42_preds.npz",
        "TCN": ROOT / "outputs/results/exp_013_tcn_seed42_preds.npz",
        "GRU": ROOT / "outputs/results/exp_013_gru_seed42_preds.npz",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    for name, path in pred_files.items():
        d = np.load(path, allow_pickle=True)
        y = np.asarray(d["y"] if name == "LogReg" else d["y_test"]).astype(int)
        score = np.asarray(d["logreg"] if name == "LogReg" else d["s_test"]).astype(float)
        fpr, tpr, _ = roc_curve(y, score)
        precision, recall, _ = precision_recall_curve(y, score)
        ap = average_precision_score(y, score)
        roc = roc_auc_score(y, score)
        axes[0].plot(fpr, tpr, lw=1.8, color=palette[name], label=f"{name}  ROC={roc:.3f}")
        axes[1].plot(recall, precision, lw=1.8, color=palette[name], label=f"{name}  AP={ap:.3f}")
    axes[0].plot([0, 1], [0, 1], color="#777777", lw=0.8, linestyle="--")
    axes[0].set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False-positive rate", ylabel="True-positive rate", title="ROC curves")
    axes[1].set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision", title="Precision–recall curves")
    for ax in axes:
        ax.grid(True, color="#dddddd", linewidth=0.5)
        ax.legend(frameon=False, fontsize=8, loc="best")
        ax.title.set_color("black")
    fig.suptitle("Model scores on the same repaired v3 test windows (seed 42)", color="black", fontsize=12)
    fig.tight_layout()
    curve_path = assets / "review3_roc_pr_curves.png"
    fig.savefig(curve_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Recording-cluster bootstrap intervals from the authoritative headline table.
    models = ["TCN", "LogReg", "TAGN", "GRU"]
    point = [0.8606, 0.8196, 0.7589, 0.7635]
    lo = [0.5722, 0.6265, 0.4713, 0.4657]
    hi = [0.9748, 0.9387, 0.9777, 0.9308]
    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    yloc = np.arange(len(models))[::-1]
    for i, (name, p, l, h) in enumerate(zip(models, point, lo, hi)):
        yy = yloc[i]
        ax.errorbar(p, yy, xerr=[[p-l], [h-p]], fmt="o", color=palette[name], ecolor=palette[name], capsize=4, markersize=5, linewidth=1.8)
        ax.text(h + 0.012, yy, f"{p:.3f}  [{l:.3f}, {h:.3f}]", va="center", fontsize=8, color="black")
    ax.set_yticks(yloc, models)
    ax.set_xlim(0.35, 1.15)
    ax.set_xlabel("PR-AUC")
    ax.set_title("95% recording-level cluster-bootstrap intervals", color="black")
    ax.grid(axis="x", color="#dddddd", linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    ci_path = assets / "review3_pr_intervals.png"
    fig.savefig(ci_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Current split's template-oracle diagnostic; the oracle uses the true type.
    subsets = ["Template\n(shipped)", "Instance-varying\n(Track 2)", "All anomalous\n(test)"]
    oracle = [0.9705, 0.6611, 0.8238]
    majority = [0.5420, 0.5625, 0.5517]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    bars1 = ax.bar(x - 0.19, oracle, width=0.36, label="True-type template oracle", color="#557fa6", edgecolor="black", linewidth=0.6)
    bars2 = ax.bar(x + 0.19, majority, width=0.36, label="Global-majority template", color="#d5a65e", edgecolor="black", linewidth=0.6)
    ax.bar_label(bars1, fmt="%.3f", padding=2, fontsize=8, color="black")
    ax.bar_label(bars2, fmt="%.3f", padding=2, fontsize=8, color="black")
    ax.set_xticks(x, subsets)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Top-k Jaccard")
    ax.set_title("RCA template-shortcut audit on anomalous test windows", color="black")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", color="#dddddd", linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    oracle_path = assets / "review3_rca_oracle.png"
    fig.savefig(oracle_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Per-seed TAGN and temporal baseline PR values from v3 registered run files.
    tag_pr = [0.7533677931, 0.7300090562, 0.7590877549, 0.7240582173, 0.8277940763]
    tag_seeds = [42, 123, 456, 789, 1337]
    tcn_pr = [0.8056592665, 0.9227116712, 0.8533220302]
    tcn_seeds = [42, 123, 456]
    fig, ax = plt.subplots(figsize=(9.0, 3.8))
    xx = np.arange(5)
    ax.bar(xx - 0.17, tag_pr, width=0.34, label="TAGN", color=palette["TAGN"], edgecolor="black", linewidth=0.5)
    for i, v in enumerate(tcn_pr):
        ax.bar(i + 0.17, v, width=0.34, color=palette["TCN"], edgecolor="black", linewidth=0.5, label="TCN" if i == 0 else None)
    ax.axhline(0.8196, color=palette["LogReg"], linestyle="--", linewidth=1.3, label="LogReg single-run reference")
    ax.set_xticks(xx, [str(s) for s in tag_seeds])
    ax.set_ylim(0.65, 1.0)
    ax.set_xlabel("Seed (TCN available for first three seeds)")
    ax.set_ylabel("PR-AUC")
    ax.set_title("Seed-level variability on v3 test set", color="black")
    ax.legend(frameon=False, fontsize=8, ncol=3)
    ax.grid(axis="y", color="#dddddd", linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    seed_path = assets / "review3_seed_variability.png"
    fig.savefig(seed_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Detection PR by anomaly subset from the v3 error-anatomy artifact.
    anatomy = json.loads((ROOT / "outputs/results/error_anatomy.json").read_text())
    fam = anatomy["family"]
    names = ["TAGN", "TCN", "GRU", "logreg"]
    labels = ["Template\n(shipped)", "Instance-varying\n(Track 2)"]
    fig, ax = plt.subplots(figsize=(9.0, 3.8))
    xx = np.arange(2)
    width = 0.18
    for i, name in enumerate(names):
        vals = [fam[name]["template_pr"], fam[name]["instvar_pr"]]
        color_key = "LogReg" if name == "logreg" else name
        ax.bar(xx + (i - 1.5) * width, vals, width=width, label=color_key, color=palette[color_key], edgecolor="black", linewidth=0.45)
    ax.set_xticks(xx, labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("PR-AUC")
    ax.set_title("Detection by anomaly-ground-truth subset (v3 test windows)", color="black")
    ax.legend(frameon=False, fontsize=8, ncol=4)
    ax.grid(axis="y", color="#dddddd", linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    family_path = assets / "review3_family_detection.png"
    fig.savefig(family_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {"curves": curve_path, "ci": ci_path, "oracle": oracle_path, "seeds": seed_path, "family": family_path}


def main():
    figures = make_result_figures()
    font_dir = Path("/System/Library/Fonts/Supplemental")
    pdf = Review3PDF(format="A4")
    pdf.set_margins(15, 14, 15)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_title("Review 3 — Validation, Results and Discussion")
    pdf.set_subject("Leakage-controlled v3 evaluation of anomaly detection and root-cause localization")
    pdf.set_author("Reddy SaiBaba; Arya Diwakar; Jeswanth G")
    for style, filename in [
        ("", "Times New Roman.ttf"),
        ("B", "Times New Roman Bold.ttf"),
        ("I", "Times New Roman Italic.ttf"),
        ("BI", "Times New Roman Bold Italic.ttf"),
    ]:
        pdf.add_font("TNR", style, str(font_dir / filename))

    # Page 1 — at-a-glance results.
    pdf.add_page()
    pdf.set_text_color(0, 0, 0)
    pdf.set_text_color(35, 55, 78)
    pdf.set_font("TNR", "B", 9)
    pdf.cell(180, 5, "RESEARCH REVIEW  •  03 / 03", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)
    pdf.set_font("TNR", "B", 17)
    pdf.multi_cell(180, 8, "REVIEW 3\nVALIDATION, RESULTS AND DISCUSSION", align="C")
    pdf.ln(1)
    pdf.set_font("TNR", "", 11.5)
    pdf.multi_cell(180, 6, "Causal-Temporal Machine Learning for Root-Cause Localization in 5G Networks Using Multimodal Observability Data", align="C")
    pdf.ln(2)
    pdf.set_font("TNR", "", 9.2)
    pdf.multi_cell(180, 5, "Reddy SaiBaba (23BDS0019)  |  Arya Diwakar (23BDS0023)  |  Jeswanth G (23BDS0212)", align="C")
    pdf.ln(3)
    section(pdf, "1. Evaluation snapshot", "This review reports results from the repaired TelecomTS v3 protocol. The held-out test set contains 5,689 windows, including 424 anomalies, across 26 recordings (7.45% prevalence). Threshold-dependent operating points are selected on validation data; PR-AUC and ROC-AUC are computed from continuous test scores. Recording-level cluster-bootstrap intervals quantify uncertainty, and older pre-repair figures are excluded.", size=9.2, after=2)

    pdf.set_font("TNR", "B", 11)
    pdf.cell(180, 5.5, "Pooled-window detection: PR-AUC", new_x="LMARGIN", new_y="NEXT")
    chart_caption_end = bar_chart(pdf, 17, pdf.get_y() + 1, chart_h=57)
    rows = [
        ["TCN", "3", "0.8606 ± 0.0589", "0.9438 ± 0.0100"],
        ["LogReg", "1", "0.8196", "0.9309"],
        ["TAGN", "5", "0.7589 ± 0.0413", "0.9543 ± 0.0069"],
        ["GRU", "3", "0.7635 ± 0.0266", "0.9399 ± 0.0078"],
    ]
    table_end = draw_table(pdf, 17, chart_caption_end + 1, [34, 18, 62, 62], ["Model", "Seeds", "PR-AUC (mean ± SD)", "ROC-AUC (mean ± SD)"], rows, row_h=7, font_size=8.2, header_fill=COLORS["gray"])
    pdf.set_y(max(190, table_end + 3))
    section(pdf, "What the headline says", "TCN has the highest mean PR-AUC (0.8606). TAGN does not beat TCN or logistic regression on PR-AUC. TAGN has the highest mean ROC-AUC among these models (0.9543), but the single-model confidence intervals overlap substantially; this is a descriptive rank, not evidence of a statistically established advantage over TCN.", size=9.1, after=1)
    section(pdf, "Test-set composition", "Test windows: 5,689  |  positives: 424  |  recordings: 26  |  prevalence: 7.45%  |  RCA ground-truth cardinality: median 9 of 16 KPIs.", size=8.8, after=1)

    pdf.set_font("TNR", "B", 10)
    pdf.set_text_color(35, 55, 78)
    pdf.cell(180, 5, "Three takeaways", new_x="LMARGIN", new_y="NEXT")
    card_y = pdf.get_y() + 1
    takeaway_cards = [
        ("DETECTION", "TCN leads mean PR-AUC\n0.8606 across 3 seeds", COLORS["blue"]),
        ("GRAPH VALUE", "No demonstrated detection gain\nin the single-seed ablation", COLORS["teal"]),
        ("RCA VALIDITY", "Template shortcuts require\ninstance-varying evaluation", COLORS["gold"]),
    ]
    for i, (label, text, fill) in enumerate(takeaway_cards):
        x = 17 + i * 60
        add_box(pdf, x, card_y, 56, 21, f"{label}\n{text}", fill, 7.6)

    # Page 2 — validation and software test evidence.
    pdf.add_page()
    heading(pdf, "Validation protocol and software testing")
    validation_flow(pdf, y=35)
    pdf.set_y(63)
    section(pdf, "2. Split and preprocessing validation", "The v3 split is chronological within recording, with windows crossing split boundaries removed. Overlapping anomaly bursts are kept together; jammer and synthetic recordings are assigned atomically where required. A synthetic recording and its clean source background are bound to the same split. The scaler, train-majority RCA templates, and graph prototypes are fit from training data only. In leave-recordings-out evaluation, each fold refits its scaler and graph prototypes on that fold's training recordings.", size=9.0, after=2)

    split_rows = [
        ["Train", "22,009", "1,235", "31"],
        ["Validation", "4,841", "185", "18"],
        ["Test", "5,689", "424", "26"],
    ]
    pdf.set_font("TNR", "B", 10)
    pdf.cell(180, 5, "Split inventory", new_x="LMARGIN", new_y="NEXT")
    end_y = draw_table(pdf, 17, pdf.get_y() + 1, [39, 45, 47, 45], ["Partition", "Windows", "Anomalous", "Recordings"], split_rows, row_h=7, font_size=8.5, header_fill=COLORS["teal"])
    pdf.set_y(end_y + 4)
    section(pdf, "3. Statistical validation", "Anomaly thresholds are selected on validation data and applied unchanged to test. Confidence intervals resample the 26 recordings rather than individual overlapping windows. Model comparisons use a paired cluster bootstrap: both systems are rescored on the same recording resamples. This preserves the pairing and avoids treating correlated windows as independent observations.", size=9.0, after=2)

    section(pdf, "4. Automated test suite", "The current test suite passes: 60 passed. Coverage includes preprocessing and pipeline checks, audit-regression tests for split integrity, model forward/shape behavior, RCA metric guards, pretraining, and the experimental backbone, fusion, intervention, and deferral utilities. These tests validate software behavior and data-handling invariants; they do not by themselves establish model quality or a research claim.", size=9.0, after=2)

    checks = [
        ("Split integrity", "No cross-split timestep overlap; paired injected/source recordings stay assigned together."),
        ("Train-only fitting", "Scaler and graph artifacts are fitted on training data, including LORO folds."),
        ("Operating point", "Threshold tuned on validation; fixed before test scoring."),
        ("Uncertainty", "Recording-level and paired cluster-bootstrap analyses."),
    ]
    pdf.set_font("TNR", "B", 10)
    pdf.cell(180, 5, "Validation checklist", new_x="LMARGIN", new_y="NEXT")
    y = pdf.get_y() + 1
    for i, (left, right) in enumerate(checks):
        fill = [COLORS["blue"], COLORS["teal"], COLORS["gold"], COLORS["lavender"]][i]
        add_box(pdf, 17, y, 40, 11, left, fill, 8.1)
        pdf.set_xy(62, y + 1)
        pdf.set_font("TNR", "", 8.8)
        pdf.multi_cell(130, 4.2, right)
        y += 14

    # Page 3 — comparative analysis, paired results, generalization, RCA.
    pdf.add_page()
    heading(pdf, "Result analysis: detection and localization")
    section(pdf, "5. Detection performance", "On the repaired pooled-window test set, TAGN's mean PR-AUC is 0.7589 across five seeds, compared with 0.8196 for logistic regression and 0.8606 for TCN. Its ROC-AUC mean is 0.9543, compared with 0.9309 for logistic regression and 0.9438 for TCN. The difference between PR-AUC and ROC-AUC rankings matters here: at 7.45% prevalence, PR-AUC directly reflects precision-recall trade-offs under class imbalance.", size=9.0, after=2)

    paired_rows = [
        ["42", "-0.0662", "[-0.1563, +0.0557]", "0.786"],
        ["123", "-0.0896", "[-0.2000, +0.0570]", "0.881"],
        ["456", "-0.0605", "[-0.1482, +0.0712]", "0.775"],
        ["789", "-0.0956", "[-0.1981, +0.0526]", "0.890"],
        ["1337", "+0.0082", "[-0.0721, +0.0698]", "0.374"],
    ]
    pdf.set_font("TNR", "B", 10)
    pdf.cell(180, 5, "Paired cluster bootstrap: TAGN minus LogReg PR-AUC", new_x="LMARGIN", new_y="NEXT")
    end_y = draw_table(pdf, 17, pdf.get_y() + 1, [25, 34, 82, 34], ["Seed", "ΔPR", "95% interval", "P(ΔPR ≤ 0)"], paired_rows, row_h=7, font_size=8.2, header_fill=COLORS["gold"])
    pdf.set_y(end_y + 3)
    section(pdf, "Interpretation", "ΔPR is negative on four of five seeds. Every reported 95% interval includes zero, so the result supports “no detectable PR-AUC advantage” over logistic regression—not a claim that TAGN is statistically worse. For ROC-AUC, TAGN's paired differences are positive across seeds, but only one interval excludes zero. This is why the paper reports the metrics separately and avoids a broad “best model” claim.", size=8.9, after=2)

    loro_rows = [["TAGN", "5 folds", "0.7689 ± 0.0737", "0.9011 ± 0.0277"]]
    pdf.set_font("TNR", "B", 10)
    pdf.cell(180, 5, "Setting II: leave-recordings-out", new_x="LMARGIN", new_y="NEXT")
    end_y = draw_table(pdf, 17, pdf.get_y() + 1, [35, 35, 55, 50], ["Model", "Protocol", "PR-AUC", "ROC-AUC"], loro_rows, row_h=8, font_size=8.2, header_fill=COLORS["blue"])
    pdf.set_y(end_y + 3)
    section(pdf, "What the generalization check changes", "TAGN's LORO PR-AUC (0.7689) is close to its pooled Setting-I mean (0.7589). After the source-background leak was repaired, the previous large gap between pooled and recording-held-out performance disappeared. This suggests the earlier gap reflected recording familiarity rather than robust transfer across sessions.", size=8.9, after=2)

    rca_rows = [["0.0624", "0.1655", "0.2871", "0.6710", "0.5334", "0.2236"]]
    pdf.set_font("TNR", "B", 10)
    pdf.cell(180, 5, "Root-cause localization: five-seed mean", new_x="LMARGIN", new_y="NEXT")
    draw_table(pdf, 17, pdf.get_y() + 1, [29, 29, 29, 29, 29, 29], ["Recall@1", "Recall@3", "Recall@5", "MRR", "MAP", "Type acc."], rca_rows, row_h=8, font_size=7.8, header_fill=COLORS["teal"])

    # Page 4 — ablations, discussion, limitations.
    pdf.add_page()
    heading(pdf, "Discussion: what the validation supports")
    section(pdf, "6. Root-cause results in context", "The ground-truth root sets contain a median of 9 out of 16 KPIs, so Recall@1 is structurally constrained and should not be read alone. Across five checkpoints, the input-driven ranking reaches mean MRR 0.6710 and MAP 0.5334; mean type accuracy is 0.2236, above the roughly 0.09 chance level but still modest. Detection quality and RCA quality do not move together across seeds, so strong detection is not a reliable proxy for good localization.", size=9.0, after=3)

    ablation_rows = [
        ["Full", "0.7439", "0.9485", "0.5310"],
        ["No graph", "0.7581", "0.9520", "0.5402"],
        ["Static graph", "0.7482", "0.9501", "0.5538"],
        ["Undirected", "0.7431", "0.9499", "0.5484"],
        ["No node conv", "0.7445", "0.9499", "0.5364"],
        ["Hazard only", "0.7305", "0.9497", "0.5640"],
    ]
    pdf.set_font("TNR", "B", 10)
    pdf.cell(180, 5, "Component ablation (single seed; indicative)", new_x="LMARGIN", new_y="NEXT")
    end_y = draw_table(pdf, 17, pdf.get_y() + 1, [48, 40, 40, 46], ["Variant", "PR-AUC", "ROC-AUC", "RCA Jaccard"], ablation_rows, row_h=7, font_size=8.2, header_fill=COLORS["lavender"])
    pdf.set_y(end_y + 4)

    section(pdf, "7. Ablation discussion", "The ablations share the same frozen summary anchor, split, optimizer, and validation threshold. The full model does not outperform the no-graph variant on detection; all PR-AUC values remain close to the anchor. The graph, node convolution, and edge directionality therefore show no clear detection gain in this experiment. Because the ablation is single-seed, its ordering is indicative rather than a significance result. The defensible conclusion is narrow: this benchmark provides no evidence that the current graph message-passing design improves detection.", size=9.0, after=3)

    section(pdf, "8. Why the benchmark repair matters", "The earlier split allowed clean source chunks from an injected test recording to appear in training. Binding each source and injected twin to the same split enlarged the test set from 4,969 to 5,689 windows and reversed the apparent TAGN-versus-baseline PR-AUC story. Synthetic labels were also repaired: 280 spurious affected-KPI labels were removed across 237 of 714 windows. These corrections are part of the evaluation result, not incidental preprocessing details.", size=9.0, after=3)

    section(pdf, "9. Limitations and discussion", "The test set has 26 recordings, so recording-level intervals are necessarily broad. Per-type support is uneven, and RCA labels on shipped windows are type templates; an input-free type oracle can score highly on that subset. Track-2 injection adds instance-varying labels but remains synthetic. The graph is a statistical dependency estimate, not an identified physical causal graph. Cross-vendor results and experimental model variants should be interpreted according to their own protocols and not mixed with the v3 headline table.", size=9.0, after=3)

    section(pdf, "10. Overall conclusion", "The validation supports a careful, useful result rather than a SOTA claim: once leakage and label issues are repaired, TCN leads on PR-AUC, TAGN has the highest mean ROC-AUC in the headline comparison, and no paired test establishes a PR-AUC advantage for TAGN. The project contributes a stronger evaluation protocol, corrected benchmark artifacts, interpretable model outputs, and clear evidence about where the current graph design does—and does not—help.", size=9.0, after=0)

    # Page 5 — model score curves from shared held-out predictions.
    pdf.add_page()
    heading(pdf, "Model-level diagnostics: ranking curves")
    pdf.set_xy(17, 35)
    pdf.image(str(figures["curves"]), x=17, y=35, w=176)
    pdf.set_y(113)
    section(pdf, "11. ROC and precision–recall behavior", "The curves use saved prediction scores for seed 42 and the same 5,689 v3 test windows. They are a single-seed diagnostic, not a replacement for the multi-seed headline table. ROC curves summarize ranking across false-positive rates; precision–recall curves expose the precision cost at this test set's 7.45% positive prevalence. In this comparison, the PR curves make clear why a strong ROC-AUC does not automatically imply the best anomaly ranking for an imbalanced deployment.", size=9.0, after=3)

    per_seed_rows = [
        ["TAGN", "42", "0.7534", "0.9505"],
        ["TAGN", "123", "0.7300", "0.9517"],
        ["TAGN", "456", "0.7591", "0.9580"],
        ["TAGN", "789", "0.7241", "0.9469"],
        ["TAGN", "1337", "0.8278", "0.9643"],
        ["TCN", "42", "0.8057", "0.9348"],
        ["TCN", "123", "0.9227", "0.9546"],
        ["TCN", "456", "0.8533", "0.9422"],
    ]
    pdf.set_font("TNR", "B", 10)
    pdf.cell(180, 5, "Per-seed test metrics (same v3 split)", new_x="LMARGIN", new_y="NEXT")
    end_y = draw_table(pdf, 17, pdf.get_y() + 1, [40, 34, 52, 52], ["Model", "Seed", "PR-AUC", "ROC-AUC"], per_seed_rows, row_h=6.3, font_size=8.0, header_fill=COLORS["blue"])
    pdf.set_y(end_y + 3)
    section(pdf, "Reading seed variability", "TAGN's seed-1337 result is notably stronger than its other four runs; the five-seed mean should therefore be read with its reported spread and cluster-bootstrap interval. TCN also varies substantially across its three runs. These distributions argue for reporting seed counts and intervals alongside point estimates rather than selecting a single favorable run.", size=8.9, after=0)

    # Page 6 — bootstrap uncertainty and seed-to-seed spread.
    pdf.add_page()
    heading(pdf, "Uncertainty and repeatability")
    pdf.set_xy(17, 35)
    pdf.image(str(figures["seeds"]), x=17, y=35, w=176)
    pdf.set_y(111)
    section(pdf, "12. Seed-level comparison", "The bars show available per-seed PR-AUC runs, with the logistic-regression result shown as a single-run reference. TCN has three seeds, TAGN five. The dashed line is not a confidence bound; it marks the LogReg point estimate. The chart is intended to make seed spread visible, not to imply that mismatched seed counts form a paired statistical test.", size=8.9, after=2)
    pdf.set_xy(17, 151)
    pdf.image(str(figures["ci"]), x=17, y=151, w=176)
    pdf.set_y(222)
    section(pdf, "13. What the intervals mean", "The PR-AUC whiskers are 95% cluster-bootstrap intervals, resampling recordings. For multi-seed models the headline table reports the union of per-seed intervals. Their width reflects substantial between-recording variability; overlapping marginal intervals should not be used as a significance test. The paired bootstrap in Review-3, page 3, is the appropriate TAGN-versus-LogReg comparison because both models are rescored on the same resampled recordings.", size=8.8, after=0)

    # Page 7 — root-cause label validity and shortcut.
    pdf.add_page()
    heading(pdf, "Root-cause validation: the template shortcut")
    pdf.set_xy(17, 36)
    pdf.image(str(figures["oracle"]), x=17, y=36, w=176)
    pdf.set_y(111)
    section(pdf, "14. Why this diagnostic matters", "Across the shipped TelecomTS anomaly corpus, the affected-KPI set is constant within each fault type (956 windows across 10 types). An input-free predictor that knows the true type and emits its KPI template therefore achieves 0.971 top-k Jaccard on the shipped test subset. On Track-2 instance-varying windows, the same oracle falls to 0.661. The oracle is deliberately privileged and is not a deployable model; it measures how much the label construction permits type lookup to explain.", size=9.0, after=2)

    subset_rows = [
        ["Template (shipped)", "223", "10", "0.9705", "0.5420"],
        ["Instance-varying (Track 2)", "201", "5", "0.6611", "0.5625"],
        ["All anomalous test", "424", "11", "0.8238", "0.5517"],
    ]
    pdf.set_font("TNR", "B", 10)
    pdf.cell(180, 5, "Oracle ceiling and global-majority reference", new_x="LMARGIN", new_y="NEXT")
    end_y = draw_table(pdf, 17, pdf.get_y() + 1, [52, 26, 22, 42, 42], ["Subset", "Windows", "Types", "Type oracle", "Global majority"], subset_rows, row_h=7, font_size=7.8, header_fill=COLORS["gold"])
    pdf.set_y(end_y + 3)
    section(pdf, "Interpretation and boundary", "A high template-based RCA score on shipped windows is not, by itself, evidence that the model localized the fault from telemetry. The instance-varying subset is a more demanding test of input-driven KPI ranking, although its ground truth is generated synthetically. The report therefore treats detection and RCA separately and avoids describing the graph as ground-truth causality.", size=8.9, after=0)

    # Page 8 — anomaly family behavior and final discussion.
    pdf.add_page()
    heading(pdf, "Error analysis and discussion by anomaly subset")
    pdf.set_xy(17, 35)
    pdf.image(str(figures["family"]), x=17, y=35, w=176)
    pdf.set_y(111)
    section(pdf, "15. Template versus instance-varying detection", "The repaired test set contains 223 template and 201 instance-varying anomalous windows. The grouped chart reports PR-AUC within each anomaly subset, scored against the shared normal pool. Performance depends on both model and subset; the template and injected cases should not be collapsed into a single narrative about fault difficulty. Because per-subset positives are fewer than the full test set, these values are descriptive and need their own uncertainty analysis before fine-grained ranking claims.", size=9.0, after=3)

    section(pdf, "16. Integrated discussion", "Three findings survive the validation checks. First, TCN leads the multi-seed PR-AUC comparison while TAGN has the highest mean ROC-AUC in the headline table; paired testing finds no detectable PR-AUC advantage for TAGN over LogReg. Second, the no-graph ablation is at least as strong as the full model on PR-AUC in the single-seed ablation, so the current graph-attention pathway has no demonstrated detection benefit. Third, the benchmark's template structure creates a large input-free RCA ceiling on shipped faults, motivating separate reporting on instance-varying labels.", size=9.0, after=3)

    section(pdf, "17. What can and cannot be claimed", "Supported: the v3 repairs remove the identified background split leak and correct synthetic affected-KPI labels; the protocol reports recording-aware uncertainty; and the measured model rankings differ by metric. Not supported: that TAGN is state of the art, that the estimated graph is a physical causal DAG, or that a model input perturbation identifies the outcome of a real network intervention. This distinction keeps the positive contribution focused on rigorous validation and diagnosis of benchmark behavior.", size=9.0, after=3)

    section(pdf, "18. Final assessment", "The model is a useful vehicle for studying temporal and KPI-graph representations, but the strongest result of this evaluation is methodological: when the split, labels, and uncertainty unit are checked, the conclusions become more modest and more reliable. Future model improvements should be evaluated on this same repaired split, against TCN and LogReg, across multiple seeds, with paired recording-level inference and no reuse of the held-out test set for tuning.", size=9.0, after=0)

    # Page 9 — provenance and practical reproduction guide.
    pdf.add_page()
    heading(pdf, "Reproducibility and evidence trail")
    section(pdf, "19. Frozen evaluation record", "The reported TAGN runs are registered against code revision 1d24faa-dirty, config hash 038f3fe328f11160, and corrected-data fingerprint f089ca31f9e6d100. The working tree was not clean at run time, so exact reproduction requires the matching source snapshot and data artifacts—not only the Git commit. The corrected split inventory is train 22,009 / validation 4,841 / test 5,689 windows; the test contains 424 positives across 26 recordings.", size=9.0, after=3)

    evidence_rows = [
        ["Protocol + split", "docs/benchmark_protocol_v3.md; data/processed/split_stats.json"],
        ["Headline results", "docs/results_v3.md; outputs/tables/headline_v3.md"],
        ["Paired inference", "outputs/results/paired_test.json (2,000 recording resamples)"],
        ["RCA ranking", "outputs/results/rca_ranking_exp_002.json"],
        ["Validity audit", "docs/benchmark_audit_v3.md; docs/data_leakage.md"],
        ["Figures in this review", "outputs/doc/review3_assets/ (derived from saved predictions/results)"],
    ]
    pdf.set_font("TNR", "B", 10)
    pdf.cell(180, 5, "Primary evidence map", new_x="LMARGIN", new_y="NEXT")
    end_y = draw_table(pdf, 17, pdf.get_y() + 1, [43, 137], ["Evidence", "Authoritative artifact"], evidence_rows, row_h=9, font_size=8.0, header_fill=COLORS["teal"])
    pdf.set_y(end_y + 4)

    section(pdf, "20. Re-run the key checks", "The following commands regenerate the central uncertainty analyses and verify the current code-level test suite. Full model training and preprocessing steps are documented in README.md and docs/benchmark_protocol_v3.md.", size=9.0, after=2)
    commands = [
        "python3 -m scripts.bootstrap_ci --seeds 42 123 456 789 1337 --tag exp_002",
        "python3 -m scripts.paired_test --baseline logreg",
        "python3 -m pytest -q",
    ]
    for i, command in enumerate(commands):
        fill = [COLORS["blue"], COLORS["gold"], COLORS["lavender"]][i]
        add_box(pdf, 17, pdf.get_y(), 180, 10, command, fill, 8.2)
        pdf.set_y(pdf.get_y() + 12)

    section(pdf, "21. Dataset citation", "Feng, A. et al. (2025). TelecomTS: A Multi-Modal Observability Dataset for Time Series and Language Analysis. arXiv:2510.06063, version 3 (May 2026). The repaired split and study-specific synthetic Track-2 labels are documented in the protocol and audit artifacts cited above.", size=8.9, after=0)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    DESKTOP.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUT))
    pdf.output(str(DESKTOP))
    print(f"Created {OUT}")
    print(f"Created {DESKTOP}")
    print(f"Pages: {pdf.page_no()}")


if __name__ == "__main__":
    main()
