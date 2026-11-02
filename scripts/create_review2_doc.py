"""Create a polished, two-page Review-2 Design and Implementation PDF."""

from pathlib import Path

from fpdf import FPDF


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "doc" / "review2_design_implementation.pdf"
DESKTOP = Path.home() / "Desktop" / "review2_design_implementation.pdf"
FONT_DIR = Path("/System/Library/Fonts/Supplemental")

COLORS = {
    "navy": (220, 230, 242),
    "blue": (226, 237, 248),
    "teal": (222, 239, 235),
    "gold": (249, 239, 216),
    "rose": (246, 228, 226),
    "lavender": (235, 231, 245),
    "gray": (242, 243, 245),
    "ink": (0, 0, 0),
    "line": (55, 55, 55),
}


class ReviewPDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("TNR", "", 8)
            self.set_text_color(0, 0, 0)
            self.cell(0, 5, "REVIEW 2  /  DESIGN AND IMPLEMENTATION", align="L")
            self.cell(0, 5, "TELECOMTS 5G RAN STUDY", align="R", new_x="LMARGIN", new_y="NEXT")
            self.set_draw_color(0, 0, 0)
            self.line(15, 20, 195, 20)
            self.set_y(23)

    def footer(self):
        self.set_y(-12)
        self.set_draw_color(0, 0, 0)
        self.line(15, self.get_y(), 195, self.get_y())
        self.set_y(-10)
        self.set_font("TNR", "", 8)
        self.set_text_color(0, 0, 0)
        self.cell(0, 5, "Reddy SaiBaba  (23BDS0019)  |  Arya Diwakar  (23BDS0023)  |  jeswanth G  (23bds0212)", align="C")


def add_box(pdf, x, y, w, h, text, fill, font_size=9, radius=2, border=(35, 35, 35)):
    pdf.set_draw_color(*border)
    pdf.set_line_width(0.35)
    pdf.set_fill_color(*fill)
    pdf.rect(x, y, w, h, style="DF")
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("TNR", "", font_size)
    lines = text.split("\n")
    line_h = 4.2
    total_h = len(lines) * line_h
    ty = y + (h - total_h) / 2 + 0.5
    for line in lines:
        pdf.set_xy(x + 1.3, ty)
        pdf.cell(w - 2.6, line_h, line, align="C")
        ty += line_h


def arrow(pdf, x1, y1, x2, y2, color=(30, 30, 30)):
    pdf.set_draw_color(*color)
    pdf.set_line_width(0.45)
    pdf.line(x1, y1, x2, y2)
    import math
    angle = math.atan2(y2 - y1, x2 - x1)
    size = 2.0
    for offset in (2.55, -2.55):
        pdf.line(x2, y2, x2 - size * math.cos(angle + offset), y2 - size * math.sin(angle + offset))


def section(pdf, title, body, width=180, size=9.3, after=3):
    x = pdf.get_x()
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("TNR", "B", 11)
    pdf.multi_cell(width, 5.5, title)
    pdf.ln(0.5)
    pdf.set_x(x)
    pdf.set_font("TNR", "", size)
    pdf.multi_cell(width, 4.8, body, align="J")
    pdf.ln(after)


def label(pdf, x, y, text, size=8.5, style=""):
    pdf.set_xy(x, y)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("TNR", style, size)
    pdf.cell(0, 4.5, text)


def architecture_figure(pdf, top=82):
    """Vector architecture diagram; text remains black, fills are pastel."""
    x0 = 17
    gap = 5
    w = 51
    h = 17
    # Inputs
    add_box(pdf, x0, top, w, h, "Telemetry window\n128 time steps x 18 channels", COLORS["blue"])
    add_box(pdf, x0 + w + gap, top, w, h, "KPI dependency graph\n16 nodes / assigned prototype", COLORS["teal"])
    add_box(pdf, x0 + 2 * (w + gap), top, w, h, "Chunk summaries\nlevel, spread, trend, sparsity", COLORS["gold"])

    # Encoders
    y2 = top + 29
    add_box(pdf, x0, y2, w, 20, "Causal temporal encoder\nDilated convolutions", COLORS["lavender"])
    add_box(pdf, x0 + w + gap, y2, w, 20, "KPI node encoder + GAT\nGraph-masked attention", COLORS["teal"])
    add_box(pdf, x0 + 2 * (w + gap), y2, w, 20, "Summary anchor\nLinear + residual readout", COLORS["gold"])

    # Fusion
    y3 = top + 63
    add_box(pdf, x0 + 25, y3, 72, 19, "Fuse temporal and node context\nGRU + temporal attention pooling", COLORS["blue"])
    add_box(pdf, x0 + 104, y3, 72, 19, "Summary statistics\nTrain-only normalization", COLORS["gold"])

    # Heads
    y4 = top + 98
    out_w = 41
    out_gap = 3
    outs = ["Anomaly\nscore", "Fault type\n(11 classes)", "KPI RCA\n(16 logits)", "Early warning\n(per-step hazard)"]
    for i, text in enumerate(outs):
        add_box(pdf, x0 + i * (out_w + out_gap), y4, out_w, 18, text, [COLORS["rose"], COLORS["lavender"], COLORS["teal"], COLORS["gray"]][i], 8.3)

    # Connectors
    arrow(pdf, x0 + w / 2, top + h, x0 + w / 2, y2)
    arrow(pdf, x0 + w + gap + w / 2, top + h, x0 + w + gap + w / 2, y2)
    arrow(pdf, x0 + 2 * (w + gap) + w / 2, top + h, x0 + 2 * (w + gap) + w / 2, y2)
    arrow(pdf, x0 + w / 2, y2 + 20, x0 + 25 + 18, y3)
    arrow(pdf, x0 + w + gap + w / 2, y2 + 20, x0 + 25 + 54, y3)
    arrow(pdf, x0 + 2 * (w + gap) + w / 2, y2 + 20, x0 + 104 + 36, y3)
    arrow(pdf, x0 + 61, y3 + 19, x0 + 61, y4)
    arrow(pdf, x0 + 140, y3 + 19, x0 + 140, y4)
    # Branching connector to the heads
    pdf.set_draw_color(30, 30, 30)
    pdf.line(x0 + 61, y4 - 4, x0 + 166, y4 - 4)
    for i in range(4):
        cx = x0 + i * (out_w + out_gap) + out_w / 2
        arrow(pdf, cx, y4 - 4, cx, y4)

    pdf.set_font("TNR", "I", 8)
    pdf.set_text_color(0, 0, 0)
    pdf.set_xy(17, y4 + 21)
    pdf.multi_cell(180, 4, "Figure 1. TAGN data flow. The graph is an estimated dependency structure; it is not treated as a verified physical causal DAG.")


def pipeline_figure(pdf, top=34):
    labels = [
        "Raw\nrecordings", "Windowing\n128 x 18", "Leakage-safe\nsplits", "Fit train-only\nscaler + graphs", "Train / validate\n/ test",
    ]
    fills = [COLORS["blue"], COLORS["teal"], COLORS["gold"], COLORS["lavender"], COLORS["rose"]]
    x0, w, gap, h = 16, 32, 4.5, 20
    for i, (text, fill) in enumerate(zip(labels, fills)):
        x = x0 + i * (w + gap)
        add_box(pdf, x, top, w, h, text, fill, 8.1)
        if i < len(labels) - 1:
            arrow(pdf, x + w, top + h / 2, x + w + gap, top + h / 2)
    pdf.set_font("TNR", "I", 8)
    pdf.set_xy(16, top + 22)
    pdf.multi_cell(180, 4, "Figure 2. Reproducible implementation pipeline. Splits are recording-aware; fitted preprocessing artifacts are learned from training data only.")


def graph_build_figure(pdf, top=38):
    labels = ["Window bins", "Sparse KPI\nskeleton", "TE direction\ncheck", "Train graph\nprototypes"]
    fills = [COLORS["blue"], COLORS["teal"], COLORS["gold"], COLORS["lavender"]]
    x0, w, gap, h = 18, 37, 9, 20
    for i, (text, fill) in enumerate(zip(labels, fills)):
        x = x0 + i * (w + gap)
        add_box(pdf, x, top, w, h, text, fill, 8.2)
        if i < len(labels) - 1:
            arrow(pdf, x + w, top + h / 2, x + w + gap, top + h / 2)
    pdf.set_xy(17, top + 22)
    pdf.set_font("TNR", "I", 8)
    pdf.multi_cell(180, 4, "Figure 3. Graph estimation sequence. Prototype fitting uses training bins; held-out bins are assigned to the nearest learned prototype.")


def illustrative_graph(pdf, x=36, y=103):
    """Illustrative network, explicitly not a plot of a measured prototype."""
    import math
    cx, cy, r = x + 38, y + 24, 20
    points = []
    for i in range(8):
        a = -math.pi / 2 + i * math.pi / 4
        points.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5), (4, 6), (5, 7), (6, 7)]
    pdf.set_draw_color(80, 80, 80)
    pdf.set_line_width(0.45)
    for i, j in edges:
        pdf.line(*points[i], *points[j])
    fills = [COLORS["blue"], COLORS["teal"], COLORS["gold"], COLORS["lavender"]]
    for i, (nx, ny) in enumerate(points):
        pdf.set_draw_color(30, 30, 30)
        pdf.set_fill_color(*fills[i % len(fills)])
        pdf.ellipse(nx - 4, ny - 4, 8, 8, style="DF")
        pdf.set_font("TNR", "", 7.2)
        pdf.set_text_color(0, 0, 0)
        pdf.set_xy(nx - 3.5, ny - 1.7)
        pdf.cell(7, 3.5, f"K{i+1}", align="C")
    pdf.set_font("TNR", "I", 8)
    pdf.set_xy(x, y + 48)
    pdf.multi_cell(78, 4, "Figure 4. Schematic KPI graph for explanation only; edges shown are illustrative, not measured results.", align="C")


def objective_figure(pdf, top=221):
    tasks = ["Anomaly\nBCE", "Type\nweighted CE", "RCA\nmultilabel BCE", "Hazard\nmasked BCE"]
    fills = [COLORS["rose"], COLORS["lavender"], COLORS["teal"], COLORS["gold"]]
    x0, w, gap = 17, 36, 4
    for i, text in enumerate(tasks):
        x = x0 + i * (w + gap)
        add_box(pdf, x, top, w, 17, text, fills[i], 8)
        arrow(pdf, x + w / 2, top + 17, 107, top + 26)
    add_box(pdf, 83, top + 26, 49, 13, "Weighted joint loss", COLORS["gray"], 8.5)
    pdf.set_font("TNR", "I", 7.8)
    pdf.set_xy(136, top + 27)
    pdf.multi_cell(57, 4, "Weights: 1.0 / 1.0 / 1.0 / 0.5", align="L")


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for style, filename in [("", "Times New Roman.ttf"), ("B", "Times New Roman Bold.ttf"), ("I", "Times New Roman Italic.ttf"), ("BI", "Times New Roman Bold Italic.ttf")]:
        path = FONT_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Times New Roman font not found: {path}")

    pdf = ReviewPDF(format="A4")
    pdf.set_margins(15, 14, 15)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_font("TNR", "", str(FONT_DIR / "Times New Roman.ttf"))
    pdf.add_font("TNR", "B", str(FONT_DIR / "Times New Roman Bold.ttf"))
    pdf.add_font("TNR", "I", str(FONT_DIR / "Times New Roman Italic.ttf"))
    pdf.add_font("TNR", "BI", str(FONT_DIR / "Times New Roman Bold Italic.ttf"))

    # Page 1: design overview and model architecture.
    pdf.add_page()
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("TNR", "B", 17)
    pdf.multi_cell(180, 8, "REVIEW 2\nDESIGN AND IMPLEMENTATION", align="C")
    pdf.ln(1)
    pdf.set_font("TNR", "", 12)
    pdf.multi_cell(180, 6, "Causal-Temporal Machine Learning for Root-Cause Localization in 5G Networks Using Multimodal Observability Data", align="C")
    pdf.ln(2)
    pdf.set_font("TNR", "", 9.3)
    pdf.multi_cell(180, 5, "Reddy SaiBaba (23BDS0019)  |  Arya Diwakar (23BDS0023)  |  jeswanth G (23bds0212)", align="C")
    pdf.ln(3)

    section(pdf, "1. Design objective", "The system turns multivariate 5G radio-access telemetry into four complementary outputs: a window-level anomaly score, an anomaly-type prediction, KPI-level root-cause scores, and a per-time-step early-warning hazard. The design pairs temporal modeling with estimated KPI dependencies, while keeping preprocessing and evaluation separated by recording and split.", size=9.4, after=1)

    pdf.set_font("TNR", "B", 11)
    pdf.cell(180, 6, "2. Model architecture", new_x="LMARGIN", new_y="NEXT")
    architecture_figure(pdf, top=91)

    pdf.set_xy(17, 236)
    pdf.set_font("TNR", "B", 9.5)
    pdf.cell(180, 5, "Design choices", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("TNR", "", 8.8)
    choices = [
        "Temporal path: causal dilated convolutions encode local change without using future samples; a GRU and attention pooling form a window representation.",
        "Graph path: each KPI has its own temporal node representation; graph attention is masked to estimated edges and retains edge strengths.",
        "Summary path: an explicit statistical readout keeps level, spread, trend, quantiles and zero fraction available to the anomaly and type heads.",
    ]
    for item in choices:
        pdf.set_x(18)
        pdf.multi_cell(178, 4.2, "- " + item)
        pdf.ln(0.3)

    # Page 2: implementation and reproducibility.
    pdf.add_page()
    pdf.set_font("TNR", "B", 15)
    pdf.cell(180, 8, "Implementation and evaluation workflow", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pipeline_figure(pdf, top=37)
    pdf.set_y(67)

    section(pdf, "3. Data preparation and split safeguards", "The corpus is organized as 128-step windows with 16 numeric KPI channels and two protocol-code channels. The chronological split is performed recording by recording; chunks that straddle boundaries are dropped, overlapping anomaly bursts are kept atomic, and source recordings paired with injected synthetic twins are assigned to the same split. Location and scale parameters are fit on training chunks only. The separate leave-one-recording-out protocol refits its scaler and graph prototypes inside each fold.", size=9.0, after=2)

    section(pdf, "4. KPI dependency graph", "For graph estimation, consecutive chunks are pooled into sliding bins (90 chunks, hop 45). GraphicalLasso estimates a sparse association skeleton from the 16 numeric KPIs; transfer entropy is used to assess edge orientation. Six topology prototypes are learned from training bins, and validation/test bins are assigned to the nearest training prototype. These are exploratory dependencies, not verified physical causes. The distinction is important: the graph constrains attention, but the evaluation does not claim that the estimated edges identify causal mechanisms.", size=9.0, after=2)

    section(pdf, "5. Model heads and optimization", "The shared encoder feeds anomaly detection, 11-way fault classification, 16-KPI multi-label RCA, and early warning. The anomaly score combines a summary-based linear anchor, a compact summary residual, and the temporal/graph trunk. Type prediction combines trunk and summary context; RCA is produced from time-averaged KPI-node states. The hazard head operates on causal step states and uses a pre-onset mask. Training uses a weighted joint objective (anomaly, type, RCA and masked hazard losses), AdamW, gradient clipping, normal-window undersampling, and validation-based early stopping.", size=9.0, after=2)

    section(pdf, "6. Implementation and reproducibility", "The implementation is organized into data loading, preprocessing, graph construction, model, experiment, evaluation and reporting modules. Configurations live under configs/; trained weights and run outputs are stored under checkpoints/ and outputs/. Experiments record seeds, configurations and data fingerprints in the results registry. The Streamlit dashboard replays held-out windows and presents telemetry, graph context, predictions and a model what-if panel. A what-if perturbation is a model sensitivity test; it is not an identified real-world intervention.", size=9.0, after=2)

    section(pdf, "7. Evaluation protocol", "Thresholds are selected on validation data and fixed before test evaluation. Detection is reported with PR-AUC and ROC-AUC, with additional operating-point metrics; RCA uses ranking and multi-label measures; uncertainty is estimated by resampling recordings rather than treating overlapping windows as independent. Paired comparisons score both models on the same resampled recordings. This protocol is designed to limit temporal, recording and pseudo-replication leakage.", size=9.0, after=2)

    # Page 3: graph and objective implementation details.
    pdf.add_page()
    pdf.set_font("TNR", "B", 15)
    pdf.cell(180, 8, "Implementation details: graph and learning", new_x="LMARGIN", new_y="NEXT")
    graph_build_figure(pdf, top=35)

    pdf.set_y(63)
    section(pdf, "8. From telemetry to a graph prototype", "The graph builder first groups nearby windows from a recording into overlapping bins. Within each bin, a sparse association skeleton is estimated from the numeric KPI channels. Transfer entropy is evaluated in both directions to assess whether an edge has directional evidence. The resulting bin-level topologies are clustered using training bins only; the assigned prototype supplies the adjacency mask used by the graph-attention layers.", size=9.0, after=1)

    illustrative_graph(pdf, x=24, y=105)
    pdf.set_xy(107, 105)
    section(pdf, "9. How graph context enters the encoder", "Each of the 16 numeric KPIs is encoded as its own temporal node representation. At each time step, attention is computed across KPI nodes and masked to the assigned prototype's edges, with a self-loop for isolated nodes. Edge strength contributes an attention bias. The node states are averaged per time step and fused with the general temporal representation before GRU aggregation. This makes graph structure an inductive bias; it does not make the edge estimates causal facts.", width=85, size=8.8, after=1)

    pdf.set_xy(17, 161)
    section(pdf, "10. Multi-task learning objective", "The shared representation supports four tasks. Anomaly detection uses binary cross-entropy; type classification uses class-weighted cross-entropy on anomalous windows; KPI localization uses multi-label binary cross-entropy; and early warning uses a masked hazard loss that only scores the pre-onset region. The configured task weights are 1.0, 1.0, 1.0, and 0.5, respectively.", size=9.0, after=1)
    objective_figure(pdf, top=216)

    # Page 4: inference path, code organization, and safeguards.
    pdf.add_page()
    pdf.set_font("TNR", "B", 15)
    pdf.cell(180, 8, "System operation, code organization, and safeguards", new_x="LMARGIN", new_y="NEXT")
    labels = ["Select held-out\nwindow", "Prepare window\n+ graph id", "TAGN\ninference", "Scores +\nexplanations", "Optional model\nwhat-if"]
    fills = [COLORS["blue"], COLORS["teal"], COLORS["lavender"], COLORS["gold"], COLORS["rose"]]
    x0, box_w, box_gap, box_h = 16, 32, 4.5, 20
    for i, (text, fill) in enumerate(zip(labels, fills)):
        x = x0 + i * (box_w + box_gap)
        add_box(pdf, x, 37, box_w, box_h, text, fill, 7.7)
        if i < len(labels) - 1:
            arrow(pdf, x + box_w, 47, x + box_w + box_gap, 47)
    pdf.set_xy(16, 59)
    pdf.set_font("TNR", "I", 8)
    pdf.multi_cell(180, 4, "Figure 5. Dashboard inference path. The what-if panel changes model inputs and reports sensitivity; it is not a real network intervention.")

    pdf.set_y(68)
    section(pdf, "11. From trained model to operator-facing view", "The Streamlit dashboard can replay a held-out telemetry window and display its KPI traces, graph context, anomaly score, predicted type, ranked KPI causes, and early-warning output. A user can perturb selected KPI values and rerun the checkpoint to inspect score sensitivity. The interface labels this as a model what-if because the observational graph and predictive model do not identify the effect of a real intervention on a live network.", size=9.0, after=2)

    section(pdf, "12. Code organization", "Data ingestion and adapters are in src/data/; split construction and train-only scaling are in src/preprocessing/; graph estimation and fold-specific graph fitting are in src/models/graphs.py and src/models/graph_fold.py; the core model is src/models/tagn.py. Experiment entry points live in src/experiments/, shared metrics in src/evaluation/, and result aggregation in src/report/. Configuration is kept separately under configs/ so model and training settings can be inspected without reading the experiment code.", size=9.0, after=2)

    section(pdf, "13. Leakage and evaluation safeguards", "The pipeline enforces several checks at the data boundary: boundary-straddling windows are excluded; overlapping fault bursts remain in one split; synthetic source recordings and injected twins share an assignment; and fitted scalers and graph prototypes never use validation or test recordings. Validation selects operating thresholds and early-stopping checkpoints. Test results are computed afterward, with recording-level bootstrap resampling and paired comparisons to account for clustered windows.", size=9.0, after=2)

    section(pdf, "14. Reproduction and verification", "A typical reproduction follows the documented sequence: obtain the dataset, build chunks, run leakage-safe preprocessing, generate graph artifacts, train the selected experiment, and regenerate its report. Each run should record its seed, configuration, code revision, and data fingerprint. Unit and regression tests cover preprocessing, model shapes, split integrity, graph checks, and evaluation behavior. The repository's README and experiment modules provide the runnable commands and artifact locations.", size=9.0, after=2)

    section(pdf, "15. Design boundaries", "The core TAGN and the newer experimental variants are kept conceptually separate. TCN-Transformer backbones, multimodal fusion, intervention utilities, and deferral components should be described as experimental variants unless a corresponding training run and evaluation artifact support the claim. In particular, a graph-based what-if result is evidence about the model's response to a perturbed input, not proof of a physical causal mechanism.", size=9.0, after=0)

    pdf.output(str(OUT))
    DESKTOP.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(DESKTOP))
    print(f"Created {OUT}")
    print(f"Created {DESKTOP}")
    print(f"Pages: {pdf.page_no()}")


if __name__ == "__main__":
    main()
