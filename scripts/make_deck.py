"""Generate the 4-slide project presentation deck."""

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

NAVY = RGBColor(0x0F, 0x2A, 0x47)
ACCENT = RGBColor(0x1F, 0x6F, 0xEB)
GREY = RGBColor(0x44, 0x4A, 0x53)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)


def _bg(slide, color=NAVY):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def _box(slide, l, t, w, h):
    return slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))


def _write(tf, lines, size, color, bold_first=False, space=6):
    tf.word_wrap = True
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.font.size = Pt(size)
        p.font.color.rgb = color
        p.font.bold = bold_first and i == 0
        p.space_after = Pt(space)


def build():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # ---- Slide 1: The plan ----
    s = prs.slides.add_slide(blank)
    _bg(s)
    _write(
        _box(s, 0.6, 0.4, 12.2, 1.0).text_frame,
        ["1 | What We Set Out To Do"], 34, WHITE, True, 0)
    _write(
        _box(s, 0.6, 1.35, 12.2, 0.6).text_frame,
        ["Build a causal-temporal graph model (TAGN) that detects and localizes 5G network faults"],
        17, ACCENT, False, 0)

    left = [
        "The Plan",
        "1. Learn a dynamic KPI dependency graph from raw 5G telemetry",
        "2. Fuse causal convolutions + graph attention + GRU in one model",
        "3. Jointly predict: anomaly, fault type, responsible KPI, warning lead time",
        "4. Evaluate strictly: 5 seeds, 6 ablations, 5-fold leave-recording-out",
        "5. Audit data leakage that inflates published TelecomTS results",
    ]
    _write(_box(s, 0.6, 2.3, 6.1, 4.5).text_frame, left, 16, WHITE, True, 12)

    right = [
        "The Claim We Intended To Test",
        "Does an adaptive, causally-structured graph beat",
        "static and non-graph baselines on 5G RCA?",
        "",
        "Success bar: beat the strongest classical baseline",
        "on PR-AUC under both random and held-out-recording",
        "splits, with confidence intervals excluding zero.",
    ]
    _write(_box(s, 6.9, 2.3, 5.9, 4.5).text_frame, right, 16, WHITE, True, 12)

    # ---- Slide 2: What we did ----
    s = prs.slides.add_slide(blank)
    _bg(s, WHITE)
    _write(
        _box(s, 0.6, 0.4, 12.2, 1.0).text_frame,
        ["2 | What We Actually Did"], 34, NAVY, True, 0)
    _write(
        _box(s, 0.6, 1.3, 12.2, 0.6).text_frame,
        ["Initial pipeline executed; the audit found a normalization failure that changed the benchmark"],
        15, ACCENT, False, 0)

    rows = [
        ("EXECUTED", "RESULT"),
        ("Scaler audit", "Zero-inflated DL_BLER: scaled std 472 → 0.98"),
        ("Corrected Logistic Regression", "PR-AUC 0.957 | ROC-AUC 0.986"),
        ("Corrected TAGN pilot (seed 42)", "PR-AUC 0.574 | ROC-AUC 0.886"),
        ("Original multi-seed/ablation suite", "Pre-fix artifacts; must be rerun under corrected scaler"),
        ("Graph validation audit", "0.000 directed edges retained; causal claim unproven"),
        ("Reproducibility", "26/26 tests pass | private repo, 147 files"),
    ]
    tbl = s.shapes.add_table(len(rows), 2, Inches(0.6), Inches(2.1),
                             Inches(12.1), Inches(4.4)).table
    tbl.columns[0].width = Inches(5.4)
    tbl.columns[1].width = Inches(6.7)
    for r, (a, b) in enumerate(rows):
        for c, val in enumerate((a, b)):
            cell = tbl.cell(r, c)
            cell.text = val
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(14 if r else 15)
            p.font.bold = (r == 0) or (c == 0 and r > 0)
            p.font.color.rgb = WHITE if r == 0 else NAVY
            cell.fill.solid()
            cell.fill.fore_color.rgb = NAVY if r == 0 else (
                RGBColor(0xEC, 0xF2, 0xFA) if r % 2 else WHITE)

    _write(
        _box(s, 0.6, 6.6, 12.2, 0.7).text_frame,
        ["Honest finding: preprocessing quality changed the benchmark; TAGN is not yet SOTA and needs a corrected full rerun."],
        15, ACCENT, True, 0)

    # ---- Slide 3: why it matters / output usefulness ----
    s = prs.slides.add_slide(blank)
    _bg(s)
    _write(
        _box(s, 0.6, 0.4, 12.2, 1.0).text_frame,
        ["3 | What The Output Is Useful For"], 34, WHITE, True, 0)
    _write(
        _box(s, 0.6, 1.3, 12.2, 0.6).text_frame,
        ["The deliverables are usable even though the headline hypothesis failed"],
        16, ACCENT, False, 0)

    cards = [
        ("Leakage-Audit Playbook",
         ["Reproducible protocol exposing recording-level",
          "and template-level shortcuts in TelecomTS RCA.",
          "Reusable for any dataset built from few long",
          "recordings with one label per recording."]),
        ("Benchmarking Harness",
         ["Chronological + burst-aware + LORO splits,",
          "validation-only thresholds, group bootstrap CIs,",
          "seed/ablation sweeps from one config.",
          "Drop-in baseline for future models."]),
        ("Negative-Result Evidence",
         ["First clean comparison showing dynamic graph",
          "underperforms static graph (0.553 vs 0.832)",
          "on this benchmark - prevents wasted work and",
          "redirects the field to better graph estimators."]),
        ("Deployable RCA Toolkit",
         ["Trained checkpoints, KPI attribution, early-warning",
          "hazard curves, and a Streamlit inspector.",
          "Operators get candidate faulty-KPI rankings,",
          "not just an anomaly flag."]),
    ]
    x = 0.6
    for title, body in cards:
        _write(_box(s, x, 2.2, 3.0, 0.6).text_frame, [title], 17, ACCENT, True, 0)
        _write(_box(s, x, 2.9, 3.0, 3.4).text_frame, body, 13, WHITE, False, 8)
        x += 3.15

    _write(
        _box(s, 0.6, 6.5, 12.2, 0.7).text_frame,
        ["Takeaway: the audit + harness is the publishable contribution; the model is the vehicle."],
        16, WHITE, True, 0)

    out = "outputs/figures/project_presentation.pptx"
    prs.save(out)
    print("wrote", out)


if __name__ == "__main__":
    build()
