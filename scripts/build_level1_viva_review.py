"""Build the Level 1 viva and demonstration review from recorded experiment results."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageTemplate,
    PageBreak,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
    HRFlowable,
)
from reportlab.pdfbase.pdfmetrics import stringWidth


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs/experiments"
OUT = ROOT / "output/pdf/PathSnap_Level1_Viva_Review.pdf"
NAVY = colors.HexColor("#10243A")
BLUE = colors.HexColor("#245B84")
TEAL = colors.HexColor("#087E8B")
PALE = colors.HexColor("#EAF2F7")
MINT = colors.HexColor("#E5F4F1")
GOLD = colors.HexColor("#F3B544")
INK = colors.HexColor("#263746")
MUTED = colors.HexColor("#5D6B78")
RED = colors.HexColor("#A33A3A")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: str | float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="CoverTitle", parent=styles["Title"], fontName="Helvetica-Bold",
                          fontSize=28, leading=33, textColor=NAVY, alignment=TA_LEFT, spaceAfter=10))
styles.add(ParagraphStyle(name="CoverSub", parent=styles["Normal"], fontName="Helvetica",
                          fontSize=13, leading=18, textColor=BLUE, spaceAfter=18))
styles.add(ParagraphStyle(name="H1x", parent=styles["Heading1"], fontName="Helvetica-Bold",
                          fontSize=19, leading=23, textColor=NAVY, spaceBefore=5, spaceAfter=10,
                          keepWithNext=True))
styles.add(ParagraphStyle(name="H2x", parent=styles["Heading2"], fontName="Helvetica-Bold",
                          fontSize=12, leading=15, textColor=BLUE, spaceBefore=8, spaceAfter=5,
                          keepWithNext=True))
styles.add(ParagraphStyle(name="Bodyx", parent=styles["BodyText"], fontName="Helvetica",
                          fontSize=9.1, leading=13.2, textColor=INK, spaceAfter=6))
styles.add(ParagraphStyle(name="Smallx", parent=styles["BodyText"], fontName="Helvetica",
                          fontSize=7.8, leading=10.4, textColor=INK, spaceAfter=3))
styles.add(ParagraphStyle(name="Tinyx", parent=styles["BodyText"], fontName="Helvetica",
                          fontSize=6.9, leading=8.8, textColor=INK))
styles.add(ParagraphStyle(name="Calloutx", parent=styles["BodyText"], fontName="Helvetica-Bold",
                          fontSize=10, leading=14, textColor=NAVY, spaceAfter=2))
styles.add(ParagraphStyle(name="CodeX", parent=styles["Code"], fontName="Courier",
                          fontSize=7.2, leading=9.5, leftIndent=7, rightIndent=7, backColor=PALE,
                          borderPadding=6, textColor=NAVY, spaceBefore=4, spaceAfter=7))
styles.add(ParagraphStyle(name="TableHeadX", parent=styles["Normal"], fontName="Helvetica-Bold",
                          fontSize=7.2, leading=8.6, textColor=colors.white, alignment=TA_LEFT))
styles.add(ParagraphStyle(name="TableCellX", parent=styles["Normal"], fontName="Helvetica",
                          fontSize=7.1, leading=8.8, textColor=INK))
styles.add(ParagraphStyle(name="ModelHeadX", parent=styles["Heading2"], fontName="Helvetica-Bold",
                          fontSize=11.2, leading=13, textColor=BLUE, spaceBefore=4, spaceAfter=2,
                          keepWithNext=True))
styles.add(ParagraphStyle(name="ModelBodyX", parent=styles["BodyText"], fontName="Helvetica",
                          fontSize=7.5, leading=9.2, textColor=INK, spaceAfter=2))


def P(text: str, style: str = "Bodyx") -> Paragraph:
    return Paragraph(text, styles[style])


def bullet(text: str) -> Paragraph:
    return Paragraph("<font color='#087E8B'><b>•</b></font>  " + text, styles["Bodyx"])


def make_table(rows, widths, header=True, font_size=7.1, repeat=1):
    normalized = []
    for ri, row in enumerate(rows):
        normalized.append([
            cell if isinstance(cell, Paragraph) else Paragraph(str(cell), styles["TableHeadX"] if header and ri == 0 else styles["TableCellX"])
            for cell in row
        ])
    table = Table(normalized, colWidths=widths, repeatRows=repeat if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D4DEE5")),
    ]
    if header:
        commands += [("BACKGROUND", (0, 0), (-1, 0), NAVY)]
        for row in range(1, len(rows)):
            if row % 2 == 0:
                commands.append(("BACKGROUND", (0, row), (-1, row), colors.HexColor("#F4F7F9")))
    table.setStyle(TableStyle(commands))
    return table


def callout(text: str, fill=PALE, border=BLUE):
    t = Table([[P(text, "Calloutx")]], colWidths=[7.05 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), fill),
        ("BOX", (0, 0), (-1, -1), 0.8, border),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    return t


def draw_bar_chart(title: str, series: list[tuple[str, float, str]], maximum: float,
                   suffix: str = "", width: float = 7.05 * inch, height: float = 2.0 * inch):
    """Small vector chart built directly from recorded CSV metrics."""
    from reportlab.graphics.shapes import Drawing, String, Rect, Line

    d = Drawing(width, height)
    left, right, top, bottom = 126, 12, 28, 22
    chart_w = width - left - right
    chart_h = height - top - bottom
    d.add(String(0, height - 12, title, fontName="Helvetica-Bold", fontSize=9.5, fillColor=NAVY))
    for i in range(4):
        v = maximum * i / 3
        y = bottom + chart_h * i / 3
        d.add(Line(left, y, width - right, y, strokeColor=colors.HexColor("#D8E1E7"), strokeWidth=.45))
        d.add(String(left - 6, y - 2, f"{v:.2f}{suffix}", textAnchor="end", fontName="Helvetica", fontSize=6.5, fillColor=MUTED))
    bar_h = min(15, chart_h / (len(series) * 1.5))
    gap = (chart_h - bar_h * len(series)) / (len(series) - 1) if len(series) > 1 else 0
    for i, (label, value, color) in enumerate(series):
        y = height - top - bar_h - i * (bar_h + gap)
        d.add(String(left - 7, y + 3.5, label, textAnchor="end", fontName="Helvetica", fontSize=7.2, fillColor=INK))
        bar_width = chart_w * max(0, value) / maximum
        d.add(Rect(left, y, bar_width, bar_h, fillColor=colors.HexColor(color), strokeColor=None))
        d.add(String(left + bar_width + 5, y + 3.5, f"{value:.4f}", fontName="Helvetica-Bold", fontSize=7, fillColor=INK))
    return d


def draw_mlp_history(csv_path: Path):
    """Render the real MLP fit histories recorded by save_mlp_history.py."""
    from reportlab.graphics.shapes import Drawing, String, Rect, Line, PolyLine

    rows = read_csv(csv_path)
    loss = [float(row["training_log_loss"]) for row in rows if row["training_log_loss"]]
    val = [float(row["validation_accuracy"]) for row in rows if row["validation_accuracy"]]
    drawing = Drawing(500, 220)
    panel_w, panel_h, panel_y = 205, 150, 32
    panels = [(15, "Training log loss", loss, TEAL), (280, "Early-stopping validation accuracy", val, colors.HexColor("#D17A22"))]
    for x0, title, values, color in panels:
        lo, hi = min(values), max(values)
        pad = max((hi-lo)*.08, .01)
        lo, hi = lo-pad, hi+pad
        drawing.add(String(x0, 203, title, fontName="Helvetica-Bold", fontSize=9.2, fillColor=NAVY))
        drawing.add(Line(x0, panel_y, x0+panel_w, panel_y, strokeColor=MUTED, strokeWidth=.6))
        drawing.add(Line(x0, panel_y, x0, panel_y+panel_h, strokeColor=MUTED, strokeWidth=.6))
        drawing.add(String(x0-5, panel_y+panel_h-2, f"{hi:.3f}", textAnchor="end", fontName="Helvetica", fontSize=6.8, fillColor=MUTED))
        drawing.add(String(x0-5, panel_y-2, f"{lo:.3f}", textAnchor="end", fontName="Helvetica", fontSize=6.8, fillColor=MUTED))
        drawing.add(String(x0+panel_w/2, 13, "Epoch", textAnchor="middle", fontName="Helvetica", fontSize=7.5, fillColor=INK))
        coords = []
        for i, value in enumerate(values):
            x = x0 + panel_w * i / max(1, len(values)-1)
            y = panel_y + panel_h * (value-lo) / (hi-lo)
            coords.extend([x, y])
        drawing.add(PolyLine(coords, strokeColor=color, strokeWidth=2, fillColor=None))
    return drawing


def page_decor(canvas, doc):
    canvas.saveState()
    w, h = letter
    if doc.page > 1:
        canvas.setStrokeColor(colors.HexColor("#D7E0E7"))
        canvas.setLineWidth(.5)
        canvas.line(0.72 * inch, h - 0.55 * inch, w - 0.72 * inch, h - 0.55 * inch)
        canvas.setFont("Helvetica-Bold", 7.5)
        canvas.setFillColor(BLUE)
        canvas.drawString(0.72 * inch, h - 0.42 * inch, "PATHSNAP  /  LEVEL 1 VIVA REVIEW")
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawRightString(w - 0.72 * inch, 0.42 * inch, f"{doc.page}")
        canvas.line(0.72 * inch, 0.55 * inch, w - 0.72 * inch, 0.55 * inch)
    canvas.restoreState()


def build():
    comparison = read_csv(RESULTS / "model_comparison.csv")
    cv = {row["model"]: row for row in comparison if row["partition"] == "cross_validation"}
    student = {row["model"]: row for row in comparison if row["partition"] == "student_test"}
    paired = read_csv(RESULTS / "paired_bootstrap.csv")
    tuned = read_csv(RESULTS / "tuned_vs_untuned_summary.csv")
    tuned_cv = read_csv(RESULTS / "tuned_vs_untuned_cv_summary.csv")
    tuning = read_json(RESULTS / "tuning_result.json")
    selected = read_json(RESULTS / "selected_model.json")
    names = ["hist_gradient_boosting", "small_mlp", "random_forest", "logistic_regression", "calibrated_linear_svm"]
    labels = {
        "hist_gradient_boosting": "HistGradientBoosting",
        "small_mlp": "Small MLP",
        "random_forest": "Random Forest",
        "logistic_regression": "Logistic Regression",
        "calibrated_linear_svm": "Calibrated Linear SVM",
    }
    colors_for = {name: ("#087E8B" if name == "hist_gradient_boosting" else "#6C8CA4") for name in names}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(str(OUT), pagesize=letter, leftMargin=.72*inch, rightMargin=.72*inch,
                          topMargin=.78*inch, bottomMargin=.72*inch,
                          title="PathSnap Level 1 Viva Review", author="PathSnap team study guide")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=page_decor)])
    story = []

    # Cover and decision summary
    story += [Spacer(1, .45*inch), P("PATHSNAP", "CoverSub"),
              P("Level 1 Viva Review", "CoverTitle"),
              P("Five-model training, comparison, selection, evidence, and a practical demo plan", "CoverSub"),
              HRFlowable(width="100%", thickness=2, color=TEAL, spaceAfter=18)]
    story.append(callout("Level 1 result: five candidate scorers were compared using 5-fold GroupKFold over 80 training trips. HistGradientBoosting is selected by the pre-declared primary metric, trip-mean cross-validation Macro F1 (0.4826). It is a narrow point-estimate winner, statistically tied with the small MLP on Macro F1.", MINT, TEAL))
    story += [Spacer(1, 13), P("What this guide covers", "H2x"),
              bullet("The problem setup, supplied files, candidate generation, features, and route-label alignment."),
              bullet("All five algorithms, actual configurations, metrics, comparison results, and selection reasoning."),
              bullet("What the evidence does and does not justify, including candidate misses and evaluation limits."),
              bullet("A step-by-step viva demo plan and commands to show the work in the repository."),
              Spacer(1, 8), P("Use this as a study guide, not a script to recite. The assessment policy requires the team to understand the implementation, write its own conclusions, and disclose AI assistance as required. Verify each claim against the code and run records before presenting it.", "Smallx"),
              Spacer(1, 8), P("Quick Level 1 snapshot", "H2x")]
    rows = [["Evidence", "Recorded result"],
            ["Training data", "80 trips; grouped 5-fold CV; 208,417 GPS observations in OOF evaluation"],
            ["Development test", "10 student_test trips; 19,160 observations"],
            ["Primary metric", "Trip-mean Macro F1, selected on CV"],
            ["Selected model", "HistGradientBoosting; CV Macro F1 0.4826"],
            ["Closest alternative", "Small MLP; CV Macro F1 0.4824"],
            ["Statistical conclusion", "Paired trip bootstrap CI for MLP minus HGB: -0.00249 to +0.00209; includes zero"]]
    story.append(make_table(rows, [1.55*inch, 5.5*inch]))
    story.append(PageBreak())

    # Challenge and data
    story += [P("1. The Level 1 task and data", "H1x"),
              P("PathSnap is a map-matching problem: noisy time-stamped GPS locations must be associated with a trip-specific directed road network and assembled into a plausible route. A GPS row is not directly paired with a road-network row, and arc IDs are local to each trip. This is why the model scores observation-candidate pairs rather than predicting a global road ID.")]
    story.append(make_table([
        ["File", "Contents and interpretation"],
        [".track", "Three whitespace-separated columns: longitude, latitude, timestamp seconds; nominal sampling is about 1 Hz but gaps occur."],
        [".nodes", "Two columns: longitude and latitude; row index is node ID."],
        [".arcs", "Two integer columns: from-node ID and to-node ID; row index is directed arc ID."],
        [".route", "Chronological sequence of reference arc IDs; not a one-label-per-GPS-row array."],
    ], [1.0*inch, 6.05*inch]))
    story += [Spacer(1, 8), P("Partitions and scope", "H2x"),
              bullet("The supplied student folder contains 80 training trips and 10 student_test trips. The hackathon brief describes a separate 10-trip evaluator partition, but that evaluator folder was not present in the supplied working data."),
              bullet("Training trips support fitting and trip-grouped cross-validation. student_test is allowed for development comparison by the brief; it is not the hidden evaluator set."),
              bullet("Do not tune or select using evaluator labels. Keep all reference labels away from the prediction/UI payload."),
              P("The inventory code checks file presence, row counts, coordinate bounds, timestamp ordering/gaps, graph components, isolated nodes, route arc validity, and route disconnected steps. The records show real trip variability, from short traces to long ones, frequent 1-second median gaps, and occasional long gaps; coordinate extents span many geographic regions.")]
    story.append(make_table([
        ["Exploration finding", "What it means for the solution"],
        ["Timestamp gaps can be much larger than 1 second", "Speed and transition reasoning must use elapsed time, and long gaps weaken local continuity assumptions."],
        ["Many trip-specific graph sizes and disconnected components", "Graph operations and arc IDs must stay trip-local; a single global arc classifier is unsuitable."],
        ["Reference routes are ordered arc sequences", "Create observation labels by aligning route progress, not by matching array positions."],
        ["GPS can deviate from its route and candidate set", "Measure alignment distance and candidate-miss rate; report limits instead of silently dropping difficult cases."],
    ], [2.05*inch, 5.0*inch]))
    story.append(PageBreak())

    # Formulation
    story += [P("2. From raw trips to training rows", "H1x"),
              P("The model is a candidate scorer. For each GPS observation, the pipeline creates a small list of nearby road arcs and estimates which candidate best explains that point. This turns route matching into binary relevance scoring over observation-arc pairs; the highest-scoring candidate is the pointwise prediction.")]
    story.append(make_table([
        ["Step", "Implemented choice", "Why it matters"],
        ["1. Project coordinates", "Per-trip UTM projection from longitude/latitude", "Distances are in meters rather than angular degrees."],
        ["2. Generate candidates", "Up to top 8 arcs within 60 m; use nearest arc if none falls in radius", "Bounds model work while retaining nearby alternatives."],
        ["3. Align labels", "Monotone Viterbi-style alignment of GPS points to occurrences in the ordered reference route; 100 m label search radius", "Accounts for route progress and the fact the route has fewer entries than GPS."],
        ["4. Build features", "Seven observation-arc features (listed below)", "Captures spatial fit, direction, position, and motion context."],
        ["5. Train", "Positive if candidate arc equals the aligned reference arc; other candidates are negatives", "Same prediction target is used for all five models."],
        ["6. Score and rank", "Predict candidate probabilities; select the highest score per observation", "Produces observation-level road assignments for evaluation."],
    ], [1.05*inch, 3.25*inch, 2.75*inch]))
    story += [Spacer(1, 8), P("Seven implemented features", "H2x")]
    story.append(make_table([
        ["Feature", "Meaning / viva explanation"],
        ["Cross-track distance (m)", "Shortest distance from GPS point to candidate arc. Smaller usually means a better spatial match."],
        ["Heading cosine", "Cosine agreement between local GPS motion direction and directed arc heading. Near +1 means aligned; near -1 means opposed."],
        ["Along-arc fraction", "Projected position along the line segment, scaled from 0 to 1."],
        ["Arc length (m)", "Length of the candidate road segment."],
        ["GPS speed (m/s)", "Distance between consecutive GPS samples divided by elapsed time; 0 if elapsed time is not positive."],
        ["GPS gap (s)", "Time since the previous observation; helps flag irregular sampling."],
        ["GPS step distance (m)", "Projected distance between the current and previous GPS observation."],
    ], [1.65*inch, 5.4*inch]))
    story += [Spacer(1, 6), P("Important limitation: candidate misses are not made to disappear. Missed observations are excluded from scorer fitting because no positive candidate exists, but remain in evaluation. Training miss rate is 12.12%; student_test miss rate is 1.58%. This large difference is a data-coverage warning and can affect comparisons, especially Log Loss.", "Smallx"), PageBreak()]

    # Experiment design and metrics
    story += [P("3. Experiment design and how to explain the metrics", "H1x"),
              P("All five models use the same candidate rows and labels. The split unit is the trip, not a GPS point. GroupKFold keeps every observation from one trip within one held-out fold, reducing the risk that a model is evaluated on near-duplicate samples from a trip it has already seen.")]
    story.append(make_table([
        ["Design element", "Actual Level 1 setup"],
        ["Training partition", "80 trips; all five models fit in 5-fold GroupKFold, then refit on all training trips to score student_test."],
        ["Random seed", "99 for the estimator constructors and recorded experiment configuration."],
        ["CV observations", "208,417 observations across the 80 out-of-fold training trips."],
        ["Student-test observations", "19,160 observations across 10 permitted development-test trips."],
        ["Primary selection metric", "Mean of per-trip Macro F1 on 5-fold CV; selected model is highest point estimate."],
        ["Aggregation", "Macro F1 / precision / recall / route continuity are first computed per trip and averaged over trips. Top-1 and Log Loss are weighted by observation count."],
        ["Timing", "Recorded model fit time sums CV fold fits and the full-training fit used for student_test scoring. Inference time is model scoring only; it excludes data loading and candidate generation."],
    ], [1.6*inch, 5.45*inch]))
    story += [Spacer(1, 8), P("Metrics in plain language", "H2x"),
              bullet("Top-1 accuracy: for each GPS row, take the candidate with the highest model score and check whether it matches the aligned reference arc."),
              bullet("Macro F1: computes F1 per represented road-arc class and averages classes equally, emphasizing rare arcs more than raw accuracy does."),
              bullet("Precision / recall: precision asks how many predicted arc assignments are correct; recall asks how many reference arc labels were recovered."),
              bullet("Log Loss: penalizes low probability assigned to the correct arc. Here it uses probability mass over the generated candidates. If the true arc is absent, the code assigns a tiny probability (1e-12), causing a large penalty."),
              bullet("Route continuity: fraction of consecutive predicted arc pairs that repeat the same arc or connect from the previous arc's endpoint to the next arc's start node. It is a simple local proxy, not a full sequence similarity metric."),
              bullet("Candidate miss rate: proportion of observations whose reference arc is missing from the generated candidate set (including reference alignments beyond the label radius).")]
    story += [Spacer(1, 6), callout("Avoid saying that Top-1 alone proves a good route. The route objective also depends on class-balanced performance, candidate coverage, probability quality, and whether consecutive arc assignments make sense.", PALE, BLUE), PageBreak()]

    # Models details
    story += [P("4. The five models: purpose, configuration, objective, trade-offs", "H1x"),
              P("Every estimator predicts candidate relevance. The probability outputs are used to rank candidates; for the calibrated SVM, probabilities come from a calibration layer on top of margin scores.")]
    model_details = [
        ("Logistic Regression", "Linear / probabilistic baseline", "StandardScaler + LogisticRegression; max_iter=600; class_weight=balanced; seed=99.", "Regularized logistic log-loss / cross-entropy.", "Fast and interpretable; linear boundary may miss nonlinear feature interactions."),
        ("Random Forest", "Bagging ensemble of trees", "120 trees; min_samples_leaf=2; class_weight=balanced_subsample; n_jobs=-1; seed=99.", "Tree impurity (Gini by default) at each split; averaged tree probabilities.", "Captures nonlinearities and interactions; larger fit/inference cost and can be less calibrated."),
        ("HistGradientBoosting", "Boosted decision trees", "max_iter=100; learning_rate=0.08; l2_regularization=1.0; class_weight=balanced; seed=99.", "Gradient boosting classifier with log-loss objective.", "Strong tabular performance and fast scoring; less directly interpretable than a linear model."),
        ("Calibrated Linear SVM", "Linear margin classifier + calibration", "StandardScaler + LinearSVC(C=1.0, class_weight=balanced, seed=99); CalibratedClassifierCV(cv=3).", "Squared hinge margin loss; calibration maps margins to probabilities.", "Effective linear decision boundary; calibration adds a second layer and its default inner CV is not trip-grouped."),
        ("Small MLP", "Neural network", "StandardScaler + hidden layers (32, 16); alpha=1e-3; max_iter=100; early_stopping=True; seed=99.", "Adam optimization of neural-network log loss, with L2 regularization.", "Can model nonlinear interactions; slower/less stable, and needs preserved learning-history evidence."),
    ]
    for title, family, config, objective, trade in model_details:
        story.append(KeepTogether([
            P(title, "ModelHeadX"),
            P(f"<b>Family:</b> {family}. <b>Actual configuration:</b> {config}", "ModelBodyX"),
            P(f"<b>Objective:</b> {objective} <b>Strength / limitation for PathSnap:</b> {trade}", "ModelBodyX"),
        ]))
    story += [Spacer(1, 3), P("The models differ in inductive bias, but the evaluation rows and split design are held constant. This makes the comparison more meaningful than comparing five unrelated experiments.", "Smallx"), PageBreak()]

    # Results table + charts
    story += [P("5. Results from the actual experiment", "H1x"),
              P("The following figures are from outputs/experiments/model_comparison.csv. Treat these as point estimates; the next section discusses uncertainty and paired comparisons.")]
    rows = [["Model", "CV<br/>Macro F1", "CV<br/>Top-1", "Student<br/>Macro F1", "Student<br/>Top-1", "Student<br/>Log Loss", "Student<br/>continuity"]]
    for name in names:
        a, b = cv[name], student[name]
        rows.append([labels[name], fmt(a["macro_f1"]), fmt(a["top1_accuracy"]), fmt(b["macro_f1"]),
                     fmt(b["top1_accuracy"]), fmt(b["log_loss"]), fmt(b["route_continuity"])])
    story.append(make_table(rows, [1.35*inch, .75*inch, .68*inch, .85*inch, .72*inch, .78*inch, .94*inch]))
    story += [Spacer(1, 10), draw_bar_chart("5-fold CV Macro F1 (primary selection metric)",
                [(labels[n], float(cv[n]["macro_f1"]), colors_for[n]) for n in names], .55),
              Spacer(1, 8), draw_bar_chart("Student-test Macro F1", [(labels[n], float(student[n]["macro_f1"]), colors_for[n]) for n in names], .60),
              Spacer(1, 7), P("Additional measured diagnostics", "H2x")]
    diag = [["Model", "CV precision", "CV recall", "Student precision", "Student recall", "Fit seconds*", "Scoring ms / obs*"]]
    for name in names:
        a, b = cv[name], student[name]
        diag.append([labels[name], fmt(a["macro_precision"]), fmt(a["macro_recall"]), fmt(b["macro_precision"]),
                     fmt(b["macro_recall"]), fmt(a["train_seconds"], 2), fmt(b["inference_ms_per_observation"], 5)])
    story.append(make_table(diag, [1.38*inch, .75*inch, .66*inch, .9*inch, .77*inch, .7*inch, .82*inch]))
    story += [Spacer(1, 3), P("Timing note: Fit seconds sum CV folds plus the full-data fit for student scoring; inference timing covers predict_proba only.", "Tinyx"),
              PageBreak(), P("5.1 MLP training history", "H1x"),
              P("This plot comes from a real refit of the registered MLP on all non-missed training candidate rows, using seed 99 and the same estimator configuration. It records the classifier's epoch-wise training loss and its early-stopping validation accuracy."),
              draw_mlp_history(RESULTS / "mlp_training_history.csv"),
              P("Scikit-learn's MLPClassifier exposes validation accuracy scores for early stopping, not an epoch-wise validation-loss history. The validation panel is therefore accuracy, not loss. The 28 recorded epochs reflect early stopping; do not compare this curve as if it came from a specific CV fold.", "Smallx"),
              PageBreak()]

    # Selection reasoning
    story += [P("6. Why HGB was selected - and what the results do not prove", "H1x"),
              P("The declared rule was to select the model with the highest trip-mean 5-fold CV Macro F1. HistGradientBoosting scored 0.482601, just above the small MLP at 0.482388. It also has the highest student-test Macro F1 and Top-1 point estimates among the five models.")]
    # lookup paired comparisons
    def find_pair(partition, metric, other):
        for row in paired:
            if row["partition"] == partition and row["metric"] == metric and row["reference_model"] == "hist_gradient_boosting" and row["comparison_model"] == other:
                return row
        return None
    cv_mlp = find_pair("cross_validation", "macro_f1", "small_mlp")
    test_mlp = find_pair("student_test", "macro_f1", "small_mlp")
    cv_rf = find_pair("cross_validation", "macro_f1", "random_forest")
    story.append(make_table([
        ["Comparison evidence", "Recorded result", "Interpretation"],
        ["HGB vs MLP, CV Macro F1", f"MLP - HGB = {float(cv_mlp['mean_paired_difference']):+.6f}; 95% CI [{float(cv_mlp['ci95_low']):+.6f}, {float(cv_mlp['ci95_high']):+.6f}] across 80 trips", "CI contains zero. Do not claim a decisive Macro F1 win."],
        ["HGB vs MLP, student Macro F1", f"MLP - HGB = {float(test_mlp['mean_paired_difference']):+.6f}; 95% CI [{float(test_mlp['ci95_low']):+.6f}, {float(test_mlp['ci95_high']):+.6f}] across 10 trips", "CI contains zero; small test sample limits certainty."],
        ["HGB vs Random Forest, CV Macro F1", f"RF - HGB = {float(cv_rf['mean_paired_difference']):+.6f}; 95% CI [{float(cv_rf['ci95_low']):+.6f}, {float(cv_rf['ci95_high']):+.6f}]", "HGB is clearly ahead on this CV comparison."],
        ["Student-test trade-off", "MLP has lower Log Loss (1.1925 vs HGB 1.2960) and slightly higher continuity (0.9132 vs 0.9117).", "There is no single metric on which HGB is best."],
    ], [1.62*inch, 3.35*inch, 2.08*inch]))
    story += [Spacer(1, 9), P("A defensible viva explanation", "H2x"),
              callout("We selected HistGradientBoosting because the pre-specified primary metric was trip-mean 5-fold CV Macro F1, where it had the highest point estimate. Its lead over the MLP is only 0.000214 and the paired bootstrap interval includes zero, so we describe HGB as the selected top-scoring model rather than a statistically decisive winner. The MLP has better student-test Log Loss, and their route continuity is nearly tied. The result is a transparent metric-based selection with acknowledged uncertainty.", MINT, TEAL),
              Spacer(1, 9), P("Generalization and overfitting: what can be said", "H2x"),
              P("The student-test Macro F1 values are broadly close to the corresponding CV values, but CV and student_test differ in trip composition and candidate-miss rates, so this is not a formal overfit test. The experiment log does not contain a dedicated train-vs-validation learning curve or per-fold training score for every model. Do not claim 'no overfitting' from these aggregate numbers alone. The MLP's early-stopping setting is part of its configuration, but its epoch-wise history must be retained as actual evidence if the report includes its training curve."),
              Spacer(1, 7), P("Known data-quality limitation", "H2x"),
              P("The training candidate miss rate is 12.12%, compared with 1.58% on student_test. Because missed reference arcs are still evaluated, this may dominate some losses and complicates train/test comparisons. Candidate generation is a major part of the overall system; the scorer cannot select the correct arc if it was never generated."), PageBreak()]

    # Bayesian tuning and HMM separate
    story += [P("7. Related follow-on work: tuning and route decoding", "H1x"),
              P("This work goes beyond the five-model Level 1 selection. Keep the distinction clear in the viva: Level 1 chooses the scorer; tuning optimizes the selected scorer; the HMM/Viterbi stage converts local candidate scores into a coherent route sequence.")]
    story += [P("Bayesian optimization", "H2x"),
              P(f"Optuna TPE ran {tuning['trials']} trials over {tuning['folds']}-fold trip-group CV, optimizing mean trip-level CV Macro F1. Best trial score: {float(tuning['best_value_cv_macro_f1']):.6f}. Search result: {tuning['best_params']}.")]
    t0 = next(row for row in tuned if row["version"] == "untuned")
    t1 = next(row for row in tuned if row["version"] == "tuned")
    c0 = next(row for row in tuned_cv if row["version"] == "untuned")
    c1 = next(row for row in tuned_cv if row["version"] == "tuned")
    story.append(make_table([
        ["Metric", "Untuned HGB", "Tuned HGB", "Change"],
        ["3-fold CV Macro F1", fmt(c0["cv_macro_f1_trip_mean"]), fmt(c1["cv_macro_f1_trip_mean"]), f"{float(c1['cv_macro_f1_trip_mean'])-float(c0['cv_macro_f1_trip_mean']):+.6f}"],
        ["Student Macro F1", fmt(t0["student_test_macro_f1_trip_mean"]), fmt(t1["student_test_macro_f1_trip_mean"]), f"{float(t1['student_test_macro_f1_trip_mean'])-float(t0['student_test_macro_f1_trip_mean']):+.6f}"],
        ["Student Top-1", fmt(t0["student_test_top1_accuracy"]), fmt(t1["student_test_top1_accuracy"]), f"{float(t1['student_test_top1_accuracy'])-float(t0['student_test_top1_accuracy']):+.6f}"],
        ["Student Log Loss", fmt(t0["student_test_log_loss"]), fmt(t1["student_test_log_loss"]), f"{float(t1['student_test_log_loss'])-float(t0['student_test_log_loss']):+.6f} (lower is better)"],
        ["Student continuity", fmt(t0["student_test_route_continuity_trip_mean"]), fmt(t1["student_test_route_continuity_trip_mean"]), f"{float(t1['student_test_route_continuity_trip_mean'])-float(t0['student_test_route_continuity_trip_mean']):+.6f}"],
    ], [1.72*inch, 1.45*inch, 1.45*inch, 2.43*inch]))
    story += [Spacer(1, 5), P("The gain is small; the paired student-trip confidence interval includes zero. The best 15-trial result should not be presented as proof of a robust improvement. The tuning results and full trial records are retained in outputs/experiments/bayesian_trials.csv and tuning_result.json.", "Smallx"),
              P("HMM / Viterbi extension", "H2x"),
              P("Hidden states are candidate road arcs at each time step. Emission evidence comes from the selected scorer's candidate scores. Transition scoring favors directed graph continuity and plausible travel progress, with tolerance adapting to GPS gap length; Viterbi finds the highest-scoring sequence. The selected decoder used base_sigma=10, gap_tolerance=4, disconnect_penalty=4. Missing candidates fall back to nearest-arc candidate generation; the decoder also handles disconnected transitions using a penalty rather than silently dropping observations."),
              make_table([
                  ["Partition", "Classifier only Macro F1", "+ HMM Macro F1", "Continuity before", "Continuity after"],
                  ["Training OOF", "0.4826", "0.5421", "0.8945", "0.9746"],
                  ["Student test", "0.5137", "0.5689", "0.9117", "0.9866"],
              ], [1.18*inch, 1.5*inch, 1.4*inch, 1.35*inch, 1.45*inch]),
              Spacer(1, 6), P("This is a measured system-level improvement beyond Level 1. Student-test paired intervals support gains in Macro F1 and continuity, while the Top-1 interval includes zero. Avoid saying that every metric improves significantly.", "Smallx"), PageBreak()]

    # Viva questions
    story += [P("8. Likely viva questions and answer points", "H1x")]
    qa = [
        ("Why is this not ordinary multiclass classification?", "Arc IDs are local to each trip and the road graph changes trip by trip. The model scores candidate arcs for each observation; sequence decoding later uses the trip's graph."),
        ("Why not match every GPS sample to the nearest road?", "Nearest geometry ignores direction, route progress, class imbalance, GPS noise, and connectivity. It is a useful baseline, not a complete route reconstruction method."),
        ("Why project longitude and latitude?", "They are angular units. Projected coordinates make the point-to-segment distances interpretable in meters for local trip geometry."),
        ("How did you turn the route into per-observation labels?", "The route is ordered but shorter/different length than the GPS stream. We align GPS points monotonically to route-arc occurrences using projected distance and progress consistency."),
        ("What does a candidate miss mean?", "The aligned reference arc is absent from the generated top-K candidate set (or alignment exceeds the label threshold). No classifier can recover it from the candidates; report miss rate separately."),
        ("Why Macro F1 as the primary metric?", "Road arcs are imbalanced. Macro F1 weights represented arc classes equally, so frequent arcs do not fully dominate model selection. We still report Top-1, Log Loss, precision, recall, continuity, misses, and timing."),
        ("Why HGB?", "It achieved the best cross-validation Macro F1 point estimate under the stated rule. Its lead over MLP is tiny and not statistically separated, so selection is evidence-based but uncertain."),
        ("What is the biggest limitation?", "Training candidate misses are high at 12.12%, and the reference route is an imperfect alignment source. Candidate coverage and label quality can limit the scorer regardless of algorithm."),
        ("Did you use the evaluator test set?", "No evaluator folder was present in the provided data. We used only training and the permitted student_test development partition."),
        ("What does the HMM add?", "It uses road connectivity and time-gap-aware transitions to turn independent candidate scores into a globally plausible sequence. It improves continuity and Macro F1 in our recorded comparison."),
    ]
    for q, a in qa:
        story.append(KeepTogether([P(q, "H2x"), P(a)]))
    story += [Spacer(1, 5), callout("A good viva answer ties an algorithm property to this task and then points to an actual experiment output. Avoid generic textbook definitions without connecting them to the trip-local candidate problem.", PALE, BLUE), PageBreak()]

    # Demo plan
    story += [P("9. Viva/demo plan: what to show and in what order", "H1x"),
              P("Aim for a 6-8 minute Level 1 walkthrough. Show the evidence in the repository, then explain the decision. Do not try to rerun the full cross-validation suite unless the evaluators specifically request it and there is enough time.")]
    demo = [
        ("1. State the task (30 sec)", "Show README.md. Explain noisy GPS-to-road matching, trip-local directed graphs, and why one global road ID classifier does not fit."),
        ("2. Show data inspection (45 sec)", "Open outputs/data_inventory.csv. Point out the 80 train / 10 student_test trips, route validity and gap/connectivity diagnostics. Mention evaluator partition was not supplied."),
        ("3. Show the pipeline (60 sec)", "Open src/pathsnap/features/candidates.py. Trace projected coordinates -> top-K candidates -> seven features -> monotone route alignment -> candidate labels/miss flags."),
        ("4. Show the five models (45 sec)", "Open src/pathsnap/models/candidates.py and configs/experiment.yaml. Identify the five families and seed 99; highlight scaling for linear/MLP models."),
        ("5. Show the comparison (90 sec)", "Open outputs/experiments/model_comparison.csv and outputs/experiments/plots/cv_macro_f1.svg. Then student_test_accuracy_macro_f1.svg and log_loss_comparison.svg. Explain the aggregation and report caveats."),
        ("6. Defend selection (60 sec)", "Open outputs/experiments/selected_model.json and paired_bootstrap.csv. Explain HGB is the primary-metric winner but not significantly different from MLP on Macro F1."),
        ("7. Demonstrate reproducibility (45 sec)", "Show scripts/prepare_data.py and scripts/run_model_suite.py. Explain the exact commands and the expected dependency on PATHSNAP_DATA_ROOT."),
        ("8. Mention follow-on work (45 sec)", "Briefly show tuning_result.json and decoder_comparison.csv as later extensions. Keep them separate from the five-model Level 1 selection."),
    ]
    for title, desc in demo:
        story += [P(title, "H2x"), P(desc)]
    story += [Spacer(1, 4), P("Useful demo commands", "H2x"),
              P("# Set this to the supplied Student folder containing train/ and student_test/<br/>export PATHSNAP_DATA_ROOT='/path/to/Student folder'<br/><br/># Show current package/test setup and reproduce data preparation + Level 1 comparison<br/>python -m pip install -e .<br/>python scripts/prepare_data.py<br/>python scripts/run_model_suite.py<br/>python scripts/analyze_comparison.py<br/><br/># If the environment is already installed, show the saved evidence directly<br/>open outputs/experiments/plots/cv_macro_f1.svg<br/>open outputs/experiments/plots/student_test_accuracy_macro_f1.svg", "CodeX"),
              PageBreak()]

    # Submission readiness
    story += [P("10. Level 1 repository evidence and final checks", "H1x"),
              P("The repository should let an evaluator trace each reported result from the saved output back to the code and configuration that produced it.")]
    story.append(make_table([
        ["What to show", "Repository evidence"],
        ["Data understanding", "outputs/data_inventory.csv; scripts/prepare_data.py; source data remains external."],
        ["Feature and label pipeline", "src/pathsnap/features/candidates.py; artifacts/intermediate/candidate_summary.csv is local/ignored; document feature names and miss policy."],
        ["Five model definitions", "src/pathsnap/models/candidates.py; configs/experiment.yaml."],
        ["Training/evaluation code", "scripts/run_model_suite.py; src/pathsnap/evaluation/metrics.py."],
        ["Results and model selection", "outputs/experiments/model_comparison.csv; per_trip_metrics.csv; selected_model.json; paired_bootstrap.csv."],
        ["Visual comparison", "outputs/experiments/plots/cv_macro_f1.svg; student_test_accuracy_macro_f1.svg; log_loss_comparison.svg."],
        ["Tuning and later decoder", "scripts/tune_selected_model.py; bayesian_trials.csv; tuning_result.json; scripts/run_decoder_comparison.py; decoder_comparison.csv."],
        ["Final scorer artifacts", "artifacts/final/candidate_scorer_untuned.joblib and candidate_scorer_tuned.joblib (include if repository size allows)."],
        ["Viva material", "This PDF and docs/level1_viva_demo.md."],
    ], [2.0*inch, 5.05*inch]))
    story += [Spacer(1, 9), P("Before the viva", "H2x"),
              bullet("Re-run the project from a clean environment and verify the saved outputs match the run logs."),
              bullet("The actual MLP training-loss and early-stopping validation-accuracy histories are saved in outputs/experiments/mlp_training_history.csv and mlp_training_history.svg. Scikit-learn does not expose validation loss for this setup, so label the validation series accurately."),
              bullet("Make sure all required source files and actual result CSVs/plots are committed; do not commit raw personal dataset files, credentials, or local environment directories."),
              bullet("A real Folium map and UI/API demo evidence are broader submission requirements, not core evidence for five-model selection. Keep the Level 1 presentation focused, but plan those deliverables separately."),
              bullet("All four teammates should be able to explain the route-label alignment, candidate misses, split design, primary metric, and why the HGB/MLP difference is uncertain."),
              Spacer(1, 10), callout("Final one-sentence takeaway: We compared five candidate-scoring algorithms on identical trip-grouped folds, selected HistGradientBoosting by CV Macro F1, documented the close MLP result and candidate-coverage limitation, and retained the experiment records so the choice can be checked and reproduced.", MINT, TEAL),
              Spacer(1, 10), P("Sources checked for this review: the PathSnap Main Guidelines, Dataset Exploration, Models and Training Guidelines, Deliverables, and Assessment and Plagiarism Policy; project README and experiment artifacts generated by the current implementation.", "Tinyx")]

    doc.build(story)
    print(OUT)


if __name__ == "__main__":
    build()
