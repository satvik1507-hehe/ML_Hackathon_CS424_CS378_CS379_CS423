"""Build the consolidated, evidence-linked PathSnap technical report PDF."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, PageBreak, PageTemplate, Paragraph,
    Spacer, Table, TableStyle, KeepTogether,
)
from PIL import Image as PILImage, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "PathSnap_Technical_Report.pdf"
BLUE = colors.HexColor("#18324b")
TEAL = colors.HexColor("#148a8a")
PALE = colors.HexColor("#eef4f8")
LIGHT = colors.HexColor("#f7f9fb")
MUTED = colors.HexColor("#536575")


def read_csv(rel: str) -> list[dict[str, str]]:
    with (ROOT / rel).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_json(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def pct(x: str | float, digits: int = 2) -> str:
    return f"{float(x) * 100:.{digits}f}%"


def num(x: str | float, digits: int = 3) -> str:
    return f"{float(x):.{digits}f}"


def img(rel: str, width: float = 174 * mm, height: float = 105 * mm) -> Image:
    im = Image(str(ROOT / rel))
    scale = min(width / im.imageWidth, height / im.imageHeight)
    im.drawWidth = im.imageWidth * scale
    im.drawHeight = im.imageHeight * scale
    return im


styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="CoverTitle", parent=styles["Title"], fontName="Helvetica-Bold",
                          fontSize=29, leading=34, textColor=BLUE, alignment=TA_CENTER,
                          spaceAfter=11 * mm))
styles.add(ParagraphStyle(name="CoverSub", parent=styles["Normal"], fontSize=13, leading=19,
                          textColor=MUTED, alignment=TA_CENTER))
styles.add(ParagraphStyle(name="H1x", parent=styles["Heading1"], fontName="Helvetica-Bold",
                          fontSize=19, leading=23, textColor=BLUE, spaceBefore=2 * mm,
                          spaceAfter=4 * mm))
styles.add(ParagraphStyle(name="H2x", parent=styles["Heading2"], fontName="Helvetica-Bold",
                          fontSize=12, leading=15, textColor=TEAL, spaceBefore=3 * mm,
                          spaceAfter=2 * mm))
styles.add(ParagraphStyle(name="Bodyx", parent=styles["BodyText"], fontSize=8.8,
                          leading=12.2, textColor=colors.HexColor("#243544"), spaceAfter=2.4 * mm))
styles.add(ParagraphStyle(name="Smallx", parent=styles["BodyText"], fontSize=7.2,
                          leading=9.2, textColor=MUTED, spaceAfter=1.4 * mm))
styles.add(ParagraphStyle(name="Captionx", parent=styles["BodyText"], fontSize=7.7,
                          leading=9.5, textColor=MUTED, alignment=TA_CENTER,
                          spaceBefore=1.5 * mm, spaceAfter=3 * mm))
styles.add(ParagraphStyle(name="Calloutx", parent=styles["BodyText"], fontSize=8.5,
                          leading=11.5, textColor=BLUE, backColor=PALE,
                          borderColor=TEAL, borderWidth=0.7, borderPadding=6,
                          spaceBefore=2 * mm, spaceAfter=3 * mm))
styles.add(ParagraphStyle(name="Tablex", parent=styles["BodyText"], fontSize=6.8,
                          leading=8.2, textColor=colors.HexColor("#1c2d3b")))
styles.add(ParagraphStyle(name="TableHeadx", parent=styles["BodyText"], fontName="Helvetica-Bold",
                          fontSize=6.6, leading=8, textColor=colors.white))


def para(text: str, style: str = "Bodyx") -> Paragraph:
    return Paragraph(text, styles[style])


def table(rows: list[list[str]], widths: list[float], repeat: int = 1) -> Table:
    data = [[Paragraph(str(v), styles["TableHeadx"] if ri == 0 else styles["Tablex"])
             for v in row] for ri, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=repeat, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BLUE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5dd")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def page(canvas, doc):
    canvas.saveState()
    w, h = doc.pagesize
    if doc.page > 1:
        canvas.setStrokeColor(colors.HexColor("#d9e1e7"))
        canvas.line(18 * mm, h - 14 * mm, w - 18 * mm, h - 14 * mm)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(18 * mm, h - 10.5 * mm, "PathSnap | consolidated technical report")
        canvas.drawRightString(w - 18 * mm, 9 * mm, f"{doc.page}")
    canvas.restoreState()


def make_chart(path: Path, title: str, labels: list[str], values: list[float],
               *, percent: bool = False, lower_is_better: bool = False) -> None:
    """Render a clean bar chart from the actual CSV metric values using Pillow."""
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = 1800, 700
    im = PILImage.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 25)
        small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 20)
        title_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 31)
    except OSError:
        font = small = title_font = ImageFont.load_default()
    draw.text((60, 25), title, fill="#18324b", font=title_font)
    left, right, top, bottom = 130, 1740, 110, 560
    vmax = max(values) * 1.18 if values else 1.0
    if lower_is_better:
        vmax = max(values) * 1.18
    for tick in range(6):
        y = bottom - (bottom - top) * tick / 5
        draw.line((left, y, right, y), fill="#d9e1e7", width=2)
        value = vmax * tick / 5
        label = f"{value*100:.0f}%" if percent else f"{value:.2f}"
        draw.text((20, y - 13), label, fill="#536575", font=small)
    step = (right - left) / max(len(values), 1)
    bar_w = min(190, step * 0.62)
    for i, (label, value) in enumerate(zip(labels, values)):
        x = left + step * (i + 0.5)
        y = bottom - (bottom - top) * value / vmax
        color = "#148a8a" if i == 0 else "#587187"
        draw.rectangle((x - bar_w/2, y, x + bar_w/2, bottom), fill=color)
        val = f"{value*100:.2f}%" if percent else f"{value:.3f}"
        box = draw.textbbox((0, 0), val, font=small)
        draw.text((x - (box[2]-box[0])/2, y - 30), val, fill="#243544", font=small)
        words = label.split(" ")
        line = ""
        lines = []
        for word in words:
            test = (line + " " + word).strip()
            if draw.textbbox((0,0), test, font=small)[2] > step - 8 and line:
                lines.append(line); line = word
            else:
                line = test
        if line:
            lines.append(line)
        yy = bottom + 12
        for text_line in lines[:2]:
            box = draw.textbbox((0,0), text_line, font=small)
            draw.text((x-(box[2]-box[0])/2, yy), text_line, fill="#243544", font=small)
            yy += 24
    im.save(path, dpi=(200, 200))


def make_metric_charts() -> None:
    out = ROOT / "reports" / "figures"
    old = read_csv("outputs/experiments/model_comparison.csv")
    names = ["hist_gradient_boosting", "small_mlp", "random_forest",
             "logistic_regression", "calibrated_linear_svm"]
    labels = ["HistGradientBoosting", "Small MLP", "Random Forest",
              "Logistic Regression", "Calibrated Linear SVM"]
    for part, metric, file, title, percent, lower in [
        ("cross_validation", "macro_f1", "cv_macro_f1.png", "Original suite CV Macro F1 (train partition)", True, False),
        ("student_test", "top1_accuracy", "student_test_top1.png", "Original suite Top-1 (student_test partition)", True, False),
        ("student_test", "macro_f1", "student_test_macro_f1.png", "Original suite Macro F1 (student_test partition)", True, False),
        ("student_test", "log_loss", "student_test_logloss.png", "Original suite log loss (student_test partition)", False, True),
    ]:
        by = {(r["model"], r["partition"]): float(r[metric]) for r in old}
        make_chart(out / file, title, labels, [by[(n, part)] for n in names],
                   percent=percent, lower_is_better=lower)
    active = read_csv("outputs/experiments/teammate_model_retraining/model_comparison.csv")
    for metric, file, title, percent, lower in [
        ("cv_macro_f1_mean", "active_cv_macro_f1.png", "Active suite CV Macro F1 (train partition)", True, False),
        ("test_top1_all", "active_student_top1.png", "Active suite Top-1 (student_test partition)", True, False),
        ("test_macro_f1", "active_student_macro_f1.png", "Active suite Macro F1 (student_test partition)", True, False),
        ("test_log_loss", "active_student_logloss.png", "Active suite log loss (student_test partition)", False, True),
    ]:
        make_chart(out / file, title, [r["display"] for r in active],
                   [float(r[metric]) for r in active], percent=percent, lower_is_better=lower)


def add_section(story, title: str, subtitle: str | None = None):
    story.append(para(title, "H1x"))
    if subtitle:
        story.append(para(subtitle, "Smallx"))


def main() -> None:
    make_metric_charts()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(str(OUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                          topMargin=21 * mm, bottomMargin=17 * mm,
                          title="PathSnap Consolidated Technical Report",
                          author="PathSnap team")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height,
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates(PageTemplate(id="main", frames=[frame], onPage=page))
    s = []

    inventory = read_csv("outputs/eda/dataset_inventory.csv")
    summary = load_json("outputs/final_evaluation/summary.json")
    active_rows = read_csv("outputs/experiments/teammate_model_retraining/model_comparison.csv")
    hmm_rows = read_csv("outputs/experiments/teammate_model_retraining/hmm_student_test_summary.csv")
    old_rows = read_csv("outputs/experiments/model_comparison.csv")
    tune = load_json("outputs/experiments/tuning_result.json")
    tune_compare = read_csv("outputs/experiments/tuned_vs_untuned_summary.csv")
    eda = load_json("outputs/eda/run_summary.json")

    # Cover
    s += [Spacer(1, 34 * mm), para("PathSnap", "CoverTitle"),
          para("Consolidated Technical Report", "CoverTitle"),
          para("GPS-to-road route reconstruction with candidate scoring and sequence decoding",
               "CoverSub"), Spacer(1, 10 * mm),
          para("One report for the full team | November 2026", "CoverSub"),
          Spacer(1, 8 * mm),
          para("Team members", "H2x"),
          table([["Name", "SRN"],
                 ["Satvik L", "PES1UG24CS423"],
                 ["Satyam Goyal", "PES1UG24CS424"],
                 ["Riya Saju", "PES1UG24CS378"],
                 ["Rohan Jadar", "PES1UG24CS379"]], [58 * mm, 58 * mm]),
          Spacer(1, 22 * mm),
          para("This report records the experiments, data partitions, implementation history, limitations, final reserved-evaluation results, and repository paths needed to reproduce each claim. It separates the original Level 1 experiment from the later API model so the two result sets are not mistaken for one experiment.", "Bodyx"),
          PageBreak()]

    add_section(s, "1. Project summary and submission status")
    s.append(para("PathSnap takes a noisy, time-ordered GPS trace and its trip-specific directed road network, then predicts an ordered sequence of road arcs. Candidate generation narrows the network around each fix; a classifier scores observation-arc pairs; HMM/Viterbi decoding uses route connectivity and distance consistency to produce a route."))
    s.append(para("The reported best final evaluation is the active 28-feature Random Forest plus HMM/Viterbi: 95.74% observation-weighted Top-1 on the reserved evaluator_test partition (10 trips, 19,674 GPS observations). This was a final evaluation only: no fitting or tuning occurred on that partition. It is distinct from student_test, which was used for model comparison and selection." , "Calloutx"))
    s.append(para("The project includes actual scripts, configurations, metrics, plots and three interactive Folium maps. The report also documents two material limitations: Bayesian tuning belongs to the earlier HistGradientBoosting pipeline, not the active Random Forest; and the active 28-feature candidate/HMM implementation was adapted from the public RealAdhi/ML2026 repository. The assessment policy's independent-work rule therefore needs an explicit team/instructor compliance check before submission."))
    s.append(para("Submission package checklist", "H2x"))
    checklist = [
        ["Required item", "Current evidence / repository location", "Status"],
        ["One consolidated technical report PDF", "reports/PathSnap_Technical_Report.pdf", "Prepared locally"],
        ["Reproducible code, configs and actual results", "scripts/, src/pathsnap/, configs/, outputs/experiments/, outputs/eda/, outputs/final_evaluation/", "Present locally"],
        ["EDA and geographic visuals", "outputs/eda/*.png; outputs/eda/maps/*.html", "Present"],
        ["Three interactive Folium maps", "normal train, high-miss train 54, student_test 56", "Present locally"],
        ["API/UI code and deployment instructions", "src/pathsnap/api/main.py, ui/app.py, docs/two_laptop_deployment.md", "Present locally"],
        ["Genuine two-system screenshots / receiving-laptop output", "Must be captured during the actual two-laptop run", "Not available in this local report"],
        ["Raw dataset and secrets excluded", "Data path is external; .env is ignored", "Keep excluded"],
    ]
    s.append(table(checklist, [43 * mm, 104 * mm, 27 * mm]))
    s.append(para("The submitted GitHub repository should include the PDF and the source/results that substantiate it. Do not add the separate viva guides or contributor-upload ZIPs. Do not push until the team approves the exact staged changes.", "Smallx"))
    s.append(PageBreak())

    # Dataset understanding
    add_section(s, "2. Dataset understanding and exploratory analysis",
                "EDA is computed only from train (80 trips) and student_test (10 trips). The EDA run summary records evaluator_partition_read=false.")
    s.append(para("Each trip supplies a GPS track, nodes, directed arcs and an ordered reference-route arc list. EDA parsed the actual files and wrote per-trip CSV tables plus titled figures. The map-matching reference route is not a row-aligned label: a monotonic geometric alignment is needed to associate route arcs to GPS observations."))
    rows = [["Partition", "Trips", "GPS rows", "Nodes", "Directed arcs", "Reference route entries" ]]
    for p in ["train", "student_test"]:
        subset = [r for r in inventory if r["partition"] == p]
        rows.append([p, str(len(subset)), f"{sum(int(r['gps_rows']) for r in subset):,}",
                     f"{sum(int(r['node_count']) for r in subset):,}",
                     f"{sum(int(r['arc_count']) for r in subset):,}",
                     f"{sum(int(r['route_arc_entries']) for r in subset):,}"])
    s.append(table(rows, [30 * mm, 19 * mm, 28 * mm, 30 * mm, 31 * mm, 36 * mm]))
    s.append(Spacer(1, 3 * mm))
    s.append(img("outputs/eda/01_dataset_inventory.png", 174 * mm, 95 * mm))
    s.append(para("Figure 1. Dataset inventory from train and student_test partitions; exact plotted values are in outputs/eda/dataset_inventory.csv.", "Captionx"))
    s.append(para("The dataset has 12,889,782 numeric cells across those partitions. EDA found zero missing, malformed, non-finite or out-of-range input values in the checked fields. It retained 578 duplicate GPS coordinate rows (572 train, 6 student_test) as observations rather than dropping them. Timestamps were non-decreasing; there were no non-positive time steps. A speed above 55 m/s was flagged as implausible for review, not removed: 989 training steps and 4 student-test steps exceeded this threshold."))
    s.append(img("outputs/eda/02_missing_invalid_values.png", 170 * mm, 78 * mm))
    s.append(para("Figure 2. Missing/invalid-value checks; partition-specific values are in outputs/eda/missing_invalid_by_trip.csv.", "Captionx"))
    s.append(PageBreak())

    add_section(s, "2.1 Sampling, trajectories and geographic extent")
    s.append(para("Sampling is mostly one second: 96.50% of train gaps and 98.62% of student_test gaps were exactly 1 s. The longest training gap was 153,064 s, and the longest student_test gap was 210 s. Gaps were retained and treated as temporal breaks where the feature/decoder code specifies; they were not interpolated."))
    s.append(img("outputs/eda/03_sampling_intervals.png", 172 * mm, 90 * mm))
    s.append(para("Figure 3. Inter-observation time-gap distributions. The plot titles identify the train and student_test partitions; underlying samples are in outputs/eda/sampling_intervals.csv.", "Captionx"))
    s.append(img("outputs/eda/04_representative_trajectories.png", 174 * mm, 103 * mm))
    s.append(para("Figure 4. Representative projected-metre GPS trajectories from train and student_test. The six selected panels include small, medium, large, high-gap, trip 54 and trip 33 cases; plotting data are in outputs/eda/trajectory_grid.csv.", "Captionx"))
    s.append(img("outputs/eda/05_geographic_extent_and_speed.png", 174 * mm, 94 * mm))
    s.append(para("Figure 5. Per-trip geographic extent and step distance / implied-speed distributions. Distances use local projected metric coordinates. Full per-trip bounding boxes and summaries are in outputs/eda/geographic_extents_speed.csv and outputs/eda/step_speed_samples.csv.", "Captionx"))
    s.append(PageBreak())

    add_section(s, "2.2 Road network and reference route")
    s.append(para("The network is directed. Train contains 1,933,485 nodes and 3,876,904 arcs; student_test contains 90,689 nodes and 185,041 arcs. No isolated nodes were found. Weakly connected component counts vary (train median 20, maximum 421; student_test median 11, maximum 100), so connectivity must be handled per trip rather than assumed globally."))
    s.append(img("outputs/eda/06_road_network_and_overlays.png", 174 * mm, 68 * mm))
    s.append(para("Figure 6. Actual GPS tracks, road network and reference-route overlays for two examples. Distances and extents are projected in metres; source plotting rows are in outputs/eda/network_overlay_plot_data.csv. Interactive Folium maps with toggleable GPS, network and reference layers are outputs/eda/maps/train-00000034-normal.html, train-00000054-high_miss.html and student_test-00000056-student_test.html.", "Captionx"))
    s.append(img("outputs/eda/13_network_connectivity.png", 174 * mm, 50 * mm))
    s.append(para("Figure 7. Per-trip network connectivity and isolated-node summaries, computed on train and student_test; source table: outputs/eda/road_network_summary.csv.", "Captionx"))
    s.append(para("Reference-route lengths do not equal the number of GPS observations. Eleven trips contain at least one disconnected adjacent route-arc pair: 10 train trips with 13 steps and one student_test trip with one step. Route alignment assigns an ordered route position to each GPS point using projected point-to-arc distance and a monotonic dynamic program; it does not assert that each GPS fix is exactly on the reference geometry. Route-length and disconnected-step detail is in outputs/eda/reference_route_summary.csv.", "Bodyx"))
    s.append(img("outputs/eda/07_reference_route_diagnostics.png", 174 * mm, 58 * mm))
    s.append(para("Figure 8. Route length versus GPS rows and disconnected route steps from the two development partitions.", "Captionx"))
    s.append(PageBreak())

    add_section(s, "2.3 Candidate generation, alignment and high-miss trips")
    s.append(para("The original candidate generator searches nearby road segments and retains up to top-k candidates (k=8 in the original Level 1 experiment). The top-k sweep shows that increasing k reduces misses on ordinary trips, but barely changes trips dominated by reference-to-GPS alignment offset. The later active pipeline instead uses a 60 m search radius and at most 12 candidates.", "Bodyx"))
    s.append(img("outputs/eda/08_candidate_alignment_diagnostics.png", 174 * mm, 92 * mm))
    s.append(para("Figure 9. Candidate count, candidate-miss and aligned-distance diagnostics; tables: candidate_topk_sweep.csv, candidate_counts_per_observation.csv, aligned_distance_samples.csv and candidate_class_balance.csv.", "Captionx"))
    s.append(para("Training trips 54 and 33 are the major outliers in the original seven-feature EDA. Trip 54 has a 90.61% top-8 miss rate; 83.66% of its GPS points are more than 100 m from their aligned reference arc (median 138.59 m, p95 210.05 m). Increasing k from 8 to 16 only lowers the miss rate to 89.78%. Trip 33 has a 67.06% top-8 miss rate; 50.96% of points are more than 100 m from the aligned reference (median 122.41 m, p95 142.06 m), and one route adjacency is disconnected. Its top-16 miss rate remains 66.67%. This is chiefly a data/reference alignment or route quality issue, not a scorer failure: most missed labels are already over 100 m from GPS, beyond candidate ranking's reach."))
    s.append(img("outputs/eda/09_high_miss_trips.png", 174 * mm, 96 * mm))
    s.append(para("Figure 10. Training-only evidence for trips 54 and 33: trajectory, network/reference overlay, candidate misses and alignment distance. The detailed numbers are in outputs/eda/high_miss_investigation.csv and high_miss_plot_data.csv.", "Captionx"))
    s.append(PageBreak())

    add_section(s, "2.4 Training-only features")
    s.append(para("The original seven-feature experiment uses cross-track distance, heading cosine, along-arc fraction, arc length, GPS speed, GPS gap and GPS step distance. Its plots use training candidate rows only, so the student_test partition does not contribute to feature-distribution or correlation summaries. Positive means the candidate arc equals the aligned reference arc."))
    s.append(img("outputs/eda/10_training_feature_distributions.png", 174 * mm, 99 * mm))
    s.append(para("Figure 11. Training-only feature distributions for positive and negative candidate rows. Binned source values are in outputs/eda/feature_distribution_bins.csv.", "Captionx"))
    s.append(img("outputs/eda/11_training_feature_correlation.png", 154 * mm, 103 * mm))
    s.append(para("Figure 12. Training-only feature correlation heatmap; source: outputs/eda/feature_correlation.csv. The along-arc fraction is independent of the other scalar features in this simple linear correlation view; nonlinear and sequence context motivated later features.", "Captionx"))
    s.append(para("The EDA also includes projected arc-length distributions in outputs/eda/12_arc_length_distribution.png. EDA operations, assumptions and choices are written in docs/data_quality.md and outputs/eda_operations.md.", "Smallx"))
    s.append(PageBreak())

    # Modeling
    add_section(s, "3. Model comparison: original Level 1 experiment")
    s.append(para("The first five-model suite used the original seven-feature pipeline and top-k=8 candidate rows. GroupKFold held whole trips together across five folds. Student_test contains 10 trips and is a development comparison set. CV Macro F1 and route continuity are trip means; Top-1 and log loss are observation-weighted. Candidate misses remain in the all-observation denominator."))
    old_by = {(r["model"], r["partition"]): r for r in old_rows}
    pretty = {"hist_gradient_boosting":"HistGradientBoosting", "small_mlp":"Small MLP",
              "random_forest":"Random Forest", "logistic_regression":"Logistic Regression",
              "calibrated_linear_svm":"Calibrated Linear SVM"}
    rows = [["Model", "CV Macro F1", "CV Top-1", "Student Macro F1", "Student Top-1", "Student log loss"]]
    for name in ["hist_gradient_boosting", "small_mlp", "random_forest", "logistic_regression", "calibrated_linear_svm"]:
        cv, st = old_by[(name, "cross_validation")], old_by[(name, "student_test")]
        rows.append([pretty[name], num(cv["macro_f1"]), pct(cv["top1_accuracy"]), num(st["macro_f1"]),
                     pct(st["top1_accuracy"]), num(st["log_loss"])])
    s.append(table(rows, [39 * mm, 26 * mm, 24 * mm, 29 * mm, 25 * mm, 24 * mm]))
    s.append(Spacer(1, 4 * mm))
    s.append(img("reports/figures/cv_macro_f1.png", 165 * mm, 65 * mm))
    s.append(para("Figure 13. Original suite five-fold trip-group CV Macro F1. Raw values: outputs/experiments/model_comparison.csv.", "Captionx"))
    s.append(PageBreak())
    add_section(s, "3.1 Original Level 1 student-test plots")
    s.append(img("reports/figures/student_test_top1.png", 165 * mm, 57 * mm))
    s.append(para("Figure 14. Original suite student_test all-observation Top-1.", "Captionx"))
    s.append(img("reports/figures/student_test_macro_f1.png", 165 * mm, 57 * mm))
    s.append(para("Figure 15. Original suite student_test trip-mean Macro F1.", "Captionx"))
    s.append(PageBreak())

    add_section(s, "3.1 Why these five models")
    suitability = [
        ["Model family", "Why suitable", "Observed result / limitation"],
        ["Logistic Regression", "Fast probabilistic linear baseline; shows whether engineered geometry features separate candidates linearly.", "Fastest simple baseline; lower original student Top-1 (69.01%) and CV Macro F1 (0.4621)."],
        ["Random Forest", "Captures nonlinear feature interactions with limited scaling assumptions; robust tabular baseline.", "Original seven-feature suite lagged HGB; retrained active suite later had strongest HMM-decoded student Top-1."],
        ["HistGradientBoosting", "Efficient nonlinear tabular boosting; captures thresholds/interactions in geometry and motion.", "Won original Level 1 point estimates (CV Macro F1 0.4826; student Top-1 70.28%), but was not the later active API model."],
        ["Calibrated Linear SVM", "Margin-based alternative for candidate ranking; calibration supplies scores for sequence decoding.", "Competitive linear margin baseline but slightly below Logistic Regression in original CV Macro F1."],
        ["Small MLP", "Tests whether a compact neural model learns nonlinear combinations beyond tree/linear baselines.", "Student log loss was best (1.1925) and continuity highest in original suite; only 10 student trips, so ranking uncertainty is material."],
    ]
    s.append(table(suitability, [31 * mm, 76 * mm, 60 * mm]))
    s.append(para("Exact original settings from src/pathsnap/models/candidates.py: Logistic Regression used StandardScaler and max_iter=600, class_weight=balanced, seed=99 (C=1, L2, lbfgs defaults); Random Forest used 120 trees, min_samples_leaf=2, class_weight=balanced_subsample, n_jobs=-1, seed=99; HistGradientBoosting used max_iter=100, learning_rate=0.08, L2=1.0, class_weight=balanced, seed=99; Calibrated Linear SVM used StandardScaler plus LinearSVC(C=1, class_weight=balanced) with 3-fold sigmoid probability calibration; Small MLP used StandardScaler, hidden layers (32,16), alpha=0.001, max_iter=100 and early stopping, seed=99. Logistic Regression and MLP optimize binary cross-entropy/log loss; Random Forest uses bootstrap trees with Gini impurity; HGB minimizes logistic deviance; the SVM uses hinge loss with sigmoid calibration. The original MLP training history is available in outputs/experiments/mlp_training_history.csv; it records training log loss and validation accuracy (not validation loss).", "Smallx"))
    s.append(para("The code and CSV are the authority for exact configuration and recorded metrics; report comparisons do not claim statistical dominance where bootstrap intervals included zero. Hyperparameters for the later five-model active suite are given in section 4.", "Calloutx"))
    s.append(img("reports/figures/student_test_logloss.png", 165 * mm, 57 * mm))
    s.append(para("Figure 16. Original suite student_test log loss. Lower is better; probability scores were generated over the original candidate set.", "Captionx"))
    s.append(PageBreak())

    # active model suite
    add_section(s, "4. Later active model comparison")
    s.append(para("After Level 1, the API pipeline moved to a 28-feature candidate generator and a retrained model zoo. This is a separate experiment from the seven-feature Level 1 table. All five candidate scorers were fit on training trips with five-fold GroupKFold, then compared on the same 10 student_test trips. The active model uses a 60 m candidate radius and max 12 candidates."))
    rows = [["Model", "CV Macro F1 +/- SD", "CV reachable Top-1", "Student all-row Top-1", "Student Macro F1", "Log loss", "Inference ms / 1k obs"]]
    by_name = {r["model"]: r for r in active_rows}
    for name in ["logreg", "random_forest", "lightgbm", "xgboost", "mlp"]:
        r = by_name[name]
        rows.append([r["display"], f"{num(r['cv_macro_f1_mean'])} +/- {num(r['cv_macro_f1_std'])}",
                     pct(r["cv_top1_reach"]), pct(r["test_top1_all"]), num(r["test_macro_f1"]),
                     num(r["test_log_loss"]), num(r["infer_ms_per_1k_obs"], 2)])
    s.append(table(rows, [27 * mm, 30 * mm, 26 * mm, 29 * mm, 25 * mm, 20 * mm, 22 * mm]))
    s.append(Spacer(1, 4 * mm))
    s.append(para("Selection: XGBoost slightly led classifier-only CV reachable Top-1 and student classifier Top-1. Random Forest was selected for the active route pipeline because RF + the fixed HMM had the highest student_test all-observation Top-1 among these five (94.16%; see section 5). This is a 10-trip development-set point comparison, not a proven statistical win. The active RF parameters were 150 trees, max depth 14, minimum leaf size 10, 40% bootstrap sample, four worker threads and seed 42. LightGBM used 300 trees, learning rate 0.05, 31 leaves, minimum child samples 20, subsample/column sample 0.8. XGBoost used 300 trees, learning rate 0.1, depth 6, subsample/column sample 0.8, histogram method and log-loss objective. Logistic Regression used C=1/L2/lbfgs. MLP used 64/32 ReLU layers, Adam (0.001), alpha 0.0001, batch 2048, early stopping. The probabilistic classifiers use binary log loss/cross-entropy; RF uses bootstrap trees with Gini split impurity; LightGBM and XGBoost optimize binary log loss. The saved comparison CSV is the source for measured settings, fit time, inference time and metrics.", "Bodyx"))
    s.append(para("The model comparison file contains fit times and metrics for each model; local inference timings are machine-specific. All candidate misses are counted in all-observation student Top-1, so a candidate generator miss cannot be hidden by conditioning on covered points.", "Bodyx"))
    s.append(para("Overfit/underfit evidence is limited to generalization comparisons: in the original suite, each model's trip-group CV Macro F1 and student_test Macro F1 are close in scale, with the student-test point estimates slightly higher; in the active suite, CV and student-test Macro F1 are also similar. This does not prove absence of overfitting because the student set has only 10 trips and the saved comparisons do not include a separate in-sample training score for every model. The MLP history is the only epoch-level history preserved.", "Smallx"))
    s.append(PageBreak())
    add_section(s, "4.1 Active model-suite plots")
    s.append(img("reports/figures/active_cv_macro_f1.png", 169 * mm, 68 * mm))
    s.append(para("Figure 17. Active five-model CV Macro F1 from the train partition; source values are in outputs/experiments/teammate_model_retraining/model_comparison.csv.", "Captionx"))
    s.append(img("reports/figures/active_student_top1.png", 169 * mm, 68 * mm))
    s.append(para("Figure 18. Active five-model all-observation Top-1 from student_test; candidate misses remain in the denominator.", "Captionx"))
    s.append(PageBreak())
    add_section(s, "4.2 Active model-suite plots, continued")
    s.append(img("reports/figures/active_student_macro_f1.png", 169 * mm, 68 * mm))
    s.append(para("Figure 19. Active five-model student_test Macro F1, averaged by trip.", "Captionx"))
    s.append(img("reports/figures/active_student_logloss.png", 169 * mm, 68 * mm))
    s.append(para("Figure 20. Active five-model student_test log loss. Lower is better; values are candidate-normalized classifier scores.", "Captionx"))
    s.append(PageBreak())

    # HPO
    add_section(s, "5. Bayesian hyperparameter optimization")
    bp = tune["best_params"]
    s.append(para("Optuna TPE ran 15 trials on the original HistGradientBoosting scorer, using three-fold trip-group CV. The objective was mean trip Macro F1. Search ranges recorded in scripts/tune_selected_model.py were: max_iter 60-220 by 20; learning rate 0.02-0.2 (log); max_leaf_nodes 7, 15, 31 or 63; min_samples_leaf 15-80 by 5; and L2 regularization 1e-4 to 10 (log). The best trial used max_iter={max_iter}, learning_rate={learning_rate:.5f}, max_leaf_nodes={max_leaf_nodes}, min_samples_leaf={min_samples_leaf}, L2={l2_regularization:.6f}; grouped-CV objective was {objective:.4f}.".format(**bp, objective=tune["best_value_cv_macro_f1"])))
    compare_rows = [["Version", "Student Macro F1", "Top-1", "Log loss", "Continuity", "Candidate miss", "Optimization time"]]
    for r in tune_compare:
        compare_rows.append([r["version"], num(r["student_test_macro_f1_trip_mean"]), pct(r["student_test_top1_accuracy"]),
                             num(r["student_test_log_loss"]), pct(r["student_test_route_continuity_trip_mean"]),
                             pct(r["student_test_candidate_miss_rate"]),
                             f"{float(r['optimization_seconds']):.1f} s" if r["optimization_seconds"] != "0.0" else "not tuned"])
    s.append(table(compare_rows, [26 * mm, 26 * mm, 20 * mm, 22 * mm, 23 * mm, 23 * mm, 29 * mm]))
    s.append(para("The tuned HGB moved student Macro F1 from 0.5137 to 0.5154 and Top-1 from 70.28% to 70.44%; log loss moved from 1.2960 to 1.2915. This small gain does not establish meaningful superiority across only 10 student trips. Crucially, this Bayesian search predates the 28-feature Random Forest API model. It is documented as HPO work completed, but it must not be described as tuning the active RF. Active RF hyperparameters were fixed from the later model implementation; no HPO was run for that selected active model.", "Calloutx"))
    s.append(para("Evidence: outputs/experiments/bayesian_trials.csv, tuning_result.json, tuned_vs_untuned_summary.csv and per-trip detail. The full trial table is included in the repository, so the reported best values can be checked.", "Bodyx"))
    s.append(PageBreak())

    # HMM
    add_section(s, "6. HMM/Viterbi route sequence decoding")
    s.append(para("For each observation, candidate scorer outputs are treated as emission scores. The decoder combines clipped/normalized log emission probability, an optional GPS-to-arc distance penalty, directed shortest-path distance between consecutive candidate arcs, a penalty for unreachable transitions, and an immediate reverse-arc penalty. Viterbi dynamic programming chooses a globally consistent arc sequence. Time gaps over the configured gap threshold restart the transition chain; very large gaps skip network-distance precomputation. If route labels are absent from a candidate set, decoding cannot recover them."))
    d = load_json("outputs/experiments/selected_decoder.json")["parameters"]
    rows = [["Parameter", "Value", "Meaning"],
            ["w_emit", str(d["w_emit"]), "Weight for classifier log-probability"],
            ["w_dist", num(d["w_dist"], 4), f"GPS distance emission weight; sigma={num(d['sigma'],2)} m"],
            ["beta", f"{num(d['beta'],2)} m", "Scale for route-distance versus GPS-step mismatch"],
            ["w_trans", num(d["w_trans"], 4), "Transition score weight"],
            ["break_pen", num(d["break_pen"], 3), "Penalty for disconnected/unreachable transition"],
            ["uturn_pen", num(d["uturn_pen"], 3), "Extra reverse-arc penalty"],
            ["gap_s", f"{num(d['gap_s'],2)} s", "Above this gap, restart transition sequence"],
            ["eps", str(d["eps"]), "Probability floor before taking log"]]
    s.append(table(rows, [30 * mm, 29 * mm, 110 * mm]))
    s.append(para("These active decoder settings came from the teammate implementation's tuned configuration, originally selected for its XGBoost pipeline, and were applied unchanged across the five retrained classifiers. They were not tuned on student_test for the active RF comparison. This is important provenance and should be explained in the oral defense."))
    hmm_by = {(r["model"], r["method"]): r for r in hmm_rows}
    rows = [["Candidate scorer", "Classifier Top-1", "+ default HMM", "+ fixed tuned HMM", "Mean trip continuity with fixed HMM"]]
    for name in ["logreg", "random_forest", "lightgbm", "xgboost", "mlp"]:
        cr = hmm_by[(name, "classifier_only")]
        dr = hmm_by[(name, "default_hmm")]
        hr = hmm_by[(name, "teammate_tuned_hmm")]
        rows.append([by_name[name]["display"], pct(cr["point_top1"]), pct(dr["point_top1"]),
                     pct(hr["point_top1"]), pct(hr["mean_trip_continuity"])])
    s.append(Spacer(1, 3 * mm))
    s.append(table(rows, [39 * mm, 30 * mm, 30 * mm, 30 * mm, 39 * mm]))
    s.append(para("The RF row improves from 85.82% classifier-only to 94.16% with the fixed HMM on student_test (+8.35 percentage points). The HMM improves sequence coherence; its benefit depends on noisy GPS, candidate coverage, graph correctness and gap behavior. A coherent route can still be the wrong route when aligned reference data is noisy or candidates omit the correct arc.", "Calloutx"))
    s.append(PageBreak())

    # final reserved eval
    add_section(s, "7. Final reserved evaluator_test evaluation")
    s.append(para("The frozen active model was run on 10 evaluator_test trips (19,674 GPS observations). Predictions were generated without loading reference routes; labels were read only after prediction was fixed, for server-side metrics. No refit or tuning occurred on evaluator_test. This partition has now been evaluated and must not be described as untouched for future model selection."))
    rows = [["Metric", "RF scorer only", "RF + HMM/Viterbi"],
            ["Observation-weighted Top-1", pct(summary["classifier_micro_top1"]), pct(summary["hmm_micro_top1"])],
            ["Mean trip Top-1", pct(summary["classifier_mean_trip_top1"]), pct(summary["hmm_mean_trip_top1"])],
            ["Mean trip Macro F1", pct(summary["classifier_mean_trip_macro_f1"]), pct(summary["hmm_mean_trip_macro_f1"])],
            ["Pooled route continuity", pct(summary["classifier_pooled_route_continuity"]), pct(summary["hmm_pooled_route_continuity"])],
            ["Mean trip route-set F1", pct(summary["classifier_mean_trip_route_f1"]), pct(summary["hmm_mean_trip_route_f1"])],
            ["Classifier observation-weighted log loss", num(summary["classifier_observation_weighted_log_loss"]), "not reported for Viterbi path"],
            ["Candidate misses", f"{summary['candidate_misses']} / {summary['observation_count']} ({pct(summary['candidate_miss_rate'])})", "same candidate support"],
            ["Mean inference time / trip", f"{summary['mean_prediction_seconds_per_trip']:.3f} s", "included in full pipeline"]]
    s.append(table(rows, [64 * mm, 52 * mm, 52 * mm]))
    s.append(Spacer(1, 4 * mm))
    s.append(para("The 95.74% headline is observation-weighted Top-1 for this one reserved partition run. Mean-trip Top-1 is 95.80%. It is not a guarantee on future trips or a claim of universal accuracy. Classifier log loss is 0.4780; Viterbi returns a best sequence rather than calibrated per-point probabilities, so no HMM log loss is claimed.", "Calloutx"))
    s.append(img("outputs/final_evaluation/plots/01_overall_model_comparison.png", 164 * mm, 83 * mm))
    s.append(para("Figure 21. Evaluator_test end-to-end comparison. The graph is generated from the final-evaluation CSV values and has a corresponding machine-readable plot-data CSV.", "Captionx"))
    s.append(img("outputs/final_evaluation/plots/02_top1_by_trip.png", 164 * mm, 78 * mm))
    s.append(para("Figure 22. Per-trip evaluator_test Top-1; all 10 trips and exact scores are in outputs/final_evaluation/per_trip_metrics.csv.", "Captionx"))
    s.append(PageBreak())

    add_section(s, "7.1 Per-trip diagnostics and error analysis")
    s.append(img("outputs/final_evaluation/plots/03_macro_f1_by_trip.png", 164 * mm, 62 * mm))
    s.append(para("Figure 23. Per-trip Macro F1 on evaluator_test, before and after HMM/Viterbi.", "Captionx"))
    s.append(img("outputs/final_evaluation/plots/04_candidate_misses_by_trip.png", 164 * mm, 62 * mm))
    s.append(para("Figure 24. Candidate-generator misses by evaluator trip; the route decoder cannot restore a reference arc absent from the candidate list.", "Captionx"))
    s.append(img("outputs/final_evaluation/plots/05_route_continuity_by_trip.png", 164 * mm, 62 * mm))
    s.append(para("Figure 25. Directed-network route continuity by evaluator trip. Underlying exact values are in outputs/final_evaluation/per_trip_metrics.csv.", "Captionx"))
    s.append(para("There were 18 missing-reference-candidate observations (0.0915%). Pooled route continuity rose from 93.69% to 97.68%. Mean trip Macro F1 increased from 74.13% to 90.46%. The largest evaluator trip, 00000015 (6,854 GPS observations), rose from 59.70% RF-only Top-1 to 94.25% after sequence decoding. Per-trip results and timing are in the CSV/JSON files under outputs/final_evaluation/.", "Bodyx"))
    s.append(PageBreak())

    # maps + api
    add_section(s, "8. Geographical visualisation and application")
    s.append(para("The report includes projected-metre trajectory and network visuals. The repository also contains three interactive Folium maps with LayerControl and map legend, each using toggleable GPS, road network and reference route layers: a normal training trip (00000034), high-miss training trip 00000054, and student_test trip 00000056. Their actual HTML and generation code are versionable and can be opened in a browser. No predicted-route layer is included in these three EDA maps because the map files were generated by the EDA script before the active prediction overlay was produced."))
    s.append(img("outputs/eda/06_road_network_and_overlays.png", 170 * mm, 97 * mm))
    s.append(para("Figure 26. Static geographic view from actual train partition GPS/network/reference data. Interactive Folium HTML maps are in outputs/eda/maps/. This figure is not a browser screenshot of the interactive Folium layers; capture and add a genuine Folium view before final submission.", "Captionx"))
    s.append(para("FastAPI exposes /health, /trips and POST /predict/{trip_id}. The active service creates candidates without labels, scores them, decodes the path, and only then loads the route to compute metrics on the server. Streamlit sends the selected trip ID and displays returned GPS points, predicted route and metrics. The code is in src/pathsnap/api/main.py and ui/app.py; the runbook is docs/two_laptop_deployment.md."))
    s.append(para("Submission guideline asks for genuine screenshots of the receiving laptop UI and successful API/terminal response from the two-system run. The supplied local artifacts do not contain a verified two-laptop screenshot, so this report does not invent or imply one. Capture the real Laptop 2 UI and Laptop 1 server terminal during the demonstration and add them before final submission.", "Calloutx"))
    s.append(PageBreak())

    add_section(s, "9. Reproducibility and evidence index")
    evidence = [
        ["Claim / artifact", "Reproduction or verification path"],
        ["Dataset file inventory", "outputs/data_inventory.csv; outputs/eda/dataset_inventory.csv; scripts/run_eda.py"],
        ["EDA findings and plots", "outputs/eda/*.csv; outputs/eda/*.png; docs/data_quality.md; make eda"],
        ["Original Level 1 models", "scripts/run_model_suite.py; outputs/experiments/model_comparison.csv; outputs/experiments/per_trip_metrics.csv"],
        ["Bayesian HPO", "scripts/tune_selected_model.py; outputs/experiments/bayesian_trials.csv; tuning_result.json; tuned_vs_untuned_summary.csv"],
        ["Later 28-feature five-model suite", "outputs/experiments/teammate_model_retraining/model_comparison.csv; scripts/run_decoder_comparison.py"],
        ["HMM parameters and per-model student comparison", "outputs/experiments/selected_decoder.json; outputs/experiments/teammate_model_retraining/hmm_student_test_summary.csv"],
        ["Final evaluator run", "scripts/evaluate_reserved.py; outputs/final_evaluation/summary.json; per_trip_metrics.csv; plots/*.png and corresponding CSVs"],
        ["Three interactive maps", "outputs/eda/maps/*.html; generated in scripts/run_eda.py"],
        ["API and UI", "src/pathsnap/api/main.py; ui/app.py; docs/two_laptop_deployment.md"],
    ]
    s.append(table(evidence, [54 * mm, 113 * mm]))
    s.append(para("Re-run commands from the repository root (after installing requirements):", "H2x"))
    s.append(para("<font name='Courier'>make eda<br/>python scripts/prepare_data.py<br/>python scripts/run_model_suite.py<br/>python scripts/tune_selected_model.py --trials 15 --folds 3<br/>python scripts/run_decoder_comparison.py<br/>python scripts/evaluate_reserved.py --data-root /path/to/evaluator_test --output outputs/final_evaluation</font>", "Bodyx"))
    s.append(para("Raw data remains outside the repository. The evaluator run can only be reproduced on the same provided evaluator files; the script records aggregate and per-trip metrics and plots, not raw evaluator tracks or observation-level labels/predictions."))
    s.append(PageBreak())

    add_section(s, "10. Limitations, risks and next actions")
    limitations = [
        "The student_test set contains only 10 trips. Small differences between models are noisy, and the selected model's ranking is a point estimate rather than a demonstrated significant win.",
        "The HPO exercise tuned original HistGradientBoosting; the current API Random Forest was not Bayesian-tuned. The report keeps these separate.",
        "The active 28-feature candidate and HMM code was adapted from RealAdhi/ML2026, commit 97a78406269b33eda31fd17a0b00125c87246946. Assessment guidance stresses independent implementation. The team should verify course authorization, cite the source and disclose the adaptation. Do not describe the adapted implementation as independently authored.",
        "AI assistance was used for project documentation, result synthesis and report preparation in this collaboration. The team should disclose AI assistance as required by the assessment rules and confirm that every member understands and can verify submitted work.",
        "The fixed HMM settings were originally chosen for the external XGBoost pipeline and then applied to the active model family. The student_test comparison was not used to retune these parameters; the final evaluator partition was not used for tuning either.",
        "The report's mapped overlays are real, but no actual two-laptop UI/API screenshot is in this local evidence set. Add genuine screenshots from the receiving-laptop presentation run.",
        "The original seven-feature and later 28-feature scores use different feature sets/candidate generation. They are separate baselines and should not be compared as if only the classifier changed.",
        "EDA flagged extreme GPS speeds and very long time gaps but retained observations. Trip 54/33 reference alignment is likely unreliable; candidate misses there mostly reflect GPS-to-reference offsets and cannot be solved by a stronger classifier alone.",
    ]
    for item in limitations:
        s.append(para("- " + item))
    s.append(para("Before final submission", "H2x"))
    for item in [
        "Capture actual screenshots from Laptop 2 (working UI with prediction/metrics) and Laptop 1 (matching successful API request log); add them to the repository and report.",
        "Have all four team members review and explain every claimed result, partition, metric definition, model setting and external-code attribution.",
        "Confirm the assessment policy permits the documented external adaptation; if not, replace it with an independently implemented pipeline and rerun experiments before claiming compliance.",
        "Review exactly what will be committed. Keep raw datasets, environment files, viva PDFs and teammate ZIP folders out of the GitHub submission.",
        "Only after review, push the final report and source/results to the team repository.",
    ]:
        s.append(para("- " + item))
    s.append(Spacer(1, 5 * mm))
    s.append(para("End of consolidated report", "Captionx"))

    doc.build(s)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
