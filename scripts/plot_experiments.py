"""Plot actual model comparison metrics from the generated experiment CSV."""

import argparse
import csv
from pathlib import Path

from xml.sax.saxutils import escape


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="outputs/experiments/model_comparison.csv")
    parser.add_argument("--output-dir", default="outputs/experiments/plots")
    args = parser.parse_args()
    rows = read_rows(Path(args.input))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cv = sorted([r for r in rows if r["partition"] == "cross_validation"],
                key=lambda r: float(r["macro_f1"]), reverse=True)
    student = sorted([r for r in rows if r["partition"] == "student_test"],
                     key=lambda r: float(r["macro_f1"]), reverse=True)
    names = [r["model"] for r in cv]

    def svg_chart(title: str, ylabel: str, series: list[tuple[str, str, list[float]]],
                  filename: str, *, max_value: float | None = None,
                  categories: list[str] | None = None) -> None:
        width, height = 1000, 590
        left, right, top, bottom = 220, 40, 74, 90
        plot_width, plot_height = width - left - right, height - top - bottom
        vals = [value for _, _, values in series for value in values]
        ceiling = max_value or max(vals) * 1.15
        cats = categories or names
        row_height = plot_height / max(len(cats), 1)
        svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
               '<rect width="100%" height="100%" fill="white"/>',
               f'<text x="{left}" y="36" font-family="Arial" font-size="22" font-weight="bold">{escape(title)}</text>',
               f'<text x="24" y="{top + plot_height/2}" transform="rotate(-90 24 {top + plot_height/2})" font-family="Arial" font-size="15">{escape(ylabel)}</text>']
        for tick in range(6):
            value = ceiling * tick / 5
            x = left + plot_width * tick / 5
            svg.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_height}" stroke="#d9dee5"/>')
            svg.append(f'<text x="{x:.1f}" y="{top+plot_height+24}" text-anchor="middle" font-family="Arial" font-size="12">{value:.2f}</text>')
        colors = [item[1] for item in series]
        for i, name in enumerate(cats):
            y = top + i * row_height + row_height * 0.16
            svg.append(f'<text x="{left-12}" y="{y+row_height*0.5:.1f}" text-anchor="end" font-family="Arial" font-size="13">{escape(name)}</text>')
            bar_h = min(24, row_height * 0.62 / max(len(series), 1))
            gap = bar_h * 0.22
            for j, (label, color, values) in enumerate(series):
                value = values[i]
                bar_y = y + j * (bar_h + gap)
                bar_w = max(0, plot_width * value / ceiling)
                svg.append(f'<rect x="{left}" y="{bar_y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}"><title>{escape(label)}: {value:.4f}</title></rect>')
                svg.append(f'<text x="{left+bar_w+5:.1f}" y="{bar_y+bar_h*0.78:.1f}" font-family="Arial" font-size="11">{value:.3f}</text>')
        legend_x = left
        for label, color, _ in series:
            svg.append(f'<rect x="{legend_x}" y="{height-28}" width="14" height="14" fill="{color}"/>')
            svg.append(f'<text x="{legend_x+20}" y="{height-16}" font-family="Arial" font-size="12">{escape(label)}</text>')
            legend_x += 185
        svg.append('</svg>')
        (output / filename).write_text("\n".join(svg), encoding="utf-8")

    cv_f1 = [float(r["macro_f1"]) for r in cv]
    svg_chart("Trip-group CV Macro F1", "Models", [("CV Macro F1", "#315b7d", cv_f1)],
              "cv_macro_f1.svg", max_value=1.0)
    student_top1 = [float(next(r["top1_accuracy"] for r in student if r["model"] == n)) for n in names]
    student_f1 = [float(next(r["macro_f1"] for r in student if r["model"] == n)) for n in names]
    svg_chart("Student-test Metrics", "Models", [("Top-1 Accuracy", "#2878b5", student_top1),
              ("Macro F1", "#24a148", student_f1)], "student_test_accuracy_macro_f1.svg", max_value=1.0)
    cv_loss = [float(next(r["log_loss"] for r in rows if r["model"] == n and r["partition"] == "cross_validation")) for n in names]
    st_loss = [float(next(r["log_loss"] for r in rows if r["model"] == n and r["partition"] == "student_test")) for n in names]
    svg_chart("Observation-level Log Loss", "Models", [("CV Log Loss", "#315b7d", cv_loss),
              ("Student-test Log Loss", "#e67e22", st_loss)], "log_loss_comparison.svg")

    decoder_file = Path("outputs/experiments/decoder_comparison.csv")
    if decoder_file.is_file():
        dec = read_rows(decoder_file)
        for partition, filename in (("cross_validation", "decoder_cv.svg"),
                                    ("student_test", "decoder_student_test.svg")):
            selected = [r for r in dec if r["partition"] == partition and not r["trip_id"]]
            if selected:
                raw = next(r for r in selected if r["stage"] == "classifier_only")
                hmm = next(r for r in selected if r["stage"] == "classifier_plus_hmm_viterbi")
                svg_chart(f"{partition.replace('_', ' ').title()}: classifier vs Viterbi", "Method",
                          [("Classifier only Macro F1", "#2878b5", [float(raw["macro_f1"])]),
                           ("Classifier + Viterbi Macro F1", "#24a148", [float(hmm["macro_f1"])]),
                           ("Classifier only continuity", "#e67e22", [float(raw["route_continuity"])]),
                           ("Classifier + Viterbi continuity", "#8e44ad", [float(hmm["route_continuity"])])],
                          filename, max_value=1.0, categories=["Selected HGB"])
    print(f"Wrote plots under {output}")


if __name__ == "__main__":
    main()

