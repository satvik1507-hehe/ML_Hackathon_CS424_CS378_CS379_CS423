"""Fit the registered MLP on all training candidate rows and save real fit history.

This is an evidence-only refit with the same seed and constructor used by the model suite. It does
not change the five-model comparison outputs or any selected-model artifact.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from pathsnap.models.candidates import build_candidates


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "artifacts/intermediate/candidate_cache/train"
OUTPUT = ROOT / "outputs/experiments/mlp_training_history.csv"
SVG = ROOT / "outputs/experiments/plots/mlp_training_history.svg"


def load_rows():
    xs, ys = [], []
    for path in sorted(CACHE.glob("*.npz")):
        with np.load(path, allow_pickle=False) as d:
            observation = d["observation_index"]
            keep = ~d["candidate_miss"][observation]
            xs.append(d["X"][keep])
            ys.append(d["y"][keep])
    if len(xs) != 80:
        raise RuntimeError(f"Expected 80 training candidate caches, found {len(xs)}")
    return np.concatenate(xs), np.concatenate(ys)


def write_svg(loss: list[float], validation_accuracy: list[float]) -> None:
    width, height = 940, 390
    left, right, top, bottom = 80, 35, 48, 55
    panel_gap = 70
    panel_width = (width - left - right - panel_gap) / 2
    panel_height = height - top - bottom

    def panel(x0, title, values, color, ylabel, y_min=None, y_max=None):
        lo = min(values) if y_min is None else y_min
        hi = max(values) if y_max is None else y_max
        if hi <= lo:
            hi = lo + 1.0
        points = []
        for i, val in enumerate(values):
            x = x0 + panel_width * i / max(1, len(values) - 1)
            y = height - top - panel_height * (float(val) - lo) / (hi - lo)
            points.append((x, y))
        path = " ".join(("M" if i == 0 else "L") + f" {x:.1f} {y:.1f}" for i, (x, y) in enumerate(points))
        lines = [
            f'<text x="{x0:.1f}" y="24" class="title">{title}</text>',
            f'<line x1="{x0:.1f}" y1="{height-top}" x2="{x0+panel_width:.1f}" y2="{height-top}" class="axis"/>',
            f'<line x1="{x0:.1f}" y1="{top}" x2="{x0:.1f}" y2="{height-top}" class="axis"/>',
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.5"/>',
            f'<text x="{x0-8:.1f}" y="{top+8}" class="tick" text-anchor="end">{hi:.3f}</text>',
            f'<text x="{x0-8:.1f}" y="{height-top+3}" class="tick" text-anchor="end">{lo:.3f}</text>',
            f'<text x="{x0+panel_width/2:.1f}" y="{height-12}" class="label" text-anchor="middle">Epoch</text>',
            f'<text x="{x0-49:.1f}" y="{height/2:.1f}" class="label" text-anchor="middle" transform="rotate(-90 {x0-49:.1f} {height/2:.1f})">{ylabel}</text>',
        ]
        return lines

    second_x = left + panel_width + panel_gap
    content = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>.title{font: bold 16px sans-serif;fill:#10243A}.axis{stroke:#8798A5;stroke-width:1}.tick{font:12px sans-serif;fill:#5D6B78}.label{font:13px sans-serif;fill:#263746}</style>',
    ]
    content += panel(left, "MLP training loss", loss, "#087E8B", "Log loss")
    content += panel(second_x, "Early-stopping validation score", validation_accuracy, "#D17A22", "Validation accuracy", 0, 1)
    content.append('</svg>')
    SVG.parent.mkdir(parents=True, exist_ok=True)
    SVG.write_text("\n".join(content), encoding="utf-8")


def main():
    X, y = load_rows()
    estimator = build_candidates(seed=99)["small_mlp"]
    estimator.fit(X, y)
    mlp = estimator.named_steps["classifier"]
    loss = [float(v) for v in mlp.loss_curve_]
    scores = [float(v) for v in mlp.validation_scores_]
    if not loss or not scores:
        raise RuntimeError("The MLP did not produce a training loss and validation score history")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "training_log_loss", "validation_accuracy"])
        for epoch in range(max(len(loss), len(scores))):
            writer.writerow([epoch + 1, loss[epoch] if epoch < len(loss) else "",
                             scores[epoch] if epoch < len(scores) else ""])
    write_svg(loss, scores)
    print(f"saved {len(loss)} training-loss epochs and {len(scores)} validation-score epochs")
    print(OUTPUT)
    print(SVG)


if __name__ == "__main__":
    main()
