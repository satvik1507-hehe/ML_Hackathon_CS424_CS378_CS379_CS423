"""Paired trip-bootstrap intervals comparing the top CV model with its alternatives."""

import argparse
import csv
from pathlib import Path

import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def paired_bootstrap(rows: list[dict[str, str]], partition: str, reference: str,
                     metric: str, n_bootstrap: int, seed: int) -> list[dict[str, object]]:
    groups: dict[str, dict[str, float]] = {}
    for row in rows:
        if row["partition"] == partition:
            groups.setdefault(row["trip_id"], {})[row["model"]] = float(row[metric])
    model_names = sorted({model for values in groups.values() for model in values})
    rng = np.random.default_rng(seed)
    trip_ids = sorted(groups)
    results = []
    for model in model_names:
        if model == reference:
            continue
        pairs = [(groups[t][model], groups[t][reference]) for t in trip_ids
                 if model in groups[t] and reference in groups[t]]
        differences = np.asarray([a - b for a, b in pairs], dtype=float)
        draws = rng.integers(0, len(differences), size=(n_bootstrap, len(differences)))
        boot = differences[draws].mean(axis=1)
        results.append({
            "partition": partition, "metric": metric, "reference_model": reference,
            "comparison_model": model, "trips": len(differences),
            "mean_paired_difference": float(differences.mean()),
            "ci95_low": float(np.quantile(boot, 0.025)),
            "ci95_high": float(np.quantile(boot, 0.975)),
            "bootstrap_resamples": n_bootstrap,
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="outputs/experiments/per_trip_metrics.csv")
    parser.add_argument("--model-summary", default="outputs/experiments/model_comparison.csv")
    parser.add_argument("--output", default="outputs/experiments/paired_bootstrap.csv")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=99)
    args = parser.parse_args()
    summary = read_rows(Path(args.model_summary))
    cv = [r for r in summary if r["partition"] == "cross_validation"]
    reference = max(cv, key=lambda r: float(r["macro_f1"]))["model"]
    rows = read_rows(Path(args.input))
    records = []
    for partition in ("cross_validation", "student_test"):
        for metric in ("macro_f1", "top1_accuracy", "route_continuity"):
            records.extend(paired_bootstrap(rows, partition, reference, metric,
                                             args.resamples, args.seed))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"Reference model: {reference}; wrote {output}")


if __name__ == "__main__":
    main()

