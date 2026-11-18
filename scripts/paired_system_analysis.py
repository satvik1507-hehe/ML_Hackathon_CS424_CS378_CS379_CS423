"""Paired trip-bootstrap intervals for decoder and Bayesian-tuning effects."""

import argparse
import csv
from pathlib import Path

import numpy as np


def load(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def compare(rows: list[dict[str, str]], *, partition_key: str, partition_value: str,
            id_key: str, stage_key: str, left: str, right: str, metrics: tuple[str, ...],
            n_bootstrap: int, seed: int) -> list[dict[str, object]]:
    paired: dict[str, dict[str, dict[str, float]]] = {}
    for row in rows:
        if row.get(partition_key) == partition_value and row.get(id_key):
            paired.setdefault(row[id_key], {})[row[stage_key]] = {
                metric: float(row[metric]) for metric in metrics if row.get(metric) not in (None, "", "N/A")
            }
    ids = sorted(k for k, v in paired.items() if left in v and right in v)
    result = []
    rng = np.random.default_rng(seed)
    for metric in metrics:
        diffs = np.asarray([paired[trip_id][right][metric] - paired[trip_id][left][metric]
                            for trip_id in ids if metric in paired[trip_id][right]
                            and metric in paired[trip_id][left]], dtype=float)
        if not len(diffs):
            continue
        draws = rng.integers(0, len(diffs), size=(n_bootstrap, len(diffs)))
        boot = diffs[draws].mean(axis=1)
        result.append({"partition": partition_value, "metric": metric,
                       "baseline": left, "comparison": right, "trips": len(diffs),
                       "mean_difference": float(diffs.mean()),
                       "ci95_low": float(np.quantile(boot, 0.025)),
                       "ci95_high": float(np.quantile(boot, 0.975)),
                       "bootstrap_resamples": n_bootstrap})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--output-dir", default="outputs/experiments")
    args = parser.parse_args()
    out = Path(args.output_dir)
    decoder_rows = load(out / "decoder_comparison.csv")
    decoder = []
    for source_partition, output_partition in (("train", "training_oof"),
                                                ("student_test", "student_test")):
        new_rows = compare(decoder_rows, partition_key="partition", partition_value=source_partition,
                           id_key="trip_id", stage_key="stage", left="classifier_only",
                           right="classifier_plus_hmm_viterbi",
                           metrics=("macro_f1", "top1_accuracy", "route_continuity"),
                           n_bootstrap=args.resamples, seed=args.seed)
        for row in new_rows:
            row["partition"] = output_partition
        decoder.extend(new_rows)
    for name, rows in (("paired_bootstrap_decoder_effect.csv", decoder),):
        with (out / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    tuning = []
    for file, partition in (("tuned_vs_untuned_cv_per_trip.csv", "cross_validation"),
                            ("tuned_vs_untuned_per_trip.csv", "student_test")):
        rows = load(out / file)
        for row in rows:
            row["partition"] = partition
        tuning += compare(rows, partition_key="partition", partition_value=partition,
                          id_key="trip_id", stage_key="version", left="untuned", right="tuned",
                          metrics=("macro_f1", "top1_accuracy", "route_continuity"),
                          n_bootstrap=args.resamples, seed=args.seed)
    with (out / "paired_bootstrap_tuning_effect.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(tuning[0]))
        writer.writeheader()
        writer.writerows(tuning)
    print(f"Wrote paired decoder and tuning intervals under {out}")


if __name__ == "__main__":
    main()

