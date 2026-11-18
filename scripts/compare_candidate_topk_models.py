"""Compare the selected scorer with a wider candidate set, without overwriting Level 1 results."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from pathsnap.data.reader import load_trip
from pathsnap.evaluation.metrics import observation_metrics
from pathsnap.features.candidates import build_candidate_dataset


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.getenv("PATHSNAP_DATA_ROOT"), required=False)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", default="outputs/experiments/candidate_topk16_comparison")
    args = parser.parse_args()
    if not args.data_root:
        parser.error("Set PATHSNAP_DATA_ROOT or pass --data-root")

    data_root = Path(args.data_root)
    datasets = {}
    trips = {}
    for partition in ("train", "student_test"):
        folder = data_root / partition
        for trip_dir in sorted(p for p in folder.iterdir() if p.is_dir() and p.name.isdigit()):
            trip = load_trip(folder, trip_dir.name)
            trips[(partition, trip.trip_id)] = trip
            datasets[(partition, trip.trip_id)] = build_candidate_dataset(
                trip, top_k=args.top_k, search_radius_m=60.0, label_radius_m=100.0
            )

    train_ids = sorted(trip_id for partition, trip_id in datasets if partition == "train")
    student_ids = sorted(trip_id for partition, trip_id in datasets if partition == "student_test")
    if len(train_ids) != 80 or len(student_ids) != 10:
        raise RuntimeError(f"Expected 80 train and 10 student_test trips, got {len(train_ids)} and {len(student_ids)}")

    xs, ys, groups = [], [], []
    for trip_id in train_ids:
        d = datasets[("train", trip_id)]
        keep = ~d.candidate_miss[d.observation_index]
        xs.append(d.X[keep])
        ys.append(d.y[keep])
        groups.append(np.full(int(keep.sum()), trip_id, dtype="U8"))
    X, y, group_ids = np.concatenate(xs), np.concatenate(ys), np.concatenate(groups)

    records: list[dict[str, object]] = []
    fit_seconds = 0.0
    splitter = GroupKFold(n_splits=args.folds)
    for fold, (fit_rows, held_rows) in enumerate(splitter.split(X, y, group_ids), start=1):
        model = HistGradientBoostingClassifier(
            max_iter=100, learning_rate=0.08, l2_regularization=1.0,
            class_weight="balanced", random_state=99,
        )
        start = time.perf_counter()
        model.fit(X[fit_rows], y[fit_rows])
        fit_seconds += time.perf_counter() - start
        held_ids = sorted(set(group_ids[held_rows].tolist()))
        for trip_id in held_ids:
            d = datasets[("train", trip_id)]
            scores = model.predict_proba(d.X)[:, 1]
            metrics = observation_metrics(d, scores, trips[("train", trip_id)].arcs)
            records.append({"partition": "cross_validation", "fold": fold,
                            "trip_id": trip_id, "model": "hist_gradient_boosting_topk16",
                            **metrics})
        print(f"completed fold {fold}/{args.folds}", flush=True)

    final_model = HistGradientBoostingClassifier(
        max_iter=100, learning_rate=0.08, l2_regularization=1.0,
        class_weight="balanced", random_state=99,
    )
    start = time.perf_counter()
    final_model.fit(X, y)
    fit_seconds += time.perf_counter() - start
    for trip_id in student_ids:
        d = datasets[("student_test", trip_id)]
        scores = final_model.predict_proba(d.X)[:, 1]
        metrics = observation_metrics(d, scores, trips[("student_test", trip_id)].arcs)
        records.append({"partition": "student_test", "fold": 0,
                        "trip_id": trip_id, "model": "hist_gradient_boosting_topk16",
                        **metrics})

    output = Path(args.output)
    write_csv(output / "per_trip_metrics.csv", records)
    summary = {}
    for partition in ("cross_validation", "student_test"):
        rows = [r for r in records if r["partition"] == partition]
        n = sum(int(r["observations"]) for r in rows)
        summary[partition] = {
            "trips": len(rows),
            "observations": n,
            "trip_mean_macro_f1": float(np.mean([float(r["macro_f1"]) for r in rows])),
            "observation_weighted_top1_accuracy": float(
                sum(float(r["top1_accuracy"]) * int(r["observations"]) for r in rows) / n
            ),
            "observation_weighted_candidate_miss_rate": float(
                sum(float(r["candidate_miss_rate"]) * int(r["observations"]) for r in rows) / n
            ),
        }

    baseline_path = Path("outputs/experiments/model_comparison.csv")
    with baseline_path.open(newline="", encoding="utf-8") as stream:
        baseline = [r for r in csv.DictReader(stream)
                    if r["model"] == "hist_gradient_boosting"]
    for row in baseline:
        name = "cross_validation" if row["partition"] == "cross_validation" else "student_test"
        if name in summary:
            summary[name]["topk8_baseline_top1_accuracy"] = float(row["top1_accuracy"])
            summary[name]["topk8_baseline_macro_f1"] = float(row["macro_f1"])
    summary["configuration"] = {
        "top_k": args.top_k, "search_radius_m": 60.0, "label_radius_m": 100.0,
        "folds": args.folds, "seed": 99, "fit_seconds_total": fit_seconds,
        "evaluator_partition_used": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
