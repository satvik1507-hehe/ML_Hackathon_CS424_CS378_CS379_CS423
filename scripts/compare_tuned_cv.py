"""Compare untuned and tuned selected models on identical grouped CV folds."""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from pathsnap.data.reader import load_trip
from pathsnap.evaluation.metrics import observation_metrics
from pathsnap.models.candidates import build_candidates
from run_model_suite import load_partition, training_arrays
from tune_selected_model import suggest_parameters_from_best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.getenv("PATHSNAP_DATA_ROOT"))
    parser.add_argument("--cache-root", default="artifacts/intermediate/candidate_cache")
    parser.add_argument("--output-root", default="outputs/experiments")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=99)
    args = parser.parse_args()
    if not args.data_root:
        parser.error("Set PATHSNAP_DATA_ROOT or pass --data-root")
    data_root, cache_root, output_root = Path(args.data_root), Path(args.cache_root), Path(args.output_root)
    train = load_partition(cache_root, "train")
    ids = sorted(train)
    X, y, groups = training_arrays(train, set(ids))
    tune_result = json.loads((output_root / "tuning_result.json").read_text(encoding="utf-8"))
    model_name = tune_result["model"]
    tuned_params = suggest_parameters_from_best(tune_result["best_params"], model_name)
    models = {"untuned": build_candidates(args.seed)[model_name],
              "tuned": build_candidates(args.seed)[model_name].set_params(**tuned_params)}
    splitter = GroupKFold(n_splits=args.folds)
    per_trip = []
    train_seconds = {name: 0.0 for name in models}
    for fold, (fit_rows, hold_rows) in enumerate(splitter.split(X, y, groups), 1):
        held_ids = set(np.unique(groups[hold_rows]).tolist())
        for version, template in models.items():
            # Reinstantiate to avoid carrying state between folds.
            model = build_candidates(args.seed)[model_name]
            if version == "tuned":
                model.set_params(**tuned_params)
            start = time.perf_counter()
            model.fit(X[fit_rows], y[fit_rows])
            fit_elapsed = time.perf_counter() - start
            train_seconds[version] += fit_elapsed
            for trip_id in sorted(held_ids):
                d = train[trip_id]
                trip = load_trip(data_root / "train", trip_id)
                start = time.perf_counter()
                scores = model.predict_proba(d.X)[:, 1]
                infer_elapsed = time.perf_counter() - start
                per_trip.append({"version": version, "fold": fold, "trip_id": trip_id,
                                 **observation_metrics(d, scores, trip.arcs),
                                 "fit_seconds": fit_elapsed, "inference_seconds": infer_elapsed})
    summary = []
    for version in models:
        rows = [r for r in per_trip if r["version"] == version]
        n = sum(int(r["observations"]) for r in rows)
        summary.append({
            "model": model_name, "version": version, "cv_folds": args.folds,
            "cv_macro_f1_trip_mean": float(np.mean([r["macro_f1"] for r in rows])),
            "cv_top1_accuracy": sum(float(r["top1_accuracy"]) * int(r["observations"]) for r in rows) / n,
            "cv_log_loss": sum(float(r["log_loss"]) * int(r["observations"]) for r in rows) / n,
            "cv_route_continuity_trip_mean": float(np.mean([r["route_continuity"] for r in rows])),
            "cv_candidate_miss_rate": sum(float(r["candidate_miss_rate"]) * int(r["observations"]) for r in rows) / n,
            "observations": n, "training_seconds": train_seconds[version],
            "inference_ms_per_observation": 1000 * sum(float(r["inference_seconds"]) for r in rows) / n,
        })
    output_root.mkdir(parents=True, exist_ok=True)
    for name, rows in (("tuned_vs_untuned_cv_per_trip.csv", per_trip),
                       ("tuned_vs_untuned_cv_summary.csv", summary)):
        with (output_root / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(output_root / "tuned_vs_untuned_cv_summary.csv")


if __name__ == "__main__":
    main()

