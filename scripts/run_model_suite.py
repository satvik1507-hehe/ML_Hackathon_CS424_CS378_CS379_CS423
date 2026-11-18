"""Run trip-grouped CV and student-test comparisons for the five registered models."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold

from pathsnap.data.reader import load_trip
from pathsnap.evaluation.metrics import observation_metrics
from pathsnap.models.candidates import build_candidates


class CachedDataset:
    def __init__(self, file: Path):
        with np.load(file, allow_pickle=False) as data:
            self.trip_id = str(data["trip_id"])
            self.X = data["X"]
            self.y = data["y"]
            self.observation_index = data["observation_index"]
            self.arc_ids = data["arc_ids"]
            self.reference_arc_ids = data["reference_arc_ids"]
            self.candidate_miss = data["candidate_miss"]
            self.aligned_distance_m = data["aligned_distance_m"]
            self.feature_names = tuple(str(x) for x in data["feature_names"])


def load_partition(root: Path, name: str) -> dict[str, CachedDataset]:
    return {p.stem: CachedDataset(p) for p in sorted((root / name).glob("*.npz"))}


def training_arrays(datasets: dict[str, CachedDataset], trip_ids: set[str]):
    xs, ys, groups = [], [], []
    for trip_id in sorted(trip_ids):
        d = datasets[trip_id]
        keep = ~d.candidate_miss[d.observation_index]
        xs.append(d.X[keep])
        ys.append(d.y[keep])
        groups.append(np.full(int(keep.sum()), trip_id, dtype="U8"))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(groups)


def score_model(model, dataset: CachedDataset) -> np.ndarray:
    return np.asarray(model.predict_proba(dataset.X)[:, 1], dtype=np.float32)


def write_csv(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def aggregate_records(records: list[dict[str, object]], *, model_name: str, partition: str,
                      train_seconds: float) -> dict[str, object]:
    rows = [row for row in records if row["model"] == model_name and row["partition"] == partition]
    observation_total = sum(int(row["observations"]) for row in rows)
    summary: dict[str, object] = {"model": model_name, "partition": partition}
    for metric in ("macro_f1", "macro_precision", "macro_recall", "route_continuity"):
        summary[metric] = float(np.mean([float(row[metric]) for row in rows])) if rows else 0.0
    for metric in ("top1_accuracy", "log_loss", "candidate_miss_rate"):
        if not rows:
            summary[metric] = 0.0
        else:
            summary[metric] = float(sum(float(row[metric]) * int(row["observations"]) for row in rows)
                                    / max(observation_total, 1))
    summary["observations"] = observation_total
    summary["train_seconds"] = float(train_seconds)
    summary["inference_ms_per_observation"] = (
        1000.0 * sum(float(row["inference_seconds"]) for row in rows) / max(observation_total, 1)
    )
    return summary


def run(data_root: Path, cache_root: Path, output_root: Path, seed: int, n_splits: int) -> Path:
    train = load_partition(cache_root, "train")
    student = load_partition(cache_root, "student_test")
    if len(train) != 80 or len(student) != 10:
        raise RuntimeError(f"Expected 80 train and 10 student_test trip caches; got {len(train)} and {len(student)}")
    trip_ids = sorted(train)
    X, y, groups = training_arrays(train, set(trip_ids))
    splitter = GroupKFold(n_splits=n_splits)
    fold_records: list[dict[str, object]] = []
    train_time: dict[str, float] = {name: 0.0 for name in build_candidates(seed)}
    score_root = cache_root.parent / "oof_scores"
    score_root.mkdir(parents=True, exist_ok=True)

    for model_name in build_candidates(seed):
        model_score_dir = score_root / model_name
        model_score_dir.mkdir(parents=True, exist_ok=True)
        print(f"Cross-validating {model_name} on {n_splits} trip-group folds", flush=True)
        for fold, (fit_rows, held_rows) in enumerate(splitter.split(X, y, groups), start=1):
            model = build_candidates(seed)[model_name]
            start = time.perf_counter()
            model.fit(X[fit_rows], y[fit_rows])
            fit_seconds = time.perf_counter() - start
            train_time[model_name] += fit_seconds
            held_trip_ids = set(np.unique(groups[held_rows]).tolist())
            for trip_id in sorted(held_trip_ids):
                dataset = train[trip_id]
                start = time.perf_counter()
                scores = score_model(model, dataset)
                inference_seconds = time.perf_counter() - start
                trip = load_trip(data_root / "train", trip_id)
                metrics = observation_metrics(dataset, scores, trip.arcs)
                fold_records.append({
                    "model": model_name, "partition": "cross_validation", "fold": fold,
                    "trip_id": trip_id, **metrics, "fit_seconds": fit_seconds,
                    "inference_seconds": inference_seconds,
                })
                np.savez_compressed(model_score_dir / f"{trip_id}.npz", scores=scores)
        print(f"  completed {model_name}; cumulative fit={train_time[model_name]:.1f}s", flush=True)

    for model_name in build_candidates(seed):
        model = build_candidates(seed)[model_name]
        X_all, y_all, _ = training_arrays(train, set(trip_ids))
        start = time.perf_counter()
        model.fit(X_all, y_all)
        fit_seconds = time.perf_counter() - start
        train_time[model_name] += fit_seconds
        student_score_dir = cache_root.parent / "student_scores" / model_name
        student_score_dir.mkdir(parents=True, exist_ok=True)
        for trip_id, dataset in student.items():
            start = time.perf_counter()
            scores = score_model(model, dataset)
            inference_seconds = time.perf_counter() - start
            trip = load_trip(data_root / "student_test", trip_id)
            metrics = observation_metrics(dataset, scores, trip.arcs)
            np.savez_compressed(student_score_dir / f"{trip_id}.npz", scores=scores)
            fold_records.append({
                "model": model_name, "partition": "student_test", "fold": 0,
                "trip_id": trip_id, **metrics, "fit_seconds": fit_seconds,
                "inference_seconds": inference_seconds,
            })

    summary_records = [
        aggregate_records(fold_records, model_name=name, partition=partition,
                          train_seconds=train_time[name])
        for name in build_candidates(seed)
        for partition in ("cross_validation", "student_test")
    ]
    summary_records.sort(key=lambda row: (row["partition"], -float(row["macro_f1"])))
    write_csv(output_root / "per_trip_metrics.csv", fold_records)
    write_csv(output_root / "model_comparison.csv", summary_records)

    cv_rows = [row for row in summary_records if row["partition"] == "cross_validation"]
    winner = max(cv_rows, key=lambda row: (float(row["macro_f1"]), float(row["top1_accuracy"]),
                                           float(row["route_continuity"])))
    winner_name = str(winner["model"])
    final_model = build_candidates(seed)[winner_name]
    X_all, y_all, _ = training_arrays(train, set(trip_ids))
    final_fit_start = time.perf_counter()
    final_model.fit(X_all, y_all)
    final_fit_seconds = time.perf_counter() - final_fit_start
    artifact_dir = Path("artifacts/final")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": final_model,
        "model_name": winner_name,
        "feature_names": list(train[trip_ids[0]].feature_names),
        "training_trip_ids": trip_ids,
        "seed": seed,
        "final_fit_seconds": final_fit_seconds,
        "selection_metric": "trip-mean 5-fold CV macro F1",
        "comparison": winner,
    }, artifact_dir / "candidate_scorer_untuned.joblib")
    (output_root / "selected_model.json").write_text(
        json.dumps({"model": winner_name, "selection": winner}, indent=2), encoding="utf-8"
    )
    print(f"Selected by CV Macro F1: {winner_name}", flush=True)
    return output_root / "model_comparison.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.getenv("PATHSNAP_DATA_ROOT"))
    parser.add_argument("--cache-root", default="artifacts/intermediate/candidate_cache")
    parser.add_argument("--output-root", default="outputs/experiments")
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    if not args.data_root:
        parser.error("Set PATHSNAP_DATA_ROOT or pass --data-root")
    path = run(Path(args.data_root), Path(args.cache_root), Path(args.output_root), args.seed, args.folds)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

