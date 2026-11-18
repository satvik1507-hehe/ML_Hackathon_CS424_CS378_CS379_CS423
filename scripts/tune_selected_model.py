"""Bayesian-optimize the CV-selected model using training-trip folds only."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
from sklearn.model_selection import GroupKFold

from pathsnap.data.reader import load_trip
from pathsnap.evaluation.metrics import observation_metrics
from pathsnap.models.candidates import build_candidates
from run_model_suite import CachedDataset, load_partition, training_arrays


def suggest_parameters(trial: optuna.Trial, model_name: str) -> dict[str, object]:
    if model_name == "logistic_regression":
        return {"classifier__C": trial.suggest_float("C", 1e-3, 1e2, log=True)}
    if model_name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 80, 240, step=40),
            "max_depth": trial.suggest_categorical("max_depth", [None, 16, 32, 48]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", 0.7, 1.0]),
        }
    if model_name == "hist_gradient_boosting":
        return {
            "max_iter": trial.suggest_int("max_iter", 60, 220, step=20),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "max_leaf_nodes": trial.suggest_categorical("max_leaf_nodes", [7, 15, 31, 63]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 15, 80, step=5),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
        }
    if model_name == "calibrated_linear_svm":
        return {"classifier__estimator__C": trial.suggest_float("C", 1e-3, 1e2, log=True)}
    if model_name == "small_mlp":
        hidden = trial.suggest_categorical("hidden_layer_sizes", ["32", "64-32", "64-32-16"])
        layers = tuple(int(x) for x in hidden.split("-"))
        return {
            "classifier__hidden_layer_sizes": layers,
            "classifier__alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
            "classifier__learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 5e-2, log=True),
        }
    raise ValueError(f"No search space for {model_name}")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(data_root: Path, cache_root: Path, output_root: Path, trials: int,
        folds: int, seed: int) -> Path:
    train = load_partition(cache_root, "train")
    student = load_partition(cache_root, "student_test")
    selection = json.loads((output_root / "selected_model.json").read_text(encoding="utf-8"))
    model_name = selection["model"]
    trip_ids = sorted(train)
    X, y, groups = training_arrays(train, set(trip_ids))
    splitter = GroupKFold(n_splits=folds)
    sampler = optuna.samplers.TPESampler(seed=seed)
    tuning_dir = Path("artifacts/tuning")
    tuning_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{(tuning_dir / f'pathsnap_{model_name}.db').resolve()}"
    study = optuna.create_study(direction="maximize", sampler=sampler, storage=storage,
                                study_name=f"pathsnap-{model_name}-{seed}", load_if_exists=True)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_parameters(trial, model_name)
        fold_trip_scores: list[float] = []
        for fit_rows, held_rows in splitter.split(X, y, groups):
            model = build_candidates(seed)[model_name]
            model.set_params(**params)
            model.fit(X[fit_rows], y[fit_rows])
            held_ids = set(np.unique(groups[held_rows]).tolist())
            for trip_id in sorted(held_ids):
                dataset = train[trip_id]
                trip = load_trip(data_root / "train", trip_id)
                scores = model.predict_proba(dataset.X)[:, 1]
                fold_trip_scores.append(observation_metrics(dataset, scores, trip.arcs)["macro_f1"])
        return float(np.mean(fold_trip_scores))

    optimization_start = time.perf_counter()
    study.optimize(objective, n_trials=trials, n_jobs=1, show_progress_bar=True)
    optimization_seconds = time.perf_counter() - optimization_start
    trial_rows = []
    for trial in study.trials:
        trial_rows.append({"number": trial.number, "state": trial.state.name, "value_macro_f1": trial.value,
                           "params_json": json.dumps(trial.params, sort_keys=True)})
    write_csv(output_root / "bayesian_trials.csv", trial_rows)
    (output_root / "tuning_result.json").write_text(
        json.dumps({"model": model_name, "best_value_cv_macro_f1": study.best_value,
                    "best_params": study.best_params, "trials": trials, "folds": folds,
                    "objective": "mean trip-level grouped-CV macro F1"}, indent=2),
        encoding="utf-8",
    )

    best_model = build_candidates(seed)[model_name]
    best_model.set_params(**suggest_parameters_from_best(study.best_params, model_name))
    X_all, y_all, _ = training_arrays(train, set(trip_ids))
    final_fit_start = time.perf_counter()
    best_model.fit(X_all, y_all)
    final_fit_seconds = time.perf_counter() - final_fit_start
    final_dir = Path("artifacts/final")
    final_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": best_model, "model_name": model_name, "params": study.best_params,
                 "feature_names": list(train[trip_ids[0]].feature_names), "training_trip_ids": trip_ids,
                 "seed": seed, "selection_metric": "Bayesian grouped-CV macro F1",
                 "optimization_seconds": optimization_seconds,
                 "final_fit_seconds": final_fit_seconds},
                final_dir / "candidate_scorer_tuned.joblib")

    # Compare the tuned and untuned selected model on the same permitted student-test trips.
    baseline_artifact = joblib.load(final_dir / "candidate_scorer_untuned.joblib")
    baseline_model = baseline_artifact["model"]
    records: list[dict[str, object]] = []
    for label, model in (("untuned", baseline_model), ("tuned", best_model)):
        for trip_id, dataset in student.items():
            trip = load_trip(data_root / "student_test", trip_id)
            inference_start = time.perf_counter()
            scores = model.predict_proba(dataset.X)[:, 1]
            inference_seconds = time.perf_counter() - inference_start
            records.append({"model": model_name, "version": label, "trip_id": trip_id,
                            **observation_metrics(dataset, scores, trip.arcs),
                            "inference_seconds": inference_seconds})
            if label == "tuned":
                score_dir = cache_root.parent / "student_scores" / f"{model_name}_tuned"
                score_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(score_dir / f"{trip_id}.npz", scores=scores)
    write_csv(output_root / "tuned_vs_untuned_per_trip.csv", records)
    summary = []
    for label in ("untuned", "tuned"):
        rows = [r for r in records if r["version"] == label]
        n = sum(int(r["observations"]) for r in rows)
        summary.append({
            "model": model_name, "version": label,
            "student_test_macro_f1_trip_mean": float(np.mean([r["macro_f1"] for r in rows])),
            "student_test_top1_accuracy": sum(float(r["top1_accuracy"]) * int(r["observations"]) for r in rows) / n,
            "student_test_log_loss": sum(float(r["log_loss"]) * int(r["observations"]) for r in rows) / n,
            "student_test_route_continuity_trip_mean": float(np.mean([r["route_continuity"] for r in rows])),
            "student_test_candidate_miss_rate": sum(float(r["candidate_miss_rate"]) * int(r["observations"]) for r in rows) / n,
            "observations": n,
            "optimization_seconds": optimization_seconds if label == "tuned" else 0.0,
            "final_fit_seconds": final_fit_seconds if label == "tuned" else baseline_artifact.get("final_fit_seconds", "N/A"),
            "inference_ms_per_observation": 1000.0 * sum(float(r["inference_seconds"]) for r in rows) / n,
        })
    write_csv(output_root / "tuned_vs_untuned_summary.csv", summary)
    return output_root / "tuned_vs_untuned_summary.csv"


def suggest_parameters_from_best(best: dict[str, object], model_name: str) -> dict[str, object]:
    params = dict(best)
    if model_name == "small_mlp":
        params["classifier__hidden_layer_sizes"] = tuple(
            int(x) for x in str(params["hidden_layer_sizes"]).split("-")
        )
        params.pop("hidden_layer_sizes")
    if model_name == "logistic_regression":
        params["classifier__C"] = params.pop("C")
    if model_name == "calibrated_linear_svm":
        params["classifier__estimator__C"] = params.pop("C")
    return params


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.getenv("PATHSNAP_DATA_ROOT"))
    parser.add_argument("--cache-root", default="artifacts/intermediate/candidate_cache")
    parser.add_argument("--output-root", default="outputs/experiments")
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=99)
    args = parser.parse_args()
    if not args.data_root:
        parser.error("Set PATHSNAP_DATA_ROOT or pass --data-root")
    print(run(Path(args.data_root), Path(args.cache_root), Path(args.output_root),
              args.trials, args.folds, args.seed))


if __name__ == "__main__":
    main()

