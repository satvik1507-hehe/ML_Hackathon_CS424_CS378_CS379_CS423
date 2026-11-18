"""Compare fixed, predeclared ensembles using OOF train predictions and student_test."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from pathsnap.data.reader import load_trip
from pathsnap.decoder.viterbi import decode_viterbi, decoded_metrics
from pathsnap.evaluation.metrics import observation_metrics
from run_model_suite import CachedDataset


MODELS = (
    "hist_gradient_boosting",
    "small_mlp",
    "random_forest",
    "logistic_regression",
    "calibrated_linear_svm",
)

ENSEMBLES = {
    "hist_gradient_boosting": ("hist_gradient_boosting", "probability"),
    "small_mlp": ("small_mlp", "probability"),
    "random_forest": ("random_forest", "probability"),
    "hgb_mlp_probability_mean": (("hist_gradient_boosting", "small_mlp"), "probability"),
    "hgb_rf_probability_mean": (("hist_gradient_boosting", "random_forest"), "probability"),
    "top3_probability_mean": (("hist_gradient_boosting", "small_mlp", "random_forest"), "probability"),
    "all5_probability_mean": (MODELS, "probability"),
    "hgb_mlp_rank_mean": (("hist_gradient_boosting", "small_mlp"), "rank"),
    "all5_rank_mean": (MODELS, "rank"),
}


def load_scores(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["scores"], dtype=np.float64)


def observation_slices(dataset: CachedDataset):
    order = np.argsort(dataset.observation_index, kind="stable")
    obs = dataset.observation_index[order]
    starts = np.r_[0, np.flatnonzero(np.diff(obs)) + 1]
    ends = np.r_[starts[1:], len(order)]
    return [(int(obs[a]), order[a:b]) for a, b in zip(starts, ends, strict=True)]


def normalized_probability(scores: np.ndarray, dataset: CachedDataset) -> np.ndarray:
    out = np.zeros_like(scores, dtype=np.float64)
    for _, rows in observation_slices(dataset):
        values = np.maximum(scores[rows], 1e-12)
        out[rows] = values / values.sum()
    return out


def normalized_rank(scores: np.ndarray, dataset: CachedDataset) -> np.ndarray:
    out = np.zeros_like(scores, dtype=np.float64)
    for _, rows in observation_slices(dataset):
        order = np.argsort(-scores[rows], kind="stable")
        ranks = np.empty(len(rows), dtype=np.float64)
        ranks[order] = np.arange(len(rows), 0, -1, dtype=np.float64)
        out[rows] = ranks / ranks.sum()
    return out


def make_ensemble(name: str, dataset: CachedDataset, model_scores: dict[str, np.ndarray]) -> np.ndarray:
    members, method = ENSEMBLES[name]
    if isinstance(members, str):
        members = (members,)
    if method == "probability":
        rows = [normalized_probability(model_scores[m], dataset) for m in members]
    else:
        rows = [normalized_rank(model_scores[m], dataset) for m in members]
    return np.mean(rows, axis=0)


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    total = sum(int(r["observations"]) for r in records)
    return {
        "trips": len(records),
        "observations": total,
        "top1_accuracy_observation_weighted": sum(
            float(r["top1_accuracy"]) * int(r["observations"]) for r in records
        ) / total,
        "macro_f1_trip_mean": float(np.mean([float(r["macro_f1"]) for r in records])),
        "route_continuity_trip_mean": float(np.mean([float(r["route_continuity"]) for r in records])),
        "candidate_miss_rate_observation_weighted": sum(
            float(r["candidate_miss_rate"]) * int(r["observations"]) for r in records
        ) / total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.getenv("PATHSNAP_DATA_ROOT"))
    parser.add_argument("--cache-root", default="artifacts/intermediate/candidate_cache")
    parser.add_argument("--experiment-root", default="outputs/experiments")
    parser.add_argument("--output", default="outputs/experiments/model_ensemble_comparison")
    args = parser.parse_args()
    if not args.data_root:
        parser.error("Set PATHSNAP_DATA_ROOT or pass --data-root")

    data_root = Path(args.data_root)
    cache_root = Path(args.cache_root)
    experiment_root = Path(args.experiment_root)
    decoder = json.loads((experiment_root / "selected_decoder.json").read_text(encoding="utf-8"))["parameters"]
    rows: list[dict[str, object]] = []
    for partition in ("train", "student_test"):
        ids = sorted(p.stem for p in (cache_root / partition).glob("*.npz"))
        for trip_id in ids:
            dataset = CachedDataset(cache_root / partition / f"{trip_id}.npz")
            trip = load_trip(data_root / partition, trip_id)
            score_partition = cache_root.parent / ("oof_scores" if partition == "train" else "student_scores")
            model_scores = {
                model: load_scores(score_partition / model / f"{trip_id}.npz") for model in MODELS
            }
            for name in ENSEMBLES:
                combined = make_ensemble(name, dataset, model_scores)
                raw = observation_metrics(dataset, combined, trip.arcs)
                decoded = decoded_metrics(trip, dataset, decode_viterbi(trip, dataset, combined, **decoder))
                rows.append({"partition": partition, "trip_id": trip_id, "ensemble": name,
                             "stage": "candidate_ranker", **raw})
                rows.append({"partition": partition, "trip_id": trip_id, "ensemble": name,
                             "stage": "ensemble_plus_hmm", **decoded})
        print(f"completed {partition}", flush=True)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_trip_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for partition in ("train", "student_test"):
        summary[partition] = {}
        for name in ENSEMBLES:
            summary[partition][name] = {}
            for stage in ("candidate_ranker", "ensemble_plus_hmm"):
                selected = [r for r in rows if r["partition"] == partition
                            and r["ensemble"] == name and r["stage"] == stage]
                summary[partition][name][stage] = summarize(selected)
    summary["configuration"] = {
        "base_predictions": "five registered model scores; train values are grouped OOF predictions",
        "ensemble_weights": "fixed equal weights; no student_test tuning",
        "decoder": decoder,
        "evaluator_partition_used": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
