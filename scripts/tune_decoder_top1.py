"""Tune route-decoder parameters for observation Top-1 using training OOF predictions only."""

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
from pathsnap.features.candidates import CandidateDataset


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


def scores(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        return data["scores"]


def aggregate(records: list[dict[str, object]]) -> dict[str, float]:
    n = sum(int(row["observations"]) for row in records)
    return {
        "observations": float(n),
        "top1_accuracy": sum(float(r["top1_accuracy"]) * int(r["observations"]) for r in records) / n,
        "trip_mean_macro_f1": float(np.mean([float(r["macro_f1"]) for r in records])),
        "trip_mean_route_continuity": float(np.mean([float(r["route_continuity"]) for r in records])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.getenv("PATHSNAP_DATA_ROOT"))
    parser.add_argument("--cache-root", default="artifacts/intermediate/candidate_cache")
    parser.add_argument("--experiment-root", default="outputs/experiments")
    parser.add_argument("--output", default="outputs/experiments/top1_decoder_tuning")
    args = parser.parse_args()
    if not args.data_root:
        parser.error("Set PATHSNAP_DATA_ROOT or pass --data-root")

    data_root, cache_root, experiment_root = Path(args.data_root), Path(args.cache_root), Path(args.experiment_root)
    model_name = json.loads((experiment_root / "selected_model.json").read_text())["model"]
    score_root = cache_root.parent / "oof_scores"
    train_ids = sorted(path.stem for path in (cache_root / "train").glob("*.npz"))
    grid = [(sigma, gap, penalty)
            for sigma in (10.0, 20.0, 40.0)
            for gap in (4.0, 12.0)
            for penalty in (4.0, 8.0, 12.0)]

    search_rows = []
    best_config = None
    best_key = None
    for sigma, gap, penalty in grid:
        config = {"base_sigma_m": sigma, "gap_tolerance_s": gap, "disconnect_penalty": penalty}
        records = []
        for trip_id in train_ids:
            dataset = CachedDataset(cache_root / "train" / f"{trip_id}.npz")
            trip = load_trip(data_root / "train", trip_id)
            emission = scores(score_root / model_name / f"{trip_id}.npz")
            pred = decode_viterbi(trip, dataset, emission, **config)
            records.append(decoded_metrics(trip, dataset, pred))
        summary = aggregate(records)
        row = {**config, **summary}
        search_rows.append(row)
        key = (summary["top1_accuracy"], summary["trip_mean_macro_f1"],
               summary["trip_mean_route_continuity"])
        if best_key is None or key > best_key:
            best_key, best_config = key, config
        print(f"tested {config}: Top-1={summary['top1_accuracy']:.4f}", flush=True)

    assert best_config is not None
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "decoder_parameter_search.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(search_rows[0]))
        writer.writeheader()
        writer.writerows(search_rows)

    partition_summaries = {}
    per_trip = []
    for partition, ids in (("train", train_ids),
                           ("student_test", sorted(p.stem for p in (cache_root / "student_test").glob("*.npz")))):
        raw_rows, decoded_rows = [], []
        score_partition = score_root if partition == "train" else cache_root.parent / "student_scores"
        for trip_id in ids:
            dataset = CachedDataset(cache_root / partition / f"{trip_id}.npz")
            trip = load_trip(data_root / partition, trip_id)
            emission = scores(score_partition / model_name / f"{trip_id}.npz")
            raw = observation_metrics(dataset, emission, trip.arcs)
            prediction = decode_viterbi(trip, dataset, emission, **best_config)
            decoded = decoded_metrics(trip, dataset, prediction)
            raw_rows.append(raw)
            decoded_rows.append(decoded)
            per_trip.append({"partition": partition, "trip_id": trip_id, "stage": "classifier_only", **raw})
            per_trip.append({"partition": partition, "trip_id": trip_id,
                             "stage": "top1_tuned_hmm_viterbi", **decoded})
        partition_summaries[partition] = {
            "classifier_only": aggregate(raw_rows),
            "top1_tuned_hmm_viterbi": aggregate(decoded_rows),
        }
    with (output / "per_trip_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_trip[0]))
        writer.writeheader()
        writer.writerows(per_trip)

    result = {
        "selected_model": model_name,
        "decoder_parameters": best_config,
        "selection_metric": "observation-weighted Top-1 across training-trip OOF predictions",
        "training_oof_grid_best": search_rows[grid.index((best_config["base_sigma_m"],
                                                            best_config["gap_tolerance_s"],
                                                            best_config["disconnect_penalty"]))],
        "partitions": partition_summaries,
        "evaluator_partition_used": False,
        "note": "Separate experiment; original selected decoder/results are unchanged.",
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
