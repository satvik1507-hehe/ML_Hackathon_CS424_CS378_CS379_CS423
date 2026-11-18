"""Tune Viterbi transition settings on OOF train predictions and compare on student_test."""

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
from pathsnap.visualization.map_trip import make_trip_map


class CachedDataset:
    def __init__(self, file: Path):
        with np.load(file, allow_pickle=False) as d:
            self.trip_id = str(d["trip_id"])
            self.X = d["X"]
            self.y = d["y"]
            self.observation_index = d["observation_index"]
            self.arc_ids = d["arc_ids"]
            self.reference_arc_ids = d["reference_arc_ids"]
            self.candidate_miss = d["candidate_miss"]
            self.aligned_distance_m = d["aligned_distance_m"]


def _load_scores(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as d:
        return d["scores"]


def _records_for_partition(data_root: Path, cache_root: Path, score_root: Path,
                           partition: str, trip_ids: list[str], model_name: str,
                           config: dict[str, float]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    raw_records, decoded_records = [], []
    for trip_id in trip_ids:
        dataset = CachedDataset(cache_root / partition / f"{trip_id}.npz")
        trip = load_trip(data_root / partition, trip_id)
        scores = _load_scores(score_root / model_name / f"{trip_id}.npz")
        raw_metrics = observation_metrics(dataset, scores, trip.arcs)
        predicted = decode_viterbi(trip, dataset, scores, **config)
        decoded = decoded_metrics(trip, dataset, predicted)
        raw_records.append({"partition": partition, "trip_id": trip_id, "stage": "classifier_only",
                            "model": model_name, **raw_metrics})
        decoded_records.append({"partition": partition, "trip_id": trip_id,
                                "stage": "classifier_plus_hmm_viterbi", "model": model_name,
                                **decoded, "log_loss": "N/A"})
        if partition == "student_test" and trip_id == "00000056":
            make_trip_map(trip, f"outputs/maps/trip-{trip_id}-prediction.html", predicted)
    return raw_records, decoded_records


def _aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    total = sum(int(r["observations"]) for r in rows)
    result: dict[str, object] = {"partition": rows[0]["partition"], "stage": rows[0]["stage"],
                                 "model": rows[0]["model"], "observations": total}
    for metric in ("macro_f1", "route_continuity"):
        result[metric] = float(np.mean([float(r[metric]) for r in rows]))
    for metric in ("top1_accuracy", "candidate_miss_rate"):
        result[metric] = sum(float(r[metric]) * int(r["observations"]) for r in rows) / max(total, 1)
    if "log_loss" in rows[0] and rows[0]["log_loss"] != "N/A":
        result["log_loss"] = sum(float(r["log_loss"]) * int(r["observations"]) for r in rows) / max(total, 1)
    else:
        result["log_loss"] = "N/A"
    return result


def _write_csv(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def run(data_root: Path, cache_root: Path, output_root: Path) -> Path:
    selected = json.loads((output_root / "selected_model.json").read_text(encoding="utf-8"))
    model_name = selected["model"]
    score_root = cache_root.parent / "oof_scores"
    # Decoder parameters are selected using only grouped out-of-fold predictions from training trips.
    grid = [(sigma, gap, penalty)
            for sigma in (10.0, 20.0, 40.0)
            for gap in (4.0, 12.0)
            for penalty in (4.0, 8.0, 12.0)]
    train_ids = sorted(p.stem for p in (cache_root / "train").glob("*.npz"))
    train_contexts = []
    for trip_id in train_ids:
        dataset = CachedDataset(cache_root / "train" / f"{trip_id}.npz")
        trip = load_trip(data_root / "train", trip_id)
        probs = _load_scores(score_root / model_name / f"{trip_id}.npz")
        train_contexts.append((dataset, trip, probs))
    best_config = None
    best_score = None
    tuning_rows = []
    for base_sigma, gap_tolerance, disconnect_penalty in grid:
        config = {"base_sigma_m": base_sigma, "gap_tolerance_s": gap_tolerance,
                  "disconnect_penalty": disconnect_penalty}
        scores = []
        for dataset, trip, probs in train_contexts:
            decoded = decode_viterbi(trip, dataset, probs, **config)
            scores.append(decoded_metrics(trip, dataset, decoded))
        mean_f1 = float(np.mean([x["macro_f1"] for x in scores]))
        mean_cont = float(np.mean([x["route_continuity"] for x in scores]))
        tuning_rows.append({**config, "cv_macro_f1": mean_f1, "cv_route_continuity": mean_cont})
        key = (mean_f1, mean_cont)
        if best_score is None or key > best_score:
            best_score, best_config = key, config
    assert best_config is not None
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "decoder_parameter_search.csv", tuning_rows)
    (output_root / "selected_decoder.json").write_text(
        json.dumps({"model": model_name, "parameters": best_config,
                    "selection": "training-trip OOF macro F1; continuity tie-break"}, indent=2),
        encoding="utf-8",
    )

    all_records: list[dict[str, object]] = []
    for partition, ids in (("train", train_ids),
                           ("student_test", sorted(p.stem for p in (cache_root / "student_test").glob("*.npz")))):
        raw, decoded = _records_for_partition(data_root, cache_root,
                                              score_root if partition == "train" else cache_root.parent / "student_scores",
                                              partition, ids, model_name, best_config)
        all_records.extend(raw + decoded)
        all_records.append(_aggregate(raw))
        all_records.append(_aggregate(decoded))
    _write_csv(output_root / "decoder_comparison.csv", all_records)
    return output_root / "decoder_comparison.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.getenv("PATHSNAP_DATA_ROOT"))
    parser.add_argument("--cache-root", default="artifacts/intermediate/candidate_cache")
    parser.add_argument("--output-root", default="outputs/experiments")
    args = parser.parse_args()
    if not args.data_root:
        parser.error("Set PATHSNAP_DATA_ROOT or pass --data-root")
    print(run(Path(args.data_root), Path(args.cache_root), Path(args.output_root)))


if __name__ == "__main__":
    main()

