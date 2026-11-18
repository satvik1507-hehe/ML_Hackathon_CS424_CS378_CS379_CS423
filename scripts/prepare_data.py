"""Build trip-local candidate features for train and student_test only."""

import argparse
import csv
import os
from pathlib import Path

import numpy as np

from pathsnap.data.reader import load_trip
from pathsnap.features.candidates import FEATURE_NAMES, build_candidate_dataset


def prepare(data_root: Path, output_root: Path, top_k: int, radius_m: float) -> Path:
    records: list[dict[str, object]] = []
    for partition in ("train", "student_test"):
        trip_root = data_root / partition
        cache_root = output_root / "candidate_cache" / partition
        cache_root.mkdir(parents=True, exist_ok=True)
        for folder in sorted(x for x in trip_root.iterdir() if x.is_dir() and x.name.isdigit()):
            trip = load_trip(trip_root, folder.name)
            dataset = build_candidate_dataset(trip, top_k=top_k, search_radius_m=radius_m)
            np.savez_compressed(
                cache_root / f"{trip.trip_id}.npz",
                trip_id=np.asarray(trip.trip_id), X=dataset.X, y=dataset.y,
                observation_index=dataset.observation_index, arc_ids=dataset.arc_ids,
                reference_arc_ids=dataset.reference_arc_ids, candidate_miss=dataset.candidate_miss,
                aligned_distance_m=dataset.aligned_distance_m,
                feature_names=np.asarray(FEATURE_NAMES),
            )
            records.append({
                "partition": partition, "trip_id": trip.trip_id,
                "gps_observations": len(trip.gps), "road_arcs": len(trip.arcs),
                "reference_route_arcs": len(trip.route), "candidate_rows": len(dataset.X),
                "candidate_miss_rate": float(dataset.candidate_miss.mean()),
                "alignment_distance_median": float(np.median(dataset.aligned_distance_m)),
                "alignment_distance_p95": float(np.percentile(dataset.aligned_distance_m, 95)),
                "positive_candidate_observations": int(
                    len(np.unique(dataset.observation_index[dataset.y == 1]))
                ),
            })
            print(f"{partition}/{trip.trip_id}: {len(trip.gps)} observations, "
                  f"miss={records[-1]['candidate_miss_rate']:.3%}")
    summary = output_root / "candidate_summary.csv"
    summary.parent.mkdir(parents=True, exist_ok=True)
    with summary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.getenv("PATHSNAP_DATA_ROOT"), required=False)
    parser.add_argument("--output-root", default="artifacts/intermediate")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--radius-m", type=float, default=60.0)
    args = parser.parse_args()
    if not args.data_root:
        parser.error("Set PATHSNAP_DATA_ROOT or pass --data-root")
    summary = prepare(Path(args.data_root), Path(args.output_root), args.top_k, args.radius_m)
    print(f"Wrote {summary}")


if __name__ == "__main__":
    main()

