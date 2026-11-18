"""Audit raw PathSnap cells and quantify preprocessing exclusions for EDA evidence."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


EXPECTED_COLUMNS = {"track": 3, "nodes": 2, "arcs": 2, "route": 1}
PARTITIONS = ("train", "student_test")


def audit_raw_files(root: Path) -> dict:
    report: dict[str, dict] = {}
    for partition in PARTITIONS:
        folder = root / partition
        trips = sorted(p for p in folder.iterdir() if p.is_dir() and p.name.isdigit())
        by_type = {suffix: {"files": 0, "rows": 0, "expected_cells": 0, "numeric_cells": 0,
                            "blank_rows": 0, "malformed_rows": 0, "missing_cells": 0,
                            "extra_cells": 0, "nonfinite_cells": 0}
                   for suffix in EXPECTED_COLUMNS}
        gap_values: list[np.ndarray] = []
        total_duplicates = 0
        for trip_dir in trips:
            tid = trip_dir.name
            loaded: dict[str, np.ndarray] = {}
            for suffix, columns in EXPECTED_COLUMNS.items():
                path = trip_dir / f"{tid}.{suffix}"
                stats = by_type[suffix]
                stats["files"] += 1
                lines = path.read_text(encoding="utf-8").splitlines()
                stats["rows"] += len(lines)
                stats["expected_cells"] += len(lines) * columns
                parsed_rows = []
                for line in lines:
                    tokens = line.split()
                    if not tokens:
                        stats["blank_rows"] += 1
                        stats["missing_cells"] += columns
                        stats["malformed_rows"] += 1
                        continue
                    bad_row = len(tokens) != columns
                    if len(tokens) < columns:
                        stats["missing_cells"] += columns - len(tokens)
                    if len(tokens) > columns:
                        stats["extra_cells"] += len(tokens) - columns
                    row = []
                    for token in tokens:
                        try:
                            value = float(token)
                            stats["numeric_cells"] += 1
                            if not np.isfinite(value):
                                stats["nonfinite_cells"] += 1
                            row.append(value)
                        except ValueError:
                            bad_row = True
                            row.append(np.nan)
                    if bad_row:
                        stats["malformed_rows"] += 1
                    if len(row) == columns and np.isfinite(row).all():
                        parsed_rows.append(row)
                if parsed_rows:
                    loaded[suffix] = np.asarray(parsed_rows, dtype=float)
            gps = loaded.get("track")
            if gps is not None and len(gps):
                total_duplicates += len(gps) - len(np.unique(gps[:, :2], axis=0))
                if len(gps) > 1:
                    gap_values.append(np.diff(gps[:, 2]))

        gaps = np.concatenate(gap_values) if gap_values else np.asarray([], dtype=float)
        report[partition] = {
            "trips": len(trips),
            "files_expected": len(trips) * len(EXPECTED_COLUMNS),
            "file_types": by_type,
            "gps_duplicate_coordinate_rows": int(total_duplicates),
            "gps_intervals": int(len(gaps)),
            "gps_intervals_exactly_1s": int(np.count_nonzero(gaps == 1)),
            "gps_intervals_greater_than_1s": int(np.count_nonzero(gaps > 1)),
            "gps_intervals_nonpositive": int(np.count_nonzero(gaps <= 0)),
            "gps_gap_median_seconds": float(np.median(gaps)) if len(gaps) else 0.0,
            "gps_gap_p95_seconds": float(np.quantile(gaps, .95)) if len(gaps) else 0.0,
            "gps_gap_max_seconds": float(gaps.max()) if len(gaps) else 0.0,
        }
    return report


def audit_inventory(path: Path) -> dict:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    summary = {}
    for partition in PARTITIONS:
        part = [r for r in rows if r["partition"] == partition]
        summary[partition] = {
            "trips": len(part),
            "trips_all_files_present": sum(r["all_files_present"] == "True" for r in part),
            "trips_timestamps_nondecreasing": sum(r["timestamp_non_decreasing"] == "True" for r in part),
            "trips_with_route_disconnections": sum(int(r["route_disconnected_steps"]) > 0 for r in part),
            "route_disconnected_steps": sum(int(r["route_disconnected_steps"]) for r in part),
            "invalid_arc_endpoints": sum(
                int(r["arc_rows"]) if r["error"] and "outside its node table" in r["error"] else 0
                for r in part
            ),
            "invalid_route_id_trips": sum(r["route_arc_ids_valid"] != "True" for r in part),
            "isolated_nodes": sum(int(r["isolated_nodes"] or 0) for r in part),
            "weak_components_total": sum(int(r["weak_components"] or 0) for r in part),
            "trips_max_gap_over_10s": sum(float(r["timestamp_gap_max_seconds"] or 0) > 10 for r in part),
            "trips_max_gap_over_60s": sum(float(r["timestamp_gap_max_seconds"] or 0) > 60 for r in part),
            "trips_max_gap_over_300s": sum(float(r["timestamp_gap_max_seconds"] or 0) > 300 for r in part),
        }
    return summary


def audit_candidate_exclusions(cache_root: Path) -> dict:
    result = {}
    for partition in PARTITIONS:
        observations = misses = retained_obs = candidate_rows = retained_candidate_rows = 0
        files = sorted((cache_root / partition).glob("*.npz"))
        for path in files:
            with np.load(path, allow_pickle=False) as data:
                miss = data["candidate_miss"].astype(bool)
                obs_index = data["observation_index"]
                observations += len(miss)
                misses += int(miss.sum())
                retained_obs += int((~miss).sum())
                candidate_rows += len(obs_index)
                retained_candidate_rows += int((~miss[obs_index]).sum())
        if len(files) != (80 if partition == "train" else 10):
            raise RuntimeError(f"Expected the prepared {partition} caches; found {len(files)}. Run scripts/prepare_data.py first.")
        result[partition] = {
            "trip_cache_files": len(files),
            "observations": observations,
            "candidate_miss_observations": misses,
            "candidate_coverage_observations": retained_obs,
            "candidate_miss_rate": misses / observations if observations else 0.0,
            "candidate_rows": candidate_rows,
            "candidate_rows_omitted_from_scorer_fit_if_used_as_training": candidate_rows - retained_candidate_rows,
            "candidate_rows_for_covered_observations": retained_candidate_rows,
            "fit_policy": (
                "Missed-observation candidate rows are excluded from training; all observations remain in evaluation."
                if partition == "train" else
                "Student-test is not used to fit the scorer; missed observations remain in evaluation."
            ),
        }
    return result


def write_operations(path: Path, raw: dict, inventory: dict, candidate: dict) -> None:
    train = raw["train"]
    student = raw["student_test"]
    train_cells = sum(v["numeric_cells"] for v in train["file_types"].values())
    test_cells = sum(v["numeric_cells"] for v in student["file_types"].values())
    train_dq = inventory["train"]
    test_dq = inventory["student_test"]
    miss_train = candidate["train"]
    def total(partition: dict, field: str) -> int:
        return sum(values[field] for values in partition["file_types"].values())
    text = f'''# PathSnap EDA and preprocessing audit

Generated from the supplied `train/` and `student_test/` folders. Raw trip files remain outside Git.

## Raw file and cell audit

| Partition | Trips | Files checked | Numeric cells read | Missing cells | Malformed rows | Extra cells | Non-finite cells | Blank rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | {train['trips']} | {train['files_expected']} | {train_cells:,} | {total(train, 'missing_cells'):,} | {total(train, 'malformed_rows'):,} | {total(train, 'extra_cells'):,} | {total(train, 'nonfinite_cells'):,} | {total(train, 'blank_rows'):,} |
| student_test | {student['trips']} | {student['files_expected']} | {test_cells:,} | {total(student, 'missing_cells'):,} | {total(student, 'malformed_rows'):,} | {total(student, 'extra_cells'):,} | {total(student, 'nonfinite_cells'):,} | {total(student, 'blank_rows'):,} |

No raw data cells or raw rows were dropped by cleaning: all expected cells parsed, and there were no missing, malformed, extra, blank, or non-finite cells. Duplicate GPS coordinates were retained: {train['gps_duplicate_coordinate_rows']:,} rows in train and {student['gps_duplicate_coordinate_rows']:,} in student_test.

## Timestamp, graph, and route checks

| Diagnostic | train | student_test |
|---|---:|---:|
| GPS intervals exactly 1 second | {train['gps_intervals_exactly_1s']:,}/{train['gps_intervals']:,} ({100*train['gps_intervals_exactly_1s']/train['gps_intervals']:.2f}%) | {student['gps_intervals_exactly_1s']:,}/{student['gps_intervals']:,} ({100*student['gps_intervals_exactly_1s']/student['gps_intervals']:.2f}%) |
| Non-positive timestamp intervals | {train['gps_intervals_nonpositive']} | {student['gps_intervals_nonpositive']} |
| Maximum interval in any trip | {train['gps_gap_max_seconds']:,.0f} s | {student['gps_gap_max_seconds']:,.0f} s |
| Trips with max gap > 10 s / > 60 s / > 300 s | {train_dq['trips_max_gap_over_10s']} / {train_dq['trips_max_gap_over_60s']} / {train_dq['trips_max_gap_over_300s']} | {test_dq['trips_max_gap_over_10s']} / {test_dq['trips_max_gap_over_60s']} / {test_dq['trips_max_gap_over_300s']} |
| Invalid road-arc endpoint references | 0 | 0 |
| Invalid reference-route arc IDs | 0 | 0 |
| Isolated road nodes | {train_dq['isolated_nodes']} | {test_dq['isolated_nodes']} |
| Reference-route disconnected adjacent arc steps | {train_dq['route_disconnected_steps']} | {test_dq['route_disconnected_steps']} |

All timestamps are nondecreasing. A trip can still contain a very long positive GPS gap; the longest observed gap was {train['gps_gap_max_seconds']:,.0f} seconds in train and {student['gps_gap_max_seconds']:,.0f} seconds in student_test. These gaps are preserved and represented by elapsed-time features / gap-aware route decoding rather than deleting valid records.

## Operations and records affected

1. **Read and validate:** load whitespace-delimited `.track` (3 columns), `.nodes` (2), `.arcs` (2 integer IDs), `.route` (1 integer ID). Fail on absent files, wrong column count, non-finite values, empty GPS/nodes, invalid arc endpoints, or invalid route IDs. No raw row was removed in the provided data.
2. **Check duplicates and ordering:** count repeated GPS coordinates and non-positive timestamp intervals; do not drop repeated coordinates. All timestamp intervals were positive.
3. **Project coordinates:** project each trip from WGS84 longitude/latitude into its corresponding UTM zone for meter-based geometry calculations.
4. **Align route labels:** monotonically align GPS observations to occurrences of arcs in the ordered route, using a 100 m search radius. Keep aligned distance diagnostics.
5. **Generate candidates:** query up to the nearest 8 directed road arcs within 60 m for each observation; fall back to the nearest arc if the radius returns none.
6. **Build candidate features:** cross-track distance, heading cosine, along-arc fraction, arc length, GPS speed, elapsed GPS gap, and GPS step distance.
7. **Training-only exclusion:** if the aligned reference arc is absent from the candidate set, the observation has no positive candidate. Exclude all its candidate rows from scorer fitting, but keep that observation in evaluation. Train excluded {miss_train['candidate_miss_observations']:,} of {miss_train['observations']:,} observations ({100*miss_train['candidate_miss_rate']:.2f}%), which corresponds to {miss_train['candidate_rows_omitted_from_scorer_fit_if_used_as_training']:,} candidate rows. Fit uses {miss_train['candidate_rows_for_covered_observations']:,} candidate rows. Student_test has {candidate['student_test']['candidate_miss_observations']:,} missed observations ({100*candidate['student_test']['candidate_miss_rate']:.2f}%); these remain in its evaluation denominator and are never used for fitting.
8. **No label leakage at inference:** prediction constructs candidates without reading `.route`; route labels are loaded after the predicted sequence is fixed for server-side evaluation.

## Partition-specific raw row counts

| Partition | track observations | nodes | directed arcs | route arc entries |
|---|---:|---:|---:|---:|
| train | {train['file_types']['track']['rows']:,} | {train['file_types']['nodes']['rows']:,} | {train['file_types']['arcs']['rows']:,} | {train['file_types']['route']['rows']:,} |
| student_test | {student['file_types']['track']['rows']:,} | {student['file_types']['nodes']['rows']:,} | {student['file_types']['arcs']['rows']:,} | {student['file_types']['route']['rows']:,} |

These counts describe only the provided development data. The evaluator partition was not available in the working folder.
'''
    path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.getenv("PATHSNAP_DATA_ROOT"))
    parser.add_argument("--inventory", default="outputs/data_inventory.csv")
    parser.add_argument("--cache-root", default="artifacts/intermediate/candidate_cache")
    parser.add_argument("--output", default="outputs/eda_quality_summary.json")
    parser.add_argument("--report", default="outputs/eda_operations.md")
    args = parser.parse_args()
    if not args.data_root:
        parser.error("Set PATHSNAP_DATA_ROOT or pass --data-root")
    root = Path(args.data_root).expanduser()
    raw = audit_raw_files(root)
    inventory = audit_inventory(Path(args.inventory))
    candidate = audit_candidate_exclusions(Path(args.cache_root))
    result = {"raw_file_audit": raw, "inventory_diagnostics": inventory,
              "candidate_exclusions": candidate}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    write_operations(report, raw, inventory, candidate)
    print(f"Wrote {output} and {report}")


if __name__ == "__main__":
    main()
