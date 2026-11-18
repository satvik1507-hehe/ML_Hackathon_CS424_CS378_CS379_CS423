"""Run the PathSnap EDA audit and save plots, backing CSVs, and three Folium maps.

Only the explicitly permitted ``train`` and ``student_test`` folders are opened. The evaluator
partition is never searched for or read.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pathsnap-matplotlib"))

import folium
import matplotlib
import numpy as np
from shapely.geometry import LineString, Point, box
from shapely.strtree import STRtree

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from pathsnap.data.reader import Trip, load_trip
from pathsnap.decoder.viterbi import decode_viterbi
from pathsnap.features.candidates import (
    FEATURE_NAMES,
    _metric_geometry,
    _route_occurrences,
    build_candidate_dataset,
)


PARTITIONS = ("train", "student_test")
EXPECTED_TRIPS = {"train": 80, "student_test": 10}
FILE_COLUMNS = {"track": 3, "nodes": 2, "arcs": 2, "route": 1}
TOP_K_VALUES = (4, 8, 12, 16)
SEARCH_RADIUS_M = 60.0
LABEL_RADIUS_M = 100.0
IMPLAUSIBLE_SPEED_MPS = 55.0  # transparent screening threshold, not a physical impossibility claim
FEATURE_LABELS = {
    "cross_track_distance_m": "Cross-track distance (m)",
    "heading_cosine": "Heading cosine",
    "along_arc_fraction": "Along-arc fraction",
    "arc_length_m": "Arc length (m)",
    "gps_speed_mps": "GPS speed (m/s)",
    "gps_gap_seconds": "GPS gap (s)",
    "gps_step_distance_m": "GPS step distance (m)",
}


def corridor_arcs(arc_lines: list[LineString], corridor, route_ids: set[int], limit: int) -> list[int]:
    """Select nearby arcs; deterministically sample dense display neighborhoods."""
    nearby = set(map(int, np.asarray(STRtree(arc_lines).query(corridor), dtype=int))) | route_ids
    local = sorted(nearby - route_ids)
    if len(local) > limit:
        local = local[::int(np.ceil(len(local) / limit))][:limit]
    return sorted(route_ids | set(local))


def gps_neighborhood(gps_xy: np.ndarray, padding_m: float = 250.0):
    """Fast display-only envelope around a projected GPS track."""
    return box(float(gps_xy[:, 0].min() - padding_m), float(gps_xy[:, 1].min() - padding_m),
               float(gps_xy[:, 0].max() + padding_m), float(gps_xy[:, 1].max() + padding_m))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def raw_file_audit(trip_dir: Path, trip: Trip) -> dict:
    """Count malformed cells directly from raw lines without dropping or repairing rows."""
    cells = {name: 0 for name in ("expected", "numeric", "missing", "extra", "nonfinite", "malformed_rows", "blank_rows")}
    per_suffix = {}
    for suffix, ncols in FILE_COLUMNS.items():
        path = trip_dir / f"{trip.trip_id}.{suffix}"
        stats = {name: 0 for name in cells}
        for line in path.read_text(encoding="utf-8").splitlines():
            tokens = line.split()
            stats["expected"] += ncols
            if not tokens:
                stats["blank_rows"] += 1
                stats["missing"] += ncols
                stats["malformed_rows"] += 1
                continue
            bad = len(tokens) != ncols
            stats["missing"] += max(0, ncols - len(tokens))
            stats["extra"] += max(0, len(tokens) - ncols)
            for token in tokens:
                try:
                    value = float(token)
                    stats["numeric"] += 1
                    stats["nonfinite"] += int(not np.isfinite(value))
                except ValueError:
                    bad = True
            stats["malformed_rows"] += int(bad)
        per_suffix[suffix] = stats
        for key in cells:
            cells[key] += stats[key]

    gps = trip.gps
    nodes = trip.nodes
    lon_bad_rows = int(np.count_nonzero((gps[:, 0] < -180) | (gps[:, 0] > 180)))
    lat_bad_rows = int(np.count_nonzero((gps[:, 1] < -90) | (gps[:, 1] > 90)))
    node_lon_bad_rows = int(np.count_nonzero((nodes[:, 0] < -180) | (nodes[:, 0] > 180)))
    node_lat_bad_rows = int(np.count_nonzero((nodes[:, 1] < -90) | (nodes[:, 1] > 90)))
    gaps = np.diff(gps[:, 2])
    invalid_endpoints = int(np.count_nonzero((trip.arcs < 0) | (trip.arcs >= len(nodes))))
    invalid_route_ids = int(np.count_nonzero((trip.route < 0) | (trip.route >= len(trip.arcs))))
    duplicate_rows = int(len(gps) - len(np.unique(gps[:, :2], axis=0)))
    nondecreasing = bool(np.all(gaps >= 0))
    route_disconnected = 0
    if len(trip.route) > 1:
        route_arcs = trip.arcs[trip.route]
        route_disconnected = int(np.count_nonzero(route_arcs[:-1, 1] != route_arcs[1:, 0]))
    return {
        **cells,
        "file_missing_count": sum(not (trip_dir / f"{trip.trip_id}.{suffix}").is_file() for suffix in FILE_COLUMNS),
        "gps_rows": len(gps),
        "gps_coordinate_out_of_range_rows": lon_bad_rows + lat_bad_rows,
        "gps_longitude_out_of_range_rows": lon_bad_rows,
        "gps_latitude_out_of_range_rows": lat_bad_rows,
        "node_coordinate_out_of_range_rows": node_lon_bad_rows + node_lat_bad_rows,
        "duplicate_gps_coordinate_rows": duplicate_rows,
        "timestamp_nondecreasing": nondecreasing,
        "nonpositive_timestamp_intervals": int(np.count_nonzero(gaps <= 0)),
        "invalid_arc_endpoint_cells": invalid_endpoints,
        "invalid_route_arc_ids": invalid_route_ids,
        "route_disconnected_steps": route_disconnected,
        "per_file": per_suffix,
    }


def weak_components(node_count: int, arcs: np.ndarray) -> tuple[int, int]:
    adjacency = [set() for _ in range(node_count)]
    for start, end in arcs:
        a, b = int(start), int(end)
        if 0 <= a < node_count and 0 <= b < node_count:
            adjacency[a].add(b)
            adjacency[b].add(a)
    isolated = sum(not neighbors for neighbors in adjacency)
    visited: set[int] = set()
    components = 0
    for node in range(node_count):
        if node in visited:
            continue
        components += 1
        stack = [node]
        visited.add(node)
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current] - visited:
                visited.add(neighbor)
                stack.append(neighbor)
    return components, isolated


def candidate_rank_sweep(trip: Trip, gps_xy: np.ndarray, arc_lines: list[LineString],
                         labels: np.ndarray, aligned_distances: np.ndarray) -> tuple[list[dict], list[dict]]:
    """Reuse the candidate search geometry and labels to compare top-k coverage."""
    tree = STRtree(arc_lines)
    obs_counts = np.zeros(len(trip.gps), dtype=int)
    totals = {k: {"candidate_rows": 0, "miss_observations": 0,
                  "alignment_over_100m_observations": 0,
                  "candidate_missing_within_100m_observations": 0} for k in TOP_K_VALUES}
    for i, xy in enumerate(gps_xy):
        point = Point(float(xy[0]), float(xy[1]))
        indices = np.asarray(tree.query(point.buffer(SEARCH_RADIUS_M)), dtype=np.int64)
        if not indices.size:
            indices = np.asarray(tree.query_nearest(point, all_matches=True), dtype=np.int64).reshape(-1)
        distances = np.asarray([point.distance(arc_lines[int(j)]) for j in indices])
        ranked = indices[np.argsort(distances)[:max(TOP_K_VALUES)]]
        n_missed_alignment = aligned_distances[i] > LABEL_RADIUS_M
        for top_k in TOP_K_VALUES:
            chosen = ranked[:top_k]
            n_missed_candidate = int(labels[i]) not in set(map(int, chosen))
            miss = bool(n_missed_alignment or n_missed_candidate)
            totals[top_k]["candidate_rows"] += len(chosen)
            totals[top_k]["miss_observations"] += int(miss)
            totals[top_k]["alignment_over_100m_observations"] += int(n_missed_alignment)
            totals[top_k]["candidate_missing_within_100m_observations"] += int((not n_missed_alignment) and n_missed_candidate)
            if top_k == 8:
                obs_counts[i] = len(chosen)
    rows = []
    for top_k in TOP_K_VALUES:
        totals_k = totals[top_k]
        rows.append({
            "partition": "", "trip_id": trip.trip_id, "top_k": top_k,
            "observations": len(trip.gps), "candidate_rows": totals_k["candidate_rows"],
            "mean_candidates_per_observation": totals_k["candidate_rows"] / max(len(trip.gps), 1),
            "miss_observations": totals_k["miss_observations"],
            "miss_rate": totals_k["miss_observations"] / max(len(trip.gps), 1),
            "alignment_over_100m_observations": totals_k["alignment_over_100m_observations"],
            "candidate_missing_within_100m_observations": totals_k["candidate_missing_within_100m_observations"],
        })
    counts = [{"trip_id": trip.trip_id, "observation_index": int(i),
               "top_k": 8, "candidate_count": int(n)} for i, n in enumerate(obs_counts)]
    return rows, counts


def reference_distance_m(gps_xy: np.ndarray, arc_lines: list[LineString], route: np.ndarray) -> np.ndarray:
    route_lines = [arc_lines[int(arc_id)] for arc_id in route]
    tree = STRtree(route_lines)
    result = np.empty(len(gps_xy), dtype=float)
    for i, xy in enumerate(gps_xy):
        point = Point(float(xy[0]), float(xy[1]))
        ids = np.asarray(tree.query(point.buffer(2000)), dtype=np.int64)
        if not ids.size:
            ids = np.asarray(tree.query_nearest(point, all_matches=True), dtype=np.int64).reshape(-1)
        result[i] = min(point.distance(route_lines[int(j)]) for j in ids)
    return result


def collect_eda(data_root: Path, output: Path) -> dict:
    partitions: dict[str, list[dict]] = {p: [] for p in PARTITIONS}
    trips: dict[tuple[str, str], Trip] = {}
    audit_rows, inventory_rows, sampling_rows, speed_rows = [], [], [], []
    trajectory_rows, extent_rows, arc_hist_rows = [], [], []
    arc_lengths_by_partition = {p: [] for p in PARTITIONS}
    route_rows, candidate_sweep_rows, candidate_balance_rows, candidate_count_rows = [], [], [], []
    aligned_rows, feature_data = [], []
    network_plot_rows, high_miss_plot_rows, high_miss_rows = [], [], []

    for partition in PARTITIONS:
        part_dir = data_root / partition
        if not part_dir.is_dir():
            raise FileNotFoundError(f"Required development partition missing: {part_dir}")
        trip_dirs = sorted(d for d in part_dir.iterdir() if d.is_dir() and d.name.isdigit())
        if len(trip_dirs) != EXPECTED_TRIPS[partition]:
            raise ValueError(f"Expected {EXPECTED_TRIPS[partition]} {partition} trips; found {len(trip_dirs)}")

        for trip_dir in trip_dirs:
            trip_id = trip_dir.name
            trip = load_trip(part_dir, trip_id)
            trips[(partition, trip_id)] = trip
            gps_xy, node_xy, arc_lines = _metric_geometry(trip)
            raw = raw_file_audit(trip_dir, trip)
            gps = trip.gps
            gaps = np.diff(gps[:, 2])
            steps = np.linalg.norm(np.diff(gps_xy, axis=0), axis=1) if len(gps) > 1 else np.asarray([])
            speed = np.divide(steps, gaps, out=np.zeros_like(steps), where=gaps > 0)
            components, isolated = weak_components(len(trip.nodes), trip.arcs)
            arc_lengths = np.asarray([line.length for line in arc_lines], dtype=float)
            route_length = float(arc_lengths[trip.route].sum()) if len(trip.route) else 0.0
            route_edges = trip.arcs[trip.route] if len(trip.route) else np.empty((0, 2), dtype=int)
            disconnected = int(np.count_nonzero(route_edges[:-1, 1] != route_edges[1:, 0])) if len(route_edges) > 1 else 0
            record = {
                "partition": partition, "trip_id": trip_id,
                "all_files_present": raw["file_missing_count"] == 0,
                "file_count_present": len(FILE_COLUMNS) - raw["file_missing_count"],
                "gps_rows": len(gps), "node_count": len(trip.nodes), "arc_count": len(trip.arcs),
                "route_arc_entries": len(trip.route), "route_length_m": route_length,
                "weak_components": components, "isolated_nodes": isolated,
                "route_disconnected_steps": disconnected,
                "max_gap_seconds": float(gaps.max()) if len(gaps) else 0.0,
                "median_gap_seconds": float(np.median(gaps)) if len(gaps) else 0.0,
                "longitude_min": float(gps[:, 0].min()), "longitude_max": float(gps[:, 0].max()),
                "latitude_min": float(gps[:, 1].min()), "latitude_max": float(gps[:, 1].max()),
                "gps_duplicate_coordinate_rows": raw["duplicate_gps_coordinate_rows"],
                "implausible_speed_steps_over_55mps": int(np.count_nonzero(speed > IMPLAUSIBLE_SPEED_MPS)),
            }
            partitions[partition].append(record)
            inventory_rows.append(record)
            audit_rows.append({k: raw[k] for k in raw if k != "per_file"} | {
                "partition": partition, "trip_id": trip_id,
                "gps_nonfinite_cells": raw["per_file"]["track"]["nonfinite"],
                "node_nonfinite_cells": raw["per_file"]["nodes"]["nonfinite"],
                "arc_nonfinite_cells": raw["per_file"]["arcs"]["nonfinite"],
                "route_nonfinite_cells": raw["per_file"]["route"]["nonfinite"],
            })
            for i, gap in enumerate(gaps, start=1):
                sampling_rows.append({"partition": partition, "trip_id": trip_id,
                                      "interval_index": i, "gap_seconds": float(gap),
                                      "exactly_1s": bool(gap == 1)})
            for i, (distance, elapsed, velocity) in enumerate(zip(steps, gaps, speed, strict=True), start=1):
                speed_rows.append({"partition": partition, "trip_id": trip_id, "interval_index": i,
                                   "step_distance_m": float(distance), "gap_seconds": float(elapsed),
                                   "speed_mps": float(velocity), "implausible_speed_flag_over_55mps": bool(velocity > IMPLAUSIBLE_SPEED_MPS)})
            extent_rows.append({**record, "step_distance_median": float(np.median(steps)) if len(steps) else 0.0,
                                "step_distance_p95": float(np.percentile(steps, 95)) if len(steps) else 0.0,
                                "step_distance_max": float(steps.max()) if len(steps) else 0.0,
                                "speed_median_mps": float(np.median(speed)) if len(speed) else 0.0,
                                "speed_p95_mps": float(np.percentile(speed, 95)) if len(speed) else 0.0,
                                "speed_max_mps": float(speed.max()) if len(speed) else 0.0,
                                "track_bbox_width_degrees": float(gps[:, 0].max() - gps[:, 0].min()),
                                "track_bbox_height_degrees": float(gps[:, 1].max() - gps[:, 1].min())})
            route_rows.append({"partition": partition, "trip_id": trip_id, "gps_rows": len(gps),
                               "route_arc_entries": len(trip.route), "route_length_m": route_length,
                               "route_length_km": route_length / 1000,
                               "route_length_per_gps_row_m": route_length / max(len(gps), 1),
                               "disconnected_route_steps": disconnected})
            arc_lengths_by_partition[partition].append(arc_lengths)

            if partition == "train" and trip_id in {"00000054", "00000033"}:
                labels, aligned = _route_occurrences(trip, arc_lines, gps_xy,
                                                      np.r_[0.0, steps] if len(steps) else np.zeros(1),
                                                      LABEL_RADIUS_M)
                ref_dist = reference_distance_m(gps_xy, arc_lines, trip.route)
                high_miss_rows.append({
                    "partition": partition, "trip_id": trip_id, "gps_rows": len(gps),
                    "network_nodes": len(trip.nodes), "network_arcs": len(trip.arcs),
                    "route_entries": len(trip.route), "route_length_m": route_length,
                    "median_alignment_distance_m": float(np.median(aligned)),
                    "p95_alignment_distance_m": float(np.percentile(aligned, 95)),
                    "max_alignment_distance_m": float(aligned.max()),
                    "share_alignment_over_100m": float(np.mean(aligned > LABEL_RADIUS_M)),
                    "median_gps_to_nearest_reference_arc_m": float(np.median(ref_dist)),
                    "p95_gps_to_nearest_reference_arc_m": float(np.percentile(ref_dist, 95)),
                    "max_gap_seconds": record["max_gap_seconds"],
                    "duplicate_gps_coordinate_rows": raw["duplicate_gps_coordinate_rows"],
                    "route_disconnected_steps": disconnected,
                })

            # Run the existing candidate feature builder once at the registered top_k=8 for feature EDA.
            candidates = build_candidate_dataset(trip, top_k=8, search_radius_m=SEARCH_RADIUS_M,
                                                  label_radius_m=LABEL_RADIUS_M)
            counts = np.bincount(candidates.observation_index, minlength=len(gps))
            if partition == "train":
                fit_rows = ~candidates.candidate_miss[candidates.observation_index]
                feature_data.append((candidates.X[fit_rows], candidates.y[fit_rows]))
                positive = int(np.count_nonzero(candidates.y[fit_rows] == 1))
                candidate_balance_rows.append({"partition": partition, "trip_id": trip_id, "top_k": 8,
                                       "candidate_rows": len(candidates.X),
                                       "candidate_rows_used_for_scorer_fit": int(fit_rows.sum()),
                                       "positive_candidate_rows_all": int(candidates.y.sum()),
                                       "positive_candidate_rows_used": positive,
                                       "positive_share_used_for_fit": positive / max(int(fit_rows.sum()), 1),
                                       "covered_observations": int((~candidates.candidate_miss).sum()),
                                       "miss_observations": int(candidates.candidate_miss.sum())})
                if trip_id in {"00000054", "00000033"}:
                    old = next(r for r in high_miss_rows if r["trip_id"] == trip_id)
                    old.update({"top8_candidate_miss_rate": float(candidates.candidate_miss.mean()),
                                "top8_candidate_miss_observations": int(candidates.candidate_miss.sum()),
                                "top8_miss_due_label_over_100m": int(np.count_nonzero(candidates.aligned_distance_m > LABEL_RADIUS_M)),
                                "top8_miss_due_candidate_absent_or_label_over_100m": int(candidates.candidate_miss.sum())})
                # Feature values are collected from covered training observations only.
            if partition == "student_test":
                candidate_balance_rows.append({"partition": partition, "trip_id": trip_id, "top_k": 8,
                                       "candidate_rows": len(candidates.X),
                                       "candidate_rows_used_for_scorer_fit": 0,
                                       "positive_candidate_rows_all": int(candidates.y.sum()),
                                       "positive_candidate_rows_used": 0,
                                       "positive_share_used_for_fit": "N/A (development test)",
                                       "covered_observations": int((~candidates.candidate_miss).sum()),
                                       "miss_observations": int(candidates.candidate_miss.sum())})

            # Candidate sweep reuses the existing projection and route-alignment implementation.
            labels, aligned_distances = _route_occurrences(
                trip, arc_lines, gps_xy, np.r_[0.0, steps] if len(steps) else np.zeros(1), LABEL_RADIUS_M
            )
            sweep, counts16 = candidate_rank_sweep(trip, gps_xy, arc_lines, labels, aligned_distances)
            for row in sweep:
                row["partition"] = partition
            candidate_sweep_rows.extend(sweep)
            if partition == "train" and trip_id in {"00000054", "00000033"}:
                top8 = next(r for r in sweep if r["top_k"] == 8)
                high_stat = next(r for r in high_miss_rows if r["trip_id"] == trip_id)
                high_stat.update({
                    "top8_candidate_miss_rate": top8["miss_rate"],
                    "top8_candidate_miss_observations": top8["miss_observations"],
                    "top8_miss_due_alignment_over_100m": top8["alignment_over_100m_observations"],
                    "top8_miss_due_candidate_rank_within_100m": top8["candidate_missing_within_100m_observations"],
                })
            candidate_count_rows.extend({"partition": partition, **row} for row in counts16)
            for i, distance in enumerate(aligned_distances):
                aligned_rows.append({"partition": partition, "trip_id": trip_id,
                                     "observation_index": i, "aligned_distance_m": float(distance),
                                     "outside_label_radius_100m": bool(distance > LABEL_RADIUS_M)})

    # Pick six training trajectories: low/median/high size, largest gap, and required trips 54/33.
    train_rows = partitions["train"]
    required = {"00000054", "00000033"}
    normal = [r for r in train_rows if r["trip_id"] not in required]
    sorted_size = sorted(normal, key=lambda r: r["gps_rows"])
    selected = [
        ("small", sorted_size[0]),
        ("medium", min(sorted_size, key=lambda r: abs(r["gps_rows"] - np.median([x["gps_rows"] for x in sorted_size])))),
        ("large", sorted_size[-1]),
        ("high_gap", max(normal, key=lambda r: r["max_gap_seconds"])),
        ("trip_54_high_miss", next(r for r in train_rows if r["trip_id"] == "00000054")),
        ("trip_33_high_miss", next(r for r in train_rows if r["trip_id"] == "00000033")),
    ]
    trajectory_roles = []
    for role, record in selected:
        trip = trips[("train", record["trip_id"])]
        gps_xy, _, _ = _metric_geometry(trip)
        trajectory_roles.append((role, trip, gps_xy))
        for i, ((lon, lat, seconds), (x, y)) in enumerate(zip(trip.gps, gps_xy, strict=True)):
            trajectory_rows.append({"partition": "train", "trip_id": trip.trip_id, "role": role,
                                    "observation_index": i, "longitude": float(lon), "latitude": float(lat),
                                    "timestamp_seconds": float(seconds), "easting_m": float(x), "northing_m": float(y)})

    # Cache named plot datasets.
    write_csv(output / "dataset_inventory.csv", inventory_rows)
    write_csv(output / "missing_invalid_by_trip.csv", audit_rows)
    write_csv(output / "sampling_intervals.csv", sampling_rows)
    write_csv(output / "step_speed_samples.csv", speed_rows)
    write_csv(output / "trajectory_grid.csv", trajectory_rows)
    write_csv(output / "geographic_extents_speed.csv", extent_rows)
    write_csv(output / "reference_route_summary.csv", route_rows)
    write_csv(output / "candidate_topk_sweep.csv", candidate_sweep_rows)
    write_csv(output / "candidate_class_balance.csv", candidate_balance_rows)
    write_csv(output / "candidate_counts_per_observation.csv", candidate_count_rows)
    write_csv(output / "aligned_distance_samples.csv", aligned_rows)
    write_csv(output / "high_miss_investigation.csv", high_miss_rows)

    # 1. Inventory plot and backing CSV.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    for partition, color in (("train", "#286A8E"), ("student_test", "#E88A2D")):
        values = partitions[partition]
        axes[0].hist([r["gps_rows"] for r in values], bins=18, alpha=.65, label=partition, color=color)
        axes[1].hist([r["node_count"] for r in values], bins=18, alpha=.65, label=partition, color=color)
        axes[2].hist([r["arc_count"] for r in values], bins=18, alpha=.65, label=partition, color=color)
    axes[0].set_title("GPS rows per trip")
    axes[1].set_title("Road nodes per trip")
    axes[2].set_title("Directed arcs per trip")
    for ax in axes:
        ax.set_yscale("log")
        ax.set_ylabel("Trip count (log scale)")
        ax.legend(frameon=False)
        ax.grid(alpha=.2)
    fig.suptitle("Dataset inventory - partitions: train (80 trips) and student_test (10 trips)")
    fig.tight_layout()
    fig.savefig(output / "01_dataset_inventory.png", dpi=200)
    plt.close(fig)

    # 2. Raw missing/invalid values table (per-trip CSV is the backing table).
    fig, ax = plt.subplots(figsize=(12, 3.4))
    ax.axis("off")
    summary_rows = []
    for p in PARTITIONS:
        rows = [r for r in audit_rows if r["partition"] == p]
        summary_rows.append([p, len(rows), sum(r["missing"] for r in rows), sum(r["nonfinite"] for r in rows),
                             sum(r["gps_coordinate_out_of_range_rows"] + r["node_coordinate_out_of_range_rows"] for r in rows),
                             sum(r["duplicate_gps_coordinate_rows"] for r in rows),
                             sum(r["timestamp_nondecreasing"] is True for r in rows)])
    table = ax.table(cellText=summary_rows,
                     colLabels=["Partition", "Trips", "Missing cells", "NaN/Inf cells", "Out-of-range coordinate rows", "Duplicate GPS rows", "Nondecreasing time trips"],
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.8)
    fig.suptitle("Missing and invalid values - train and student_test")
    fig.text(.02, .06, "No missing/NaN/Inf/malformed/out-of-range values found. Duplicate coordinates are counted, not dropped.", fontsize=9)
    fig.tight_layout(rect=(0, .13, 1, .9))
    fig.savefig(output / "02_missing_invalid_values.png", dpi=200)
    plt.close(fig)

    # 3. Gap histogram and largest-gap trip table.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    gap_values = {p: np.asarray([r["gap_seconds"] for r in sampling_rows if r["partition"] == p]) for p in PARTITIONS}
    bins = np.geomspace(1, max(2, max(v.max() for v in gap_values.values())), 45)
    for p, color in (("train", "#286A8E"), ("student_test", "#E88A2D")):
        axes[0].hist(gap_values[p], bins=bins, alpha=.65, label=p, color=color, log=True)
        share = float(np.mean(gap_values[p] == 1))
        axes[0].plot([], [], color=color, label=f"{p}: exactly 1 s = {share:.2%}")
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("Interval between consecutive GPS observations (s, log scale)")
    axes[0].set_ylabel("Interval count (log scale)"); axes[0].legend(fontsize=8, frameon=False)
    largest = sorted(inventory_rows, key=lambda r: r["max_gap_seconds"], reverse=True)[:8]
    axes[1].barh([f"{r['partition']} {r['trip_id']}" for r in largest][::-1],
                 [r["max_gap_seconds"] for r in largest][::-1], color="#7A6FAC")
    axes[1].set_xscale("log"); axes[1].set_xlabel("Largest gap in trip (s, log scale)")
    axes[1].set_title("Trips with the largest gaps")
    fig.suptitle("GPS sampling intervals - partitions: train and student_test")
    fig.tight_layout()
    fig.savefig(output / "03_sampling_intervals.png", dpi=200)
    plt.close(fig)
    write_csv(output / "largest_gap_trips.csv", [{"partition": r["partition"], "trip_id": r["trip_id"],
                                                  "gps_rows": r["gps_rows"], "max_gap_seconds": r["max_gap_seconds"]}
                                                  for r in largest])

    # 4. Six GPS trajectories in local projected metres.
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    for ax, (role, trip, xy) in zip(axes.ravel(), trajectory_roles, strict=True):
        ax.plot(xy[:, 0], xy[:, 1], color="#286A8E", linewidth=.65)
        ax.scatter(xy[::max(1, len(xy)//300), 0], xy[::max(1, len(xy)//300), 1], s=3, color="#E88A2D", alpha=.55)
        ax.set_title(f"train trip {trip.trip_id} - {role} ({len(trip.gps):,} GPS rows)", fontsize=9)
        ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)"); ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=.15)
    fig.suptitle("GPS trajectories in projected metres - partition: train")
    fig.tight_layout()
    fig.savefig(output / "04_representative_trajectories.png", dpi=200)
    plt.close(fig)

    # 5. Geographic bounds, projected step distances and speed distributions.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for p, marker, color in (("train", "o", "#286A8E"), ("student_test", "^", "#E88A2D")):
        r = [x for x in extent_rows if x["partition"] == p]
        axes[0].scatter([x["longitude_min"] for x in r], [x["latitude_min"] for x in r],
                        marker=marker, color=color, alpha=.7, label=p)
    axes[0].set_title("Trip southwest bounds (degrees)"); axes[0].set_xlabel("Longitude"); axes[0].set_ylabel("Latitude"); axes[0].legend(frameon=False)
    for p, color in (("train", "#286A8E"), ("student_test", "#E88A2D")):
        vals = [r["step_distance_m"] for r in speed_rows if r["partition"] == p and r["step_distance_m"] > 0]
        axes[1].hist(vals, bins=np.geomspace(max(min(vals), .01), max(vals), 45), alpha=.6, color=color, label=p, log=True)
        vals_speed = [r["speed_mps"] for r in speed_rows if r["partition"] == p and r["speed_mps"] > 0]
        axes[2].hist(vals_speed, bins=np.geomspace(max(min(vals_speed), .01), max(vals_speed), 45), alpha=.6, color=color, label=p, log=True)
    axes[1].set_xscale("log"); axes[1].set_title("GPS step distance (projected m)"); axes[1].set_xlabel("Metres (log scale)")
    axes[2].set_xscale("log"); axes[2].axvline(IMPLAUSIBLE_SPEED_MPS, color="#A33A3A", linestyle="--", label="screening threshold 55 m/s")
    axes[2].set_title("Implied GPS speed"); axes[2].set_xlabel("m/s (log scale)")
    for ax in axes[1:]: ax.legend(frameon=False, fontsize=8); ax.set_ylabel("Interval count (log scale)")
    fig.suptitle("Geographic extent and motion diagnostics - partitions: train and student_test")
    fig.text(.02, .015, "Distance/speed use per-trip projected metres. 55 m/s is a screening flag, not a universal vehicle limit.", fontsize=8)
    fig.tight_layout(rect=(0, .07, 1, .92))
    fig.savefig(output / "05_geographic_extent_and_speed.png", dpi=200)
    plt.close(fig)

    # 6. Road network summary, arc lengths, and two real network overlays.
    for p in PARTITIONS:
        lengths = np.concatenate(arc_lengths_by_partition[p])
        bins = np.geomspace(max(float(lengths[lengths > 0].min()), .01), float(lengths.max()), 55)
        counts, edges = np.histogram(lengths, bins=bins)
        for i, (lo, hi, count) in enumerate(zip(edges[:-1], edges[1:], counts, strict=True)):
            arc_hist_rows.append({"partition": p, "bin_low_m": float(lo), "bin_high_m": float(hi),
                                  "arc_count": int(count), "total_arcs": int(len(lengths)),
                                  "underflow_count": 0, "overflow_count": 0})
    network_summary_rows = []
    for p in PARTITIONS:
        for r in [x for x in inventory_rows if x["partition"] == p]:
            network_summary_rows.append({"partition": p, "trip_id": r["trip_id"], "node_count": r["node_count"],
                                         "arc_count": r["arc_count"], "weak_components": r["weak_components"],
                                         "isolated_nodes": r["isolated_nodes"]})
    write_csv(output / "road_network_summary.csv", network_summary_rows)
    write_csv(output / "arc_length_histogram.csv", arc_hist_rows)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for p, color in (("train", "#286A8E"), ("student_test", "#E88A2D")):
        rows = [r for r in arc_hist_rows if r["partition"] == p]
        edges = [rows[0]["bin_low_m"], *[r["bin_high_m"] for r in rows]]
        ax.stairs([r["arc_count"] for r in rows], edges, color=color, label=p)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Projected arc length (m, log scale)"); ax.set_ylabel("Arc count (log scale)")
    ax.set_title("Road arc length distribution - partitions: train and student_test")
    ax.grid(alpha=.2); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(output / "12_arc_length_distribution.png", dpi=200); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 5))
    component_rows = sorted(network_summary_rows, key=lambda r: (r["partition"], r["trip_id"]))
    labels = [f"{r['partition']} {r['trip_id']}" for r in component_rows]
    colors = ["#286A8E" if r["partition"] == "train" else "#E88A2D" for r in component_rows]
    ax.bar(range(len(component_rows)), [r["weak_components"] for r in component_rows], color=colors)
    ax.set_xticks(range(len(component_rows)), labels, rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("Weakly connected components")
    ax.set_title("Weak components per trip - partitions: train and student_test")
    ax.grid(axis="y", alpha=.2); fig.tight_layout()
    fig.savefig(output / "13_network_connectivity.png", dpi=200); plt.close(fig)
    overlay_trip_ids = ["00000034", "00000054"]
    overlay_fig, overlay_axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, tid in zip(overlay_axes, overlay_trip_ids, strict=True):
        trip = trips[("train", tid)]
        gps_xy, node_xy, arc_lines = _metric_geometry(trip)
        # This padded envelope is only for selecting display roads; metrics use full coordinates.
        corridor = gps_neighborhood(gps_xy)
        route_set = set(map(int, trip.route))
        selected_arcs = corridor_arcs(arc_lines, corridor, route_set, limit=800)
        local_segments, reference_segments = [], []
        for aid in selected_arcs:
            a, b = trip.arcs[aid]
            start, end = node_xy[int(a)], node_xy[int(b)]
            is_ref = aid in route_set
            (reference_segments if is_ref else local_segments).append([start, end])
            network_plot_rows.append({"partition": "train", "trip_id": tid, "layer": "reference_arc" if is_ref else "local_network",
                                     "record_index": aid, "arc_id": aid, "x1_m": float(start[0]), "y1_m": float(start[1]),
                                     "x2_m": float(end[0]), "y2_m": float(end[1])})
        if local_segments:
            ax.add_collection(LineCollection(local_segments, colors="#C7CDD1", linewidths=.35,
                                             alpha=.4, zorder=1))
        if reference_segments:
            ax.add_collection(LineCollection(reference_segments, colors="#2E8B57", linewidths=1.15,
                                             alpha=.9, zorder=2))
        ax.plot(gps_xy[:, 0], gps_xy[:, 1], color="#2878B5", linewidth=.7, alpha=.9, label="GPS track", zorder=3)
        for obs_i, (x, y) in enumerate(gps_xy):
            network_plot_rows.append({"partition": "train", "trip_id": tid, "layer": "gps_observation",
                                     "record_index": obs_i, "x1_m": float(x), "y1_m": float(y),
                                     "x2_m": "", "y2_m": ""})
        for route_pos, aid in enumerate(trip.route):
            a, b = trip.arcs[int(aid)]
            start, end = node_xy[int(a)], node_xy[int(b)]
            network_plot_rows.append({"partition": "train", "trip_id": tid, "layer": "ordered_reference_route",
                                     "record_index": route_pos, "x1_m": float(start[0]), "y1_m": float(start[1]),
                                     "x2_m": float(end[0]), "y2_m": float(end[1])})
        ax.set_title(f"train trip {tid} - GPS + sampled local network + reference route")
        ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)"); ax.set_aspect("equal", adjustable="datalim"); ax.grid(alpha=.15)
        handles = [Line2D([0],[0],color="#C7CDD1",lw=2,label="Local road network"),
                   Line2D([0],[0],color="#2E8B57",lw=2,label="Reference route arcs"),
                   Line2D([0],[0],color="#2878B5",lw=2,label="GPS track")]
        ax.legend(handles=handles, fontsize=8, frameon=True)
    write_csv(output / "network_overlay_plot_data.csv", network_plot_rows)
    overlay_fig.suptitle("Road network overlays - partition: train")
    overlay_fig.tight_layout()
    overlay_fig.savefig(output / "06_road_network_and_overlays.png", dpi=200)
    plt.close(overlay_fig)

    # 7. Route length versus GPS row count, plus disconnected route steps.
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for p, marker, color in (("train", "o", "#286A8E"), ("student_test", "^", "#E88A2D")):
        r = [x for x in route_rows if x["partition"] == p]
        axes[0].scatter([x["gps_rows"] for x in r], [x["route_length_km"] for x in r], marker=marker, color=color, label=p, alpha=.75)
    axes[0].set_xlabel("GPS rows"); axes[0].set_ylabel("Reference route length (km, projected)"); axes[0].set_title("Ordered route length vs GPS samples"); axes[0].legend(frameon=False)
    nonzero = [r for r in route_rows if r["disconnected_route_steps"] > 0]
    axes[1].bar([f"{r['partition']} {r['trip_id']}" for r in nonzero], [r["disconnected_route_steps"] for r in nonzero], color="#A75D72")
    axes[1].tick_params(axis="x", labelrotation=65); axes[1].set_ylabel("Disconnected adjacent route arc pairs")
    axes[1].set_title(f"Trips with disconnections ({len(nonzero)} trips)")
    fig.suptitle("Reference route diagnostics - partitions: train and student_test")
    fig.tight_layout()
    fig.savefig(output / "07_reference_route_diagnostics.png", dpi=200)
    plt.close(fig)

    # 8. Candidate coverage, count, alignment distance, and class-balance evidence.
    sweep_rows = candidate_sweep_rows
    balance_rows = candidate_balance_rows
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for p, color in (("train", "#286A8E"), ("student_test", "#E88A2D")):
        rows = sorted([r for r in sweep_rows if r["partition"] == p], key=lambda r:r["top_k"])
        ks = sorted(set(r["top_k"] for r in rows))
        miss = [sum(r["miss_observations"] for r in rows if r["top_k"] == k) / sum(r["observations"] for r in rows if r["top_k"] == k) for k in ks]
        axes[0,0].plot(ks, miss, marker="o", color=color, label=p)
        axes[0,0].set_xticks(ks); axes[0,0].set_xlabel("top_k"); axes[0,0].set_ylabel("Candidate miss rate"); axes[0,0].set_title("Miss rate vs candidate count"); axes[0,0].legend(frameon=False)
        distances = [r["aligned_distance_m"] for r in aligned_rows if r["partition"] == p]
        axes[0,1].hist(distances, bins=np.linspace(0, max(200, np.percentile(distances, 99)), 50), alpha=.6, color=color, label=p)
        count_vals = [r["candidate_count"] for r in candidate_count_rows if r["partition"] == p]
        axes[1,0].hist(count_vals, bins=np.arange(0, 18)-.5, alpha=.6, color=color, label=p)
    axes[0,1].axvline(LABEL_RADIUS_M, color="#A33A3A", linestyle="--", label="label radius 100 m")
    axes[0,1].set_xlabel("Aligned GPS-to-route distance (m)"); axes[0,1].set_ylabel("Observation count"); axes[0,1].set_title("Route alignment distances"); axes[0,1].legend(frameon=False, fontsize=8)
    axes[1,0].set_xlabel("Candidates per observation (top_k=8)"); axes[1,0].set_ylabel("Observation count"); axes[1,0].set_title("Candidate list sizes"); axes[1,0].legend(frameon=False)
    train_balance = [r for r in balance_rows if r["partition"] == "train"]
    names = [r["trip_id"] for r in train_balance]
    shares = [float(r["positive_share_used_for_fit"]) for r in train_balance]
    axes[1,1].hist(shares, bins=25, color="#6B9B64", alpha=.8)
    axes[1,1].set_xlabel("Positive candidate rows / fit candidate rows"); axes[1,1].set_ylabel("Training trip count"); axes[1,1].set_title(f"Positive share by trip (train only, {len(names)} trips)")
    fig.suptitle("Candidate and alignment diagnostics - partitions: train and student_test; class balance: train only")
    fig.tight_layout()
    fig.savefig(output / "08_candidate_alignment_diagnostics.png", dpi=200)
    plt.close(fig)

    # 9. High-miss trips - compare alignment distance with candidate rank failures.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    high_stats = []
    for ax, tid in zip(axes, ("00000054", "00000033"), strict=True):
        trip = trips[("train", tid)]
        gps_xy, node_xy, arc_lines = _metric_geometry(trip)
        labels, distances = _route_occurrences(trip, arc_lines, gps_xy,
                                                np.r_[0.0, np.linalg.norm(np.diff(gps_xy, axis=0), axis=1)], LABEL_RADIUS_M)
        corridor = gps_neighborhood(gps_xy); route_set = set(map(int, trip.route))
        selected_arcs = corridor_arcs(arc_lines, corridor, route_set, limit=800)
        local_segments, reference_segments = [], []
        for aid in sorted(selected_arcs):
            a, b = trip.arcs[aid]; start, end = node_xy[int(a)], node_xy[int(b)]
            (reference_segments if aid in route_set else local_segments).append([start, end])
        if local_segments:
            ax.add_collection(LineCollection(local_segments, colors="#C8CDD0", linewidths=.35,
                                             alpha=.55, zorder=1))
        if reference_segments:
            ax.add_collection(LineCollection(reference_segments, colors="#2E8B57", linewidths=1.1,
                                             alpha=.9, zorder=2))
        ax.plot(gps_xy[:,0], gps_xy[:,1], color="#2878B5", linewidth=.8, label="GPS")
        route_x, route_y = [], []
        for aid in trip.route:
            a, b = trip.arcs[int(aid)]
            route_x.extend([node_xy[int(a), 0], node_xy[int(b), 0], np.nan])
            route_y.extend([node_xy[int(a), 1], node_xy[int(b), 1], np.nan])
        if route_x:
            ax.plot(route_x, route_y, color="#2E8B57", linewidth=1.2, label="Reference route")
        for obs_i, (x, y) in enumerate(gps_xy):
            high_miss_plot_rows.append({"partition": "train", "trip_id": tid, "layer": "gps_observation",
                                        "record_index": obs_i, "x1_m": float(x), "y1_m": float(y), "x2_m": "", "y2_m": ""})
        for route_pos, aid in enumerate(trip.route):
            a, b = trip.arcs[int(aid)]
            start, end = node_xy[int(a)], node_xy[int(b)]
            high_miss_plot_rows.append({"partition": "train", "trip_id": tid, "layer": "ordered_reference_route",
                                        "record_index": route_pos, "x1_m": float(start[0]), "y1_m": float(start[1]),
                                        "x2_m": float(end[0]), "y2_m": float(end[1])})
        for aid in sorted(selected_arcs):
            a, b = trip.arcs[aid]
            start, end = node_xy[int(a)], node_xy[int(b)]
            high_miss_plot_rows.append({"partition": "train", "trip_id": tid,
                                        "layer": "reference_arc_network" if aid in route_set else "local_road_network",
                                        "record_index": aid, "x1_m": float(start[0]), "y1_m": float(start[1]),
                                        "x2_m": float(end[0]), "y2_m": float(end[1])})
        ax.set_title(f"train trip {tid}: GPS, local network envelope, reference route")
        ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)"); ax.set_aspect("equal", adjustable="datalim"); ax.legend(fontsize=8, frameon=True); ax.grid(alpha=.15)
        stat = next(r for r in high_miss_rows if r["trip_id"] == tid)
        stat["alignment_over_100m_count"] = int(np.count_nonzero(distances > LABEL_RADIUS_M))
        stat["gps_route_median_distance_m"] = float(np.median(distances))
        high_stats.append(stat)
    fig.suptitle("High candidate-miss investigation - partition: train (trips 00000054 and 00000033)")
    fig.tight_layout()
    fig.savefig(output / "09_high_miss_trips.png", dpi=200)
    plt.close(fig)
    write_csv(output / "high_miss_investigation.csv", high_stats)
    write_csv(output / "high_miss_plot_data.csv", high_miss_plot_rows)

    # 10. Training-only feature distributions by class and correlation matrix.
    X_all = np.vstack([values for values, _ in feature_data])
    y_all = np.concatenate([labels for _, labels in feature_data]).astype(int)
    feature_hist_rows = []
    fig, axes = plt.subplots(2, 4, figsize=(15, 8.5))
    for j, name in enumerate(FEATURE_NAMES):
        ax = axes.ravel()[j]
        values = X_all[:, j]
        lo, hi = np.quantile(values, [0.01, 0.99])
        if not hi > lo:
            hi = lo + 1e-6
        edges = np.linspace(lo, hi, 41)
        for cls, label, color in ((1, "positive", "#238B6E"), (0, "negative", "#A7B8C4")):
            vals = values[y_all == cls]
            counts, bins = np.histogram(vals, bins=edges)
            under = int(np.count_nonzero(vals < lo)); over = int(np.count_nonzero(vals > hi))
            for bi, (left, right, count) in enumerate(zip(bins[:-1], bins[1:], counts, strict=True)):
                feature_hist_rows.append({"partition": "train", "feature": name, "class": label,
                                          "bin_low": float(left), "bin_high": float(right), "count": int(count),
                                          "under_1pct_count": under if bi == 0 else 0,
                                          "over_99pct_count": over if bi == len(counts)-1 else 0,
                                          "class_rows_total": int(len(vals))})
            ax.hist(vals, bins=edges, density=True, histtype="step", linewidth=1.5, label=label, color=color)
        ax.set_title(FEATURE_LABELS[name], fontsize=9); ax.grid(alpha=.15); ax.legend(fontsize=7, frameon=False)
    axes.ravel()[-1].axis("off")
    fig.suptitle("Candidate feature distributions by class - partition: train only; covered observations used for fitting")
    fig.tight_layout()
    fig.savefig(output / "10_training_feature_distributions.png", dpi=200)
    plt.close(fig)
    write_csv(output / "feature_distribution_bins.csv", feature_hist_rows)
    corr = np.corrcoef(X_all, rowvar=False)
    write_csv(output / "feature_correlation.csv", [
        {"partition": "train", "feature": FEATURE_NAMES[i],
         **{FEATURE_NAMES[j]: float(corr[i,j]) for j in range(len(FEATURE_NAMES))}}
        for i in range(len(FEATURE_NAMES))
    ])
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(FEATURE_NAMES)), [FEATURE_LABELS[x] for x in FEATURE_NAMES], rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(len(FEATURE_NAMES)), [FEATURE_LABELS[x] for x in FEATURE_NAMES], fontsize=8)
    for i in range(len(FEATURE_NAMES)):
        for j in range(len(FEATURE_NAMES)):
            ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    fig.suptitle("Candidate feature correlation - partition: train only; covered fit rows")
    fig.tight_layout()
    fig.savefig(output / "11_training_feature_correlation.png", dpi=200)
    plt.close(fig)

    # Folium maps for normal train, trip 54, and one student_test trip.
    map_paths = write_requested_maps(data_root, output)

    summary = {
        "partitions_read": {p: len(partitions[p]) for p in PARTITIONS},
        "evaluator_partition_read": False,
        "raw_numeric_cells": {p: sum(r["numeric"] for r in [a for a in audit_rows if a["partition"] == p]) for p in PARTITIONS},
        "plots": sorted(p.name for p in output.glob("*.png")),
        "interactive_maps": [str(p) for p in map_paths],
    }
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"EDA outputs: {output}")
    for path in map_paths:
        print(f"map {path} ({path.stat().st_size / 1024:.1f} KB)")


def write_requested_maps(data_root: Path, output: Path) -> list[Path]:
    """Generate the three requested map examples, optionally adding model predictions."""
    maps_dir = output / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path("artifacts/final/candidate_scorer_tuned.joblib")
    decoder_path = Path("outputs/experiments/selected_decoder.json")
    model = decoder = None
    if model_path.is_file() and decoder_path.is_file():
        import joblib
        model = joblib.load(model_path)["model"]
        decoder = json.loads(decoder_path.read_text(encoding="utf-8"))["parameters"]
    map_specs = [("train", "00000034", "normal"), ("train", "00000054", "high_miss"),
                 ("student_test", "00000056", "student_test")]
    map_paths = []
    for partition, trip_id, label in map_specs:
        trip = load_trip(data_root / partition, trip_id, include_route=True)
        predicted = None
        if model is not None:
            # Prediction inputs omit route labels; reference routes are only map display layers.
            inference_trip = load_trip(data_root / partition, trip_id, include_route=False)
            dataset = build_candidate_dataset(inference_trip, align_labels=False)
            scores = model.predict_proba(dataset.X)[:, 1]
            predicted = decode_viterbi(inference_trip, dataset, scores, **decoder)
        map_path = write_folium_map(trip, maps_dir / f"{partition}-{trip_id}-{label}.html", partition, predicted)
        if map_path.stat().st_size >= 5 * 1024 * 1024:
            raise RuntimeError(f"Folium map exceeded 5 MB: {map_path}")
        map_paths.append(map_path)
    return map_paths


def write_folium_map(trip: Trip, output_path: Path, partition: str, predicted_arc_ids=None) -> Path:
    """Create a layered map with local road arcs and sampled GPS point markers."""
    gps_xy, node_xy, arc_lines = _metric_geometry(trip)
    route_ids = set(map(int, trip.route))
    corridor = gps_neighborhood(gps_xy)
    selected_arcs = corridor_arcs(arc_lines, corridor, route_ids, limit=500)
    center = [float(np.mean(trip.gps[:, 1])), float(np.mean(trip.gps[:, 0]))]
    fmap = folium.Map(location=center, zoom_start=13, control_scale=True, tiles="OpenStreetMap")
    gps_layer = folium.FeatureGroup(name="GPS observations / track", show=True)
    # Full observed path remains visible as a polyline; only marker glyphs are sampled to keep the HTML small.
    folium.PolyLine([[float(lat), float(lon)] for lon, lat in trip.gps[:, :2]],
                    color="#2878B5", weight=2, opacity=.8, tooltip="GPS track").add_to(gps_layer)
    stride = max(1, int(np.ceil(len(trip.gps) / 500)))
    for lon, lat, seconds in trip.gps[::stride]:
        folium.CircleMarker([float(lat), float(lon)], radius=2, color="#2878B5", fill=True,
                            fill_opacity=.7, tooltip=f"GPS t={seconds:g}s").add_to(gps_layer)
    gps_layer.add_to(fmap)

    network_layer = folium.FeatureGroup(name="Sampled local road network", show=False)
    for aid in selected_arcs:
        a, b = trip.arcs[aid]
        start, end = trip.nodes[int(a)], trip.nodes[int(b)]
        folium.PolyLine([[float(start[1]), float(start[0])], [float(end[1]), float(end[0])]],
                        color="#AEB7BE", weight=1, opacity=.55, tooltip=f"road arc {aid}").add_to(network_layer)
    network_layer.add_to(fmap)

    route_layer = folium.FeatureGroup(name="Reference route", show=True)
    ref_coords = []
    previous_end = None
    for aid in trip.route:
        a, b = trip.arcs[int(aid)]
        start, end = trip.nodes[int(a)], trip.nodes[int(b)]
        if previous_end is not None and previous_end != int(a):
            if ref_coords:
                folium.PolyLine(ref_coords, color="#218739", weight=4, opacity=.9,
                                tooltip="Reference route").add_to(route_layer)
            ref_coords = []
        if not ref_coords:
            ref_coords.append([float(start[1]), float(start[0])])
        ref_coords.append([float(end[1]), float(end[0])])
        previous_end = int(b)
    if ref_coords:
        folium.PolyLine(ref_coords, color="#218739", weight=4, opacity=.9,
                        tooltip="Reference route").add_to(route_layer)
    route_layer.add_to(fmap)

    if predicted_arc_ids is not None:
        pred_layer = folium.FeatureGroup(name="Predicted route", show=True)
        coords = []
        geo_segments = []
        previous_end = None
        for aid in predicted_arc_ids:
            start_id, end_id = trip.arcs[int(aid)]
            start, end = trip.nodes[int(start_id)], trip.nodes[int(end_id)]
            if previous_end is None or previous_end != int(start_id):
                if coords:
                    geo_segments.append([[lon, lat] for lat, lon in coords])
                coords = [[float(start[1]), float(start[0])]]
            coords.append([float(end[1]), float(end[0])])
            previous_end = int(end_id)
        if coords:
            geo_segments.append([[lon, lat] for lat, lon in coords])
        if geo_segments:
            folium.GeoJson(
                {"type": "Feature", "properties": {},
                 "geometry": {"type": "MultiLineString", "coordinates": geo_segments}},
                name="Predicted route",
                style_function=lambda _: {"color": "#E67E22", "weight": 4, "opacity": .9},
                tooltip="Predicted route",
            ).add_to(pred_layer)
        pred_layer.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    legend = f"""<div style='position:fixed;top:12px;left:50px;z-index:9999;background:white;padding:8px 10px;border:1px solid #888;font-size:12px'>
    <b>PathSnap EDA | Partition: {partition} | Trip: {trip.trip_id}</b><br>
    <span style='color:#2878B5'>━━</span> GPS track / sampled points<br><span style='color:#AEB7BE'>━━</span> Sampled local road network<br><span style='color:#218739'>━━</span> Reference route<br><span style='color:#E67E22'>━━</span> Predicted route (if available)
    </div>"""
    fmap.get_root().html.add_child(folium.Element(legend))
    bounds = [[float(trip.gps[:,1].min()), float(trip.gps[:,0].min())],
              [float(trip.gps[:,1].max()), float(trip.gps[:,0].max())]]
    fmap.fit_bounds(bounds, padding=(15, 15))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(output_path))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.getenv("PATHSNAP_DATA_ROOT"),
                        help="Supplied Student folder containing train/ and student_test/ only")
    parser.add_argument("--output-dir", default="outputs/eda")
    parser.add_argument("--maps-only", action="store_true",
                        help="Regenerate only the three Folium maps from train/student_test")
    args = parser.parse_args()
    if not args.data_root:
        parser.error("Set PATHSNAP_DATA_ROOT or pass --data-root")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root).expanduser()
    if args.maps_only:
        paths = write_requested_maps(data_root, output)
        inventory = list(csv.DictReader((output / "dataset_inventory.csv").open(encoding="utf-8")))
        audit = list(csv.DictReader((output / "missing_invalid_by_trip.csv").open(encoding="utf-8")))
        summary = {
            "partitions_read": {p: sum(row["partition"] == p for row in inventory) for p in PARTITIONS},
            "evaluator_partition_read": False,
            "raw_numeric_cells": {p: sum(int(row["numeric"]) for row in audit if row["partition"] == p)
                                   for p in PARTITIONS},
            "plots": sorted(p.name for p in output.glob("*.png")),
            "interactive_maps": [str(path) for path in paths],
        }
        (output / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        for path in paths:
            print(f"map {path} ({path.stat().st_size / 1024:.1f} KB)")
    else:
        collect_eda(data_root, output)


if __name__ == "__main__":
    main()
