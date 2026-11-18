"""Trip-local candidate generation and weak reference-route alignment.

The reference route is an ordered arc sequence, not an observation-level label array. This module
aligns GPS observations monotonically to route-arc occurrences using projected geometry and path
progress, while retaining alignment diagnostics for review.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyproj import CRS, Transformer
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

from pathsnap.data.reader import Trip

FEATURE_NAMES = (
    "cross_track_distance_m",
    "heading_cosine",
    "along_arc_fraction",
    "arc_length_m",
    "gps_speed_mps",
    "gps_gap_seconds",
    "gps_step_distance_m",
)


@dataclass
class CandidateDataset:
    trip_id: str
    X: np.ndarray
    y: np.ndarray
    observation_index: np.ndarray
    arc_ids: np.ndarray
    reference_arc_ids: np.ndarray
    candidate_miss: np.ndarray
    aligned_distance_m: np.ndarray
    feature_names: tuple[str, ...] = FEATURE_NAMES


def _projector(gps: np.ndarray) -> Transformer:
    lon0 = float(np.median(gps[:, 0]))
    lat0 = float(np.median(gps[:, 1]))
    zone = int((lon0 + 180.0) // 6.0) + 1
    epsg = (32600 if lat0 >= 0 else 32700) + zone
    return Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(epsg), always_xy=True)


def _metric_geometry(trip: Trip):
    transformer = _projector(trip.gps)
    gps_x, gps_y = transformer.transform(trip.gps[:, 0], trip.gps[:, 1])
    node_x, node_y = transformer.transform(trip.nodes[:, 0], trip.nodes[:, 1])
    gps_xy = np.column_stack((gps_x, gps_y))
    node_xy = np.column_stack((node_x, node_y))
    arc_lines = [LineString((node_xy[int(a)], node_xy[int(b)])) for a, b in trip.arcs]
    return gps_xy, node_xy, arc_lines


def _route_occurrences(trip: Trip, arc_lines: list[LineString], gps_xy: np.ndarray,
                       step_distance: np.ndarray, search_radius_m: float) -> tuple[np.ndarray, np.ndarray]:
    if trip.route.size == 0:
        raise ValueError(f"Trip {trip.trip_id} has no reference route")
    route_ids = trip.route.astype(np.int64)
    route_lines = [arc_lines[int(arc_id)] for arc_id in route_ids]
    tree = STRtree(route_lines)
    route_lengths = np.asarray([line.length for line in route_lines], dtype=float)
    route_cumulative = np.r_[0.0, np.cumsum(route_lengths)]
    candidates: list[list[tuple[int, float]]] = []
    for i, xy in enumerate(gps_xy):
        point = Point(float(xy[0]), float(xy[1]))
        indices = np.asarray(tree.query(point.buffer(search_radius_m)), dtype=np.int64)
        if not indices.size:
            nearest = tree.query_nearest(point, all_matches=True)
            indices = np.asarray(nearest, dtype=np.int64).reshape(-1)
        distances = [(int(j), float(point.distance(route_lines[int(j)]))) for j in indices]
        # Keep the geometrically nearest occurrences. Raw GPS cumulative distance is noisy at 1 Hz
        # and must not be used to force a progress estimate across the reference sequence.
        candidates.append(sorted(distances, key=lambda item: item[1])[:12])

    # Viterbi alignment over sparse route-occurrence candidates. States may stay on an arc or move
    # forward; movement cost compares route progress with observed GPS displacement.
    costs: list[np.ndarray] = []
    backpointers: list[np.ndarray] = []
    first = candidates[0]
    costs.append(np.asarray([d for _, d in first], dtype=float))
    backpointers.append(np.full(len(first), -1, dtype=np.int32))
    for i in range(1, len(candidates)):
        previous, current = candidates[i - 1], candidates[i]
        prev_cost = costs[-1]
        this_cost = np.full(len(current), np.inf, dtype=float)
        back = np.full(len(current), -1, dtype=np.int32)
        # Cap isolated GPS jumps; timestamp gaps widen the transition tolerance separately.
        observed = min(max(float(step_distance[i]), 0.0), 100.0)
        scale = 12.0 + 0.35 * observed + 4.0 * max(float(trip.gps[i, 2] - trip.gps[i-1, 2]) - 1.0, 0.0)
        for cj, (current_idx, emission) in enumerate(current):
            for pj, (previous_idx, _) in enumerate(previous):
                if current_idx < previous_idx:
                    continue
                route_progress = float(route_cumulative[current_idx] - route_cumulative[previous_idx])
                transition = abs(route_progress - observed) / scale
                value = prev_cost[pj] + transition + emission / max(search_radius_m, 1.0)
                if value < this_cost[cj]:
                    this_cost[cj], back[cj] = value, pj
        if not np.isfinite(this_cost).any():
            # A route label can be geometrically discontinuous; recover with the best spatial fit
            # and mark the final route continuity metrics rather than dropping the observation.
            for cj, (_, emission) in enumerate(current):
                pj = int(np.argmin(prev_cost))
                this_cost[cj], back[cj] = prev_cost[pj] + emission / max(search_radius_m, 1.0), pj
        costs.append(this_cost)
        backpointers.append(back)

    selected = np.empty(len(candidates), dtype=np.int32)
    selected[-1] = int(np.argmin(costs[-1]))
    for i in range(len(candidates) - 1, 0, -1):
        selected[i - 1] = backpointers[i][selected[i]]
    aligned_occurrence = np.asarray([candidates[i][selected[i]][0] for i in range(len(candidates))])
    aligned_arc_ids = route_ids[aligned_occurrence]
    aligned_distances = np.asarray([candidates[i][selected[i]][1] for i in range(len(candidates))])
    return aligned_arc_ids, aligned_distances


def align_reference_route(trip: Trip, *, search_radius_m: float = 100.0) -> tuple[np.ndarray, np.ndarray]:
    """Align a supplied route to GPS observations for training or post-prediction evaluation."""
    gps_xy, _, arc_lines = _metric_geometry(trip)
    step = np.zeros(len(gps_xy), dtype=float)
    if len(gps_xy) > 1:
        step[1:] = np.linalg.norm(np.diff(gps_xy, axis=0), axis=1)
    return _route_occurrences(trip, arc_lines, gps_xy, step, search_radius_m)


def build_candidate_dataset(trip: Trip, *, top_k: int = 8, search_radius_m: float = 60.0,
                            label_radius_m: float = 100.0,
                            align_labels: bool = True) -> CandidateDataset:
    """Create candidate features and optional route-aligned binary labels for one trip.

    Use ``align_labels=False`` for inference so reference routes cannot enter candidate features or
    decoding. Evaluation code may align labels after predictions have been produced.
    """
    gps_xy, _, arc_lines = _metric_geometry(trip)
    arc_tree = STRtree(arc_lines)
    n_obs = len(trip.gps)
    step = np.zeros(n_obs, dtype=float)
    gaps = np.zeros(n_obs, dtype=float)
    if n_obs > 1:
        step[1:] = np.linalg.norm(np.diff(gps_xy, axis=0), axis=1)
        gaps[1:] = np.diff(trip.gps[:, 2])
    if align_labels:
        labels, label_distances = align_reference_route(trip, search_radius_m=label_radius_m)
    else:
        labels = np.full(n_obs, -1, dtype=np.int64)
        label_distances = np.zeros(n_obs, dtype=float)

    headings = np.zeros(n_obs, dtype=float)
    for i in range(n_obs):
        before = max(i - 1, 0)
        after = min(i + 1, n_obs - 1)
        delta = gps_xy[after] - gps_xy[before]
        headings[i] = np.arctan2(delta[0], delta[1]) if np.linalg.norm(delta) else 0.0
    arc_headings = np.asarray([
        np.arctan2(line.coords[-1][0] - line.coords[0][0],
                   line.coords[-1][1] - line.coords[0][1]) if line.length else 0.0
        for line in arc_lines
    ])
    arc_lengths = np.asarray([line.length for line in arc_lines], dtype=float)

    X_rows: list[list[float]] = []
    y_rows: list[int] = []
    obs_rows: list[int] = []
    arc_rows: list[int] = []
    misses = np.zeros(n_obs, dtype=bool)
    for i, xy in enumerate(gps_xy):
        point = Point(float(xy[0]), float(xy[1]))
        indices = np.asarray(arc_tree.query(point.buffer(search_radius_m)), dtype=np.int64)
        if not indices.size:
            indices = np.asarray(arc_tree.query_nearest(point, all_matches=True), dtype=np.int64).reshape(-1)
        distances = np.asarray([point.distance(arc_lines[int(j)]) for j in indices])
        order = np.argsort(distances)[:top_k]
        chosen = indices[order]
        dists = distances[order]
        if align_labels and int(labels[i]) not in set(int(j) for j in chosen):
            misses[i] = True
        for arc_id, distance in zip(chosen, dists, strict=True):
            line = arc_lines[int(arc_id)]
            along = float(line.project(point) / line.length) if line.length else 0.0
            heading_cosine = float(np.cos(headings[i] - arc_headings[int(arc_id)]))
            X_rows.append([
                float(distance), heading_cosine, along, float(arc_lengths[int(arc_id)]),
                float(step[i] / gaps[i]) if gaps[i] > 0 else 0.0,
                float(gaps[i]), float(step[i]),
            ])
            y_rows.append(int(align_labels and int(arc_id) == int(labels[i])))
            obs_rows.append(i)
            arc_rows.append(int(arc_id))
    return CandidateDataset(
        trip_id=trip.trip_id,
        X=np.asarray(X_rows, dtype=np.float32),
        y=np.asarray(y_rows, dtype=np.int8),
        observation_index=np.asarray(obs_rows, dtype=np.int32),
        arc_ids=np.asarray(arc_rows, dtype=np.int32),
        reference_arc_ids=labels,
        candidate_miss=(misses | (label_distances > label_radius_m)) if align_labels else misses,
        aligned_distance_m=label_distances.astype(np.float32),
    )

