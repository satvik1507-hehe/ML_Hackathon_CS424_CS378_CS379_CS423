"""Gap-aware Viterbi decoder over each observation's trip-local candidate arcs."""

from __future__ import annotations

import numpy as np

from pathsnap.data.reader import Trip
from pathsnap.features.candidates import CandidateDataset, _projector


def decode_viterbi(trip: Trip, dataset: CandidateDataset, candidate_scores: np.ndarray,
                   *, base_sigma_m: float = 20.0, gap_tolerance_s: float = 10.0,
                   disconnect_penalty: float = 8.0) -> np.ndarray:
    """Decode a topological route; longer GPS gaps reduce the disconnected-transition penalty."""
    if len(candidate_scores) != len(dataset.arc_ids):
        raise ValueError("There must be one emission score per candidate row")
    endpoints = trip.arcs.astype(np.int64)
    transformer = _projector(trip.gps)
    gps_x, gps_y = transformer.transform(trip.gps[:, 0], trip.gps[:, 1])
    node_x, node_y = transformer.transform(trip.nodes[:, 0], trip.nodes[:, 1])
    gps_xy = np.column_stack((gps_x, gps_y))
    node_xy = np.column_stack((node_x, node_y))
    arc_lengths = np.linalg.norm(node_xy[endpoints[:, 1]] - node_xy[endpoints[:, 0]], axis=1)
    gps_steps = np.zeros(len(trip.gps), dtype=float)
    if len(trip.gps) > 1:
        gps_steps[1:] = np.linalg.norm(np.diff(gps_xy, axis=0), axis=1)
    gaps = np.zeros(len(trip.gps), dtype=float)
    if len(trip.gps) > 1:
        gaps[1:] = np.diff(trip.gps[:, 2])

    state_rows: list[np.ndarray] = []
    sorted_rows = np.argsort(dataset.observation_index, kind="stable")
    obs_sorted = dataset.observation_index[sorted_rows]
    starts = np.r_[0, np.flatnonzero(np.diff(obs_sorted)) + 1]
    ends = np.r_[starts[1:], len(sorted_rows)]
    for start, end in zip(starts, ends, strict=True):
        state_rows.append(sorted_rows[start:end])
    if len(state_rows) != len(trip.gps):
        raise ValueError("Candidate rows must include at least one state for every GPS observation")

    emissions = []
    for rows in state_rows:
        probs = np.maximum(np.asarray(candidate_scores[rows], dtype=float), 1e-12)
        probs /= probs.sum()
        emissions.append(np.log(probs))

    dp = [emissions[0].copy()]
    back: list[np.ndarray] = [np.full(len(state_rows[0]), -1, dtype=np.int32)]
    for i in range(1, len(state_rows)):
        prev_rows, cur_rows = state_rows[i - 1], state_rows[i]
        prev_arcs, cur_arcs = dataset.arc_ids[prev_rows], dataset.arc_ids[cur_rows]
        prev_fraction = dataset.X[prev_rows, 2]
        cur_fraction = dataset.X[cur_rows, 2]
        gap = max(float(gaps[i]), 1.0)
        observed = min(float(gps_steps[i]), 300.0)
        sigma = base_sigma_m + max(gap - 1.0, 0.0) * gap_tolerance_s + 0.25 * observed
        transition_penalty = disconnect_penalty / np.sqrt(gap)
        cur_dp = np.full(len(cur_rows), -np.inf, dtype=float)
        cur_back = np.full(len(cur_rows), -1, dtype=np.int32)
        same_arc = prev_arcs[:, None] == cur_arcs[None, :]
        directly_connected = endpoints[prev_arcs, 1][:, None] == endpoints[cur_arcs, 0][None, :]
        prev_len = arc_lengths[prev_arcs][:, None]
        curr_len = arc_lengths[cur_arcs][None, :]
        prev_frac = prev_fraction[:, None]
        curr_frac = cur_fraction[None, :]
        same_distance = np.abs(curr_frac - prev_frac) * curr_len
        connected_distance = (1.0 - prev_frac) * prev_len + curr_frac * curr_len
        route_distance = np.where(same_arc, same_distance,
                                  np.where(directly_connected, connected_distance, observed))
        connected = directly_connected | (same_arc & (curr_frac >= prev_frac))
        transition = -np.abs(route_distance - observed) / max(sigma, 1.0)
        transition -= (~connected) * transition_penalty
        candidate_scores = dp[-1][:, None] + transition + emissions[i][None, :]
        cur_back = np.argmax(candidate_scores, axis=0).astype(np.int32)
        cur_dp = candidate_scores[cur_back, np.arange(len(cur_rows))]
        if not np.isfinite(cur_dp).any():
            cur_dp = emissions[i].copy()
            cur_back[:] = 0
        dp.append(cur_dp)
        back.append(cur_back)

    selected = np.empty(len(state_rows), dtype=np.int32)
    selected[-1] = int(np.argmax(dp[-1]))
    for i in range(len(state_rows) - 1, 0, -1):
        previous = back[i][selected[i]]
        selected[i - 1] = max(int(previous), 0)
    return np.asarray([dataset.arc_ids[state_rows[i][selected[i]]] for i in range(len(state_rows))],
                      dtype=np.int64)


def decoded_metrics(trip: Trip, dataset: CandidateDataset, predicted_arc_ids: np.ndarray) -> dict[str, float]:
    """Metrics for hard Viterbi outputs; log loss is not reported for a single best path."""
    from sklearn.metrics import f1_score, precision_score, recall_score

    truth = dataset.reference_arc_ids
    pred = np.asarray(predicted_arc_ids, dtype=np.int64)
    labels = np.unique(np.r_[truth, pred])
    connected = (pred[:-1] == pred[1:]) | (trip.arcs[pred[:-1], 1] == trip.arcs[pred[1:], 0])
    return {
        "observations": float(len(truth)),
        "top1_accuracy": float(np.mean(truth == pred)),
        "macro_f1": float(f1_score(truth, pred, labels=labels, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(truth, pred, labels=labels,
                                                   average="macro", zero_division=0)),
        "macro_recall": float(recall_score(truth, pred, labels=labels,
                                            average="macro", zero_division=0)),
        "route_continuity": float(np.mean(connected)) if len(connected) else 1.0,
        "candidate_miss_rate": float(np.mean(dataset.candidate_miss)),
    }

