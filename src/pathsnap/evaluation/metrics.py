"""Observation-level map-matching metrics for trip-local candidate predictions."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def _observation_predictions(dataset, candidate_scores: np.ndarray):
    n_obs = len(dataset.reference_arc_ids)
    predicted = np.full(n_obs, -1, dtype=np.int64)
    target_probability = np.full(n_obs, 1e-12, dtype=float)
    # Candidate rows are generated in observation order; stable sorting also supports external
    # callers that persist/reload arrays with a different row order.
    order = np.argsort(dataset.observation_index, kind="stable")
    obs_sorted = dataset.observation_index[order]
    starts = np.r_[0, np.flatnonzero(np.diff(obs_sorted)) + 1]
    ends = np.r_[starts[1:], len(order)]
    for start, end in zip(starts, ends, strict=True):
        row_ids = order[start:end]
        obs_id = int(obs_sorted[start])
        probs = np.maximum(candidate_scores[row_ids], 1e-12)
        probs = probs / probs.sum()
        best = int(np.argmax(probs))
        predicted[obs_id] = int(dataset.arc_ids[row_ids[best]])
        target_hits = np.flatnonzero(dataset.arc_ids[row_ids] == dataset.reference_arc_ids[obs_id])
        if target_hits.size:
            target_probability[obs_id] = probs[int(target_hits[0])]
    return predicted, target_probability


def observation_metrics(dataset, candidate_scores: np.ndarray, arc_endpoints: np.ndarray) -> dict[str, float]:
    """Compute metrics with a per-observation probability distribution over candidates."""
    y_true = np.asarray(dataset.reference_arc_ids, dtype=np.int64)
    y_pred, p_true = _observation_predictions(dataset, candidate_scores)
    valid = y_pred >= 0
    y_true, y_pred, p_true = y_true[valid], y_pred[valid], p_true[valid]
    labels = np.unique(np.r_[y_true, y_pred])
    correct = y_true == y_pred
    if len(y_pred) > 1:
        connected = (y_pred[:-1] == y_pred[1:]) | (
            arc_endpoints[y_pred[:-1], 1] == arc_endpoints[y_pred[1:], 0]
        )
        continuity = float(np.mean(connected))
    else:
        continuity = 1.0
    return {
        "observations": float(len(y_true)),
        "top1_accuracy": float(np.mean(correct)) if len(correct) else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels,
                                                   average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels,
                                            average="macro", zero_division=0)),
        "log_loss": float(-np.log(np.clip(p_true, 1e-12, 1.0)).mean()) if len(p_true) else 0.0,
        "route_continuity": continuity,
        "candidate_miss_rate": float(np.mean(dataset.candidate_miss)),
    }

