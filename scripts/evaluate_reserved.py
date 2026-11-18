"""Evaluate the frozen active PathSnap pipeline on a reserved trip partition.

This script never fits or tunes a model. It predicts each trip without loading its reference
route, then reads the route only after classifier and HMM predictions are fixed. It saves metrics
and plots, but does not save observation-level labels, predictions, or raw trip data.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

# Keep Matplotlib's font/config cache in a writable temporary folder on student laptops.
_mpl_config = Path(tempfile.gettempdir()) / "pathsnap-mplconfig"
_xdg_cache = Path(tempfile.gettempdir()) / "pathsnap-cache"
_mpl_config.mkdir(parents=True, exist_ok=True)
_xdg_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_config))
os.environ.setdefault("XDG_CACHE_HOME", str(_xdg_cache))
os.environ.setdefault("MPLBACKEND", "Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score

from pathsnap.config import settings
from pathsnap.data.reader import load_trip
from pathsnap.decoder.retrained_hmm import DecoderParams, precompute, viterbi
from pathsnap.features.retrained_candidates import FEATURES, feature_trip, generate


def _read_artifacts():
    """Load the active scorer and fixed decoder settings; do not fit either one."""
    model_path = Path(settings.model_path)
    decoder_path = Path(settings.decoder_config_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Active model artifact not found: {model_path}")
    if not decoder_path.is_file():
        raise FileNotFoundError(f"Active decoder configuration not found: {decoder_path}")
    artifact = joblib.load(model_path)
    if tuple(artifact["feature_names"]) != tuple(FEATURES):
        raise ValueError("The feature order in the active artifact does not match the code")
    decoder_record = json.loads(decoder_path.read_text(encoding="utf-8"))
    return artifact["model"], DecoderParams(**decoder_record["parameters"])


def _classifier_predictions(candidates: pd.DataFrame, scores: np.ndarray, n_obs: int):
    """Choose each observation's highest-scoring candidate and retain its candidate probabilities."""
    predicted = np.full(n_obs, -1, dtype=np.int64)
    probability_by_observation: list[np.ndarray] = [np.empty(0)] * n_obs
    candidate_arcs: list[np.ndarray] = [np.empty(0, dtype=np.int64)] * n_obs
    for obs, rows in candidates.groupby("obs", sort=True):
        row_ids = rows.index.to_numpy()
        arcs = rows["arc"].to_numpy(dtype=np.int64)
        p = np.clip(scores[row_ids], 0.0, None)
        total = p.sum()
        p = p / total if total > 0 else np.full(len(p), 1.0 / len(p))
        predicted[int(obs)] = int(arcs[int(np.argmax(p))])
        probability_by_observation[int(obs)] = p
        candidate_arcs[int(obs)] = arcs
    return predicted, probability_by_observation, candidate_arcs


def _route_continuity(predicted: np.ndarray, arcs: np.ndarray) -> tuple[float, int, int]:
    """Count adjacent predicted arc pairs that share an arc or connect end-to-start."""
    if len(predicted) < 2:
        return 1.0, 0, 0
    connected = ((predicted[:-1] == predicted[1:])
                 | (arcs[predicted[:-1], 1] == arcs[predicted[1:], 0]))
    return float(np.mean(connected)), int(connected.sum()), int(len(connected))


def _route_f1(reference_route: np.ndarray, predicted: np.ndarray) -> tuple[float, float, float]:
    """Compute set-based route precision, recall, and F1, matching the PathSnap API definition."""
    ref_set, pred_set = set(reference_route.tolist()), set(predicted.tolist())
    common = len(ref_set & pred_set)
    precision = common / max(len(pred_set), 1)
    recall = common / max(len(ref_set), 1)
    f1 = 2 * precision * recall / (precision + recall) if common else 0.0
    return f1, precision, recall


def _score_predictions(truth: np.ndarray, predicted: np.ndarray, arcs: np.ndarray,
                       route_reference: np.ndarray, miss: np.ndarray,
                       *, true_probability: np.ndarray | None = None) -> dict:
    """Return observation and sequence metrics with candidate misses kept in the denominator."""
    labels = np.unique(np.r_[truth, predicted])
    continuity, connected_pairs, transition_count = _route_continuity(predicted, arcs)
    route_f1, route_precision, route_recall = _route_f1(route_reference, predicted)
    metrics = {
        "observations": int(len(truth)),
        "top1_accuracy": float(np.mean(truth == predicted)),
        "macro_f1": float(f1_score(truth, predicted, labels=labels, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(
            truth, predicted, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(
            truth, predicted, labels=labels, average="macro", zero_division=0)),
        "route_continuity": continuity,
        "connected_transitions": connected_pairs,
        "transitions": transition_count,
        "candidate_miss_rate": float(np.mean(miss)),
        "candidate_misses": int(np.sum(miss)),
        "route_f1": route_f1,
        "route_precision": route_precision,
        "route_recall": route_recall,
    }
    if true_probability is not None:
        metrics["log_loss"] = float(log_loss(
            np.ones(len(true_probability)), np.column_stack([1 - true_probability, true_probability]),
            labels=[0, 1]))
    return metrics


def evaluate_trip(partition: Path, trip_id: str, model, decoder: DecoderParams) -> tuple[dict, dict]:
    """Predict label-blind first, then load labels and evaluate the frozen predictions."""
    started = time.perf_counter()
    prediction_trip = load_trip(partition, trip_id, include_route=False)
    model_trip = feature_trip(prediction_trip)
    candidates, _ = generate(model_trip, with_labels=False)
    if candidates.empty:
        raise ValueError(f"Trip {trip_id} generated no road candidates")

    scores = model.predict_proba(candidates[FEATURES].to_numpy(dtype="float32"))[:, 1]
    classifier_pred, _, candidate_arcs = _classifier_predictions(
        candidates, scores, len(prediction_trip.gps))
    decoded = viterbi(precompute(model_trip, candidates), scores, decoder)
    prediction_seconds = time.perf_counter() - started

    # Loading the reference route happens only after both prediction sequences are fixed.
    labeled_trip = load_trip(partition, trip_id, include_route=True)
    labeled_model_trip = feature_trip(labeled_trip)
    labeled_candidates, info = generate(labeled_model_trip, with_labels=True)
    if not np.array_equal(candidates[["obs", "arc"]].to_numpy(),
                          labeled_candidates[["obs", "arc"]].to_numpy()):
        raise ValueError(f"Candidate list changed while aligning reference labels for trip {trip_id}")
    truth = info["ref_arc"]
    candidate_miss = np.ones(len(truth), dtype=bool)
    p_true = np.full(len(truth), 1e-12, dtype=float)
    for obs, rows in labeled_candidates.groupby("obs", sort=True):
        obs = int(obs)
        arcs_for_obs = candidate_arcs[obs]
        target_position = np.flatnonzero(arcs_for_obs == truth[obs])
        if target_position.size:
            candidate_miss[obs] = False
            probabilities = np.clip(scores[rows.index.to_numpy()], 0.0, None)
            total = probabilities.sum()
            probabilities = (probabilities / total if total > 0
                             else np.full(len(probabilities), 1.0 / len(probabilities)))
            p_true[obs] = probabilities[int(target_position[0])]

    classifier_metrics = _score_predictions(
        truth, classifier_pred, labeled_trip.arcs, labeled_trip.route, candidate_miss,
        true_probability=p_true)
    hmm_metrics = _score_predictions(truth, decoded, labeled_trip.arcs, labeled_trip.route,
                                     candidate_miss)
    per_trip = {
        "trip_id": trip_id,
        "prediction_seconds": float(prediction_seconds),
        "gps_rows": len(truth),
        "classifier": classifier_metrics,
        "hmm": hmm_metrics,
    }
    summary_row = {
        "trip_id": trip_id,
        "observations": len(truth),
        "candidate_misses": int(candidate_miss.sum()),
        "candidate_miss_rate": float(candidate_miss.mean()),
        "classifier_top1": classifier_metrics["top1_accuracy"],
        "hmm_top1": hmm_metrics["top1_accuracy"],
        "classifier_macro_f1": classifier_metrics["macro_f1"],
        "hmm_macro_f1": hmm_metrics["macro_f1"],
        "classifier_macro_precision": classifier_metrics["macro_precision"],
        "hmm_macro_precision": hmm_metrics["macro_precision"],
        "classifier_macro_recall": classifier_metrics["macro_recall"],
        "hmm_macro_recall": hmm_metrics["macro_recall"],
        "classifier_log_loss": classifier_metrics["log_loss"],
        "classifier_route_continuity": classifier_metrics["route_continuity"],
        "hmm_route_continuity": hmm_metrics["route_continuity"],
        "classifier_route_f1": classifier_metrics["route_f1"],
        "hmm_route_f1": hmm_metrics["route_f1"],
        "classifier_connected_transitions": classifier_metrics["connected_transitions"],
        "hmm_connected_transitions": hmm_metrics["connected_transitions"],
        "transitions": hmm_metrics["transitions"],
        "prediction_seconds": float(prediction_seconds),
    }
    return per_trip, summary_row


def _plot_outputs(frame: pd.DataFrame, overall: dict, out_dir: Path, partition_name: str) -> None:
    """Create actual-result plots and matching CSVs for auditability."""
    frame = frame.copy()
    frame["trip_id"] = frame["trip_id"].astype(str).str.zfill(8)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 12, "axes.labelsize": 10})

    overall_rows = [
        {"metric": "Top-1 accuracy (all observations)", "method": "RF classifier",
         "value": overall["classifier_micro_top1"]},
        {"metric": "Top-1 accuracy (all observations)", "method": "RF + HMM/Viterbi",
         "value": overall["hmm_micro_top1"]},
        {"metric": "Macro F1 (mean across trips)", "method": "RF classifier",
         "value": overall["classifier_mean_trip_macro_f1"]},
        {"metric": "Macro F1 (mean across trips)", "method": "RF + HMM/Viterbi",
         "value": overall["hmm_mean_trip_macro_f1"]},
        {"metric": "Route continuity (pooled transitions)", "method": "RF classifier",
         "value": overall["classifier_pooled_route_continuity"]},
        {"metric": "Route continuity (pooled transitions)", "method": "RF + HMM/Viterbi",
         "value": overall["hmm_pooled_route_continuity"]},
    ]
    plot_data = pd.DataFrame(overall_rows)
    plot_data.to_csv(plot_dir / "01_overall_model_comparison.csv", index=False)
    pivot = plot_data.pivot(index="metric", columns="method", values="value")
    ax = pivot.plot(kind="bar", figsize=(10, 5), color=["#2878B5", "#E87522"])
    ax.set_title(f"{partition_name}: Random Forest before and after HMM/Viterbi")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(title="Prediction system")
    ax.grid(axis="y", alpha=.25)
    plt.tight_layout()
    plt.savefig(plot_dir / "01_overall_model_comparison.png", dpi=200)
    plt.close()

    per_trip = frame[["trip_id", "classifier_top1", "hmm_top1"]].melt(
        id_vars="trip_id", var_name="method", value_name="top1_accuracy")
    per_trip["method"] = per_trip["method"].map({
        "classifier_top1": "RF classifier", "hmm_top1": "RF + HMM/Viterbi"})
    per_trip.to_csv(plot_dir / "02_top1_by_trip.csv", index=False)
    pivot = per_trip.pivot(index="trip_id", columns="method", values="top1_accuracy")
    ax = pivot.plot(kind="bar", figsize=(11, 5), color=["#2878B5", "#E87522"])
    ax.set_title(f"{partition_name}: Top-1 accuracy by trip")
    ax.set_ylabel("Top-1 accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Trip ID")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="Prediction system")
    ax.grid(axis="y", alpha=.25)
    plt.tight_layout()
    plt.savefig(plot_dir / "02_top1_by_trip.png", dpi=200)
    plt.close()

    per_trip = frame[["trip_id", "classifier_macro_f1", "hmm_macro_f1"]].melt(
        id_vars="trip_id", var_name="method", value_name="macro_f1")
    per_trip["method"] = per_trip["method"].map({
        "classifier_macro_f1": "RF classifier", "hmm_macro_f1": "RF + HMM/Viterbi"})
    per_trip.to_csv(plot_dir / "03_macro_f1_by_trip.csv", index=False)
    pivot = per_trip.pivot(index="trip_id", columns="method", values="macro_f1")
    ax = pivot.plot(kind="bar", figsize=(11, 5), color=["#2878B5", "#E87522"])
    ax.set_title(f"{partition_name}: Macro F1 by trip")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Trip ID")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="Prediction system")
    ax.grid(axis="y", alpha=.25)
    plt.tight_layout()
    plt.savefig(plot_dir / "03_macro_f1_by_trip.png", dpi=200)
    plt.close()

    diagnostics = frame[["trip_id", "observations", "candidate_misses", "candidate_miss_rate"]].copy()
    diagnostics.to_csv(plot_dir / "04_candidate_misses_by_trip.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(diagnostics["trip_id"], diagnostics["candidate_misses"], color="#6750A4")
    for x, row in enumerate(diagnostics.itertuples(index=False)):
        ax.text(x, row.candidate_misses, f"{row.candidate_misses}\n({row.candidate_miss_rate:.2%})",
                ha="center", va="bottom", fontsize=7)
    ax.set_title(f"{partition_name}: observations whose reference arc was not a candidate")
    ax.set_ylabel("Candidate misses (observations)")
    ax.set_xlabel("Trip ID")
    ax.tick_params(axis="x", rotation=35)
    ax.set_ylim(bottom=0, top=max(2, diagnostics["candidate_misses"].max() * 1.35))
    ax.grid(axis="y", alpha=.25)
    plt.tight_layout()
    plt.savefig(plot_dir / "04_candidate_misses_by_trip.png", dpi=200)
    plt.close()

    continuity = frame[["trip_id", "classifier_route_continuity", "hmm_route_continuity"]].melt(
        id_vars="trip_id", var_name="method", value_name="route_continuity")
    continuity["method"] = continuity["method"].map({
        "classifier_route_continuity": "RF classifier",
        "hmm_route_continuity": "RF + HMM/Viterbi"})
    continuity.to_csv(plot_dir / "05_route_continuity_by_trip.csv", index=False)
    pivot = continuity.pivot(index="trip_id", columns="method", values="route_continuity")
    ax = pivot.plot(kind="bar", figsize=(11, 5), color=["#2878B5", "#E87522"])
    ax.set_title(f"{partition_name}: route continuity by trip")
    ax.set_ylabel("Connected adjacent predictions")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Trip ID")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="Prediction system")
    ax.grid(axis="y", alpha=.25)
    plt.tight_layout()
    plt.savefig(plot_dir / "05_route_continuity_by_trip.png", dpi=200)
    plt.close()

    latency = frame[["trip_id", "observations", "prediction_seconds"]].copy()
    latency["observations_per_second"] = latency["observations"] / latency["prediction_seconds"]
    latency.to_csv(plot_dir / "06_prediction_time_by_trip.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(latency["trip_id"], latency["prediction_seconds"], color="#238B6E")
    ax.set_title(f"{partition_name}: prediction pipeline time by trip")
    ax.set_ylabel("Seconds (candidate generation + scoring + HMM)")
    ax.set_xlabel("Trip ID")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=.25)
    plt.tight_layout()
    plt.savefig(plot_dir / "06_prediction_time_by_trip.png", dpi=200)
    plt.close()

    loss = frame[["trip_id", "classifier_log_loss"]].copy()
    loss.to_csv(plot_dir / "07_classifier_log_loss_by_trip.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(loss["trip_id"], loss["classifier_log_loss"], color="#9A5B9A")
    ax.set_title(f"{partition_name}: classifier log loss by trip")
    ax.set_ylabel("Candidate-normalized log loss (lower is better)")
    ax.set_xlabel("Trip ID")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=.25)
    plt.tight_layout()
    plt.savefig(plot_dir / "07_classifier_log_loss_by_trip.png", dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",
                        help="Partition directory containing one subdirectory per 8-digit trip")
    parser.add_argument("--output", default="outputs/final_evaluation",
                        help="Output directory for summary CSVs, JSON, and graphs")
    parser.add_argument("--plots-only", action="store_true",
                        help="Build graphs from saved final-evaluation CSV/JSON without rereading data")
    args = parser.parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.plots_only:
        metrics_path = out_dir / "per_trip_metrics.csv"
        summary_path = out_dir / "summary.json"
        if not metrics_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError("Plots-only mode requires saved per_trip_metrics.csv and summary.json")
        saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        _plot_outputs(pd.read_csv(metrics_path), saved_summary, out_dir,
                      saved_summary["partition"])
        print(f"Rebuilt plots from saved metrics in {out_dir.resolve()}")
        return
    if not args.data_root:
        parser.error("--data-root is required unless --plots-only is selected")
    partition = Path(args.data_root).expanduser().resolve()
    if not partition.is_dir():
        raise FileNotFoundError(f"Evaluation partition does not exist: {partition}")

    trip_ids = sorted(path.name for path in partition.iterdir()
                      if path.is_dir() and path.name.isdigit() and len(path.name) == 8
                      and (path / f"{path.name}.track").is_file())
    if not trip_ids:
        raise ValueError(f"No 8-digit trip folders found in {partition}")

    model, decoder = _read_artifacts()
    trip_records, summary_rows = [], []
    for trip_id in trip_ids:
        print(f"Evaluating {trip_id} ...", flush=True)
        detailed, row = evaluate_trip(partition, trip_id, model, decoder)
        trip_records.append(detailed)
        summary_rows.append(row)

    frame = pd.DataFrame(summary_rows)
    frame.to_csv(out_dir / "per_trip_metrics.csv", index=False)
    observations = int(frame["observations"].sum())
    transitions = int(frame["transitions"].sum())
    overall = {
        "partition": partition.name,
        "trip_count": len(frame),
        "observation_count": observations,
        "model": "active Random Forest + fixed HMM/Viterbi",
        "model_artifact": settings.model_path,
        "decoder_config": settings.decoder_config_path,
        "refit_or_tuned_on_evaluator": False,
        "classifier_micro_top1": float(
            (frame["classifier_top1"] * frame["observations"]).sum() / observations),
        "hmm_micro_top1": float((frame["hmm_top1"] * frame["observations"]).sum() / observations),
        "classifier_mean_trip_top1": float(frame["classifier_top1"].mean()),
        "hmm_mean_trip_top1": float(frame["hmm_top1"].mean()),
        "classifier_mean_trip_macro_f1": float(frame["classifier_macro_f1"].mean()),
        "hmm_mean_trip_macro_f1": float(frame["hmm_macro_f1"].mean()),
        "classifier_mean_trip_macro_precision": float(frame["classifier_macro_precision"].mean()),
        "hmm_mean_trip_macro_precision": float(frame["hmm_macro_precision"].mean()),
        "classifier_mean_trip_macro_recall": float(frame["classifier_macro_recall"].mean()),
        "hmm_mean_trip_macro_recall": float(frame["hmm_macro_recall"].mean()),
        "classifier_observation_weighted_log_loss": float(
            (frame["classifier_log_loss"] * frame["observations"]).sum() / observations),
        "candidate_miss_rate": float(frame["candidate_misses"].sum() / observations),
        "candidate_misses": int(frame["candidate_misses"].sum()),
        "classifier_pooled_route_continuity": float(
            frame["classifier_connected_transitions"].sum() / transitions) if transitions else 1.0,
        "hmm_pooled_route_continuity": float(
            frame["hmm_connected_transitions"].sum() / transitions) if transitions else 1.0,
        "classifier_mean_trip_route_f1": float(frame["classifier_route_f1"].mean()),
        "hmm_mean_trip_route_f1": float(frame["hmm_route_f1"].mean()),
        "mean_prediction_seconds_per_trip": float(frame["prediction_seconds"].mean()),
    }
    (out_dir / "summary.json").write_text(json.dumps(overall, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(trip_records).to_json(out_dir / "per_trip_metrics_detailed.json",
                                       orient="records", indent=2)
    _plot_outputs(frame, overall, out_dir, partition.name)
    print(json.dumps(overall, indent=2))
    print(f"Saved final evaluation outputs to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
