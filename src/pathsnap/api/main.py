"""Laptop 1 API entry point. Prediction is label-blind; evaluation stays server-side."""

from functools import lru_cache
import json
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pathsnap.config import settings
from pathsnap.data.reader import load_trip
from pathsnap.decoder.retrained_hmm import DecoderParams, precompute, viterbi
from pathsnap.features.retrained_candidates import FEATURES, feature_trip, generate

app = FastAPI(title=settings.api_title, version=settings.api_version)


class PredictionResponse(BaseModel):
    trip_id: str
    gps_points: list[list[float]]
    predicted_route: list[list[float]]
    predicted_route_segments: list[list[list[float]]]
    metrics: dict[str, float] | None = None


@lru_cache(maxsize=1)
def _model_artifact() -> dict[str, object]:
    path = Path(settings.model_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return joblib.load(path)


@lru_cache(maxsize=1)
def _decoder_settings() -> dict[str, float]:
    path = Path(settings.decoder_config_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))["parameters"]


def _retrained_metrics(trip, predicted: np.ndarray) -> dict[str, float]:
    """Score fixed predictions with the route alignment used to train the active model."""
    from sklearn.metrics import f1_score, precision_score, recall_score

    candidates, info = generate(feature_trip(trip), with_labels=True)
    truth = info["ref_arc"]
    labels = np.unique(np.r_[truth, predicted])
    connected = ((predicted[:-1] == predicted[1:])
                 | (trip.arcs[predicted[:-1], 1] == trip.arcs[predicted[1:], 0]))
    reference_route = set(trip.route.tolist())
    predicted_route = set(predicted.tolist())
    common = len(reference_route & predicted_route)
    route_precision = common / max(len(predicted_route), 1)
    route_recall = common / max(len(reference_route), 1)
    route_f1 = (2 * route_precision * route_recall / (route_precision + route_recall)
                if common else 0.0)
    reachable_share = candidates.groupby("obs")["label"].max().mean()
    return {
        "top1_accuracy": float(np.mean(truth == predicted)),
        "macro_f1": float(f1_score(truth, predicted, labels=labels, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(
            truth, predicted, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(
            truth, predicted, labels=labels, average="macro", zero_division=0)),
        "route_continuity": float(np.mean(connected)) if len(connected) else 1.0,
        "candidate_miss_rate": float(1.0 - reachable_share),
        "route_f1": float(route_f1),
        "route_precision": float(route_precision),
        "route_recall": float(route_recall),
    }


def _evaluation_dir() -> Path:
    if not settings.evaluation_root:
        raise HTTPException(status_code=503, detail="PATHSNAP_EVALUATION_ROOT is not configured")
    directory = Path(settings.evaluation_root).expanduser()
    if not directory.is_dir():
        raise HTTPException(status_code=503, detail="Evaluation-trip directory is unavailable")
    return directory


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "pathsnap-api"}


@app.get("/trips")
def trips() -> dict[str, list[str]]:
    """Expose evaluator trip IDs only; no route labels are returned."""
    folder = _evaluation_dir()
    ids = sorted(p.name for p in folder.iterdir()
                 if p.is_dir() and p.name.isdigit() and len(p.name) == 8
                 and (p / f"{p.name}.track").is_file())
    if not ids:
        raise HTTPException(status_code=503, detail="No evaluator trips are available")
    return {"trip_ids": ids}


@app.post("/predict/{trip_id}", response_model=PredictionResponse)
def predict(trip_id: str) -> PredictionResponse:
    """Predict without loading reference routes; load labels only after route decoding."""
    folder = _evaluation_dir()
    try:
        trip = load_trip(folder, trip_id, include_route=False)
        model_trip = feature_trip(trip)
        dataset, _ = generate(model_trip, with_labels=False)
        artifact = _model_artifact()
        model = artifact["model"]
        if tuple(artifact["feature_names"]) != tuple(FEATURES):
            raise ValueError("The active model feature order does not match its trained artifact")
        scores = model.predict_proba(dataset[FEATURES].to_numpy(dtype="float32"))[:, 1]
        precomputed = precompute(model_trip, dataset)
        predicted = viterbi(precomputed, scores, DecoderParams(**_decoder_settings()))
        # Evaluation labels are accessed only after the inference outputs have been fixed.
        labeled_trip = load_trip(folder, trip_id, include_route=True)
        metrics = _retrained_metrics(labeled_trip, predicted)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Required model/data file is missing: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    route_segments: list[list[list[float]]] = []
    current_segment: list[list[float]] = []
    for position, arc_id in enumerate(predicted):
        start_id, end_id = trip.arcs[int(arc_id)]
        start, end = trip.nodes[int(start_id)], trip.nodes[int(end_id)]
        if position == 0 or int(trip.arcs[int(predicted[position - 1]), 1]) != int(start_id):
            if current_segment:
                route_segments.append(current_segment)
            current_segment = [[float(start[0]), float(start[1])]]
        current_segment.append([float(end[0]), float(end[1])])
    if current_segment:
        route_segments.append(current_segment)
    route_coords = [point for segment in route_segments for point in segment]
    gps_points = [[float(lon), float(lat)] for lon, lat in trip.gps[:, :2]]
    safe_metrics = {key: float(value) for key, value in metrics.items()}
    return PredictionResponse(trip_id=trip_id, gps_points=gps_points,
                              predicted_route=route_coords, predicted_route_segments=route_segments,
                              metrics=safe_metrics)

