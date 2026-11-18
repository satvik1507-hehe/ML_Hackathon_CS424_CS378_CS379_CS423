"""Laptop 1 API entry point. Evaluation labels must remain server-side."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pathsnap.config import settings

app = FastAPI(title=settings.api_title, version=settings.api_version)


class PredictionResponse(BaseModel):
    trip_id: str
    gps_points: list[list[float]]
    predicted_route: list[list[float]]
    metrics: dict[str, float] | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "pathsnap-api"}


@app.get("/trips")
def trips() -> dict[str, list[str]]:
    """Expose selectable IDs only after an inference partition is configured."""
    raise HTTPException(status_code=503, detail="Trip catalog is not configured yet")


@app.post("/predict/{trip_id}", response_model=PredictionResponse)
def predict(trip_id: str) -> PredictionResponse:
    """Run candidate scoring and route decoding once model artifacts are registered."""
    raise HTTPException(status_code=503, detail=f"Prediction pipeline is not configured for {trip_id}")

