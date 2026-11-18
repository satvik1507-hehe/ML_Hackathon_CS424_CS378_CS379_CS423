"""Evaluator-facing PathSnap client; trip data and labels stay on the API server."""

import os

import folium
import requests
import streamlit as st
import streamlit.components.v1 as components

API_URL = os.getenv("PATHSNAP_API_URL", "http://127.0.0.1:8000").rstrip("/")
MAX_MARKERS = 500
METRIC_LABELS = {
    "top1_accuracy": "Top-1 accuracy",
    "macro_f1": "Macro F1",
    "macro_precision": "Macro precision",
    "macro_recall": "Macro recall",
    "route_continuity": "Route continuity",
    "candidate_miss_rate": "Candidate miss rate",
}


def build_trip_map(gps: list[list[float]], route_segments: list[list[list[float]]], trip_id: str):
    """Render GPS and predicted geometry only; no reference route is sent to the browser."""
    center = [sum(point[1] for point in gps) / len(gps), sum(point[0] for point in gps) / len(gps)]
    route_map = folium.Map(location=center, zoom_start=14, control_scale=True)
    gps_layer = folium.FeatureGroup(name="GPS track and observations", show=True)
    folium.PolyLine([[lat, lon] for lon, lat in gps], color="#2878B5", weight=2,
                    opacity=.85, tooltip="GPS track").add_to(gps_layer)
    stride = max(1, (len(gps) + MAX_MARKERS - 1) // MAX_MARKERS)
    for lon, lat in gps[::stride]:
        folium.CircleMarker([lat, lon], radius=3, color="#2878B5", fill=True,
                            fill_opacity=.8, tooltip="GPS observation").add_to(gps_layer)
    gps_layer.add_to(route_map)

    predicted_layer = folium.FeatureGroup(name="Predicted route", show=True)
    all_points = [[lat, lon] for lon, lat in gps]
    for segment in route_segments:
        coords = [[lat, lon] for lon, lat in segment]
        if coords:
            folium.PolyLine(coords, color="#E87522", weight=5, opacity=.9,
                            tooltip="Predicted route").add_to(predicted_layer)
            all_points.extend(coords)
    predicted_layer.add_to(route_map)
    folium.LayerControl(collapsed=False).add_to(route_map)
    folium.Element(
        """<div style='position:fixed;bottom:26px;left:26px;z-index:9999;background:white;
        padding:8px 11px;border:1px solid #aaa;border-radius:5px;font-size:12px'>
        <b>PathSnap | Trip """ + trip_id + """</b><br>
        <span style='color:#2878B5'>━━</span> GPS observations / track<br>
        <span style='color:#E87522'>━━</span> Predicted route</div>"""
    ).add_to(route_map.get_root().html)
    route_map.fit_bounds(all_points, padding=(18, 18))
    return route_map


st.set_page_config(page_title="PathSnap", page_icon="🗺️", layout="wide")
st.title("PathSnap")
st.caption("Choose an evaluation trip to reconstruct its route from GPS observations.")

try:
    response = requests.get(f"{API_URL}/trips", timeout=4)
    response.raise_for_status()
    trip_ids = response.json().get("trip_ids", [])
except requests.RequestException as exc:
    st.error(f"Cannot reach the PathSnap API at {API_URL}: {exc}")
    st.stop()

if not trip_ids:
    st.info("The API has no evaluation trips configured.")
    st.stop()

st.success(f"Connected to PathSnap API · {len(trip_ids)} trips available")
trip_id = st.selectbox("Evaluation trip", trip_ids)
if st.button("Reconstruct route", type="primary", use_container_width=True):
    with st.spinner("Matching GPS observations to the road network…"):
        try:
            result = requests.post(f"{API_URL}/predict/{trip_id}", timeout=120)
            result.raise_for_status()
            payload = result.json()
        except requests.RequestException as exc:
            st.error(f"Prediction request failed: {exc}")
            st.stop()

    gps = payload.get("gps_points", [])
    # The API gives segment boundaries so the map does not draw through a route break.
    segments = payload.get("predicted_route_segments")
    if segments is None:
        route = payload.get("predicted_route", [])
        segments = [route] if route else []
    if not gps:
        st.warning("The API returned no GPS observations to display.")
        st.stop()

    components.html(build_trip_map(gps, segments, trip_id).get_root().render(),
                    height=620, scrolling=True)
    metrics = payload.get("metrics") or {}
    if metrics:
        st.subheader("Evaluation results")
        primary = ["top1_accuracy", "macro_f1", "route_continuity", "candidate_miss_rate"]
        columns = st.columns(len(primary))
        for column, key in zip(columns, primary, strict=True):
            if key in metrics:
                column.metric(METRIC_LABELS[key], f"{metrics[key]:.3f}")
        with st.expander("All returned metrics"):
            st.json({METRIC_LABELS.get(key, key): value for key, value in metrics.items()})
