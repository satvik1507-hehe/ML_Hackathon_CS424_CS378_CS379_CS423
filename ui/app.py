"""Thin evaluator-facing client; all data and evaluation remain on the API server."""

import os

import folium
import requests
import streamlit as st
import streamlit.components.v1 as components

API_URL = os.getenv("PATHSNAP_API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="PathSnap", page_icon="🗺️", layout="wide")
st.title("PathSnap")
st.caption("GPS trajectory map matching and route reconstruction")

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

trip_id = st.selectbox("Select a trip", trip_ids)
if st.button("Reconstruct route", type="primary"):
    try:
        result = requests.post(f"{API_URL}/predict/{trip_id}", timeout=120)
        result.raise_for_status()
        payload = result.json()
    except requests.RequestException as exc:
        st.error(f"Prediction request failed: {exc}")
        st.stop()

    gps = payload.get("gps_points", [])
    route = payload.get("predicted_route", [])
    if not gps:
        st.warning("The API returned no GPS observations to display.")
        st.stop()

    center = [sum(point[1] for point in gps) / len(gps), sum(point[0] for point in gps) / len(gps)]
    route_map = folium.Map(location=center, zoom_start=14, control_scale=True)
    folium.FeatureGroup(name="GPS observations").add_to(route_map)
    for lon, lat in gps:
        folium.CircleMarker([lat, lon], radius=3, color="#2878b5", fill=True,
                            fill_opacity=0.8).add_to(route_map)
    if route:
        folium.PolyLine([[lat, lon] for lon, lat in route], color="#24a148", weight=5,
                        tooltip="Predicted route").add_to(route_map)
    folium.LayerControl().add_to(route_map)
    components.html(route_map.get_root().render(), height=620, scrolling=True)
    metrics = payload.get("metrics") or {}
    if metrics:
        st.subheader("Evaluation metrics")
        st.json(metrics)

