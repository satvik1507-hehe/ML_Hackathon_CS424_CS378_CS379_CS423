"""Generate a Folium map for one real trip and its available route layers."""

import argparse
from pathlib import Path

import folium

from pathsnap.data.reader import load_trip


def make_trip_map(trip, output: str | Path) -> Path:
    """Render GPS observations, network arcs, and reference route when available."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    center = [float(trip.gps[:, 1].mean()), float(trip.gps[:, 0].mean())]
    fmap = folium.Map(location=center, zoom_start=14, control_scale=True, tiles="CartoDB positron")

    gps_layer = folium.FeatureGroup(name="GPS observations", show=True)
    for lon, lat, seconds in trip.gps:
        folium.CircleMarker(
            location=[float(lat), float(lon)], radius=3, color="#2878b5", fill=True,
            fill_opacity=0.75, tooltip=f"t={seconds:g}s",
        ).add_to(gps_layer)
    gps_layer.add_to(fmap)

    network_layer = folium.FeatureGroup(name="Road network", show=False)
    route_set = set(int(arc_id) for arc_id in trip.route)
    for arc_id, (from_id, to_id) in enumerate(trip.arcs):
        start = trip.nodes[int(from_id)]
        end = trip.nodes[int(to_id)]
        is_reference = arc_id in route_set
        folium.PolyLine(
            locations=[[float(start[1]), float(start[0])], [float(end[1]), float(end[0])]],
            color="#24a148" if is_reference else "#9aa0a6",
            weight=4 if is_reference else 1,
            opacity=0.9 if is_reference else 0.45,
            tooltip=f"arc {arc_id}" + (" (reference)" if is_reference else ""),
        ).add_to(network_layer)
    network_layer.add_to(fmap)

    ordered_route_layer = folium.FeatureGroup(name="Reference route order", show=True)
    if trip.route.size:
        route_coords = []
        for arc_id in trip.route:
            from_id, to_id = trip.arcs[int(arc_id)]
            start, end = trip.nodes[int(from_id)], trip.nodes[int(to_id)]
            if not route_coords:
                route_coords.append([float(start[1]), float(start[0])])
            route_coords.append([float(end[1]), float(end[0])])
        folium.PolyLine(route_coords, color="#24a148", weight=5, tooltip="Reference route").add_to(
            ordered_route_layer
        )
    ordered_route_layer.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.save(str(output_path))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="Path to supplied Student folder")
    parser.add_argument("--partition", choices=("train", "student_test"), default="train")
    parser.add_argument("--trip-id", required=True)
    parser.add_argument("--output", default="outputs/maps/trip.html")
    args = parser.parse_args()
    trip = load_trip(Path(args.data_root) / args.partition, args.trip_id)
    print(make_trip_map(trip, args.output))


if __name__ == "__main__":
    main()

