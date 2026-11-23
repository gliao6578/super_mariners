# Path locates project data files reliably. json reads the GeoJSON overlay files.
from pathlib import Path
import json

# Folium builds the interactive Leaflet map; pandas processes the AIS table;
# Streamlit creates the dashboard and its controls.
import folium
import pandas as pd
import streamlit as st
from folium.plugins import Fullscreen
from streamlit_folium import st_folium


# These bounds define the San Diego study area used to exclude unrelated data.
MIN_LAT, MAX_LAT = 32.50, 32.95
MIN_LON, MAX_LON = -117.65, -117.05

# A segment with a longer gap is shown as uncertain because its exact path is
# unknown between the last and next AIS observations.
AIS_GAP_BREAK_MINUTES = 45

# Each entry contains the file stem, display color, and optional dash pattern.
# Different patterns help distinguish overlapping geospatial layers.
OVERLAY_CONFIG = {
    "San Diego Bay": ("san_diego_bay", "#0077b6", None),
    "Navigation channels": ("navigation_channel", "#6a3d9a", "8 5"),
    "Anchorages": ("anchorages", "#f28e2b", "3 3"),
    "Restricted areas": ("restricted_areas", "#e31a1c", "10 4"),
    "Named restricted areas": ("named_restricted_areas", "#d95f9f", "2 6"),
    "Military areas": ("military_areas", "#8c510a", "12 3 2 3"),
    "Degaussing ranges": ("degaussing_ranges", "#2ca25f", "5 2"),
    "Offshore safety zones": ("offshore_safety_zones", "#00a6d6", "1 4"),
}


def project_root():
    # The page is in project_root/website/pages/, so two parent levels reach
    # the project root regardless of the directory used to launch Streamlit.
    return Path(__file__).resolve().parents[2]


def find_csv_file():
    # The wildcard supports copied filenames such as (1) or (2).
    processed = project_root() / "data" / "processed"
    candidates = sorted(processed.glob("san_diego_30min_features_weather*.csv"))
    if not candidates:
        st.error(
            "Could not find the San Diego feature CSV. Expected a file matching "
            f"san_diego_30min_features_weather*.csv in {processed}"
        )
        st.stop()
    return candidates[0]


@st.cache_data
def load_data(uploaded_file=None):
    # Cache the large CSV so filter changes do not reread and reprocess it.
    source = uploaded_file if uploaded_file is not None else find_csv_file()
    df = pd.read_csv(source, low_memory=False)

    required = ["mmsi", "window_start", "start_lat", "start_lon", "end_lat", "end_lon"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    df["mmsi"] = df["mmsi"].astype(str)
    df["window_start"] = pd.to_datetime(df["window_start"], utc=True, errors="coerce")
    df = df.dropna(subset=required).copy()

    # These fields are optional in case an older engineered CSV is used.
    numeric_defaults = {
        "max_ais_gap_minutes": 0.0,
        "max_speed_knots": 0.0,
        "max_turn_rate_deg_min": 0.0,
        "significant_turn_count": 0.0,
        "position_jump_count": 0.0,
        "large_ais_gap_count": 0.0,
        "inside_military_area": 0,
        "avg_speed_knots": 0.0,
    }
    for column, default in numeric_defaults.items():
        if column not in df.columns:
            df[column] = default
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(default)

    if "vessel_group" not in df.columns:
        # Fall back to vessel_type if an older engineered file uses that name.
        if "vessel_type" in df.columns:
            df["vessel_group"] = df["vessel_type"]
        else:
            df["vessel_group"] = "Unknown"
    df["vessel_group"] = df["vessel_group"].fillna("Unknown").astype(str)

    # Transparent first-pass rules create explainable unusual-window flags.
    # These can later be replaced or supplemented by a trained model.
    df["abrupt_maneuver"] = (df["max_turn_rate_deg_min"] >= 120) | (df["significant_turn_count"] >= 15)
    df["position_jump"] = df["position_jump_count"] > 0
    df["ais_gap_flag"] = df["large_ais_gap_count"] > 0
    df["very_high_speed"] = df["max_speed_knots"] >= 20
    df["military_context"] = df["inside_military_area"].astype(int) == 1

    reason_columns = [
        "abrupt_maneuver",
        "position_jump",
        "ais_gap_flag",
        "very_high_speed",
        "military_context",
    ]
    reason_labels = {
        "abrupt_maneuver": "Abrupt maneuver",
        "position_jump": "Position jump",
        "ais_gap_flag": "AIS gap flag",
        "very_high_speed": "Very high speed",
        "military_context": "Military-area context",
    }
    # A window is unusual if at least one rule is true. The reasons are saved
    # for the tooltip and the explanation table.
    df["unusual"] = df[reason_columns].any(axis=1)
    df["reasons"] = df.apply(
        lambda row: ", ".join(
            reason_labels[column] for column in reason_columns if bool(row[column])
        ) or "No rule triggered",
        axis=1,
    )
    df["day"] = df["window_start"].dt.date

    # Keep only rows whose start and end coordinates are inside the study area.
    return df[
        df["start_lat"].between(MIN_LAT, MAX_LAT)
        & df["end_lat"].between(MIN_LAT, MAX_LAT)
        & df["start_lon"].between(MIN_LON, MAX_LON)
        & df["end_lon"].between(MIN_LON, MAX_LON)
    ].copy()


def find_geojson(stem):
    # Search by filename stem so suffixes such as (1) are accepted.
    folder = project_root() / "data" / "geospatial"
    matches = sorted(folder.glob(f"{stem}*.geojson"))
    return matches[0] if matches else None


def add_geojson_overlay(map_object, display_name, stem, color):
    # Add each selected GeoJSON as an independent toggleable map layer.
    path = find_geojson(stem)
    if path is None:
        st.sidebar.warning(f"Overlay file not found: {stem}*.geojson")
        return

    with path.open("r", encoding="utf-8") as file:
        geojson = json.load(file)

    layer = folium.FeatureGroup(name=display_name, overlay=True, show=True)
    dash_pattern = OVERLAY_CONFIG[display_name][2]

    def style_function(feature):
        geometry_type = feature.get("geometry", {}).get("type", "")
        style = {
            "color": color,
            "weight": 4 if geometry_type == "LineString" else 3,
            "fillColor": color,
            "fillOpacity": 0.025 if geometry_type != "LineString" else 0,
            "opacity": 0.95,
        }
        if dash_pattern:
            style["dashArray"] = dash_pattern
        return style

    folium.GeoJson(
        geojson,
        style_function=style_function,
        highlight_function=lambda feature: {
            "color": "#111827",
            "weight": 5,
            "fillOpacity": 0.12,
        },
    ).add_to(layer)
    layer.add_to(map_object)


def add_overlay_legend(map_object, selected_overlays):
    # Describe active geospatial overlays outside the route-segment legend.
    if not selected_overlays:
        return
    rows = "".join(
        f"<div><span style='display:inline-block;width:13px;height:13px;"
        f"background:{OVERLAY_CONFIG[name][1]};border:1px solid #374151;"
        f"margin-right:8px;vertical-align:-2px;'></span>{name}</div>"
        for name in selected_overlays
    )
    legend_html = f"""
    <div style="position: fixed; bottom: 32px; left: 32px; z-index: 9999;
                background: #ffffff !important; color: #152536 !important;
                padding: 12px 16px; min-width: 220px;
                border: 2px solid #374151; border-radius: 7px;
                font-family: Arial, sans-serif; font-size: 14px;
                line-height: 1.9; box-shadow: 0 2px 8px rgba(0,0,0,.35);">
        <div style="color:#152536 !important; font-weight:700; margin-bottom:3px;">
            Active geospatial overlays
        </div>
        {rows}
    </div>
    """
    map_object.get_root().html.add_child(folium.Element(legend_html))


def add_route_legend(map_object):
    # Use Leaflet's own control positioning instead of fixed/absolute CSS.
    # This guarantees that the legend is anchored inside the map viewport.
    map_name = map_object.get_name()
    legend_script = f"""
    <script>
    (function() {{
        var routeLegend = L.control({{position: 'bottomright'}});
        routeLegend.onAdd = function(map) {{
            var div = L.DomUtil.create('div', 'route-legend leaflet-control');
            div.style.background = '#ffffff';
            div.style.color = '#152536';
            div.style.padding = '10px 14px';
            div.style.minWidth = '190px';
            div.style.border = '2px solid #374151';
            div.style.borderRadius = '7px';
            div.style.fontFamily = 'Arial, sans-serif';
            div.style.fontSize = '14px';
            div.style.lineHeight = '1.9';
            div.style.boxShadow = '0 2px 8px rgba(0,0,0,.3)';
            div.innerHTML =
                '<div style="font-weight:700;margin-bottom:3px;">Route segments</div>' +
                '<div><span style="color:#147d8d;font-size:20px;">━━</span> Normal</div>' +
                '<div><span style="color:#d64545;font-size:20px;">━━</span> Unusual</div>' +
                '<div><span style="color:#808080;font-size:20px;">╌╌</span> AIS gap &gt; 45 min</div>';
            return div;
        }};
        routeLegend.addTo({map_name});
    }})();
    </script>
    """
    map_object.get_root().script.add_child(folium.Element(legend_script))


st.set_page_config(page_title="San Diego Vessel Routes", page_icon="⚓", layout="wide")
st.title("San Diego vessel route dashboard")
st.caption("Routes are drawn as individual 30-minute start-to-end segments. A route may continue before or after the selected date range.")

# The dashboard uses the project CSV from data/processed.
# The uploader is intentionally not shown in the sidebar.
df = load_data()
available_days = sorted(df["day"].dropna().unique())

default_filters = {
    "start_day": available_days[0],
    "end_day": available_days[-1],
    "vessel": "All ships",
    "route_filter": "All ships",
    "show_points": True,
    "overlays": [],
}


def normalize_date(value, fallback):
    """Convert old string/session-state dates into datetime.date objects."""
    try:
        return pd.to_datetime(value).date()
    except (TypeError, ValueError, AttributeError):
        return fallback

# Streamlit preserves widget state between reruns. This migration converts the
# old single-day state into the newer start_day/end_day date-range state.
if "applied_filters" not in st.session_state:
    st.session_state.applied_filters = default_filters.copy()
else:
    existing_filters = st.session_state.applied_filters

    if "start_day" not in existing_filters:
        legacy_day = existing_filters.get("day", available_days[0])
        existing_filters = {
            "start_day": legacy_day,
            "end_day": legacy_day,
            "vessel": existing_filters.get("vessel", "All ships"),
            "route_filter": existing_filters.get("route_filter", "All ships"),
            "show_points": existing_filters.get("show_points", True),
            "overlays": existing_filters.get("overlays", []),
        }

    # Fill any missing keys if the state came from an intermediate version.
    st.session_state.applied_filters = {**default_filters, **existing_filters,}

# Older sessions may store dates as strings, while df["day"] contains date
# objects. Normalize them before performing date comparisons.
st.session_state.applied_filters["start_day"] = normalize_date(
    st.session_state.applied_filters["start_day"],
    available_days[0],
)
st.session_state.applied_filters["end_day"] = normalize_date(
    st.session_state.applied_filters["end_day"],
    available_days[-1],
)

applied = st.session_state.applied_filters

with st.sidebar:
    st.header("Filters")
    st.caption("Change settings, then click Apply map changes.")

    # Using a form prevents every individual control change from rebuilding the
    # map. The map updates only after Apply map changes is clicked.
    with st.form("map_filters"):
        form_start_day, form_end_day = st.date_input(
            "Date range",
            value=(applied["start_day"], applied["end_day"]),
            min_value=available_days[0],
            max_value=available_days[-1],
        )
        if form_start_day > form_end_day:
            form_start_day, form_end_day = form_end_day, form_start_day

        form_day_mask = df["day"].between(form_start_day, form_end_day)
        vessel_options = ["All ships"] + sorted(df.loc[form_day_mask, "mmsi"].unique())
        form_vessel = st.selectbox(
            "Vessel route",
            vessel_options,
            index=(vessel_options.index(applied["vessel"]) if applied["vessel"] in vessel_options else 0),
            help="Choose one MMSI to view its route across the selected date range.",
        )
        route_choices = ["All ships", "Unusual only", "Normal only"]
        form_route_filter = st.radio(
            "Show routes", route_choices, index=route_choices.index(applied["route_filter"])
        )
        form_show_points = st.checkbox("Show endpoints for selected vessel", value=applied["show_points"])
        form_overlays = st.multiselect(
            "Geospatial overlays",
            list(OVERLAY_CONFIG),
            default=[name for name in applied["overlays"] if name in OVERLAY_CONFIG],
        )
        apply_changes = st.form_submit_button("Apply map changes", type="primary", use_container_width=True)

if apply_changes:
    # Save all submitted controls together, then rerun once so the map,
    # statistics, overlays, and table use the same filter state.
    st.session_state.applied_filters = {
        "start_day": form_start_day,
        "end_day": form_end_day,
        "vessel": form_vessel,
        "route_filter": form_route_filter,
        "show_points": form_show_points,
        "overlays": form_overlays,
    }
    st.rerun()

applied = st.session_state.applied_filters
selected_start = applied["start_day"]
selected_end = applied["end_day"]
selected_vessel = applied["vessel"]
route_filter = applied["route_filter"]
selected_overlays = applied["overlays"]

# Apply the date range only after the user submits the filter form.
range_df = df[df["day"].between(selected_start, selected_end)].copy()

# Count ships once per MMSI, rather than counting every 30-minute window.
ship_summary = range_df.groupby("mmsi", as_index=False).agg(
    unusual=("unusual", "any"), windows=("window_start", "size")
)

if route_filter == "Unusual only":
    visible_ids = set(ship_summary.loc[ship_summary["unusual"], "mmsi"])
elif route_filter == "Normal only":
    visible_ids = set(ship_summary.loc[~ship_summary["unusual"], "mmsi"])
else:
    visible_ids = set(ship_summary["mmsi"])

if selected_vessel != "All ships":
    visible_ids &= {selected_vessel}

visible = range_df[range_df["mmsi"].isin(visible_ids)].copy()

# Rendering one Folium PolyLine per window can overwhelm the browser when all
# ships and an entire month are selected. Keep complete detail for one vessel,
# and use a small per-vessel overview sample otherwise.
MAX_OVERVIEW_SEGMENTS = 5000
map_visible = visible
overview_sampled = False

if selected_vessel == "All ships" and len(visible) > MAX_OVERVIEW_SEGMENTS:
    overview_sampled = True
    vessel_count = max(1, visible["mmsi"].nunique())
    segments_per_vessel = max(1, MAX_OVERVIEW_SEGMENTS // vessel_count)
    sampled_routes = []

    for _, vessel_route in visible.sort_values("window_start").groupby("mmsi"):
        step = max(1, len(vessel_route) // segments_per_vessel)
        sampled_routes.append(vessel_route.iloc[::step].head(segments_per_vessel))

    map_visible = pd.concat(sampled_routes, ignore_index=True)

metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("Ships in range", f"{len(ship_summary):,}")
metric2.metric("Unusual ships", f"{int(ship_summary['unusual'].sum()):,}")
metric3.metric("Route windows", f"{len(range_df):,}")
metric4.metric("AIS-gap windows", f"{int((range_df['max_ais_gap_minutes'] > AIS_GAP_BREAK_MINUTES).sum()):,}")

if overview_sampled:
    st.info(
        f"The all-ship map is showing an overview sample of {len(map_visible):,} "
        f"route segments instead of all {len(visible):,}. Select one vessel for "
        "its complete route."
    )

st.markdown(f"### Routes: {selected_start} through {selected_end}")

# Create the OpenStreetMap base layer. Routes and GeoJSON overlays are added to
# this map before it is embedded in the Streamlit page.
m = folium.Map(
    location=[(MIN_LAT + MAX_LAT) / 2, (MIN_LON + MAX_LON) / 2],
    zoom_start=9,
    tiles="OpenStreetMap",
    control_scale=True,
)
Fullscreen().add_to(m)

for overlay_name in selected_overlays:
    stem, color, _ = OVERLAY_CONFIG[overlay_name]
    add_geojson_overlay(m, overlay_name, stem, color)

add_overlay_legend(m, selected_overlays)

# Draw each window separately instead of connecting all start points. This
# preserves the engineered start-to-end movement and avoids false long routes.
for mmsi, route in map_visible.sort_values("window_start").groupby("mmsi"):
    route = route.sort_values("window_start")

    for row in route.itertuples():
        ais_gap = float(row.max_ais_gap_minutes)

        # Gray dashed means the route is not continuously observed.
        if ais_gap > AIS_GAP_BREAK_MINUTES:
            segment_color = "#808080"
            segment_style = "dashed"
            segment_weight = 3
            status = f"AIS gap > {AIS_GAP_BREAK_MINUTES} minutes"
        # Red identifies only this unusual window, not the entire voyage.
        elif row.unusual:
            segment_color = "#d64545"
            segment_style = "solid"
            segment_weight = 4
            status = f"Unusual: {row.reasons}"
        # Blue means none of the current screening rules triggered.
        else:
            segment_color = "#147d8d"
            segment_style = "solid"
            segment_weight = 2
            status = "Normal"

        folium.PolyLine(
            locations=[[row.start_lat, row.start_lon], [row.end_lat, row.end_lon]],
            color=segment_color,
            weight=segment_weight,
            opacity=0.85,
            dash_array="8 6" if segment_style == "dashed" else None,
            tooltip=(
                f"MMSI: {row.mmsi}<br>Window: {row.window_start}<br>"
                f"Status: {status}<br>AIS gap: {ais_gap:.1f} minutes<br>"
                f"Average speed: {row.avg_speed_knots:.2f} knots"
            ),
        ).add_to(m)

    # Endpoints are limited to single-vessel mode so the all-ship map stays
    # readable and does not contain hundreds of extra markers.
    if selected_vessel != "All ships" and applied["show_points"]:
        first, last = route.iloc[0], route.iloc[-1]
        folium.CircleMarker(
            [first["start_lat"], first["start_lon"]], radius=7,
            color="#111827", fill=True, fill_color="white", fill_opacity=1,
            tooltip="First observed window in selected range",
        ).add_to(m)
        folium.CircleMarker(
            [last["end_lat"], last["end_lon"]], radius=8,
            color="#111827", fill=True, fill_color="#d64545", fill_opacity=1,
            tooltip="Last observed window in selected range",
        ).add_to(m)

folium.LayerControl(collapsed=False).add_to(m)
st_folium(m, use_container_width=True, height=720, returned_objects=[])

if selected_vessel != "All ships":
    selected_summary = ship_summary[ship_summary["mmsi"] == selected_vessel]
    if not selected_summary.empty:
        selected_summary = selected_summary.iloc[0]
        status = "UNUSUAL" if selected_summary["unusual"] else "Normal"
        st.info(
            f"MMSI {selected_vessel}: {status}. {selected_summary['windows']} windows "
            f"in the selected range. Gray dashed segments indicate AIS gaps over "
            f"{AIS_GAP_BREAK_MINUTES} minutes."
        )

# Show the vessel group and feature values behind every red segment.
st.markdown("### Unusual-window explanations")
unusual_rows = visible[visible["unusual"]].copy()
if unusual_rows.empty:
    st.success("No unusual windows are present in the current map selection.")
else:
    unusual_rows["window_start"] = unusual_rows["window_start"].dt.strftime("%Y-%m-%d %H:%M UTC")
    st.dataframe(
        unusual_rows[["mmsi", "vessel_group", "window_start", "reasons", "max_ais_gap_minutes", "max_speed_knots", "max_turn_rate_deg_min"]].rename(
            columns={
                "mmsi": "MMSI", "vessel_group": "Vessel group",
                "window_start": "Window", "reasons": "Reason(s)",
                "max_ais_gap_minutes": "Max AIS gap (min)",
                "max_speed_knots": "Max speed (knots)",
                "max_turn_rate_deg_min": "Max turn rate (deg/min)",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

with st.expander("How to interpret the route lines"):
    st.markdown(
        f"""
        <div style="font-size:15px; line-height:1.9;">
            <div><span style="color:#147d8d;font-size:24px;">━━</span>
                <b>Normal segment</b> — no unusual-behavior rule was triggered.</div>
            <div><span style="color:#d64545;font-size:24px;">━━</span>
                <b>Unusual segment</b> — at least one screening rule was triggered.</div>
            <div><span style="color:#808080;font-size:24px;">╌╌</span>
                <b>AIS-gap segment</b> — the window contains an AIS gap greater than
                {AIS_GAP_BREAK_MINUTES} minutes, so the exact path is unknown.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "Each segment represents one 30-minute feature window from its start "
        "coordinate to its end coordinate. A route can continue before or after "
        "the selected date range."
    )
