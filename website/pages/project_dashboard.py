"""San Diego vessel route dashboard.

This page combines the three-month 30-minute route-window feature file with
the teammate's vessel-level categorized anomaly output. Only vessels that
were scored by the teammate's model are displayed.
"""

from pathlib import Path
import json

import folium
import pandas as pd
import streamlit as st
from folium.plugins import Fullscreen, PolyLineTextPath
from streamlit_folium import st_folium


# These are the bounds used by the San Diego feature-engineering workflow.
MIN_LAT, MAX_LAT = 32.50, 32.95
MIN_LON, MAX_LON = -117.65, -117.05
AIS_GAP_BREAK_MINUTES = 45

# Route colors follow the vessel-classification guide supplied for the project.
VESSEL_GROUP_COLORS = {
    "Cargo": "#0052cc",
    "Tanker": "#d90429",
    "Passenger": "#1b9e3e",
    "Fishing": "#e67e00",
    "Tug Tow": "#7b2cbf",
    "Military": "#555555",
    "Pleasure Craft/Sailing": "#d63384",
    "Unknown": "#147d8d",
}

# Each overlay has a file stem, a unique color, and a line pattern.
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


def search_roots():
    """Return common project locations without assuming one launch directory."""
    here = Path(__file__).resolve().parent
    roots = [here, Path.cwd(), here / "data", here / "upload"]
    roots.extend(list(here.parents[:3]))
    return list(dict.fromkeys(p for p in roots if p.exists()))


def find_csv(pattern):
    """Find a CSV in data/processed, data, beside this file, or upload."""
    candidates = []
    for root in search_roots():
        candidates.extend(root.glob(pattern))
        candidates.extend((root / "data" / "processed").glob(pattern))
        candidates.extend((root / "data").glob(pattern))
    candidates = sorted(set(p.resolve() for p in candidates if p.is_file()))
    if not candidates:
        st.error(f"Could not find a CSV matching {pattern!r}.")
        st.info("Put the file in data/processed, beside this script, or an upload folder.")
        st.stop()
    return candidates[0]


def find_geojson(stem):
    """Find overlay files even when copied names contain suffixes such as (1)."""
    candidates = []
    for root in search_roots():
        candidates.extend(root.glob(f"{stem}*.geojson"))
        candidates.extend((root / "data" / "geospatial").glob(f"{stem}*.geojson"))
    candidates = sorted(set(p.resolve() for p in candidates if p.is_file()))
    return candidates[0] if candidates else None


@st.cache_data
def load_data():
    """Load mapped windows and attach the teammate's final model labels."""
    # This file supplies the map geometry and window-level AIS features for the
    # three-month study period (January through March 2020 in this dataset).
    features_path = find_csv("san_diego_30min_features*.csv")
    # This is the final model output after anomaly categories were assigned.
    anomaly_path = find_csv("san_diego_categorized_anomalies*.csv")

    required = [
        "mmsi", "window_start", "start_lat", "start_lon", "end_lat", "end_lon"
    ]
    # Read only the columns used by the dashboard. The feature export has
    # roughly 900k rows, so loading all engineered/weather columns wastes
    # memory and slows every first page load.
    feature_columns = set(pd.read_csv(features_path, nrows=0).columns)
    feature_usecols = [c for c in required + [
        "max_ais_gap_minutes", "max_speed_knots", "max_turn_rate_deg_min",
        "significant_turn_count", "position_jump_count", "large_ais_gap_count",
        "avg_speed_knots", "vessel_group", "vessel_type", "inside_military_area",
    ] if c in feature_columns]
    df = pd.read_csv(features_path, usecols=feature_usecols, low_memory=False)
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"The route feature file is missing required columns: {missing}")
        st.stop()

    # Loading only map and explanation fields reduces memory use for the
    # 900k-row feature file while preserving everything shown on this page.
    df = df.copy()
    df["mmsi"] = df["mmsi"].astype(str)
    df["window_start"] = pd.to_datetime(df["window_start"], utc=True, errors="coerce")
    df = df.dropna(subset=required).copy()

    model_columns = set(pd.read_csv(anomaly_path, nrows=0).columns)
    model_usecols = [c for c in [
        "mmsi", "is_anomalous", "anomaly_category",
        "predicted_behavior_class", "reported_class_probability", "vessel_group",
    ] if c in model_columns]
    model = pd.read_csv(anomaly_path, usecols=model_usecols, low_memory=False)
    model_required = ["mmsi", "is_anomalous", "anomaly_category"]
    missing_model = [c for c in model_required if c not in model.columns]
    if missing_model:
        st.error(f"The categorized anomaly file is missing required columns: {missing_model}")
        st.stop()

    # The classification file is one row per vessel. An inner merge is used so
    # unscored AIS vessels are not incorrectly presented as normal vessels.
    model_cols = [
        "mmsi", "is_anomalous", "anomaly_category",
        "predicted_behavior_class", "reported_class_probability", "vessel_group",
    ]
    model = model[[c for c in model_cols if c in model.columns]].drop_duplicates("mmsi")
    model["mmsi"] = model["mmsi"].astype(str)
    model = model.rename(columns={"vessel_group": "model_vessel_group"})
    df = df.merge(model, on="mmsi", how="inner")

    # The teammate's final anomaly label is authoritative for every remaining
    # row because the inner merge retained model-scored MMSIs only.
    df["unusual"] = df["is_anomalous"].fillna(False).astype(bool)
    df["anomaly_category"] = df["anomaly_category"].fillna("Unspecified model category")
    df["predicted_behavior_class"] = df["predicted_behavior_class"].fillna("Unknown")
    if "reported_class_probability" not in df:
        df["reported_class_probability"] = 0.0
    df["reported_class_probability"] = pd.to_numeric(
        df["reported_class_probability"], errors="coerce"
    ).fillna(0.0)
    if "vessel_group" not in df.columns:
        df["vessel_group"] = df["model_vessel_group"]
    df["vessel_group"] = (
        df["vessel_group"].fillna(df["model_vessel_group"])
        .fillna("Unknown").astype(str)
    )

    # Defaults keep older feature exports usable if a diagnostic column is absent.
    defaults = {
        "max_ais_gap_minutes": 0.0, "max_speed_knots": 0.0,
        "max_turn_rate_deg_min": 0.0, "significant_turn_count": 0.0,
        "position_jump_count": 0.0, "large_ais_gap_count": 0.0,
        "avg_speed_knots": 0.0, "inside_military_area": 0,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)

    df["ais_gap_flag"] = df["max_ais_gap_minutes"] > AIS_GAP_BREAK_MINUTES
    df["day"] = df["window_start"].dt.date

    # Keep only windows whose endpoints remain inside the study-area bounds.
    return df[
        df["start_lat"].between(MIN_LAT, MAX_LAT)
        & df["end_lat"].between(MIN_LAT, MAX_LAT)
        & df["start_lon"].between(MIN_LON, MAX_LON)
        & df["end_lon"].between(MIN_LON, MAX_LON)
    ].copy()


def add_geojson_overlay(map_object, name):
    """Add one independently colored and toggleable geospatial layer."""
    stem, color, dash = OVERLAY_CONFIG[name]
    path = find_geojson(stem)
    if path is None:
        st.sidebar.warning(f"Overlay file not found: {stem}*.geojson")
        return

    with path.open("r", encoding="utf-8") as file:
        geojson = json.load(file)
    layer = folium.FeatureGroup(name=name, overlay=True, show=True)

    def style(feature):
        is_line = feature.get("geometry", {}).get("type") == "LineString"
        result = {
            "color": color,
            "weight": 4 if is_line else 3,
            "fillColor": color,
            "fillOpacity": 0 if is_line else 0.05,
            "opacity": 0.95,
        }
        if dash:
            result["dashArray"] = dash
        return result

    folium.GeoJson(geojson, style_function=style).add_to(layer)
    layer.add_to(map_object)


def add_overlay_legend(map_object, selected):
    """Add a readable color key for active overlays inside the map."""
    if not selected:
        return
    rows = "".join(
        f"<div><span style='display:inline-block;width:13px;height:13px;"
        f"background:{OVERLAY_CONFIG[name][1]};border:1px solid #374151;"
        f"margin-right:8px;vertical-align:-2px;'></span>{name}</div>"
        for name in selected
    )
    legend_html = f"""<div style="position:fixed;bottom:32px;left:32px;z-index:9999;
        background:#fff;color:#152536;padding:12px 16px;min-width:220px;
        border:2px solid #374151;border-radius:7px;font:14px Arial;line-height:1.9;
        box-shadow:0 2px 8px rgba(0,0,0,.35)">
        <b>Active geospatial overlays</b>{rows}</div>"""
    map_object.get_root().html.add_child(folium.Element(legend_html))


st.set_page_config(page_title="San Diego Vessel Routes", page_icon="⚓", layout="wide")
st.title("San Diego three-month vessel route dashboard")

df = load_data()
available_days = sorted(df["day"].dropna().unique())
if not available_days:
    st.error("No valid route dates were found in the San Diego study area.")
    st.stop()
st.caption(
    f"Study period loaded from the feature file: {available_days[0]} through "
    f"{available_days[-1]} ({len(available_days)} days). Routes use 30-minute "
    "engineered start-to-end windows. Only vessels with a record in the "
    "teammate's final model output are shown. Red dashed segments indicate "
    "model-anomalous vessels."
)

defaults = {
    "start_day": available_days[0], "end_day": available_days[-1],
    "vessel_groups": ["All groups"], "vessels": ["All ships"],
    "route_filter": "All ships",
    "show_points": True, "overlays": [],
}
if "applied_filters" not in st.session_state:
    st.session_state.applied_filters = defaults.copy()
st.session_state.applied_filters = {**defaults, **st.session_state.applied_filters}


def as_date(value, fallback):
    """Convert old string/session-state dates into datetime.date objects."""
    try:
        return pd.to_datetime(value).date()
    except (TypeError, ValueError):
        return fallback


# Normalize state from earlier dashboard versions before date comparisons.
applied_state = st.session_state.applied_filters
applied_state["start_day"] = as_date(applied_state["start_day"], available_days[0])
applied_state["end_day"] = as_date(applied_state["end_day"], available_days[-1])
# Migrate the old single-vessel state into the new group and multi-vessel state.
if "vessel_groups" not in applied_state:
    applied_state["vessel_groups"] = ["All groups"]
if "vessels" not in applied_state:
    old_vessel = applied_state.get("vessel", "All ships")
    applied_state["vessels"] = [old_vessel]
if applied_state["start_day"] > applied_state["end_day"]:
    applied_state["start_day"], applied_state["end_day"] = (
        applied_state["end_day"], applied_state["start_day"]
    )

with st.sidebar:
    st.header("Filters")
    st.caption("Change settings, then click Apply map changes.")
    with st.form("map_filters"):
        start, end = st.date_input(
            "Date range",
            value=(applied_state["start_day"], applied_state["end_day"]),
            min_value=available_days[0],
            max_value=available_days[-1],
        )
        if start > end:
            start, end = end, start

        date_mask = df["day"].between(start, end)
        group_options = ["All groups"] + sorted(df.loc[date_mask, "vessel_group"].unique())
        selected_groups = st.multiselect(
            "Vessel groups",
            group_options,
            default=[g for g in applied_state["vessel_groups"] if g in group_options]
            or ["All groups"],
            help="Choose one or more vessel groups. Select All groups to include every group.",
        )
        group_mask = date_mask
        if selected_groups and "All groups" not in selected_groups:
            group_mask &= df["vessel_group"].isin(selected_groups)
        vessel_options = ["All ships"] + sorted(df.loc[group_mask, "mmsi"].unique())
        selected_vessels = st.multiselect(
            "Individual ships",
            vessel_options,
            default=[v for v in applied_state["vessels"] if v in vessel_options]
            or ["All ships"],
            help="Choose one or more MMSI values. Select All ships to include every ship in the selected groups.",
        )
        route_choices = ["All ships", "Unusual only", "Normal only"]
        route_filter = st.radio(
            "Show routes", route_choices,
            index=route_choices.index(applied_state["route_filter"]),
        )
        show_points = st.checkbox(
            "Show endpoints for selected vessel",
            value=applied_state["show_points"],
        )
        overlays = st.multiselect(
            "Geospatial overlays",
            list(OVERLAY_CONFIG),
            default=[x for x in applied_state["overlays"] if x in OVERLAY_CONFIG],
        )
        apply = st.form_submit_button(
            "Apply map changes", type="primary", use_container_width=True
        )

if apply:
    st.session_state.applied_filters = {
        "start_day": start, "end_day": end,
        "vessel_groups": selected_groups or ["All groups"],
        "vessels": selected_vessels or ["All ships"],
        "route_filter": route_filter, "show_points": show_points,
        "overlays": overlays,
    }
    st.rerun()

applied = st.session_state.applied_filters
range_df = df[df["day"].between(applied["start_day"], applied["end_day"])].copy()
# If specific groups are selected, they take priority over the default
# "All groups" token even if the user did not manually remove that token.
specific_groups = [g for g in applied["vessel_groups"] if g != "All groups"]
if specific_groups:
    range_df = range_df[range_df["vessel_group"].isin(specific_groups)]
ship_summary = range_df.groupby("mmsi", as_index=False).agg(
    unusual=("unusual", "any"),
    windows=("window_start", "size"),
    vessel_group=("vessel_group", "first"),
)

if applied["route_filter"] == "Unusual only":
    visible_ids = set(ship_summary.loc[ship_summary["unusual"], "mmsi"])
elif applied["route_filter"] == "Normal only":
    visible_ids = set(ship_summary.loc[~ship_summary["unusual"], "mmsi"])
else:
    visible_ids = set(ship_summary["mmsi"])
# If specific MMSIs are selected, they take priority over the default
# "All ships" token. This prevents a searched ship from silently rendering
# the entire date range because All ships remained selected.
specific_vessels = [v for v in applied["vessels"] if v != "All ships"]
if specific_vessels:
    visible_ids &= set(specific_vessels)
visible = range_df[range_df["mmsi"].isin(visible_ids)].copy()
single_vessel_selected = len(specific_vessels) == 1

# Cap the overview to keep Folium responsive. Selecting one vessel retains all
# of its windows, which is the detailed inspection mode.
# Folium creates a JavaScript object for every line. Keeping the overview at a
# lower cap prevents the browser from freezing when all vessels are selected.
MAX_OVERVIEW_SEGMENTS = 1500
MAX_DETAIL_SEGMENTS = 300
MAX_DIRECTION_ARROWS = 12
map_visible = visible
overview_sampled = False
if not single_vessel_selected and len(visible) > MAX_OVERVIEW_SEGMENTS:
    overview_sampled = True
    per_vessel = max(
        1, MAX_OVERVIEW_SEGMENTS // max(1, visible["mmsi"].nunique())
    )
    chunks = []
    for _, group in visible.sort_values("window_start").groupby("mmsi"):
        step = max(1, len(group) // per_vessel)
        chunks.append(group.iloc[::step].head(per_vessel))
    map_visible = pd.concat(chunks, ignore_index=True)


def limit_detail_route(route, maximum=MAX_DETAIL_SEGMENTS):
    """Keep important windows while thinning repetitive normal windows."""
    original_length = len(route)

    # A vessel can remain nearly stationary for many consecutive windows. Do
    # not send visually identical lines to the browser repeatedly. Rounding
    # coordinates to roughly 11 meters preserves the route shape while
    # collapsing tiny GPS fluctuations.
    compact = route.copy()
    compact["_start_lat_key"] = compact["start_lat"].round(4)
    compact["_start_lon_key"] = compact["start_lon"].round(4)
    compact["_end_lat_key"] = compact["end_lat"].round(4)
    compact["_end_lon_key"] = compact["end_lon"].round(4)
    compact = compact.drop_duplicates(
        subset=[
            "_start_lat_key", "_start_lon_key", "_end_lat_key", "_end_lon_key",
            "unusual", "ais_gap_flag",
        ],
        keep="first",
    ).drop(columns=[
        "_start_lat_key", "_start_lon_key", "_end_lat_key", "_end_lon_key",
    ])

    if len(compact) <= maximum:
        return compact, len(compact) < original_length

    route = compact

    # Always prioritize anomalous and uncertain AIS-gap windows because they
    # carry more investigative value than repeated normal movement windows.
    priority_mask = route["unusual"] | route["ais_gap_flag"]
    priority = route[priority_mask]
    normal = route[~priority_mask]

    if len(priority) > maximum:
        step = max(1, (len(priority) + maximum - 1) // maximum)
        priority = priority.iloc[::step].head(maximum)

    normal_budget = max(0, maximum - len(priority))
    if normal_budget and len(normal) > normal_budget:
        step = max(1, (len(normal) + normal_budget - 1) // normal_budget)
        normal = normal.iloc[::step].head(normal_budget)
    else:
        normal = normal.head(normal_budget)

    result = pd.concat([priority, normal]).sort_values("window_start")
    return result, True

c1, c2, c3, c4 = st.columns(4)
c1.metric("Model-scored ships in range", f"{len(ship_summary):,}")
c2.metric("Model-anomalous ships", f"{int(ship_summary['unusual'].sum()):,}")
c3.metric("Route windows", f"{len(range_df):,}")
c4.metric(
    "AIS-gap windows",
    f"{int(range_df['ais_gap_flag'].sum()):,}",
)
if overview_sampled:
    st.info(
        f"The all-ship map shows {len(map_visible):,} of {len(visible):,} "
        "route windows. Select one vessel for complete detail and direction arrows."
    )

detail_sampled = False
if single_vessel_selected and len(map_visible) > MAX_DETAIL_SEGMENTS:
    map_visible, detail_sampled = limit_detail_route(map_visible)
    st.info(
        f"This vessel has more than {MAX_DETAIL_SEGMENTS:,} route windows in the "
        "selected period. The map keeps anomaly/AIS-gap windows and samples "
        "repetitive normal windows for responsiveness."
    )

# Arrowheads are separate Leaflet layers and are the most expensive visual
# element. They are enabled for one-vessel inspection, but omitted from the
# all-ship overview so the browser remains responsive.
show_arrows = single_vessel_selected

st.markdown(f"### Routes: {applied['start_day']} through {applied['end_day']}")
m = folium.Map(
    location=[(MIN_LAT + MAX_LAT) / 2, (MIN_LON + MAX_LON) / 2],
    zoom_start=9, tiles="OpenStreetMap", control_scale=True,
)
Fullscreen().add_to(m)
for overlay_name in applied["overlays"]:
    add_geojson_overlay(m, overlay_name)
add_overlay_legend(m, applied["overlays"])

# Draw each engineered window separately. We never connect across a missing AIS
# period because the actual path between observations is unknown.
for _, route in map_visible.sort_values("window_start").groupby("mmsi"):
    route = route.sort_values("window_start")
    # A few arrows communicate direction without creating hundreds of extra
    # browser layers for a long route.
    arrow_step = max(1, len(route) // MAX_DIRECTION_ARROWS)
    tooltip_step = max(1, len(route) // 50)
    for segment_number, row in enumerate(route.itertuples()):
        vessel_color = VESSEL_GROUP_COLORS.get(row.vessel_group, "#147d8d")
        if row.ais_gap_flag:
            # Gray dotted segments show missing observations; the route color
            # cannot reliably describe an unobserved path.
            color, weight, dash, status = (
                "#808080", 3, "3 6", f"AIS gap > {AIS_GAP_BREAK_MINUTES} min"
            )
        elif row.unusual:
            # Suspicious vessels retain their vessel-group color but use a
            # dashed line, so color identifies type and dashes identify risk.
            color, weight, dash, status = (
                vessel_color, 4, "8 6", f"Model anomaly: {row.anomaly_category}"
            )
        else:
            color, weight, dash, status = (
                vessel_color, 2, None, "Normal model classification"
            )

        tooltip = None
        if segment_number % tooltip_step == 0:
            tooltip = (
                f"MMSI: {row.mmsi}<br>Window: {row.window_start}<br>"
                f"Vessel group: {row.vessel_group}<br>Status: {status}<br>"
                f"Predicted class: {row.predicted_behavior_class}<br>"
                f"Class probability: {row.reported_class_probability:.3f}"
            )

        segment = folium.PolyLine(
            [[row.start_lat, row.start_lon], [row.end_lat, row.end_lon]],
            color=color, weight=weight, opacity=0.85, dash_array=dash,
            tooltip=tooltip,
        ).add_to(m)
        if show_arrows and segment_number % arrow_step == 0:
            # Add repeated arrowheads in detailed mode so the line shows travel
            # direction from the engineered start point toward its end point.
            PolyLineTextPath(
                segment, "➤", repeat=True, offset=8,
                attributes={"fill": color, "font-size": "14", "font-weight": "bold"},
            ).add_to(m)

    if single_vessel_selected and applied["show_points"]:
        # Use the full filtered route for endpoint markers even when the map
        # line rendering sampled repetitive windows for performance.
        endpoint_route = visible[visible["mmsi"] == route["mmsi"].iloc[0]].sort_values("window_start")
        first, last = endpoint_route.iloc[0], endpoint_route.iloc[-1]
        folium.CircleMarker(
            [first["start_lat"], first["start_lon"]], radius=7,
            color="#111827", fill=True, fill_color="white", fill_opacity=1,
            tooltip="First observed window",
        ).add_to(m)
        folium.CircleMarker(
            [last["end_lat"], last["end_lon"]], radius=8,
            color="#111827", fill=True, fill_color="#d64545", fill_opacity=1,
            tooltip="Last observed window",
        ).add_to(m)

folium.LayerControl(collapsed=False).add_to(m)
st_folium(m, use_container_width=True, height=720, returned_objects=[])

st.markdown("### Model-anomaly explanations")
unusual = visible[visible["unusual"]].copy()
if unusual.empty:
    st.success("No model-anomalous windows are present in the current selection.")
else:
    # The model label is vessel-level, but the table is grouped by calendar
    # day so a ship appears once for each day containing anomalous windows.
    # This avoids repeating the same ship for every 30-minute window.
    unusual_summary = unusual.groupby(["mmsi", "day"], as_index=False).agg(
        vessel_group=("vessel_group", "first"),
        predicted_behavior_class=("predicted_behavior_class", "first"),
        reported_class_probability=("reported_class_probability", "first"),
        first_window=("window_start", "min"),
        last_window=("window_start", "max"),
        selected_windows=("window_start", "size"),
        anomaly_category=("anomaly_category", "first"),
        max_ais_gap_minutes=("max_ais_gap_minutes", "max"),
    )
    unusual_summary["first_window"] = unusual_summary["first_window"].dt.strftime(
        "%Y-%m-%d %H:%M UTC"
    )
    unusual_summary["last_window"] = unusual_summary["last_window"].dt.strftime(
        "%Y-%m-%d %H:%M UTC"
    )
    unusual_summary["day"] = pd.to_datetime(unusual_summary["day"]).dt.strftime(
        "%Y-%m-%d"
    )
    st.dataframe(
        unusual_summary.rename(columns={
            "mmsi": "MMSI",
            "day": "Anomaly day",
            "vessel_group": "Vessel group",
            "predicted_behavior_class": "Predicted class",
            "reported_class_probability": "Class probability",
            "first_window": "First selected window",
            "last_window": "Last selected window",
            "selected_windows": "Windows in selection",
            "anomaly_category": "Model explanation",
            "max_ais_gap_minutes": "Largest AIS gap (min)",
        }),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "One row represents one anomalous vessel on one day. A vessel appears "
        "on five rows only when it has anomalous route windows on five selected "
        "days. The model classification is vessel-level; the day grouping is "
        "a dashboard summary of when that vessel was observed."
    )

with st.expander("How to interpret the route lines"):
    st.markdown(
        f"""<div style="font-size:15px;line-height:1.9">
        <b>Vessel classification colors</b>
        <div style="color:#0052cc">━━ Cargo</div>
        <div style="color:#d90429">━━ Tanker</div>
        <div style="color:#1b9e3e">━━ Passenger</div>
        <div style="color:#e67e00">━━ Fishing</div>
        <div style="color:#7b2cbf">━━ Tug Tow</div>
        <div style="color:#555555">━━ Military</div>
        <div style="color:#d63384">━━ Pleasure Craft/Sailing</div>
        <br><b>Track and anomaly key</b>
        <div>━━ <b>Normal trajectory</b> — solid line.</div>
        <div>╌╌ <b>Model-anomalous trajectory</b> — dashed line.</div>
        <div>➤ <b>Direction of travel</b> — arrows point from start to end.</div>
        <div><span style="color:#808080;font-size:24px">╌╌</span>
        <b>AIS gap</b> — max gap exceeds {AIS_GAP_BREAK_MINUTES} minutes;
        the exact path is unknown.</div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.write(
        "The model output is vessel-level, while map geometry is window-level. "
        "Therefore, an anomalous vessel's visible windows are red unless an "
        "AIS-gap segment is shown gray."
    )

