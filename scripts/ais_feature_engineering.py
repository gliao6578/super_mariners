#5
from pathlib import Path
import pandas as pd
import numpy as np
import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
input_file = PROJECT_ROOT / "data" / "processed" / "san_diego_ais_cleaned.csv"
output_file = PROJECT_ROOT / "data" / "processed" / "san_diego_30min_features"
output_file2 = PROJECT_ROOT / "data" / "processed" / "san_diego_30min_features.csv"

GEOSPATIAL_DIR = PROJECT_ROOT / "data" / "geospatial"
# Change these filenames to the actual files you download.
BAY_FILE = GEOSPATIAL_DIR / "san_diego_bay.geojson"
ANCHORAGE_FILE = GEOSPATIAL_DIR / "anchorages.geojson"
CHANNEL_FILE = GEOSPATIAL_DIR / "navigation_channel.geojson"
RESTRICTED_FILE = GEOSPATIAL_DIR / "restricted_areas.geojson"

MILITARY_FILE = (GEOSPATIAL_DIR / "military_areas.geojson")
OFFSHORE_SAFETY_FILE = (GEOSPATIAL_DIR / "offshore_safety_zones.geojson")
DEGAUSSING_FILE = (GEOSPATIAL_DIR / "degaussing_ranges.geojson")


# Geographic CRS used by AIS latitude/longitude.
GEOGRAPHIC_CRS = "EPSG:4326"

# Projected CRS used for distance calculations around San Diego.
PROJECTED_CRS = "EPSG:32611"

SAN_DIEGO_BAY_ENTRANCE_LAT = 32.6817
SAN_DIEGO_BAY_ENTRANCE_LON = -117.2300

#helper functions
def haversine_nm(lat1, lon1, lat2, lon2): #calculate geographic distance in nautical miles
    earth_radius = 6371.0088

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))   #c = 2 * np.arcsin(np.sqrt(a))

    distance_km = earth_radius * c
    return distance_km / 1.852

def angular_difference(current,previous): #return smallest signed angular difference
    return (current - previous + 180) % 360 - 180 #(current - previous) / np.pi * 180

def classify_movement_state(row): #classifies movement speed
    if row["avg_speed_knots"] <= 0.5:
        return "stationary"
    elif row["avg_speed_knots"] <=2:
        return "slow"
    else: return "underway"

def classify_vessel_group(vessel_type):
    if pd.isna(vessel_type) or vessel_type == -1:
        return "Not Available"

    vessel_type = int(vessel_type) # Make sure the value is an integer.

    if vessel_type == 0: #vessel not avail
        return "Not Available"

    elif vessel_type == 30: # Fishing vessel
        return "Fishing"

    elif vessel_type == 35:# Military operations
        return "Military"

    elif vessel_type in [36, 37]: # Sailing vessel or pleasure craft
        return "Pleasure Craft/Sailing"

    elif vessel_type in [21, 22, 31, 32, 52]: # Tug / towing vessels
        return "Tug Tow"

    elif 60 <= vessel_type <= 69: # Passenger vessels
        return "Passenger"

    elif 70 <= vessel_type <= 79: # Cargo vessels
        return "Cargo"

    elif 80 <= vessel_type <= 89: # Tankers
        return "Tanker"

    elif vessel_type in [1001, 1002]: # Fishing
        return "Fishing"

    elif vessel_type in [1003, 1004, 1016]: # Cargo
        return "Cargo"

    elif vessel_type in [1012, 1013, 1014, 1015]: # Passenger
        return "Passenger"

    elif vessel_type in [1017, 1024]: # Tanker
        return "Tanker"

    elif vessel_type == 1019: # Recreational
        return "Pleasure Craft/Sailing"

    elif vessel_type == 1021: # Military
        return "Military"

    elif vessel_type in [1023, 1025]: # Tug / towing
        return "Tug Tow"

    else: #pilot vessels, SAR vessels, dredging, diving, law enforcement, research vessels, high-speed craft, reserved codes, etc.
        return "Other"

# determines if each vessel is in the polygon; 1 if inside polygon, 0 if not
def add_inside_polygon_feature(vessel_points,polygon_file, feature_name):
    polygons = gpd.read_file(polygon_file)
    polygons = polygons.to_crs(vessel_points.crs) #convert to same coordinates as vessels
    polygons = polygons[["geometry"]].copy() #get only geometry
    polygons["inside_match"] = 1

    joined = gpd.sjoin( #for every vessel point, determine if its in any of the polygons
        vessel_points,
        polygons,
        how="left",
        predicate="within")

    #point can theoretically match more than one overlapping polygon.
    #Collapse the result back to the original row.
    result = (
        joined.groupby(joined.index)["inside_match"]
        .max()
        .reindex(vessel_points.index)
        .fillna(0)
        .astype(int))

    result.name = feature_name
    return result

def add_distance_to_feature(vessel_points,feature_file):
    features = gpd.read_file(feature_file)
    vessel_points = vessel_points.to_crs(GEOGRAPHIC_CRS)
    features = features.to_crs(GEOGRAPHIC_CRS)
    vessel_projected = vessel_points.to_crs("EPSG:32611")
    features_projected = features.to_crs("EPSG:32611")
    # features = features.to_crs(vessel_points.crs)  # convert to same coordinates as vessels
    # features = features[["geometry"]].copy()  # get only geometry
    # features["inside_match"] = 1

    nearest = gpd.sjoin_nearest(
        vessel_projected,
        features_projected,
        how="left",
        distance_col="distance_m"
    )

    # could be duplicate matches if two features are equally
    # close. Keep the minimum distance for each original row.
    distance_m = (nearest.groupby(nearest.index)["distance_m"].min().reindex(vessel_projected.index))

    return distance_m / 1852 # 1 nautical mile = 1852 meters.


#========================= End of methods ===========================================
df = pd.read_csv(input_file).copy()

df["base_date_time"] = pd.to_datetime( #cvs does not save type so have to change it again
    df["base_date_time"],
    errors="coerce",
    utc=True)

#sort by vessel and time
df = df.sort_values(by=["mmsi", "base_date_time"]).reset_index(drop=True)

grouped = df.groupby("mmsi", sort=False)

#previous observations
df["prev_time"] = (grouped["base_date_time"].shift(1))
df["prev_lat"] = (grouped["latitude"].shift(1))
df["prev_lon"] = (grouped["longitude"].shift(1))
df["prev_sog"] = (grouped["sog"].shift(1))
df["prev_cog"] = (grouped["cog"].shift(1))
df["prev_heading"] = (grouped["heading"].shift(1))

#time difference
df["time_diff_seconds"] = (df["base_date_time"] - df["prev_time"]).dt.total_seconds()
df["time_diff_minutes"] = (df["time_diff_seconds"] / 60)
df["time_diff_hours"] = (df["time_diff_minutes"] / 60)

invalid_time = (df["time_diff_seconds"] <= 0)
df.loc[invalid_time, ["time_diff_seconds", "time_diff_minutes", "time_diff_hours"]] = np.nan

#GROUP 1
#movement features

df["speed_change_knots"] = (df["sog"]-df["prev_sog"])   #speed change

df["acceleration_mps2"] = (df["speed_change_knots"] * 0.514444) / df["time_diff_seconds"] #acceleration, 1knot = 0.514444m/s

df["course_change_deg"] = (angular_difference(df["cog"],df["prev_cog"])) #course change
df["abs_course_change_deg"] = df["course_change_deg"].abs()

df["turn_rate_deg_min"] = (df["course_change_deg"] / df["time_diff_minutes"]) #turn rate
df["abs_turn_rate_deg_min"] = df["turn_rate_deg_min"].abs()

df["heading_change_deg"] = (angular_difference(df["heading"],df["prev_heading"])) #heading change
df["abs_heading_change_deg"] = df["heading_change_deg"].abs()

df["heading_cog_difference"] = np.abs(angular_difference(df["heading"],df["cog"])) #difference between heading and course

#GROUP 2
#position /route features

df["distance_nm"] = haversine_nm(df["prev_lat"],df["prev_lon"],df["latitude"],df["longitude"]) #distance from previous observation
df["implied_speed_knots"] = (df["distance_nm"] / df["time_diff_hours"]) #position derived speed
df["speed_position_difference"] = np.abs(df["implied_speed_knots"] - df["sog"]) #difference between reported and position derived speed

#GROUP 3
#loitering features

#INITIAL THRESHOLD ONLY !!!!! TUNE LATER BY VESSEL TYPE AND LOCATION
df["low_speed_flag"] = (df["sog"] <= 2).astype(int)

df["significant_turn_flag"] = (df["abs_course_change_deg"] >=20).astype(int)

#GROUP 4
#ais reporting features

df["large_ais_gap_flag"] = (df["time_diff_minutes"] >= 30).astype(int) #large gaps

df["position_jump_flag"] = (df["implied_speed_knots"] > 60).astype(int) #high speed may indicate jumped coordinates
df["implied_speed_knots_clean"] = (df["implied_speed_knots"].where(df["position_jump_flag"] == 0, np.nan))

#create 30 minute intervals
df["window_start"] = (df["base_date_time"].dt.floor("30min"))

window_features = (
    df.groupby(["mmsi", "window_start"], as_index=False)
    .agg(
        #movement
        observation_count = ("base_date_time", "count"),
        avg_speed_knots = ("sog", "mean"),
        max_speed_knots = ("sog", "max"),
        speed_std_knots = ("sog", "std"),
        avg_acceleration_mps2 = ("acceleration_mps2", "mean"),
        max_abs_acceleration_mps2 = ("acceleration_mps2", lambda x: x.abs().max()),
        avg_turn_rate_deg_min = ("abs_turn_rate_deg_min", "mean"),
        max_turn_rate_deg_min = ("abs_turn_rate_deg_min", "max"),
        avg_heading_change_deg = ("abs_heading_change_deg", "mean"),
        max_heading_change_deg = ("abs_heading_change_deg", "max"),
        avg_heading_cog_difference = ("heading_cog_difference", "mean"),

        #route / position
        total_distance_nm = ("distance_nm", "sum"),
        avg_implied_speed_knots=("implied_speed_knots_clean", "mean"),
        max_implied_speed_knots=("implied_speed_knots_clean", "max"),
        avg_speed_position_difference = ("speed_position_difference", "mean"),

        #loitering
        low_speed_observations = ("low_speed_flag", "sum"),
        significant_turn_count = ("significant_turn_flag", "sum"),

        #ais reporting
        avg_ais_gap_minutes = ("time_diff_minutes", "mean"),
        max_ais_gap_minutes = ("time_diff_minutes", "max"),
        large_ais_gap_count = ("large_ais_gap_flag", "sum"),
        position_jump_count = ("position_jump_flag", "sum"),

        #data quality
        sog_missing_fraction = ("sog_missing", "mean"),
        cog_missing_fraction = ("cog_missing", "mean"),
        heading_missing_fraction = ("heading_missing", "mean"),

        #window position
        start_lat = ("latitude", "first"),
        start_lon = ("longitude", "first"),
        end_lat = ("latitude", "last"),
        end_lon = ("longitude", "last"),

        #vessel context
        vessel_type = ("vessel_type_clean", "first"),
        nav_status = ("status_clean", "first"),
        vessel_length = ("length", "first"),
        vessel_width = ("width", "first"),
        vessel_draft = ("draft", "first")
    ))

#net displacement
window_features["net_displacement_nm"] = (
    haversine_nm(window_features["start_lat"],
                 window_features["start_lon"],
                 window_features["end_lat"],
                 window_features["end_lon"])
)

#movement efficiency
window_features["movement_efficiency"] = np.where(
    window_features["total_distance_nm"] > 0,
    ( window_features["net_displacement_nm"] / window_features["total_distance_nm"] ),
    np.nan
)

window_features["movement_efficiency"] = (
    window_features["movement_efficiency"].clip(0,1)
)

#low speed fraction
window_features["low_speed_fraction"] = (
    window_features["low_speed_observations"] / window_features["observation_count"]
)

#data completeness
window_features["sog_completeness"] = (1 - window_features["sog_missing_fraction"])
window_features["cog_completeness"] = (1 - window_features["cog_missing_fraction"])
window_features["heading_completeness"] = (1 - window_features["heading_missing_fraction"])
window_features["movement_data_completeness"] = (
    window_features[[
        "sog_completeness",
        "cog_completeness",
        #"heading_completeness"
    ]].mean(axis=1)
)

#remove windows with too few observations
window_features = window_features[
    window_features["observation_count"] >= 3
].copy()

# classify each type of vessel
window_features["vessel_group"] = (
    window_features["vessel_type"].apply(classify_vessel_group)
)

#GROUP 5
#Geospatial features

#convert final position to a geospatial point
vessel_points = gpd.GeoDataFrame(
    window_features.copy(),

    geometry=gpd.points_from_xy(
        window_features["end_lon"],
        window_features["end_lat"] ),
    crs=GEOGRAPHIC_CRS
)

# distance to sd bay entrance
window_features["distance_to_bay_entrance_nm"] = (
    haversine_nm(
        window_features["end_lat"],
        window_features["end_lon"],
        SAN_DIEGO_BAY_ENTRANCE_LAT,
        SAN_DIEGO_BAY_ENTRANCE_LON)
)

# apply movement state for each row
window_features["movement_state"] = (
    window_features.apply(classify_movement_state, axis=1)
)

vessel_projected = vessel_points.to_crs(PROJECTED_CRS)

if BAY_FILE.exists():
    window_features["inside_bay"] = ( #create feature whether vessel is inside sd bay
        add_inside_polygon_feature(
            vessel_points,
            BAY_FILE,
            "inside_bay")
    )
else: print(f"Bay polygon not found: {BAY_FILE}")

if ANCHORAGE_FILE.exists():
    window_features["inside_anchorage"] = ( #feature whether vessel is inside anchorage area
        add_inside_polygon_feature(
            vessel_points,
            ANCHORAGE_FILE,
            "inside_anchorage")
    )

    window_features["distance_to_anchorage_nm"] = ( #distance to nearest anchorage area (nautical mile)
        add_distance_to_feature(
            vessel_projected,
            ANCHORAGE_FILE
        )
    )
else: print(f"Anchorage file not found: {ANCHORAGE_FILE}")

if CHANNEL_FILE.exists():
    window_features["distance_to_shipping_channel_nm"] = ( #distance to nearest navigation channel (nautical mile)
        add_distance_to_feature(
            vessel_projected,
            CHANNEL_FILE)
    )
else: print(f"navigation channel file not found: {CHANNEL_FILE}")

if RESTRICTED_FILE.exists():
    window_features["inside_restricted_area"] = ( #whether vessel is inside restricted area
        add_inside_polygon_feature(
            vessel_points,
            RESTRICTED_FILE,
            "inside_restricted_area")
    )

    window_features["distance_to_restricted_area_nm"] = ( #distance to nearest restricted area (nautical mile)
        add_distance_to_feature(
            vessel_projected,
            RESTRICTED_FILE
        )
    )

    if MILITARY_FILE.exists():
        window_features["inside_military_area"] = (
            add_inside_polygon_feature(
                vessel_points,
                MILITARY_FILE,
                "inside_military_area"
            )
        )

        window_features[
            "distance_to_military_area_nm"
        ] = add_distance_to_feature(
            vessel_points,
            MILITARY_FILE
        )

    if OFFSHORE_SAFETY_FILE.exists():
        window_features[
            "inside_offshore_safety_zone"
        ] = add_inside_polygon_feature(
            vessel_points,
            OFFSHORE_SAFETY_FILE,
            "inside_offshore_safety_zone"
        )

        window_features[
            "distance_to_offshore_safety_zone_nm"
        ] = add_distance_to_feature(
            vessel_points,
            OFFSHORE_SAFETY_FILE
        )

    if DEGAUSSING_FILE.exists():
        window_features[
            "inside_degaussing_range"
        ] = add_inside_polygon_feature(
            vessel_points,
            DEGAUSSING_FILE,
            "inside_degaussing_range"
        )

        window_features[
            "distance_to_degaussing_range_nm"] = add_distance_to_feature(
            vessel_points,
            DEGAUSSING_FILE
        )

else: print(f"Restricted area file not found: {RESTRICTED_FILE}")

#window quality flag
# CAN CHANGE THRESHOLD AFTER MORE TESTING
window_features["usable_for_model"] = (
    (
        window_features["observation_count"] >= 3
    )
    &
    (
        window_features["movement_data_completeness"] >= 0.50
    )
)

# saving results
output_file.parent.mkdir(parents=True,exist_ok=True)
df.to_csv(output_file,index=False)
window_features.to_csv(output_file2,index=False)


#printing results
print()
print("Feature engineering complete.")
print()

print(
    f"Point-level observations: "
    f"{len(df):,}")

print(
    f"30-minute windows: "
    f"{len(window_features):,}")

print(
    f"Usable model windows: "
    f"{window_features['usable_for_model'].sum():,}")

print()
print(
    f"Point features saved to:\n"
    f"{output_file}")

print()
print(
    f"Window features saved to:\n"
    f"{output_file2}")

# print(
#     window_features[
#         [
#             "distance_to_bay_entrance_nm",
#             "distance_to_anchorage_nm",
#             "distance_to_shipping_channel_nm",
#             "distance_to_restricted_area_nm"
#         ]
#     ].describe()
# )