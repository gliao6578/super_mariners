#4
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
input_file = PROJECT_ROOT / "data" / "processed" / "san_diego_ais_cleaned.csv"
output_file = PROJECT_ROOT / "data" / "processed" / "san_diego_30min_features"
output_file2 = PROJECT_ROOT / "data" / "processed" / "san_diego_30min_features.csv"

df = pd.read_csv(input_file).copy()

df["base_date_time"] = pd.to_datetime( #cvs does not save type so have to change it again
    df["base_date_time"],
    errors="coerce",
    utc=True)

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
        avg_implied_speed_knots = ("implied_speed_knots", "mean"),
        max_implied_speed_knots = ("implied_speed_knots", "max"),
        avg_speed_position_difference = ("speed_position_difference", "mean"),

        #loitering
        low_speed_observations = ("low_speed_flag", "sum"),
        significant_turn_flag = ("significant_turn_flag", "sum"),

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
        "heading_completeness"
    ]].mean(axis=1)
)

#remove windows with too few observations
window_features = window_features[
    window_features["observation_count"] >= 3
].copy()

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
