#7
from pathlib import Path

import numpy as np
import pandas as pd


# FILE PATHS
PROJECT_ROOT = Path(__file__).resolve().parent.parent
AIS_FILE = (PROJECT_ROOT / "data" / "processed" / "san_diego_30min_featuresOLD.csv")
WEATHER_FILE = ( PROJECT_ROOT / "data" / "weather" / "ndbc_weather.csv")
OUTPUT_FILE = ( PROJECT_ROOT / "data" / "processed" / "san_diego_30min_features_weather.csv")


# WEATHER STATION INFORMATION
SDBC1_LAT = 32.714
SDBC1_LON = -117.174

LJAC1_LAT = 32.867
LJAC1_LON = -117.257

POINT_LOMA_LAT = 32.517
POINT_LOMA_LON = -117.425


# HELPER FUNCTION
# HAVERSINE DISTANCE
def haversine_nm(lat1, lon1, lat2, lon2):
    """
    Calculate geographic distance between two coordinates.
    Returns distance in nautical miles.
    """

    earth_radius_km = 6371.0088

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)

    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance_km = earth_radius_km * c
    return distance_km / 1.852


# LOAD AIS FEATURES
print()
print("=" * 70)
print("LOADING AIS FEATURE DATA")
print("=" * 70)

ais = pd.read_csv(AIS_FILE, low_memory=False)


# Convert AIS window timestamp back into datetime.
ais["window_start"] = pd.to_datetime(ais["window_start"], errors="coerce", utc=True)

# Remove rows where the timestamp could not be parsed.
ais = ais.dropna(subset=["window_start"]).copy()

print(f"AIS windows loaded: {len(ais):,}")



# LOAD WEATHER DATA
print()
print("=" * 70)
print("LOADING NDBC WEATHER DATA")
print("=" * 70)

weather = pd.read_csv(WEATHER_FILE, low_memory=False)
weather["timestamp"] = pd.to_datetime(weather["timestamp"], errors="coerce",utc=True)
weather = weather.dropna(subset=["timestamp"]).copy()


print(f"Weather observations loaded: {len(weather):,}")
print(f"Weather stations: {weather['station_id'].nunique()}")


# SPLIT WEATHER DATA BY STATION
# SDBC1
#
# San Diego Bay environmental station.
#
# We use this station for:
#
# pressure
# air temperature
# water temperature
# dew point
# visibility

sdbc1 = weather[weather["station_id"].astype(str).str.upper() == "SDBC1"].copy()

# LJAC1
#
# La Jolla station.
#
# We use this station primarily for:
#
# wind direction
# wind speed
# wind gust
# ------------------------------------------------------------

ljac1 = weather[weather["station_id"].astype(str).str.upper() == "LJAC1"].copy()

# ------------------------------------------------------------
# 46232
#
# Point Loma South offshore wave buoy.
#
# We use this station primarily for:
#
# wave height
# dominant wave period
# average wave period
# wave direction
# ------------------------------------------------------------

point_loma = weather[weather["station_id"].astype(str) == "46232"].copy()


print()
print("Station observations:")
print(f"SDBC1: {len(sdbc1):,}")
print(f"LJAC1: {len(ljac1):,}")
print(f"46232: {len(point_loma):,}")


# PREPARE SDBC1 DATA
bay_columns = [
    "timestamp",
    "pressure_hpa",
    "air_temperature_c",
    "water_temperature_c"
    # "dewpoint_temperature_c",
    # "visibility_nmi"
]


# Keep only columns that actually exist.
bay_columns = [column for column in bay_columns if column in sdbc1.columns]
bay_weather = (sdbc1[bay_columns].copy())


# Rename timestamp so we retain the actual weather
# observation time after merging.
bay_weather = bay_weather.rename(columns={"timestamp":"bay_weather_timestamp"})

# Sort because merge_asof requires sorted timestamps.
bay_weather = bay_weather.sort_values("bay_weather_timestamp")


# PREPARE LJAC1 WIND DATA
wind_columns = [
    "timestamp",
    "wind_direction_deg",
    "wind_speed_mps",
    "wind_gust_mps"
]


wind_columns = [column for column in wind_columns if column in ljac1.columns]
wind_weather = (ljac1[wind_columns].copy())

# ------------------------------------------------------------
# IMPORTANT:
#
# LJAC1 contains some observations where wind variables
# are unavailable.
#
# Remove rows where ALL wind measurements are missing.
#
# This prevents merge_asof from selecting a timestamp that
# contains no useful wind information.
# ------------------------------------------------------------

available_wind_columns = [column
    for column in [
        "wind_direction_deg",
        "wind_speed_mps",
        "wind_gust_mps"
    ]
    if column in wind_weather.columns
]


if available_wind_columns:
    wind_weather = wind_weather.dropna(subset=available_wind_columns, how="all")

wind_weather = wind_weather.rename(columns={"timestamp": "wind_weather_timestamp"})
wind_weather = wind_weather.sort_values("wind_weather_timestamp")


# PREPARE POINT LOMA WAVE DATA
wave_columns = [
    "timestamp",
    "wave_height_m",
    "dominant_wave_period_s",
    "average_wave_period_s",
    "wave_direction_deg",
    "water_temperature_c"
]


wave_columns = [column for column in wave_columns if column in point_loma.columns]
wave_weather = (point_loma[wave_columns].copy())


# Remove observations where ALL wave measurements are missing.
available_wave_columns = [
    column
    for column in [
        "wave_height_m",
        "dominant_wave_period_s",
        "average_wave_period_s",
        "wave_direction_deg"
    ]
    if column in wave_weather.columns
]


if available_wave_columns:
    wave_weather = wave_weather.dropna(subset=available_wave_columns,how="all")


# Rename Point Loma water temperature so it does not collide
# with SDBC1's bay water temperature.

if "water_temperature_c" in wave_weather.columns:
    wave_weather = wave_weather.rename(columns={"water_temperature_c": "offshore_water_temperature_c"})

wave_weather = wave_weather.rename(columns={"timestamp":"wave_weather_timestamp"})
wave_weather = wave_weather.sort_values("wave_weather_timestamp")


# SORT AIS DATA
ais = ais.sort_values("window_start").reset_index(drop=True)

# MERGE SDBC1 BAY WEATHER
print()
print("Merging SDBC1 bay weather...")


merged = pd.merge_asof(
    ais,
    bay_weather,
    left_on="window_start",
    right_on="bay_weather_timestamp",

    direction="nearest",

    # Do not accept a weather observation more than
    # 30 minutes away from the AIS window.
    tolerance=pd.Timedelta("30min"))


# MERGE LJAC1 WIND
print("Merging LJAC1 wind...")
merged = merged.sort_values("window_start")
merged = pd.merge_asof(
    merged,
    wind_weather,
    left_on="window_start",
    right_on="wind_weather_timestamp",
    direction="nearest",
    tolerance=pd.Timedelta("30min")
)

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# print("\nPOINT LOMA WAVE DEBUG")
# print("=" * 60)
#
# print("Wave rows available:", len(wave_weather))
#
# print("\nFirst wave observations:")
# print(
#     wave_weather[
#         [
#             "wave_weather_timestamp",
#             "wave_height_m",
#             "dominant_wave_period_s",
#             "average_wave_period_s",
#             "wave_direction_deg"
#         ]
#     ].head(20)
# )
#
# print("\nWave timestamp range:")
# print(
#     wave_weather["wave_weather_timestamp"].min()
# )
# print(
#     wave_weather["wave_weather_timestamp"].max()
# )
#
# print("\nAIS timestamp range:")
# print(
#     merged["window_start"].min()
# )
# print(
#     merged["window_start"].max()
# )#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# ============================================================
# MERGE POINT LOMA WAVES
# ============================================================

print("Merging Point Loma wave data...")

merged = merged.sort_values("window_start")
merged = pd.merge_asof(
    merged,
    wave_weather,
    left_on="window_start",
    right_on="wave_weather_timestamp",
    direction="nearest",
    tolerance=pd.Timedelta("30min"))



# CALCULATE WEATHER TIME DIFFERENCES
merged["bay_weather_time_diff_min"] = ((merged["window_start"]-merged["bay_weather_timestamp"]).abs().dt.total_seconds() / 60) # SDBC1 time difference
merged["wind_time_diff_min"] = ((merged["window_start"] -merged["wind_weather_timestamp"]).abs().dt.total_seconds() / 60) # LJAC1 wind time difference
merged["wave_time_diff_min"] = ((merged["window_start"] - merged["wave_weather_timestamp"]).abs().dt.total_seconds() / 60) # Point Loma wave time difference


# CALCULATE DISTANCE FROM VESSEL TO EACH WEATHER STATION
# We use the vessel's final location in each 30-minute window.
merged["distance_to_sdbc1_nm"] = (haversine_nm(merged["end_lat"], merged["end_lon"], SDBC1_LAT, SDBC1_LON)) # Distance to SDBC1
merged["distance_to_ljac1_nm"] = (haversine_nm(merged["end_lat"], merged["end_lon"], LJAC1_LAT, LJAC1_LON)) # Distance to LJAC1
merged["distance_to_46232_nm"] = (haversine_nm(merged["end_lat"],merged["end_lon"],POINT_LOMA_LAT,POINT_LOMA_LON)) # Distance to Point Loma South

# CREATE WEATHER MATCH FLAGS
# These flags indicate whether an observation from each weather
# station was successfully matched to the AIS window.
#
# IMPORTANT:
# A successful station match does NOT necessarily mean that
# every weather variable from that station is available.


# SDBC1 observation successfully matched
merged["bay_weather_available"] = (merged["bay_weather_timestamp"].notna().astype(int))


# LJAC1 observation successfully matched
merged["wind_weather_available"] = (merged["wind_weather_timestamp"].notna().astype(int))


# Point Loma 46232 observation successfully matched
merged["wave_weather_available"] = (merged["wave_weather_timestamp"].notna().astype(int))


# CREATE INDIVIDUAL WEATHER FEATURE AVAILABILITY FLAGS
# These flags indicate whether the actual environmental
# measurement is available for this AIS window.
# 1 = measurement available
# 0 = measurement missing

merged["wind_available"] = (merged["wind_speed_mps"].notna().astype(int)) # Wind speed
merged["pressure_available"] = (merged["pressure_hpa"].notna().astype(int)) # Atmospheric pressure
merged["water_temperature_available"] = (merged["water_temperature_c"].notna().astype(int)) # Water temperature
merged["air_temperature_available"] = (merged["air_temperature_c"].notna().astype(int)) # Air temperature
merged["wave_available"] = (merged["wave_height_m"].notna().astype(int)) # Wave height


# CONVERT DIRECTION VARIABLES
# Direction is circular.
#
# Example:
# 359 degrees and 1 degree are only 2 degrees apart.
# But numerically: 359 - 1 = 358
# That is bad for machine learning.
# Convert directions into sine/cosine components instead.

# Wind direction
if "wind_direction_deg" in merged.columns:
    wind_radians = np.radians(merged["wind_direction_deg"])
    merged["wind_direction_sin"] = (np.sin(wind_radians))
    merged["wind_direction_cos"] = (np.cos(wind_radians))


# Wave direction
if "wave_direction_deg" in merged.columns:
    wave_radians = np.radians(merged["wave_direction_deg"])
    merged["wave_direction_sin"] = (np.sin(wave_radians))
    merged["wave_direction_cos"] = (np.cos(wave_radians))


# WEATHER COMPLETENESS
# Count how many of our most important weather variables
# are available for each vessel window.

important_weather_features = [
    "wind_speed_mps",
    # "wave_height_m", omit for now at least for v1
    # "dominant_wave_period_s", omit for now at least for v1
    "pressure_hpa",
    # "air_temperature_c", #sdbc1 does not suply air temp so we remove
    "water_temperature_c"
    # "visibility_nmi"
]


important_weather_features = [column for column in important_weather_features if column in merged.columns]


merged["weather_completeness"] = (merged[important_weather_features].notna().mean(axis=1))

# SAVE FINAL AIS + WEATHER DATASET
OUTPUT_FILE.parent.mkdir(parents=True,exist_ok=True)
merged.to_csv(OUTPUT_FILE,index=False)

# PRINT RESULTS
print()
print("=" * 70)
print("WEATHER MERGE COMPLETE")
print("=" * 70)
print()
print(f"Original AIS windows: {len(ais):,}")
print(f"Final merged windows: {len(merged):,}")
print()


# Weather availability
print("Weather availability:")
print(f"SDBC1 bay weather: {merged['bay_weather_available'].mean() * 100:.1f}%")
print(f"LJAC1 wind: {merged['wind_weather_available'].mean() * 100:.1f}%")
print(f"46232 waves: {merged['wave_weather_available'].mean() * 100:.1f}%")
print()


# Time-match quality
print("Weather time differences:")
print()
print("Bay weather:")
print(merged["bay_weather_time_diff_min"].describe())
print()
print("Wind:")
print(merged["wind_time_diff_min"].describe())
print()
print("Waves:")
print(merged["wave_time_diff_min"].describe())
print()

# Station distance
print("Weather station distances:")
print()
print("SDBC1:")
print(merged["distance_to_sdbc1_nm"].describe())


print()
print("LJAC1:")
print(merged["distance_to_ljac1_nm"].describe())
print()
print("46232:")
print(merged["distance_to_46232_nm"].describe())

print()

# Missing weather values
print("Important weather feature missingness:")

for column in important_weather_features:
    missing_percent = (merged[column].isna().mean()* 100)
    print(f"{column}: {missing_percent:.1f}% missing")

print()
print(f"Average weather completeness: {merged['weather_completeness'].mean() * 100:.1f}%")
print()

print()
print("Individual weather feature availability:")

availability_columns = [
    "wind_available",
    "pressure_available",
    "water_temperature_available",
    "air_temperature_available",
    "wave_available"
]

for column in availability_columns:
    print(
        f"{column}: "
        f"{merged[column].mean() * 100:.1f}%"
    )

print("Final dataset saved to:")
print(OUTPUT_FILE)