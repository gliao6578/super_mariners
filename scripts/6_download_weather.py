#6
from pathlib import Path
import gzip
import io

import numpy as np
import pandas as pd
import requests

# FILE PATHS
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEATHER_DIR = (PROJECT_ROOT / "data" / "weather")
WEATHER_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = (WEATHER_DIR / "ndbc_weather.csv")


# SETTINGS
# Match this range to the AIS range. These can span months and years.
START_DATE = pd.Timestamp("2020-01-01", tz="UTC")
END_DATE = pd.Timestamp("2020-01-31 23:59:59", tz="UTC")
YEARS = list(range(START_DATE.year, END_DATE.year + 1))

# WEATHER STATIONS
STATIONS = {
    # ===============SAN DIEGO BAY================
    # Useful primarily for:
    #
    # pressure
    # air temperature
    # water temperature
    # dew point
    # visibility

    "SDBC1": {
        "name": "San Diego, CA",
        "latitude": 32.714,
        "longitude": -117.174,
        "role": "bay_environment"
    },


    # ===========LA JOLLA============
    # Useful primarily for coastal meteorological conditions:
    # wind_direction_deg
    # wind_speed_mps
    # wind_gust_mps

    "LJAC1": {
        "name": "La Jolla, CA",
        "latitude": 32.867,
        "longitude": -117.257,
        "role": "wind"
    },


    # ==========POINT LOMA SOUTH==========
    # Useful primarily for:
    #
    # wave height
    # wave period
    # wave direction
    # water temperature

    "46232": {
        "name": "Point Loma South, CA",
        "latitude": 32.517,
        "longitude": -117.425,
        "role": "waves"
    }
}

# DOWNLOAD HISTORICAL WEATHER DATA
def download_station_year(station_id,station_name,station_role,year,latitude,longitude):
    """
    Download one year of NDBC Standard Meteorological data for a weather station
    The returned dataframe contains:
        station information
        timestamp
        wind measurements
        wave measurements
        temperature
        pressure
        visibility
    Not every station measures every weather variable; missing measurements remain NaN"""

    # Build NDBC historical-data URL
    # NDBC historical Standard Meteorological files use:
    # station + "h" + year + ".txt.gz"
    # Example: sdbc1h2020.txt.gz

    url = (
        "https://www.ndbc.noaa.gov/data/historical/stdmet/"
        f"{station_id.lower()}h{year}.txt.gz")


    print()
    print("=" * 70)
    print(f"Downloading weather station: {station_id} - {station_name}")
    print(f"Year: {year}")
    print(f"URL: {url}")
    print("=" * 70)

    # Download file
    print(f"Downloading {station_id} for {year}: {url}")
    response = requests.get(url,timeout=120)
    response.raise_for_status()

    # Decompress .gz file
    decompressed_data = gzip.decompress(response.content)


    # Read whitespace-delimited NDBC text
    # NDBC files not normal comma-separated CSV files
    # EX:
    # #YY MM DD hh mm WDIR WSPD GST ...
    # 2020 01 01 00 00 270  4.2 ...
    # sep=r"\s+" ---> tells pandas that one+ spaces separate the columns
    # "MM" --> missing values

    weather = pd.read_csv(io.BytesIO(decompressed_data),sep=r"\s+", na_values=["MM"],low_memory=False)

    # CLEAN COLUMN NAMES
    # The first column may appear as: #YY remove the # so it becomes: YY
    weather.columns = [column.replace("#", "") for column in weather.columns]


    # Create timestamp
    if "YYYY" in weather.columns: #sometimes in YYYY or YY format
        year_column = "YYYY"

    elif "YY" in weather.columns:
        year_column = "YY"

    else:
        raise ValueError("Could not find year column in NDBC data.")


    # Some files contain minute ("mm"), but uf minutes are unavailable, assume minute = 0

    if "mm" not in weather.columns:
        weather["mm"] = 0


    weather["timestamp"] = pd.to_datetime(
        {
            "year":weather[year_column],
            "month":weather["MM"],
            "day":weather["DD"],
            "hour":weather["hh"],
            "minute":weather["mm"]},
        errors="coerce", utc=True)

    # Add station metadata
    weather["station_id"] = station_id
    weather["station_name"] = station_name
    weather["station_role"] = station_role
    weather["station_latitude"] = latitude
    weather["station_longitude"] = longitude

    # CONVERT WEATHER VARIABLES TO NUMERIC
    weather_columns = [

        "WDIR",
        "WSPD",
        "GST",

        "WVHT",
        "DPD",
        "APD",
        "MWD",

        "PRES",
        "BAR",

        "ATMP",
        "WTMP"
        # "DEWP", #sd bay does not capture these data (checked only yr:2020), so we remove
        #
        # "VIS"
    ]


    for column in weather_columns:
        if column in weather.columns:
            weather[column] = pd.to_numeric(weather[column],errors="coerce")


    # HANDLE NDBC MISSING / INVALID VALUES
    #bc NDBC uses large values for unavailable measurements in some datasets,
    #we replace obviously invalid values with NaN

    # Wind direction
    if "WDIR" in weather.columns:
        weather.loc[(weather["WDIR"] < 0) | (weather["WDIR"] > 360), "WDIR"] = np.nan

    # Wind speed
    if "WSPD" in weather.columns:
        weather.loc[(weather["WSPD"] < 0) | (weather["WSPD"] >= 99), "WSPD"] = np.nan

    # Wind gust
    if "GST" in weather.columns:
        weather.loc[(weather["GST"] < 0) | (weather["GST"] >= 99),"GST"] = np.nan

    # Wave height
    if "WVHT" in weather.columns:
        weather.loc[(weather["WVHT"] < 0) | (weather["WVHT"] >= 99),"WVHT"] = np.nan

    # Dominant wave period
    if "DPD" in weather.columns:
        weather.loc[(weather["DPD"] < 0) | (weather["DPD"] >= 99),"DPD"] = np.nan

    # Average wave period
    if "APD" in weather.columns:
        weather.loc[(weather["APD"] < 0) | (weather["APD"] >= 99), "APD"] = np.nan

    # Mean wave direction
    if "MWD" in weather.columns:
        weather.loc[(weather["MWD"] < 0) | (weather["MWD"] > 360),"MWD"] = np.nan

    # Pressure
    for column in ["PRES", "BAR"]:
        if column in weather.columns:
            weather.loc[weather[column] >= 9999, column] = np.nan

    # Temperature
    for column in ["ATMP", "WTMP"]: #, "DEWP"
        if column in weather.columns:
            weather.loc[weather[column] >= 999, column] = np.nan

    # # Visibility
    # if "VIS" in weather.columns:
    #     weather.loc[weather["VIS"] >= 99, "VIS"] = np.nan


    # RENAME VARIABLES
    rename_columns = {
        "WDIR":"wind_direction_deg",
        "WSPD":"wind_speed_mps",
        "GST":"wind_gust_mps",
        "WVHT":"wave_height_m",
        "DPD":"dominant_wave_period_s",
        "APD":"average_wave_period_s",
        "MWD":"wave_direction_deg",
        "PRES":"pressure_hpa",
        "BAR":"pressure_hpa",
        "ATMP":"air_temperature_c",
        "WTMP":"water_temperature_c"
        # "DEWP":"dewpoint_temperature_c",
        # "VIS":"visibility_nmi"
    }


    # Only rename columns that actually exist bc not every station measures every variable
    rename_columns = {old_name: new_name for old_name, new_name in rename_columns.items()
        if old_name in weather.columns
    }

    weather = weather.rename(columns=rename_columns)


    # KEEP USEFUL COLUMNS
    desired_columns = [

        "station_id",
        "station_name",
        "station_role",

        "station_latitude",
        "station_longitude",

        "timestamp",

        "wind_direction_deg",
        "wind_speed_mps",
        "wind_gust_mps",

        "wave_height_m",
        "dominant_wave_period_s",
        "average_wave_period_s",
        "wave_direction_deg",

        "pressure_hpa",

        "air_temperature_c",
        "water_temperature_c"
        # "dewpoint_temperature_c",
        #
        # "visibility_nmi"
    ]


    # Keep only columns that this station actually provides.
    available_columns = [column for column in desired_columns
        if column in weather.columns]


    weather = weather[available_columns].copy()

    # Remove observations with invalid timestamps
    weather = weather.dropna(subset=["timestamp"])

    # Sort chronologically
    weather = weather.sort_values("timestamp").reset_index(drop=True)

    print()
    print(f"Weather observations downloaded: {len(weather):,}")


    print()
    print("Available columns:")
    print(weather.columns.tolist())

    return weather


# DOWNLOAD ALL WEATHER STATIONS
all_weather = []

for year in YEARS:
    for station_id, station in STATIONS.items():
        try:
            station_weather = download_station_year(
                station_id=station_id,
                station_name=station["name"],
                station_role=station["role"],
                year=year,
                latitude=station["latitude"],
                longitude=station["longitude"],
            )
            all_weather.append(station_weather)
        except Exception as error:
            print(f"ERROR downloading {station_id} for {year}: {error}")

if not all_weather:
    raise RuntimeError("No weather stations were successfully downloaded.")

weather = pd.concat(all_weather, ignore_index=True) # COMBINE STATIONS

weather = weather[weather["timestamp"].between(START_DATE, END_DATE, inclusive="both")].copy()

# SORT FINAL DATASET
weather = weather.sort_values(["timestamp", "station_id"]).reset_index(drop=True)
weather.to_csv(OUTPUT_FILE, index=False) # SAVE WEATHER DATA

# COMBINE STATIONS
# if len(all_weather) == 0:
#     raise RuntimeError("No weather stations were successfully downloaded.")

# weather = pd.concat(all_weather,ignore_index=True)

# SORT FINAL DATASET
# weather = weather.sort_values(["timestamp","station_id"]).reset_index(drop=True)


# SAVE WEATHER DATA
# weather.to_csv(OUTPUT_FILE, index=False)


# PRINT SUMMARY
print()
print("=" * 70)
print("NDBC WEATHER DOWNLOAD COMPLETE")

print("=" * 70)
print()

print(f"Years requested: {YEARS}")
print(f"Date range: {START_DATE} through {END_DATE}")
print(f"Total weather observations: {len(weather):,}")
print(f"Stations: {weather['station_id'].nunique()}")
print(f"Start time: {weather['timestamp'].min()}")
print(f"End time: {weather['timestamp'].max()}")
print(f"Saved to: {OUTPUT_FILE}")
# print("Missing values:")


# weather_columns = [column for column in weather.columns
#     if column not in [
#
#         "station_id",
#         "station_name",
#         "station_latitude",
#         "station_longitude",
#         "timestamp"
#     ]
# ]

# for column in weather_columns:
#     missing_percent = (weather[column].isna().mean()* 100)
#     print(f"{column}: {missing_percent:.1f}% missing")
#
# print()
# print(f"Saved weather dataset to:")
# print(OUTPUT_FILE)