#3
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
input_file = PROJECT_ROOT / "data" / "processed" / "san_diego_ais.csv"
output_file = PROJECT_ROOT / "data" / "processed" / "san_diego_ais_cleaned.csv"

df = pd.read_csv(input_file).copy()

#convert timestamps
df["base_date_time"] = pd.to_datetime(df["base_date_time"])

#do not need to convert numeric columns to numeric data type bc they already floats

#remove vessels missing essential information
df = df.dropna(subset=["mmsi","base_date_time","longitude","latitude"])

#remove invalid coordinates
df = df[df["latitude"].between(-90,90) & df["longitude"].between(-180,180)].copy()

#set invalid sog to nan
df.loc[(df["sog"] < 0) | (df["sog"] >= 102.3),"sog"] = np.nan

#set invalid cog to nan
df.loc[(df["cog"] < 0) | (df["cog"] >= 360), "cog"] = np.nan

#set invalid heading to nan
df.loc[(df["heading"] < 0) | (df["heading"] >= 360), "heading"] = np.nan

#remove duplicates (although unlikely to have duplicates)
df = df.drop_duplicates(subset=["mmsi","base_date_time","longitude","latitude"])

#sort vessels by mmsi and datetime
df = df.sort_values(["mmsi", "base_date_time"]).reset_index(drop=True)

#fill in values for static columns that appear in one record but not another for same mmsi (ie. length)
static_columns = [
    "imo",
    "call_sign",
    "vessel_type",
    "status",
    "transceiver",
    "length",
    "width"
]

for column in static_columns:
    if column in df.columns:
        df[column] = (df.groupby("mmsi")[column].transform(lambda x: x.ffill().bfill()))

#clean categorical variables
df["vessel_type_clean"] = (df["vessel_type"].fillna(-1).astype(int))
df["status_clean"] = (df["status"].fillna(-1).astype(int))
df["cargo_clean"] = (df["cargo"].astype("Int64").astype("string").fillna("Unknown"))

#missing  column indicator (WHAT DOES THIS DO DO RESEARCH
missing_indicator_columns = [
    "sog",
    "cog",
    "heading",
    "vessel_type",
    "status",
    "length",
    "width",
    "draft",
    "cargo"
]

for column in missing_indicator_columns:
    if column in df.columns:
        df[f"{column.lower()}_missing"] = (
            df[column]
            .isna()
            .astype(int)
        )


#quality report
null_summary = pd.DataFrame({

    "null_count":
        df.isna().sum(),

    "null_percent":
        df.isna().mean() * 100

}).sort_values(
    "null_percent",
    ascending=False
)


print("\nMissing data after cleaning:")
print(null_summary.to_string())
print(df.dtypes)

df.to_csv(output_file, index=False)