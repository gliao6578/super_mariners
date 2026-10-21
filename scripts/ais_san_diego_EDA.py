#2
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
input_file = PROJECT_ROOT / "data" / "processed" / "san_diego_ais.csv"

df = pd.read_csv(input_file)
dc = pd.read_csv(input_file).copy()

print(df.head())
print(df.shape)
print(df["mmsi"].nunique())

print(df.isna().any())

print(df.dtypes)

dd = df.dropna(subset=["mmsi","base_date_time","longitude","latitude"])
print(dd["mmsi"].nunique())

dc = dc.drop(index = 0)

print(df.shape)
print(dc.shape)