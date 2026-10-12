from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
input_file = PROJECT_ROOT / "data" / "processed" / "san_diego_ais.csv"

df = pd.read_csv(input_file)

print(df.head())
print(df.shape)
print(df["mmsi"].nunique())