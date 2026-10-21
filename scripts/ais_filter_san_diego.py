#1
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
input_file = PROJECT_ROOT / "data" / "raw" / "ais-2020-01-01.csv.zst"
output_file = PROJECT_ROOT / "data" / "processed" / "san_diego_ais.csv"

df = pd.read_csv(input_file, compression='zstd')

san_diego = df[
    (df["latitude"] >= 32.50) &
    (df["latitude"] <= 32.95) &
    (df["longitude"] >= -117.65) &
    (df["longitude"] <= -117.05)
]

san_diego.to_csv(output_file, index=False)
