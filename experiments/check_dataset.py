import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/primary")

csv_files = list(DATA_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError("No CSV found in data/primary")

if len(csv_files) > 1:
    print("Multiple CSV files found:")
    for f in csv_files:
        print(f)

csv_path = csv_files[0]

print("=" * 60)
print("DATASET:", csv_path)
print("=" * 60)

df = pd.read_csv(csv_path)

print("\nSHAPE")
print(df.shape)

print("\nCOLUMNS")
print(df.columns.tolist())

print("\nDATA TYPES")
print(df.dtypes)

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nUNIQUE VALUES")
print(df.nunique())

print("\nNUMERICAL SUMMARY")
print(df.describe().T)

print("\nFIRST 5 ROWS")
print(df.head())

print("\nLAST 5 ROWS")
print(df.tail())