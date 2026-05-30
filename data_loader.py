"""
data_loader.py — Chicago Crime Arrest Prediction
Downloads the dataset from the Chicago Data Portal (no API key required)
and caches it locally. Subsequent calls load from cache instantly.
"""

import requests
from pathlib import Path
import pandas as pd

PORTAL_URL = (
    "https://data.cityofchicago.org/api/views/ijzp-q8t2/rows.csv"
    "?accessType=DOWNLOAD"
)
CACHE_PATH = Path(__file__).parent / "data" / "chicago_crimes.csv"

DTYPES = {
    "ID":                   "int32",
    "Case Number":          "category",
    "Block":                "category",
    "IUCR":                 "category",
    "Primary Type":         "category",
    "Description":          "category",
    "Location Description": "category",
    "Arrest":               "bool",
    "Domestic":             "bool",
    "Beat":                 "int16",
    "District":             "float32",
    "Ward":                 "float32",
    "Community Area":       "float32",
    "FBI Code":             "category",
    "X Coordinate":         "float32",
    "Y Coordinate":         "float32",
    "Year":                 "int16",
    "Latitude":             "float64",
    "Longitude":            "float64",
}


def download(dest: Path) -> None:
    """Stream the CSV from the Chicago Data Portal with progress output."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading from Chicago Data Portal (~1.8 GB, one-time)...")
    with requests.get(PORTAL_URL, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done  = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / 1e9:.2f} / {total / 1e9:.2f} GB", end="")
    print(f"\nSaved to {dest}")


def load_data() -> pd.DataFrame:
    """
    Load the Chicago Crime dataset, downloading it on first call.

    Returns
    -------
    pd.DataFrame
        Raw dataset with all original columns and memory-optimised dtypes.
    """
    if not CACHE_PATH.exists():
        download(CACHE_PATH)
    else:
        print(f"Using cached file: {CACHE_PATH}")

    return pd.read_csv(CACHE_PATH, dtype=DTYPES, low_memory=False)


if __name__ == "__main__":
    df = load_data()
    print(f"Shape : {df.shape}")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")
    print(df.head())
