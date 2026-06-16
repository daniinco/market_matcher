"""Module 1: Load and normalize all supplier CSV files."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

SUPPLIERS_DIR = Path(__file__).parent.parent / "data" / "suppliers"

# Price sanity bounds (rub)
PRICE_MIN = 10.0
PRICE_MAX = 5_000_000.0


def _normalize_name(name: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation noise for comparison."""
    if not isinstance(name, str):
        return ""
    name = name.lower().strip()
    # collapse multiple spaces
    name = re.sub(r"\s+", " ", name)
    return name


def load_supplier(path: Path) -> pd.DataFrame:
    """Load a single supplier CSV and return a normalized DataFrame."""
    df = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8")

    # Ensure required columns exist
    if "product_name" not in df.columns or "price_rub" not in df.columns:
        raise ValueError(f"Missing required columns in {path.name}: {list(df.columns)}")

    # 1-based row_id matching the row number in the original file
    df.insert(0, "row_id", range(1, len(df) + 1))

    # Keep original name, add normalized copy
    df["product_name"] = df["product_name"].fillna("").str.strip()
    df["product_name_norm"] = df["product_name"].apply(_normalize_name)

    # Validate price
    df["price_rub"] = pd.to_numeric(df["price_rub"], errors="coerce")
    df = df[df["price_rub"].notna()].copy()
    df = df[(df["price_rub"] >= PRICE_MIN) & (df["price_rub"] <= PRICE_MAX)].copy()
    df["price_rub"] = df["price_rub"].astype(float)

    # Drop rows with empty product names
    df = df[df["product_name_norm"].str.len() > 0].copy()

    df["file"] = path.name
    df = df.reset_index(drop=True)

    return df[["row_id", "product_name", "product_name_norm", "price_rub", "file"]]


def load_all_suppliers(
    suppliers_dir: Path = SUPPLIERS_DIR,
    max_rows: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Load all supplier CSVs from the given directory.

    Args:
        suppliers_dir: directory containing supplier CSV files
        max_rows: if set, keep only the first N rows per supplier (for testing)

    Returns a dict keyed by filename (e.g. 'supplier_S1_РозничМаркет.csv').
    """
    result: dict[str, pd.DataFrame] = {}
    csv_files = sorted(suppliers_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {suppliers_dir}")

    for path in csv_files:
        df = load_supplier(path)
        if max_rows is not None:
            df = df.head(max_rows).copy()
        result[path.name] = df
        print(f"  Loaded {path.name}: {len(df)} valid rows")

    return result


if __name__ == "__main__":
    print("Loading supplier data...")
    data = load_all_suppliers()
    for fname, df in data.items():
        print(f"\n{fname}")
        print(df.head(3).to_string(index=False))
