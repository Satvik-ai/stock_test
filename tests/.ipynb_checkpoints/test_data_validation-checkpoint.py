import pandas as pd
import pytest
import re

def test_parquet_data_integrity():
    # Load Parquet data
    df = pd.read_parquet("data/stock_data.parquet")

    # 1. Check shape — should have 11 columns
    assert df.shape[1] == 11, f"Expected 11 columns, found {df.shape[1]}"

    # 2. Check for missing values
    assert df.isnull().sum().sum() == 0, "Dataset contains missing values"

    # 3. Expected base columns (excluding versioned ID)
    expected_base_cols = {
        "event_timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "ma_15_min",
        "ma_60_min",
        "rsi_14",
        "target",
    }
    # Find the versioned ID column dynamically
    versioned_id_col = next(
        (col for col in df.columns if re.match(r"stock_v\d+_id", col)),
        None
    )

    assert versioned_id_col is not None, \
        "No versioned ID column found (expected something like 'stock_v1_id' or 'stock_v2_id')"

    # 4. Verify all expected columns are present
    expected_cols = expected_base_cols.union({versioned_id_col})
    assert expected_cols.issubset(df.columns), \
        f"Missing columns: {expected_cols - set(df.columns)}"

    # 5. Verify event_timestamp is datetime type
    assert pd.api.types.is_datetime64_any_dtype(df["event_timestamp"]), \
        "'event_timestamp' column should be datetime type"

def test_numeric_columns_are_numeric_parquet():
    df = pd.read_parquet("data/stock_data.parquet")
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'ma_15_min', 'ma_60_min', 'rsi_14']

    for col in numeric_cols:
        assert pd.api.types.is_numeric_dtype(df[col]), f"Column {col} is not numeric"