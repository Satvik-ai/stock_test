import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
import pytest

def test_model_performance_parquet():
    # Load data from Parquet file
    df = pd.read_parquet("data/stock_data.parquet")

    # Drop target and non-feature columns
    X = df[['open', 'high', 'low', 'close', 'volume', 'ma_15_min', 'ma_60_min', 'rsi_14']]
    y = df["target"]

    # Load trained model
    model = joblib.load("artifacts/model.joblib")

    # Generate predictions
    preds = model.predict(X)

    # Compute accuracy
    acc = accuracy_score(y, preds)

    # Assert model meets minimum performance threshold
    assert acc > 0.5, f"Model accuracy too low: {acc:.3f}"
