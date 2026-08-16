from pathlib import Path

import polars as pl

from polars_tsfresh import extract_features
from polars_tsfresh import features


def float_close(a: float, b: float, tol: float = 1e-5) -> bool:
    """Return True if ``a`` and ``b`` are within ``tol`` of each other."""
    return abs(a - b) < tol


def test_distribution():
    x_csv_path = Path(__file__).parent / "data" / "sp500_raw.csv"
    y_csv_path = Path(__file__).parent / "data" / "sp500_distribution_features.csv"

    df = pl.read_csv(x_csv_path).with_columns(pl.lit("sp500").alias("kind"))
    extracted = extract_features(
        df, column_id="kind", column_sort="date", feature_sets=[features.distribution_feature_set]
    )
    expected = pl.read_csv(y_csv_path)

    for col in expected.columns:
        val = extracted[col][0]
        val_true = expected[col][0]
        assert float_close(val, val_true), f"Feature {col} does not match: {val} != {val_true}"
