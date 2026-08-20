import math
from pathlib import Path

import polars as pl

from polars_tsfresh import extract_features, features


def float_close(a: float, b: float, tol: float = 1e-5) -> bool:
    """Return True if ``a`` and ``b`` are within ``tol`` of each other."""
    return abs(a - b) < tol


def test_distribution():
    x_csv_path = Path(__file__).parent / "data" / "sp500_raw.csv"
    y_csv_path = Path(__file__).parent / "data" / "sp500_distribution_features.csv"

    # Reference values generated with upstream tsfresh's
    # tsfresh.feature_extraction.feature_calculators (skewness, kurtosis,
    # variation_coefficient, quantile, binned_entropy, benford_correlation)
    # applied to sp500_raw.csv, so this test guards tsfresh parity.

    df = pl.read_csv(x_csv_path).with_columns(pl.lit("sp500").alias("kind"))
    extracted = extract_features(
        df, column_id="kind", column_sort="date", feature_sets=[features.distribution_feature_set]
    )
    expected = pl.read_csv(y_csv_path)

    for col in expected.columns:
        val = extracted[col][0]
        val_true = expected[col][0]
        assert float_close(val, val_true), f"Feature {col} does not match: {val} != {val_true}"


def test_benford_correlation_excludes_infinity():
    with_inf = pl.DataFrame({"value": [float("inf"), 1.0, 2.0, 3.0]}).select(
        features.benford_correlation("value")
    )
    without_inf = pl.DataFrame({"value": [1.0, 2.0, 3.0]}).select(
        features.benford_correlation("value")
    )
    assert math.isclose(
        with_inf["value__benford_correlation"][0],
        without_inf["value__benford_correlation"][0],
        rel_tol=1e-12,
    )


def test_binned_entropy_small_integer_range():
    # Regression test: integer columns whose range is smaller than ``max_bins``
    # used to raise ``ValueError: Too many bins for data range`` (np.histogram limit).
    result = pl.DataFrame({"value": pl.Series([1, 2, 3], dtype=pl.Int64)}).select(
        features.binned_entropy("value")
    )
    assert math.isfinite(result["value__binned_entropy"][0])
