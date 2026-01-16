from pathlib import Path

import polars as pl

from polars_tsfresh import extract_features
from polars_tsfresh.feature_extraction.settings import from_columns


def float_close(a: float, b: float, tol=1e-5) -> bool:
    return abs(a - b) < tol


def test_minimal():
    x_csv_path = Path(__file__).parent.parent / "data" / "sp500_raw.csv"
    y_csv_path = Path(__file__).parent.parent / "data" / "sp500_tsfresh_features.csv"

    df = pl.read_csv(x_csv_path).with_columns(pl.lit("sp500").alias("kind"))
    features = extract_features(df, column_id="kind", column_sort="date")
    features_true = pl.read_csv(y_csv_path)

    for col in features.columns:
        # check only close
        if not col.startswith("close"):
            continue

        val = features[col][0]
        val_true = features_true[col][0]
        assert float_close(val, val_true), (
            f"Feature {col} does not match: {val} != {val_true}"
        )


def test_feats():
    x_csv_path = Path(__file__).parent.parent / "data" / "sp500_raw.csv"
    y_csv_path = Path(__file__).parent.parent / "data" / "sp500_tsfresh_features.csv"

    df = pl.read_csv(x_csv_path).with_columns(pl.lit("sp500").alias("kind"))
    features_true = pl.read_csv(y_csv_path)

    features_columns = [
        "open__mean",
        "open__standard_deviation",
        "close__mean",
        "close__median",
        "close__variance",
    ]

    features = extract_features(
        df,
        column_id="kind",
        column_sort="date",
        kind_to_fc_parameters=from_columns(features_columns),
    )
    assert features.columns[1:] == features_columns

    for col in features.columns:
        # check only close
        if not col.startswith("close"):
            continue

        val = features[col][0]
        val_true = features_true[col][0]
        assert float_close(val, val_true), (
            f"Feature {col} does not match: {val} != {val_true}"
        )
