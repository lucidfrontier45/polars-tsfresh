from collections import UserDict
from pathlib import Path

import polars as pl
import pytest

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


def _sample_df() -> pl.DataFrame:
    x_csv_path = Path(__file__).parent.parent / "data" / "sp500_raw.csv"
    return pl.read_csv(x_csv_path).with_columns(pl.lit("sp500").alias("kind"))


def test_mixed_schema_raises():
    df = _sample_df()
    mixed: dict[str, dict | None] = {"open": None, "close": {"mean": None}}

    with pytest.raises(ValueError, match="mix"):
        extract_features(df, column_id="kind", column_sort="date", kind_to_fc_parameters=mixed)


def test_mapping_schema_accepts_non_dict_mappings():
    df = _sample_df()
    params = UserDict({"close": UserDict({"mean": None})})

    features = extract_features(
        df, column_id="kind", column_sort="date", kind_to_fc_parameters=params
    )

    assert features.columns == ["kind", "close__mean"]


def test_empty_kind_to_fc_parameters_yields_no_features():
    df = _sample_df()
    features = extract_features(
        df, column_id="kind", column_sort="date", kind_to_fc_parameters={}
    )

    assert features.columns == ["kind"]


def test_unknown_feature_name_raises():
    df = _sample_df()
    unknown = {"close": {"not_a_real_feature": None}}

    with pytest.raises(ValueError, match="not_a_real_feature"):
        extract_features(
            df, column_id="kind", column_sort="date", kind_to_fc_parameters=unknown
        )


def test_from_columns_missing_separator_raises():
    with pytest.raises(ValueError, match="__"):
        from_columns(["no_separator"])


def test_from_columns_splits_on_last_separator():
    # column name itself contains "__"; the feature name is the final segment
    result = from_columns(["a__b__mean"])

    assert result == {"a__b": {"mean": None}}


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
