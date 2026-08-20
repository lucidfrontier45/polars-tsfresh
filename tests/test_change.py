import math

import numpy as np
import polars as pl

from polars_tsfresh import extract_features, features


def test_change_and_rate_feature_set():
    df = pl.DataFrame(
        {
            "id": ["series"] * 4,
            "time": [0, 1, 2, 3],
            "value": [1.0, 3.0, 2.0, 6.0],
        }
    )

    result = extract_features(
        df,
        column_id="id",
        column_sort="time",
        feature_sets=[features.change_and_rate_feature_set],
    )

    assert math.isclose(result["value__mean_abs_change"][0], 7 / 3)
    assert math.isclose(result["value__mean_change"][0], 5 / 3)
    assert math.isclose(result["value__mean_second_derivative_central"][0], 0.5)
    assert math.isclose(result["value__absolute_sum_of_changes"][0], 7.0)

    normalized = (np.array([1.0, 3.0, 2.0, 6.0]) - 3.0) / np.std([1.0, 3.0, 2.0, 6.0])
    expected_cid_ce = np.sqrt(np.sum(np.diff(normalized) ** 2))
    assert math.isclose(result["value__cid_ce"][0], expected_cid_ce)


def test_cid_ce_without_normalization():
    result = pl.DataFrame({"value": [1.0, 3.0, 2.0, 6.0]}).select(
        features.cid_ce("value", normalize=False)
    )

    assert math.isclose(result["value__cid_ce"][0], math.sqrt(21.0))


def test_change_features_short_series():
    result = pl.DataFrame({"value": [1.0]}).select(
        features.mean_abs_change("value"),
        features.mean_change("value"),
        features.mean_second_derivative_central("value"),
        features.absolute_sum_of_changes("value"),
        features.cid_ce("value"),
    )

    assert math.isnan(result["value__mean_abs_change"][0])
    assert math.isnan(result["value__mean_change"][0])
    assert math.isnan(result["value__mean_second_derivative_central"][0])
    assert result["value__absolute_sum_of_changes"][0] == 0.0
    assert result["value__cid_ce"][0] == 0.0


def test_change_features_propagate_nan_and_null():
    nan_result = pl.DataFrame({"value": [1.0, 2.0, float("nan"), 4.0, 5.0]}).select(
        features.mean_change("value"),
        features.mean_second_derivative_central("value"),
    )
    null_result = pl.DataFrame({"value": [1.0, None, 3.0]}).select(
        features.mean_change("value"),
    )

    assert math.isnan(nan_result["value__mean_change"][0])
    assert math.isnan(nan_result["value__mean_second_derivative_central"][0])
    assert math.isnan(null_result["value__mean_change"][0])


def test_change_features_cast_unsigned_values():
    result = pl.DataFrame({"value": pl.Series("value", [3, 1], dtype=pl.UInt8)}).select(
        features.mean_abs_change("value"),
        features.mean_change("value"),
        features.absolute_sum_of_changes("value"),
        features.cid_ce("value", normalize=False),
    )
    second_result = pl.DataFrame({"value": pl.Series("value", [3, 1, 0], dtype=pl.UInt8)}).select(
        features.mean_second_derivative_central("value")
    )

    assert result["value__mean_abs_change"][0] == 2.0
    assert result["value__mean_change"][0] == -2.0
    assert result["value__absolute_sum_of_changes"][0] == 2.0
    assert result["value__cid_ce"][0] == 2.0
    assert math.isclose(
        second_result["value__mean_second_derivative_central"][0],
        0.5,
    )
