import math

import polars as pl

from polars_tsfresh import extract_features, features


def test_recurring_value_features_match_tsfresh_docstring_example():
    result = pl.DataFrame({"value": [2, 2, 2, 2, 1]}).select(
        features.recurring_value_feature_set("value")
    )

    assert result.row(0, named=True) == {
        "value__percentage_of_reoccurring_datapoints_to_all_datapoints": 0.8,
        "value__percentage_of_reoccurring_values_to_all_values": 0.5,
        "value__sum_of_reoccurring_data_points": 8.0,
        "value__sum_of_reoccurring_values": 2.0,
    }


def test_recurring_value_features_return_zero_for_all_unique_series():
    result = pl.DataFrame({"value": [1.0, 2.0, 3.0]}).select(
        features.recurring_value_feature_set("value")
    )

    assert result.row(0) == (0.0, 0.0, 0.0, 0.0)


def test_recurring_value_features_sum_multiple_distinct_recurring_values():
    result = pl.DataFrame({"value": [1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 4.0, 4.0]}).select(
        features.recurring_value_feature_set("value")
    )
    values = result.row(0, named=True)

    assert math.isclose(
        values["value__percentage_of_reoccurring_datapoints_to_all_datapoints"],
        7 / 8,
    )
    assert values["value__percentage_of_reoccurring_values_to_all_values"] == 3 / 4
    assert values["value__sum_of_reoccurring_data_points"] == 16.0
    assert values["value__sum_of_reoccurring_values"] == 7.0


def test_recurring_value_features_cast_integer_input_to_float_output():
    result = pl.DataFrame({"value": [1, 1, 2]}).select(
        features.recurring_value_feature_set("value")
    )

    assert result.dtypes == [pl.Float64] * 4
    assert result.row(0) == (2 / 3, 0.5, 2.0, 1.0)


def test_recurring_value_features_via_grouped_extract_features():
    df = pl.DataFrame(
        {
            "id": ["a", "a", "a", "a", "a", "b", "b", "b"],
            "time": [2, 0, 1, 3, 4, 0, 1, 2],
            "value": [2, 2, 2, 2, 1, 1, 2, 3],
        }
    )

    result = extract_features(
        df,
        column_id="id",
        column_sort="time",
        feature_sets=[features.recurring_value_feature_set],
    )
    a = result.filter(pl.col("id") == "a").row(0, named=True)
    b = result.filter(pl.col("id") == "b").row(0, named=True)

    assert a["value__percentage_of_reoccurring_datapoints_to_all_datapoints"] == 0.8
    assert a["value__percentage_of_reoccurring_values_to_all_values"] == 0.5
    assert a["value__sum_of_reoccurring_data_points"] == 8.0
    assert a["value__sum_of_reoccurring_values"] == 2.0
    assert b["value__percentage_of_reoccurring_datapoints_to_all_datapoints"] == 0.0
    assert b["value__percentage_of_reoccurring_values_to_all_values"] == 0.0
    assert b["value__sum_of_reoccurring_data_points"] == 0.0
    assert b["value__sum_of_reoccurring_values"] == 0.0
