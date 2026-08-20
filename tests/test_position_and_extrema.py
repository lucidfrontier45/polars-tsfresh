import math

import polars as pl

from polars_tsfresh import extract_features, features


def test_location_features_with_ties_and_reversal():
    df = pl.DataFrame({"value": [5.0, 1.0, 5.0, 3.0, 5.0]}).select(
        features.first_location_of_maximum("value"),
        features.last_location_of_maximum("value"),
        features.first_location_of_minimum("value"),
        features.last_location_of_minimum("value"),
    )
    assert df["value__first_location_of_maximum"][0] == 0.0
    assert df["value__last_location_of_maximum"][0] == 1.0
    assert math.isclose(df["value__first_location_of_minimum"][0], 0.2)
    assert math.isclose(df["value__last_location_of_minimum"][0], 0.4)


def test_count_above_and_below_mean():
    df = pl.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 5.0]}).select(
        features.count_above_mean("value"),
        features.count_below_mean("value"),
    )
    assert df["value__count_above_mean"][0] == 2
    assert df["value__count_below_mean"][0] == 2


def test_count_above_and_below_inclusive_threshold():
    df = pl.DataFrame({"value": [0.0, 1.0, 2.0, 3.0, 4.0]}).select(
        features.count_above("value", t=2.0),
        features.count_below("value", t=2.0),
    )
    assert math.isclose(df["value__count_above"][0], 0.6)
    assert math.isclose(df["value__count_below"][0], 0.6)


def test_has_duplicate_uses_three_cases():
    df = pl.DataFrame(
        {
            "extrema": [5.0, 1.0, 5.0],
            "unique": [1.0, 2.0, 3.0],
        }
    ).select(
        features.has_duplicate("extrema"),
        features.has_duplicate_max("extrema"),
        features.has_duplicate_min("extrema"),
        features.has_duplicate("unique"),
        features.has_duplicate_max("unique"),
        features.has_duplicate_min("unique"),
    )
    assert df["extrema__has_duplicate"][0]
    assert df["extrema__has_duplicate_max"][0]
    assert not df["extrema__has_duplicate_min"][0]
    assert not df["unique__has_duplicate"][0]
    assert not df["unique__has_duplicate_max"][0]
    assert not df["unique__has_duplicate_min"][0]


def test_has_duplicate_with_nulls():
    df = pl.DataFrame(
        {
            "null_then_unique": [None, 1.0, 2.0],
            "duplicate_with_null": [1.0, None, 1.0],
        }
    ).select(
        features.has_duplicate("null_then_unique"),
        features.has_duplicate("duplicate_with_null"),
    )
    assert not df["null_then_unique__has_duplicate"][0]
    assert df["duplicate_with_null__has_duplicate"][0]


def test_longest_strike_start_middle_end():
    df = pl.DataFrame(
        {
            "start": [10.0, 11.0, 12.0, 1.0, 2.0],
            "middle": [1.0, 10.0, 11.0, 12.0, 2.0],
            "end": [1.0, 2.0, 10.0, 11.0, 12.0],
        }
    ).select(
        features.longest_strike_above_mean("start"),
        features.longest_strike_below_mean("start"),
        features.longest_strike_above_mean("middle"),
        features.longest_strike_below_mean("middle"),
        features.longest_strike_above_mean("end"),
        features.longest_strike_below_mean("end"),
    )
    assert df["start__longest_strike_above_mean"][0] == 3
    assert df["start__longest_strike_below_mean"][0] == 2
    assert df["middle__longest_strike_above_mean"][0] == 3
    assert df["middle__longest_strike_below_mean"][0] == 1
    assert df["end__longest_strike_above_mean"][0] == 3
    assert df["end__longest_strike_below_mean"][0] == 2


def test_longest_strike_is_zero_when_no_match():
    df = pl.DataFrame({"value": [1.0, 1.0, 1.0]}).select(
        features.longest_strike_above_mean("value"),
    )
    assert df["value__longest_strike_above_mean"][0] == 0


def test_position_and_extrema_via_extract_features():
    df = pl.DataFrame(
        {
            "id": ["a", "a", "b", "b"],
            "time": [0, 1, 0, 1],
            "value": [1.0, 5.0, 3.0, 6.0],
        }
    )
    result = extract_features(
        df,
        column_id="id",
        column_sort="time",
        feature_sets=[features.position_and_extrema_feature_set],
    )
    a = result.filter(pl.col("id") == "a").row(0, named=True)
    b = result.filter(pl.col("id") == "b").row(0, named=True)
    assert a["value__first_location_of_maximum"] == 0.5
    assert math.isclose(a["value__count_above_mean"], 1.0)
    assert not a["value__has_duplicate_max"]
    assert b["value__has_duplicate"] is False
    assert b["value__first_location_of_maximum"] == 0.5
