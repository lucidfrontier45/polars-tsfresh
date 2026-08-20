import polars as pl

from polars_tsfresh import extract_features, features


def test_number_peaks_strict_neighbors_and_boundaries():
    result = pl.DataFrame(
        {
            "n1": [0.0, 3.0, 0.0, 2.0, 2.0, 1.0, 0.0],
            "n2": [0.0, 1.0, 4.0, 1.0, 0.0, 3.0, 0.0],
            "n0": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        }
    ).select(
        features.number_peaks("n1"),
        features.number_peaks("n2", n=2),
        features.number_peaks("n0", n=0),
    )
    short_result = pl.DataFrame({"value": [1.0, 2.0, 1.0]}).select(
        features.number_peaks("value", n=2),
    )
    boundary_result = pl.DataFrame({"value": [3.0, 0.0, 1.0]}).select(
        features.number_peaks("value"),
    )

    assert result.row(0) == (1, 1, 0)
    assert short_result.row(0) == (0,)
    assert boundary_result.row(0) == (0,)


def test_number_crossing_m_counts_boolean_threshold_transitions():
    result = pl.DataFrame(
        {
            "up_down": [-2.0, 3.0, 1.0, -4.0, 0.0],
            "touching": [-1.0, 0.0, 1.0, 0.0, -1.0],
            "nonzero": [1.0, 4.0, 6.0, 2.0, 7.0],
            "constant": [2.0, 2.0, 2.0, 2.0, 2.0],
        }
    ).select(
        features.number_crossing_m("up_down"),
        features.number_crossing_m("touching"),
        features.number_crossing_m("nonzero", m=5.0),
        features.number_crossing_m("constant", m=1.0),
    )

    assert result.row(0) == (2, 2, 3, 0)


def test_number_crossing_m_counts_threshold_as_not_above():
    result = pl.DataFrame(
        {
            "ending_at_threshold": [-1.0, 0.0, 1.0],
            "starting_at_threshold": [0.0, 1.0, 0.0],
        }
    ).select(
        features.number_crossing_m("ending_at_threshold"),
        features.number_crossing_m("starting_at_threshold"),
    )

    assert result.row(0) == (1, 2)


def test_peak_valley_and_crossing_features_via_extract_features():
    df = pl.DataFrame(
        {
            "id": ["a", "a", "a", "b", "b", "b"],
            "time": [2, 0, 1, 1, 2, 0],
            "value": [0.0, 0.0, 3.0, 2.0, -1.0, -2.0],
        }
    )
    result = extract_features(
        df,
        column_id="id",
        column_sort="time",
        feature_sets=[features.peak_valley_and_crossing_feature_set],
    )

    a = result.filter(pl.col("id") == "a").row(0, named=True)
    b = result.filter(pl.col("id") == "b").row(0, named=True)
    assert a["value__number_crossing_m"] == 2
    assert a["value__number_peaks"] == 1
    assert b["value__number_crossing_m"] == 2
    assert b["value__number_peaks"] == 1
