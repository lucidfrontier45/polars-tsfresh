from collections.abc import Callable

import polars as pl

from . import features
from .features import (
    basic_statistics,
    change_and_rate_features,
    distribution_features,
    position_and_extrema_features,
)


def extract_features(
    df: pl.DataFrame,
    column_id: str,
    column_sort: str,
    feature_sets: list[Callable[[str], list[pl.Expr]]] | None = None,
) -> pl.DataFrame:
    """
    Extract features from a Polars DataFrame using tsfresh-like functionality.

    Parameters:
    df (polars.DataFrame): Input Polars DataFrame.
    column_id (str): The name of the column containing the IDs.
    column_sort (str): The name of the column to sort by.
    feature_sets (list[Callable[[str], list[pl.Expr]]]): Optional list of feature set
        functions to apply. Each function takes a column name and returns a list of
        Polars expressions. Defaults to ``[features.minimal_feature_set]``.

    Returns:
    polars.DataFrame: DataFrame with extracted features.
    """
    if feature_sets is None:
        feature_sets = [features.minimal_feature_set]

    # get all columns other than column_id and column_sort
    feature_columns = [col for col in df.columns if col not in (column_id, column_sort)]
    feature_exprs = []
    for col in feature_columns:
        for feature_set in feature_sets:
            feature_exprs.extend(feature_set(col))

    grouped = df.sort(column_sort).group_by(column_id)

    result = grouped.agg(feature_exprs)

    return result


__all__ = [
    "basic_statistics",
    "change_and_rate_features",
    "distribution_features",
    "extract_features",
    "features",
    "position_and_extrema_features",
]
