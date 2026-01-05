import polars as pl

from . import features


def extract_features(df: pl.DataFrame, column_id: str, column_sort: str):
    """
    Extract features from a Polars DataFrame using tsfresh-like functionality.

    Parameters:
    df (polars.DataFrame): Input Polars DataFrame.
    column_id (str): The name of the column containing the IDs.
    column_sort (str): The name of the column to sort by.

    Returns:
    polars.DataFrame: DataFrame with extracted features.
    """

    # get all columns other than column_id and column_sort
    feature_columns = [col for col in df.columns if col not in (column_id, column_sort)]
    feature_exprs = []
    for col in feature_columns:
        feature_exprs.extend(features.minimal_feature_set(col))

    grouped = df.sort(column_sort).group_by(column_id)

    result = grouped.agg(feature_exprs)

    return result
