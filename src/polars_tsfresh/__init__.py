import polars as pl

from polars_tsfresh.feature_extraction.settings import minimal_fc_parameters

from . import features


def extract_features(
    df: pl.DataFrame,
    column_id: str,
    column_sort: str,
    kind_to_fc_parameters: dict[str, dict | None] | None = None,
) -> pl.DataFrame:
    """
    Extract features from a Polars DataFrame using tsfresh-like functionality.

    Parameters:
    df (polars.DataFrame): Input Polars DataFrame.
    column_id (str): The name of the column containing the IDs.
    column_sort (str): The name of the column to sort by.
    kind_to_fc_parameters (dict, optional): A dictionary mapping column names to feature.
    i.e. {'column_name': {'feature_name': None, ...}, ...}. or {'feature_name: None, ...}.

    Returns:
    polars.DataFrame: DataFrame with extracted features.
    """

    # get all columns other than column_id and column_sort
    feature_exprs = []
    # Default to minimal fc parameters when not provided
    if not kind_to_fc_parameters:
        kind_to_fc_parameters = minimal_fc_parameters()

    # If parameters are None or contain None values, use minimal features for all non-id columns
    if next(iter(kind_to_fc_parameters.values())) is None:
        for col in df.columns:
            if col not in (column_id, column_sort):
                feature_exprs.extend(features.minimal_feature_set(col))
    else:
        for col, fnames in kind_to_fc_parameters.items():
            for fname in fnames:
                feature_exprs.append(getattr(features, fname)(col))

    grouped = df.sort(column_sort).group_by(column_id)
    result = grouped.agg(feature_exprs)

    return result
