from collections.abc import Mapping

import polars as pl

from polars_tsfresh.feature_extraction.settings import minimal_fc_parameters

from . import features


def extract_features(
    df: pl.DataFrame,
    column_id: str,
    column_sort: str,
    kind_to_fc_parameters: Mapping[str, Mapping | None] | None = None,
) -> pl.DataFrame:
    """
    Extract features from a Polars DataFrame using tsfresh-like functionality.

    Parameters:
    df (polars.DataFrame): Input Polars DataFrame.
    column_id (str): The name of the column containing the IDs.
    column_sort (str): The name of the column to sort by.
    kind_to_fc_parameters (dict, optional): Either of two schemas, matching
        tsfresh's convention:
        - fc-parameters style: {"feature_name": None, ...}. Applies the given
          features to every non-id, non-sort column.
        - kind-to-fc-parameters style: {"column_name": {"feature_name": None,
          ...}, ...}. Applies the given features to each named column.
        Defaults to the minimal feature set applied to every column. An
        explicitly empty dict computes no features, matching tsfresh.

    Returns:
    polars.DataFrame: DataFrame with extracted features.
    """

    if kind_to_fc_parameters is None:
        kind_to_fc_parameters = minimal_fc_parameters()

    feature_exprs = _build_feature_exprs(
        df, column_id, column_sort, kind_to_fc_parameters
    )

    grouped = df.sort(column_sort).group_by(column_id)
    result = grouped.agg(feature_exprs)

    return result


def _build_feature_exprs(
    df: pl.DataFrame,
    column_id: str,
    column_sort: str,
    kind_to_fc_parameters: Mapping[str, Mapping | None],
) -> list[pl.Expr]:
    values = list(kind_to_fc_parameters.values())
    if not values:
        return []

    is_fc_parameters = all(v is None for v in values)
    is_kind_to_fc_parameters = all(isinstance(v, Mapping) for v in values)

    if not (is_fc_parameters or is_kind_to_fc_parameters):
        raise ValueError(
            "kind_to_fc_parameters must not mix feature-name keys (value=None) "
            "with column-name keys (value=dict); got a mix of both."
        )

    if is_fc_parameters:
        fnames = list(kind_to_fc_parameters.keys())
        feature_exprs = []
        for col in df.columns:
            if col not in (column_id, column_sort):
                for fname in fnames:
                    feature_exprs.append(_get_feature_fn(col, fname))
        return feature_exprs

    feature_exprs = []
    for col, fparams in kind_to_fc_parameters.items():
        for fname in fparams or {}:
            feature_exprs.append(_get_feature_fn(col, fname))
    return feature_exprs


def _get_feature_fn(col: str, fname: str) -> pl.Expr:
    fn = getattr(features, fname, None)
    if fn is None:
        raise ValueError(
            f"Unknown feature calculator {fname!r} requested for column {col!r}."
        )
    return fn(col)
