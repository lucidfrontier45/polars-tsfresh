"""Features describing recurring values and data points in one series."""

import polars as pl


def percentage_of_reoccurring_values_to_all_values(col_name: str) -> pl.Expr:
    """Compute fraction of distinct values that occur more than once.

    Args:
        col_name (str): The name of the column to compute the feature for.

    Returns:
        pl.Expr: A Polars expression computing the fraction of recurring values.
    """
    x = pl.col(col_name).cast(pl.Float64)
    return (
        ((x.is_duplicated() & x.is_first_distinct()).sum() / x.n_unique())
        .cast(pl.Float64)
        .alias(f"{col_name}__percentage_of_reoccurring_values_to_all_values")
    )


def percentage_of_reoccurring_datapoints_to_all_datapoints(col_name: str) -> pl.Expr:
    """Compute fraction of data points whose value occurs more than once.

    Args:
        col_name (str): The name of the column to compute the feature for.

    Returns:
        pl.Expr: A Polars expression computing the fraction of recurring data points.
    """
    x = pl.col(col_name).cast(pl.Float64)
    return (
        (x.is_duplicated().sum() / pl.len())
        .cast(pl.Float64)
        .alias(f"{col_name}__percentage_of_reoccurring_datapoints_to_all_datapoints")
    )


def sum_of_reoccurring_values(col_name: str) -> pl.Expr:
    """Compute the sum of each distinct value that occurs more than once.

    Args:
        col_name (str): The name of the column to compute the feature for.

    Returns:
        pl.Expr: A Polars expression computing the sum of recurring values.
    """
    x = pl.col(col_name).cast(pl.Float64)
    return (
        x.filter(x.is_duplicated() & x.is_first_distinct())
        .sum()
        .cast(pl.Float64)
        .alias(f"{col_name}__sum_of_reoccurring_values")
    )


def sum_of_reoccurring_data_points(col_name: str) -> pl.Expr:
    """Compute the sum of all data points whose value occurs more than once.

    Args:
        col_name (str): The name of the column to compute the feature for.

    Returns:
        pl.Expr: A Polars expression computing the sum of recurring data points.
    """
    x = pl.col(col_name).cast(pl.Float64)
    return (
        x.filter(x.is_duplicated())
        .sum()
        .cast(pl.Float64)
        .alias(f"{col_name}__sum_of_reoccurring_data_points")
    )


def recurring_value_feature_set(col_name: str) -> list[pl.Expr]:
    """Get recurring-value features for a column.

    Args:
        col_name (str): The name of the column to compute features for.

    Returns:
        list[pl.Expr]: Polars expressions for recurring-value features.
    """
    return [
        percentage_of_reoccurring_datapoints_to_all_datapoints(col_name),
        percentage_of_reoccurring_values_to_all_values(col_name),
        sum_of_reoccurring_data_points(col_name),
        sum_of_reoccurring_values(col_name),
    ]
