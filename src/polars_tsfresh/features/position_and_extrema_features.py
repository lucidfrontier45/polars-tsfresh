"""Position- and extrema-based features extracted from a single time-series column."""

import polars as pl


def _is_empty(x: pl.Expr) -> pl.Expr:
    """Return an expression that is True when the group has no rows."""
    return x.count() == 0


def _longest_strike(mask: pl.Expr) -> pl.Expr:
    """Maximum length of a consecutive run of True in ``mask``.

    Returns 0 when the group has no True values. Uses native Polars RLE.
    """
    runs = mask.rle()
    lengths = runs.struct.field("len")
    is_true = runs.struct.field("value")
    return (
        pl.when(lengths.filter(is_true).max().is_null())
        .then(0)
        .otherwise(lengths.filter(is_true).max())
    )


def first_location_of_maximum(col_name: str) -> pl.Expr:
    """Relative position of the first occurrence of the maximum value.

    Args:
        col_name (str): The name of the column to compute the feature for.

    Returns:
        pl.Expr: A Polars expression computing the relative position.
    """
    x = pl.col(col_name)
    return (
        pl.when(_is_empty(x))
        .then(float("nan"))
        .otherwise(x.arg_max().cast(pl.Float64) / x.count().cast(pl.Float64))
        .alias(f"{col_name}__first_location_of_maximum")
    )


def last_location_of_maximum(col_name: str) -> pl.Expr:
    """Relative position of the last occurrence of the maximum value.

    Args:
        col_name (str): The name of the column to compute the feature for.

    Returns:
        pl.Expr: A Polars expression computing the relative position.
    """
    x = pl.col(col_name)
    n = x.count().cast(pl.Float64)
    return (
        pl.when(_is_empty(x))
        .then(float("nan"))
        .otherwise(1.0 - x.reverse().arg_max().cast(pl.Float64) / n)
        .alias(f"{col_name}__last_location_of_maximum")
    )


def first_location_of_minimum(col_name: str) -> pl.Expr:
    """Relative position of the first occurrence of the minimum value.

    Args:
        col_name (str): The name of the column to compute the feature for.

    Returns:
        pl.Expr: A Polars expression computing the relative position.
    """
    x = pl.col(col_name)
    return (
        pl.when(_is_empty(x))
        .then(float("nan"))
        .otherwise(x.arg_min().cast(pl.Float64) / x.count().cast(pl.Float64))
        .alias(f"{col_name}__first_location_of_minimum")
    )


def last_location_of_minimum(col_name: str) -> pl.Expr:
    """Relative position of the last occurrence of the minimum value.

    Args:
        col_name (str): The name of the column to compute the feature for.

    Returns:
        pl.Expr: A Polars expression computing the relative position.
    """
    x = pl.col(col_name)
    n = x.count().cast(pl.Float64)
    return (
        pl.when(_is_empty(x))
        .then(float("nan"))
        .otherwise(1.0 - x.reverse().arg_min().cast(pl.Float64) / n)
        .alias(f"{col_name}__last_location_of_minimum")
    )


def count_above_mean(col_name: str) -> pl.Expr:
    """Number of values strictly greater than the mean of the group.

    Args:
        col_name (str): The name of the column to compute the feature for.

    Returns:
        pl.Expr: A Polars expression computing the count above the mean.
    """
    x = pl.col(col_name).cast(pl.Float64)
    mean = x.mean()
    return (x > mean).sum().cast(pl.Int64).alias(f"{col_name}__count_above_mean")


def count_below_mean(col_name: str) -> pl.Expr:
    """Number of values strictly less than the mean of the group.

    Args:
        col_name (str): The name of the column to compute the feature for.

    Returns:
        pl.Expr: A Polars expression computing the count below the mean.
    """
    x = pl.col(col_name).cast(pl.Float64)
    mean = x.mean()
    return (x < mean).sum().cast(pl.Int64).alias(f"{col_name}__count_below_mean")


def count_above(col_name: str, t: float = 0.0) -> pl.Expr:
    """Fraction of values greater than or equal to threshold ``t``.

    Args:
        col_name (str): The name of the column to compute the feature for.
        t (float): The threshold to compare against.

    Returns:
        pl.Expr: A Polars expression computing the fraction above ``t``.
    """
    x = pl.col(col_name).cast(pl.Float64)
    n = x.count().cast(pl.Float64)
    return ((x >= t).sum() / n).alias(f"{col_name}__count_above")


def count_below(col_name: str, t: float = 0.0) -> pl.Expr:
    """Fraction of values less than or equal to threshold ``t``.

    Args:
        col_name (str): The name of the column to compute the feature for.
        t (float): The threshold to compare against.

    Returns:
        pl.Expr: A Polars expression computing the fraction below ``t``.
    """
    x = pl.col(col_name).cast(pl.Float64)
    n = x.count().cast(pl.Float64)
    return ((x <= t).sum() / n).alias(f"{col_name}__count_below")


def has_duplicate(col_name: str) -> pl.Expr:
    """Whether any value occurs more than once.

    Args:
        col_name (str): The name of the column to compute the feature for.

    Returns:
        pl.Expr: A Polars expression computing the duplicate flag.
    """
    x = pl.col(col_name)
    return (pl.len() != x.n_unique()).alias(f"{col_name}__has_duplicate")


def has_duplicate_max(col_name: str) -> pl.Expr:
    """Whether the maximum value is observed more than once.

    Args:
        col_name (str): The name of the column to compute the feature for.

    Returns:
        pl.Expr: A Polars expression computing the duplicate-max flag.
    """
    x = pl.col(col_name)
    return ((x == x.max()).sum() >= 2).alias(f"{col_name}__has_duplicate_max")


def has_duplicate_min(col_name: str) -> pl.Expr:
    """Whether the minimum value is observed more than once.

    Args:
        col_name (str): The name of the column to compute the feature for.

    Returns:
        pl.Expr: A Polars expression computing the duplicate-min flag.
    """
    x = pl.col(col_name)
    return ((x == x.min()).sum() >= 2).alias(f"{col_name}__has_duplicate_min")


def longest_strike_above_mean(col_name: str) -> pl.Expr:
    """Length of the longest consecutive subsequence above the group mean.

    Args:
        col_name (str): The name of the column to compute the feature for.

    Returns:
        pl.Expr: A Polars expression computing the longest strike above mean.
    """
    x = pl.col(col_name).cast(pl.Float64)
    mean = x.mean()
    return _longest_strike(x > mean).alias(f"{col_name}__longest_strike_above_mean")


def longest_strike_below_mean(col_name: str) -> pl.Expr:
    """Length of the longest consecutive subsequence below the group mean.

    Args:
        col_name (str): The name of the column to compute the feature for.

    Returns:
        pl.Expr: A Polars expression computing the longest strike below mean.
    """
    x = pl.col(col_name).cast(pl.Float64)
    mean = x.mean()
    return _longest_strike(x < mean).alias(f"{col_name}__longest_strike_below_mean")


def position_and_extrema_feature_set(col_name: str) -> list[pl.Expr]:
    """Get position and extrema features for a column.

    Args:
        col_name (str): The name of the column to compute features for.

    Returns:
        list[pl.Expr]: Polars expressions for position and extrema features.
    """
    return [
        first_location_of_maximum(col_name),
        last_location_of_maximum(col_name),
        first_location_of_minimum(col_name),
        last_location_of_minimum(col_name),
        count_above_mean(col_name),
        count_below_mean(col_name),
        count_above(col_name),
        count_below(col_name),
        has_duplicate(col_name),
        has_duplicate_max(col_name),
        has_duplicate_min(col_name),
        longest_strike_above_mean(col_name),
        longest_strike_below_mean(col_name),
    ]
