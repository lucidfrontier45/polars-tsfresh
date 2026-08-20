"""Change- and rate-based features extracted from a single time-series column."""

import math

import polars as pl


def _poisoned(x: pl.Expr) -> pl.Expr:
    """Return an expression that is True if the group contains any null or NaN value."""
    return x.is_null().any() | x.is_nan().any()


def _diffs(col_name: str) -> pl.Expr:
    """Compute consecutive differences as Float64.

    Differences are exact while they fit the column's diff dtype (the
    supersigned promotion of the native dtype). Polars wraps signed overflow
    (e.g. Int64 differences beyond ±2**63) and nulls unsigned overflow, so
    such groups are poisoned with NaN instead of returning a wrong number.
    """
    x = pl.col(col_name)
    d = x.diff()
    overflow = (
        ((x >= 0) & (x.shift(1) < 0) & (d < 0))
        | ((x < 0) & (x.shift(1) > 0) & (d > 0))
        | (d.is_null() & x.shift(1).is_not_null())
    )
    return pl.when(overflow).then(math.nan).otherwise(d).drop_nulls()


def mean_abs_change(col_name: str) -> pl.Expr:
    """Compute the mean absolute change between consecutive values.

    Args:
        col_name (str): The name of the column to compute the feature for.

    Returns:
        pl.Expr: A Polars expression computing the mean absolute change.
    """
    x = pl.col(col_name).cast(pl.Float64)
    return (
        pl.when(_poisoned(x))
        .then(math.nan)
        .otherwise(_diffs(col_name).abs().mean())
        .fill_null(math.nan)
        .alias(f"{col_name}__mean_abs_change")
    )


def mean_change(col_name: str) -> pl.Expr:
    """Compute the mean change between consecutive values.

    Args:
        col_name (str): The name of the column to compute the feature for.

    Returns:
        pl.Expr: A Polars expression computing the mean change.
    """
    x = pl.col(col_name).cast(pl.Float64)
    return (
        pl.when(_poisoned(x))
        .then(math.nan)
        .otherwise(_diffs(col_name).mean())
        .fill_null(math.nan)
        .alias(f"{col_name}__mean_change")
    )


def mean_second_derivative_central(col_name: str) -> pl.Expr:
    """Compute the mean central approximation of the second derivative.

    Args:
        col_name (str): The name of the column to compute the feature for.

    Returns:
        pl.Expr: A Polars expression computing the mean central second derivative.
    """
    x = pl.col(col_name).cast(pl.Float64)
    return (
        pl.when(_poisoned(x))
        .then(math.nan)
        .otherwise(_diffs(col_name).diff().drop_nulls().mean() / 2)
        .fill_null(math.nan)
        .alias(f"{col_name}__mean_second_derivative_central")
    )


def absolute_sum_of_changes(col_name: str) -> pl.Expr:
    """Compute the sum of absolute changes between consecutive values.

    Args:
        col_name (str): The name of the column to compute the feature for.

    Returns:
        pl.Expr: A Polars expression computing the absolute sum of changes.
    """
    x = pl.col(col_name).cast(pl.Float64)
    return (
        pl.when(_poisoned(x))
        .then(math.nan)
        .otherwise(_diffs(col_name).abs().sum())
        .fill_null(0.0)
        .alias(f"{col_name}__absolute_sum_of_changes")
    )


def cid_ce(col_name: str, normalize: bool = True) -> pl.Expr:
    """Compute the cumulative incremental deviation complexity estimate.

    Args:
        col_name (str): The name of the column to compute the feature for.
        normalize (bool): Whether to z-normalize values before computing complexity.

    Returns:
        pl.Expr: A Polars expression computing the complexity estimate.
    """
    x = pl.col(col_name).cast(pl.Float64)
    diffs = _diffs(col_name)
    if not normalize:
        return (
            pl.when(_poisoned(x))
            .then(math.nan)
            .otherwise(diffs.pow(2).sum().sqrt())
            .fill_null(0.0)
            .alias(f"{col_name}__cid_ce")
        )
    # Min-center in the native dtype while the span fits it: exact for
    # large-offset integer series (e.g. ns timestamps). When the span
    # overflows the native dtype, native centering would wrap, so fall back
    # to Float64 centering, which never wraps.
    centered_native = pl.col(col_name) - pl.col(col_name).min()
    fits = centered_native.min() >= 0
    centered = pl.when(fits).then(centered_native.cast(pl.Float64)).otherwise(x - x.min())
    std = centered.std(0)
    normalized = (centered - centered.mean()) / std
    inner = (
        pl.when(std == 0)
        .then(diffs.pow(2).sum().sqrt())
        .otherwise(normalized.diff().drop_nulls().pow(2).sum().sqrt())
    )
    return (
        pl.when(_poisoned(x))
        .then(math.nan)
        .otherwise(inner)
        .fill_null(0.0)
        .alias(f"{col_name}__cid_ce")
    )


def change_and_rate_feature_set(col_name: str) -> list[pl.Expr]:
    """Get change and rate features for a column.

    Args:
        col_name (str): The name of the column to compute features for.

    Returns:
        list[pl.Expr]: Polars expressions for change and rate features.
    """
    return [
        mean_abs_change(col_name),
        mean_change(col_name),
        mean_second_derivative_central(col_name),
        absolute_sum_of_changes(col_name),
        cid_ce(col_name),
    ]
