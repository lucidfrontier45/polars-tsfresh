import math
from collections.abc import Callable

import numpy as np
import polars as pl


def mean(col_name: str) -> pl.Expr:
    """Compute the mean of a column.

    Args:
        col_name (str): The name of the column to compute the mean for.

    Returns:
        pl.Expr: A Polars expression that computes the mean.
    """
    return pl.col(col_name).mean().alias(f"{col_name}__mean")


def median(col_name: str) -> pl.Expr:
    """Compute the median of a column.

    Args:
        col_name (str): The name of the column to compute the median for.

    Returns:
        pl.Expr: A Polars expression that computes the median.
    """

    return pl.col(col_name).median().alias(f"{col_name}__median")


def variance(col_name: str) -> pl.Expr:
    """Compute the variance of a column.

    Args:
        col_name (str): The name of the column to compute the variance for.

    Returns:
        pl.Expr: A Polars expression that computes the variance.
    """
    return pl.col(col_name).var(0).alias(f"{col_name}__variance")


def standard_deviation(col_name: str) -> pl.Expr:
    """Compute the standard deviation of a column.

    Args:
        col_name (str): The name of the column to compute the standard deviation for.

    Returns:
        pl.Expr: A Polars expression that computes the standard deviation.
    """
    return pl.col(col_name).std(0).alias(f"{col_name}__standard_deviation")


def length(col_name: str) -> pl.Expr:
    """Compute the length of a column.

    Args:
        col_name (str): The name of the column to compute the length for.

    Returns:
        pl.Expr: A Polars expression that computes the length.
    """
    return pl.col(col_name).len().alias(f"{col_name}__length")


def maximum(col_name: str) -> pl.Expr:
    """Compute the maximum value of a column.

    Args:
        col_name (str): The name of the column to compute the maximum for.

    Returns:
        pl.Expr: A Polars expression that computes the maximum.
    """
    return pl.col(col_name).max().alias(f"{col_name}__maximum")


def minimum(col_name: str) -> pl.Expr:
    """Compute the minimum value of a column.

    Args:
        col_name (str): The name of the column to compute the minimum for.

    Returns:
        pl.Expr: A Polars expression that computes the minimum.
    """
    return pl.col(col_name).min().alias(f"{col_name}__minimum")


def absolute_maximum(col_name: str) -> pl.Expr:
    """Compute the maximum absolute value of a column.

    Args:
        col_name (str): The name of the column to compute the absolute maximum for.

    Returns:
        pl.Expr: A Polars expression that computes the absolute maximum.
    """
    return pl.col(col_name).abs().max().alias(f"{col_name}__absolute_maximum")


def root_mean_square(col_name: str) -> pl.Expr:
    """Compute the root mean square of a column.

    Args:
        col_name (str): The name of the column to compute the root mean square for.

    Returns:
        pl.Expr: A Polars expression that computes the root mean square.
    """
    return pl.col(col_name).pow(2).mean().sqrt().alias(f"{col_name}__root_mean_square")


def sum_values(col_name: str) -> pl.Expr:
    """Compute the sum of all values in a column.

    Args:
        col_name (str): The name of the column to compute the sum for.

    Returns:
        pl.Expr: A Polars expression that computes the sum.
    """
    return pl.col(col_name).sum().alias(f"{col_name}__sum_values")


def minimal_feature_set(col_name: str) -> list[pl.Expr]:
    """Get a minimal set of features for a column.

    Args:
        col_name (str): The name of the column to compute features for.

    Returns:
        list[pl.Expr]: A list of Polars expressions for the minimal feature set.
    """
    return [
        mean(col_name),
        median(col_name),
        variance(col_name),
        standard_deviation(col_name),
        length(col_name),
        maximum(col_name),
        minimum(col_name),
        absolute_maximum(col_name),
        root_mean_square(col_name),
        sum_values(col_name),
    ]


def _scalar_expr(
    col_name: str,
    func: "Callable[[pl.Series], float]",
    name: str,
    dtype: "pl.DataType | None" = None,
) -> pl.Expr:
    """Wrap a scalar ``func(pl.Series) -> float`` into a per-group aggregation expression.

    Args:
        col_name (str): The input column to apply ``func`` to.
        func (Callable[[pl.Series], float]): A function taking a ``pl.Series`` and returning a scalar.
        name (str): The name of the output column.
        dtype (pl.DataType): The Polars dtype of the scalar result.

    Returns:
        pl.Expr: A Polars expression that yields a scalar per group.
    """
    return (
        pl.col(col_name)
        .map_batches(lambda s: pl.Series([func(s)]), return_dtype=dtype or pl.Float64)
        .first()
        .alias(name)
    )


def skewness(col_name: str) -> pl.Expr:
    """Compute the sample skewness (adjusted Fisher-Pearson standardized moment coefficient G1).

    Args:
        col_name (str): The name of the column to compute skewness for.

    Returns:
        pl.Expr: A Polars expression that computes the bias-corrected skewness.
    """
    return pl.col(col_name).skew(bias=False).alias(f"{col_name}__skewness")


def kurtosis(col_name: str) -> pl.Expr:
    """Compute the kurtosis (adjusted Fisher-Pearson standardized moment coefficient G2, excess kurtosis).

    Args:
        col_name (str): The name of the column to compute kurtosis for.

    Returns:
        pl.Expr: A Polars expression that computes the excess kurtosis.
    """
    return pl.col(col_name).kurtosis(fisher=True, bias=False).alias(f"{col_name}__kurtosis")


def variation_coefficient(col_name: str) -> pl.Expr:
    """Compute the variation coefficient (standard deviation / mean).

    Returns NaN if the mean is zero.

    Args:
        col_name (str): The name of the column to compute the variation coefficient for.

    Returns:
        pl.Expr: A Polars expression that computes the variation coefficient.
    """

    def _calc(s: pl.Series) -> float:
        arr = s.to_numpy()
        if arr.size == 0:
            return float("nan")
        m = float(np.nanmean(arr))
        if m == 0 or not math.isfinite(m):
            return float("nan")
        return float(np.nanstd(arr, ddof=0) / m)

    return _scalar_expr(col_name, _calc, f"{col_name}__variation_coefficient")


def quantile(col_name: str, q: float = 0.5) -> pl.Expr:
    """Compute the q-th quantile of a column.

    Uses linear interpolation to match tsfresh's reference implementation.

    Args:
        col_name (str): The name of the column to compute the quantile for.
        q (float): The quantile to compute, in the range [0, 1]. Defaults to 0.5.

    Returns:
        pl.Expr: A Polars expression that computes the q-th quantile.
    """
    return (
        pl.col(col_name).quantile(q, interpolation="linear").alias(f"{col_name}__quantile__q_{q}")
    )


def binned_entropy(col_name: str, max_bins: int = 10) -> pl.Expr:
    """Compute the binned entropy of a column.

    Values are first binned into ``max_bins`` equidistant bins, then the entropy
    ``-sum(p * log(p))`` of the normalized bin probabilities is computed.

    Returns NaN if the input is empty or contains NaN values.

    Args:
        col_name (str): The name of the column to compute the binned entropy for.
        max_bins (int): The number of equidistant bins. Defaults to 10.

    Returns:
        pl.Expr: A Polars expression that computes the binned entropy.
    """

    def _calc(s: pl.Series) -> float:
        arr = s.to_numpy()
        if arr.size == 0 or np.any(np.isnan(arr)):
            return float("nan")
        counts, _ = np.histogram(arr, bins=max_bins)
        total = counts.sum()
        if total == 0:
            return float("nan")
        p = counts / total
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)))

    return _scalar_expr(col_name, _calc, f"{col_name}__binned_entropy")


def benford_correlation(col_name: str) -> pl.Expr:
    """Compute the correlation between the first-digit distribution of ``x`` and Benford's law.

    Useful for anomaly detection. Returns NaN for empty series or if Benford's
    law produces zero variance.

    Args:
        col_name (str): The name of the column to compute the correlation for.

    Returns:
        pl.Expr: A Polars expression that computes the Benford correlation.
    """
    benford = np.log10(1.0 + 1.0 / np.arange(1, 10))

    def _calc(s: pl.Series) -> float:
        arr = s.to_numpy()
        if arr.size == 0:
            return float("nan")
        first_digits = np.array(
            [int(np.format_float_scientific(i)[:1]) for i in np.abs(np.nan_to_num(arr))]
        )
        observed = np.array([(first_digits == n).mean() for n in range(1, 10)])
        return float(np.corrcoef(benford, observed)[0, 1])

    return _scalar_expr(col_name, _calc, f"{col_name}__benford_correlation")


def distribution_feature_set(col_name: str) -> list[pl.Expr]:
    """Get the distribution feature set for a column.

    Args:
        col_name (str): The name of the column to compute features for.

    Returns:
        list[pl.Expr]: A list of Polars expressions for the distribution feature set.
    """
    return [
        skewness(col_name),
        kurtosis(col_name),
        variation_coefficient(col_name),
        quantile(col_name),
        binned_entropy(col_name),
        benford_correlation(col_name),
    ]
