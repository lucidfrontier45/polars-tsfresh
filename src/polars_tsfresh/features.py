import math

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


NAN = float("nan")


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
    return pl.when(overflow).then(NAN).otherwise(d).drop_nulls()


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
    """Compute the variation coefficient (population standard deviation / mean).

    Uses the population standard deviation (ddof=0) to match tsfresh, whose
    reference implementation relies on ``numpy.std``.

    Returns NaN if the mean is zero or non-finite, or the series is empty.

    Args:
        col_name (str): The name of the column to compute the variation coefficient for.

    Returns:
        pl.Expr: A Polars expression that computes the variation coefficient.
    """
    # NaN/null values are skipped, matching the numpy.nanmean/numpy.nanstd
    # semantics of the tsfresh reference implementation.
    x = pl.col(col_name).cast(pl.Float64).drop_nulls().drop_nans()
    m = x.mean()
    return (
        pl.when((m == 0) | ~m.is_finite())
        .then(NAN)
        .otherwise(x.std(0) / m)
        .fill_null(NAN)
        .alias(f"{col_name}__variation_coefficient")
    )


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

    Returns NaN if the input is empty or contains NaN or null values.

    Unlike ``numpy.histogram``, integer inputs whose value range is smaller
    than ``max_bins`` are binned without error.

    Args:
        col_name (str): The name of the column to compute the binned entropy for.
        max_bins (int): The number of equidistant bins. Defaults to 10.

    Returns:
        pl.Expr: A Polars expression that computes the binned entropy.
    """
    x = pl.col(col_name).cast(pl.Float64)
    lo, hi = x.min(), x.max()
    idx = ((x - lo) / (hi - lo) * max_bins).floor().clip(0, max_bins - 1)
    counts = idx.unique_counts().cast(pl.Float64)
    p = counts / counts.sum()
    entropy = (-(p * p.log())).sum()
    body = (
        pl.when(x.is_nan().any() | x.is_null().any() | (x.len() == 0))
        .then(NAN)
        .otherwise(pl.when(hi == lo).then(0.0).otherwise(entropy))
    )
    return body.fill_null(NAN).alias(f"{col_name}__binned_entropy")


def benford_correlation(col_name: str) -> pl.Expr:
    """Compute the correlation between the first-digit distribution of ``x`` and Benford's law.

    Useful for anomaly detection. Returns NaN for empty series or if Benford's
    law produces zero variance.

    Note: Zero, NaN, and infinity values are excluded: their first
    significant digit is undefined, so they are ignored rather than
    rejected. (The numpy reference mapped NaN to 0 via ``np.nan_to_num`` and
    ±inf to a large finite value with first digit 1; excluding inf instead
    is a deliberate deviation.)

    Args:
        col_name (str): The name of the column to compute the correlation for.

    Returns:
        pl.Expr: A Polars expression that computes the Benford correlation.
    """
    benford = [math.log10(1.0 + 1.0 / d) for d in range(1, 10)]
    sb1, sb2 = 1.0, sum(b * b for b in benford)  # sum(benford) == 1
    x = pl.col(col_name).cast(pl.Float64)
    n = x.len()
    digit = x.abs().cast(pl.String).str.extract(r"^[0.]*([1-9])", 1).cast(pl.Int64)
    mapping = {d: benford[d - 1] for d in range(1, 10)}
    sfb = digit.replace_strict(mapping, default=0.0, return_dtype=pl.Float64).sum() / n
    counts = digit.drop_nulls().unique_counts().cast(pl.Float64)
    sf1 = counts.sum() / n
    sf2 = counts.pow(2).sum() / (n * n)
    a_mean = sf1 / 9.0
    b_mean = sb1 / 9.0
    corr = (sfb / 9.0 - a_mean * b_mean) / (
        (sf2 / 9.0 - a_mean**2) * (sb2 / 9.0 - b_mean**2)
    ).sqrt()
    return corr.fill_null(NAN).alias(f"{col_name}__benford_correlation")


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
        .then(NAN)
        .otherwise(_diffs(col_name).abs().mean())
        .fill_null(NAN)
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
        .then(NAN)
        .otherwise(_diffs(col_name).mean())
        .fill_null(NAN)
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
        .then(NAN)
        .otherwise(_diffs(col_name).diff().drop_nulls().mean() / 2)
        .fill_null(NAN)
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
        .then(NAN)
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
            .then(NAN)
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
        pl.when(_poisoned(x)).then(NAN).otherwise(inner).fill_null(0.0).alias(f"{col_name}__cid_ce")
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
