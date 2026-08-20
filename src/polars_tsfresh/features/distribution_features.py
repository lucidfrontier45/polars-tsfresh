"""Distribution-shape features extracted from a single time-series column."""

import math

import polars as pl


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
        .then(math.nan)
        .otherwise(x.std(0) / m)
        .fill_null(math.nan)
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

    Integer inputs use exact native-dtype centering when their span fits the
    dtype. Unlike ``numpy.histogram``, this avoids Float64 edge degradation above
    ``2**53`` and handles value ranges smaller than ``max_bins`` without error.

    Args:
        col_name (str): The name of the column to compute the binned entropy for.
        max_bins (int): The number of equidistant bins. Defaults to 10.

    Returns:
        pl.Expr: A Polars expression that computes the binned entropy.
    """
    # Native centering preserves large integer bin boundaries. Signed integer
    # spans can overflow, detected by wrapped negative values; Float64 centering
    # loses some precision but cannot silently wrap.
    x_native = pl.col(col_name)
    centered_native = x_native - x_native.min()
    x_float = x_native.cast(pl.Float64)
    x = (
        pl.when(centered_native.min() >= 0)
        .then(centered_native.cast(pl.Float64))
        .otherwise(x_float - x_float.min())
    )
    lo, hi = pl.lit(0.0), x.max()
    idx = ((x - lo) / (hi - lo) * max_bins).floor().clip(0, max_bins - 1)
    counts = idx.unique_counts().cast(pl.Float64)
    p = counts / counts.sum()
    entropy = (-(p * p.log())).sum()
    body = (
        pl.when(x.is_nan().any() | x.is_null().any() | (x.len() == 0))
        .then(math.nan)
        .otherwise(pl.when(hi == lo).then(0.0).otherwise(entropy))
    )
    return body.fill_null(math.nan).alias(f"{col_name}__binned_entropy")


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
    return corr.fill_null(math.nan).alias(f"{col_name}__benford_correlation")


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
