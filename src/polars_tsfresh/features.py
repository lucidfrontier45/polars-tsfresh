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
