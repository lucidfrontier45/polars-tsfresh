"""Peak, valley, and threshold-crossing features extracted from one series."""

import polars as pl


def number_peaks(col_name: str, n: int = 1) -> pl.Expr:
    """Count points strictly greater than their ``n`` neighbors on each side.

    Args:
        col_name (str): The name of the column to compute the feature for.
        n (int): Number of neighboring points to compare on each side.

    Returns:
        pl.Expr: A Polars expression computing the number of peaks.
    """
    x = pl.col(col_name)
    is_peak = pl.lit(True)
    for offset in range(1, n + 1):
        is_peak &= (x > x.shift(offset)) & (x > x.shift(-offset))
    return is_peak.sum().cast(pl.Int64).alias(f"{col_name}__number_peaks")


def number_crossing_m(col_name: str, m: float = 0.0) -> pl.Expr:
    """Count adjacent pairs that cross threshold ``m`` strictly.

    Args:
        col_name (str): The name of the column to compute the feature for.
        m (float): The threshold to cross.

    Returns:
        pl.Expr: A Polars expression computing the number of threshold crossings.
    """
    x = pl.col(col_name).cast(pl.Float64)
    return (
        ((x - m) * (x.shift(1) - m) < 0)
        .sum()
        .cast(pl.Int64)
        .alias(f"{col_name}__number_crossing_m")
    )


def peak_valley_and_crossing_feature_set(col_name: str) -> list[pl.Expr]:
    """Get peak and threshold-crossing features for a column.

    Args:
        col_name (str): The name of the column to compute features for.

    Returns:
        list[pl.Expr]: Polars expressions for peak and crossing features.
    """
    return [number_crossing_m(col_name), number_peaks(col_name)]
