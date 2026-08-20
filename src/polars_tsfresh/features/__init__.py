"""Feature group modules. Re-exports keep the public ``features.X`` API stable."""

from . import basic_statistics, change_and_rate_features, distribution_features
from .basic_statistics import (
    absolute_maximum,
    length,
    maximum,
    mean,
    median,
    minimal_feature_set,
    minimum,
    root_mean_square,
    standard_deviation,
    sum_values,
    variance,
)
from .change_and_rate_features import (
    absolute_sum_of_changes,
    change_and_rate_feature_set,
    cid_ce,
    mean_abs_change,
    mean_change,
    mean_second_derivative_central,
)
from .distribution_features import (
    benford_correlation,
    binned_entropy,
    distribution_feature_set,
    kurtosis,
    quantile,
    skewness,
    variation_coefficient,
)

__all__ = [
    "absolute_maximum",
    "absolute_sum_of_changes",
    "basic_statistics",
    "benford_correlation",
    "binned_entropy",
    "change_and_rate_feature_set",
    "change_and_rate_features",
    "cid_ce",
    "distribution_feature_set",
    "distribution_features",
    "kurtosis",
    "length",
    "maximum",
    "mean",
    "mean_abs_change",
    "mean_change",
    "mean_second_derivative_central",
    "median",
    "minimal_feature_set",
    "minimum",
    "quantile",
    "root_mean_square",
    "skewness",
    "standard_deviation",
    "sum_values",
    "variance",
    "variation_coefficient",
]
