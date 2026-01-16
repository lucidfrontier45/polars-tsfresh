# Use local features names to mirror tsfresh MinimalFCParameters
_FEATURE_NAMES_MINIMAL = [
    "mean",
    "median",
    "variance",
    "standard_deviation",
    "length",
    "maximum",
    "minimum",
    "absolute_maximum",
    "root_mean_square",
    "sum_values",
]


def minimal_fc_parameters() -> dict[str, dict|None]:
    """Return minimal feature calculator parameters, tsfresh-style.

    Matches tsfresh's MinimalFCParameters schema: a dict mapping
    feature calculator names to (usually empty) parameter dicts.
    """
    return {name: None for name in _FEATURE_NAMES_MINIMAL}


def from_columns(columns: list[str]) -> dict[str, dict[str, None]]:
    """Build `kind_to_fc_parameters` from a list of column names.

    Equivalent to `tsfresh.feature_extraction.settings.from_columns` using
    this project's minimal feature set.

    Args:
        columns: List of column names to compute features for.

    Returns:
        Dict mapping each column name to the minimal feature parameters dict.
    """
    ret = {}
    for col in columns:
        k, v = col.split("__")
        ret.setdefault(k, {})[v] = None
    return ret
