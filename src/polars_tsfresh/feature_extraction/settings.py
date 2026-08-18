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


def minimal_fc_parameters() -> dict[str, None]:
    """Return minimal feature calculator parameters, tsfresh-style.

    Matches tsfresh's MinimalFCParameters schema: a dict mapping feature
    calculator names to None, since none of the minimal calculators take
    parameters.
    """
    return {name: None for name in _FEATURE_NAMES_MINIMAL}


def from_columns(columns: list[str]) -> dict[str, dict]:
    """Build `kind_to_fc_parameters` from a list of feature column names.

    Equivalent to `tsfresh.feature_extraction.settings.from_columns`.
    Each entry must be named "<column>__<feature_name>"; the final "__" is
    used as the separator, so column names may contain "__" themselves.

    Args:
        columns: List of "<column>__<feature_name>" column names.

    Returns:
        Dict mapping each column name to a dict of its requested feature
        names.

    Raises:
        ValueError: If an entry does not contain the "__" separator.
    """
    ret: dict[str, dict[str, None]] = {}
    for col in columns:
        if "__" not in col:
            raise ValueError(
                f"Column name {col!r} is not in '<column>__<feature_name>' format."
            )
        k, _, v = col.rpartition("__")
        ret.setdefault(k, {})[v] = None
    return ret
