# Feature Sets

A **feature set** in `polars-tsfresh` is a callable that takes a column
name and returns a list of `polars.Expr` aggregations. `extract_features`
applies one or more feature sets to every column except the id and sort
columns, then groups by the id column to compute one row of features
per series.

```python
from polars_tsfresh import extract_features
from polars_tsfresh.features import minimal_feature_set

df = pl.read_csv("data.csv")
features = extract_features(df, column_id="id", column_sort="date")
# equivalently, pass feature sets explicitly:
features = extract_features(
    df,
    column_id="id",
    column_sort="date",
    feature_sets=[minimal_feature_set],
)
```

The output has one row per unique id and one column per feature. Feature
column names follow the convention `{column_name}__{feature_name}`.

## Minimal Feature Set

The minimal feature set is the default feature set used by
`extract_features`. It produces a compact battery of ten summary
statistics that describe the central tendency, spread, range, and gross
energy of a time series.

### The ten features

For a column `x` with `n` observations, the minimal feature set computes:

| Feature | Expression | Output column |
| ------- | ---------- | ------------- |
| Mean | `mean(x)` | `x__mean` |
| Median | `median(x)` | `x__median` |
| Variance | population variance, `var(ddof=0)` | `x__variance` |
| Standard deviation | population std, `std(ddof=0)` | `x__standard_deviation` |
| Length | number of observations, `n` | `x__length` |
| Maximum | `max(x)` | `x__maximum` |
| Minimum | `min(x)` | `x__minimum` |
| Absolute maximum | `max(\|x\|)` | `x__absolute_maximum` |
| Root mean square | `sqrt(mean(x^2))` | `x__root_mean_square` |
| Sum | `sum(x)` | `x__sum_values` |

#### Notes on specific features

- **Variance** and **standard deviation** use `ddof=0` (population
  estimator). This matches the column-level `polars` aggregations but may
  differ from sample-based tsfresh conventions; check carefully if
  cross-validating against a sample-variance reference.
- **Length** is the number of non-null observations in the group, not the
  theoretical length of the series.
- **Root mean square** is computed as `sqrt(mean(x^2))`, equivalent to
  the L2 norm normalized by `sqrt(n)`.
- **Absolute maximum** ignores sign; useful for symmetric quantities
  such as accelerometer magnitudes.

### Conventions

- Nulls propagate through every aggregation. A group consisting entirely
  of nulls yields a null for every feature except `length`, which is
  always defined.
- Empty groups (no rows for a given id) are dropped by `group_by` and
  will not appear in the output.
- Feature column names use double underscores (`__`) as the separator
  between source column and feature name, matching the tsfresh naming
  convention.

### Extending the set

To compute only a subset of the minimal features, either pass a custom
list of feature sets to `extract_features`, or call the individual
feature functions directly and aggregate them yourself:

```python
from polars_tsfresh.features import mean, variance, root_mean_square

result = df.group_by("id").agg(
    [
        mean("value"),
        variance("value"),
        root_mean_square("value"),
    ]
)
```

## Distribution Feature Set

The distribution feature set captures the shape of a time series'
empirical distribution: skewness, tail weight, relative spread,
quantile position, information content, and conformity to a
theoretical reference distribution. It produces six features per
column.

### The six features

For a column `x` with `n` observations, the distribution feature set
computes:

| Feature | Definition | Output column |
| ------- | ---------- | ------------- |
| Skewness | Adjusted Fisher-Pearson standardized moment coefficient G1 (bias-corrected) | `x__skewness` |
| Kurtosis | Adjusted Fisher-Pearson standardized moment coefficient G2 (excess kurtosis, bias-corrected) | `x__kurtosis` |
| Variation coefficient | Population standard deviation divided by mean | `x__variation_coefficient` |
| Quantile | The `q`-th quantile with linear interpolation (default `q=0.5`) | `x__quantile__q_{q}` |
| Binned entropy | Entropy of the histogram of `x` over `max_bins` equidistant bins | `x__binned_entropy` |
| Benford correlation | Pearson correlation between the leading-digit distribution of `|x|` and Benford's law | `x__benford_correlation` |

#### Notes on specific features

- **Skewness** uses the bias-corrected formula
  `(n / ((n-1)(n-2))) * sum(((x - mean) / std_sample)^3)`,
  which matches `pandas.Series.skew()` and `tsfresh`'s reference
  implementation. Polars' default `skew()` is biased, so the
  implementation passes `bias=False` explicitly.
- **Kurtosis** returns the excess kurtosis (G2 minus 3), matching
  `pandas.Series.kurtosis()` and `tsfresh`. The implementation uses
  `kurtosis(fisher=True, bias=False)` to ensure both the Fisher
  adjustment and the bias-correction are applied.
- **Variation coefficient** is computed as
  `std(x, ddof=0) / mean(x)` and returns NaN whenever the mean is
  zero, non-finite, or undefined (empty group).
- **Quantile** uses `interpolation="linear"` (the default in
  `numpy.quantile` and `pandas`). Polars' default interpolation is
  `nearest`, which yields different values for non-exact quantiles
  and would diverge from tsfresh's reference output.
- **Binned entropy** computes `-sum(p * log(p))` where `p` is the
  normalized histogram of `x` over `max_bins` equidistant bins. The
  bins are computed via `numpy.histogram(x, bins=max_bins)`. Returns
  NaN for empty input or if the input contains NaN values.
- **Benford correlation** extracts the leading digit of each value
  using `numpy.format_float_scientific` (matching tsfresh's
  implementation), then correlates the resulting digit distribution
  with Benford's law
  `P(d) = log10(1 + 1/d)` for `d` in `1..9` via
  `numpy.corrcoef`. Returns NaN for empty input.

### Conventions

- Features that involve fixed-parameterized parameters (quantile,
  binned entropy) carry the parameter value in the output column name
  (e.g. `x__quantile__q_0.5`, `x__binned_entropy`).
- Empty groups are dropped by `group_by` and will not appear in the
  output.
- The two simple-moment features (skewness, kurtosis) are built on
  Polars expressions and stream with the rest of the aggregation. The
  four features that require per-group numpy computation
  (variation coefficient, binned entropy, Benford correlation) are
  implemented via `pl.col(...).map_batches(...).first()` and execute
  per group in the engine's Python-UDF path.

### Composing feature sets

Distribution features can be combined with the minimal feature set by
passing both to `extract_features`:

```python
from polars_tsfresh import extract_features
from polars_tsfresh.features import minimal_feature_set, distribution_feature_set

features = extract_features(
    df,
    column_id="id",
    column_sort="date",
    feature_sets=[minimal_feature_set, distribution_feature_set],
)
```

The order of `feature_sets` controls the order of the resulting
columns.


## Change and Rate Feature Set

The change and rate feature set describes first differences and consecutive-value
complexity. It produces five features per column.

| Feature | Definition | Output column |
| ------- | ---------- | ------------- |
| Mean absolute change | Mean absolute difference between consecutive values | `x__mean_abs_change` |
| Mean change | Mean difference between consecutive values | `x__mean_change` |
| Mean central second derivative | Mean central second-derivative approximation | `x__mean_second_derivative_central` |
| Absolute sum of changes | Sum of absolute consecutive differences | `x__absolute_sum_of_changes` |
| CID-CE | L2 norm of consecutive differences after z-normalization | `x__cid_ce` |

Use `cid_ce(col_name, normalize=False)` to compute CID-CE without z-normalization.
