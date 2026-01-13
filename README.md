# polars-tsfresh

A high-performance Polars-based reimplementation of [tsfresh](https://tsfresh.readthedocs.io/) for time series feature extraction.

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-green.svg)](https://github.com/lucidfrontier45/polars-tsfresh/actions)
[![Coverage](https://img.shields.io/badge/coverage-100%25-green.svg)](https://github.com/lucidfrontier45/polars-tsfresh)

`polars-tsfresh` extracts statistical features from time series data stored in Polars DataFrames. It's designed as a faster, more type-safe alternative to tsfresh, leveraging Polars' efficient columnar operations for better performance on grouped time series data.

## Installation

### With uv (Recommended)

```bash
uv add polars-tsfresh
```

### With pip

```bash
pip install polars-tsfresh
```

### Requirements

- Python 3.12+
- Polars >= 1.36.1

## Quick Start

```python
import polars as pl
from polars_tsfresh import extract_features

# Load your time series data
df = pl.read_csv("data.csv")
print(df.head())
# shape: (5, 5)
# ┌────────────┬─────────┬─────────┬─────────┬─────────┐
# │ date       ┆ open    ┆ high    ┆ low     ┆ close   │
# │ ---        ┆ ---     ┆ ---     ┆ ---     ┆ ---     │
# │ str        ┆ f64     ┆ f64     ┆ f64     ┆ f64     │
# ╞════════════╪═════════╪═════════╪═════════╪═════════╡
# │ 2025/12/03 ┆ 6815.29 ┆ 6862.42 ┆ 6810.43 ┆ 6849.72 │
# │ 2025/12/04 ┆ 6866.47 ┆ 6866.47 ┆ 6827.12 ┆ 6857.12 │
# │ 2025/12/05 ┆ 6866.32 ┆ 6895.78 ┆ 6858.29 ┆ 6870.40 │
# │ ...        ┆ ...     ┆ ...     ┆ ...     ┆ ...     │
# └────────────┴─────────┴─────────┴─────────┴─────────┘
```

Add an ID column to group by (here we're treating all data as one time series):

```python
df = df.with_columns(pl.lit("sp500").alias("kind"))
```

Extract features:

```python
features = extract_features(df, column_id="kind", column_sort="date")
print(features)
# shape: (1, 11)
# ┌────────────┬────────────┬────────────┬────────────┬────────────┬────────────┬────────────┬────────────┬────────────┬────────────┬────────────┐
# │ kind       ┆ close__sum ┆ close__med ┆ close__mea ┆ close__len ┆ close__std ┆ close__var ┆ close__rms ┆ close__max ┆ close__abs ┆ close__min │
# │ ---        ┆ _values    ┆ ian        ┆ n          ┆ gth        ┆            ┆            ┆            ┆            ┆ _maximum   ┆            │
# │ str        ┆ ---        ┆ ---        ┆ ---        ┆ ---        ┆ ---        ┆ ---        ┆ ---        ┆ ---        ┆ ---        ┆ ---        │
# │ f64        ┆ f64        ┆ f64        ┆ f64        ┆ f64        ┆ f64        ┆ f64        ┆ f64        ┆ f64        ┆ f64        ┆ f64        │
# ╞════════════╪════════════╪════════════╪════════════╪════════════╪════════════╪════════════╪════════════╪════════════╪════════════╪════════════╡
# │ sp500      ┆ 137124.56  ┆ 6853.42    ┆ 6856.228   ┆ 20.0       ┆ 51.503     ┆ 2652.59    ┆ 6856.421   ┆ 6932.05    ┆ 6932.05    ┆ 6721.43    │
# └────────────┴────────────┴────────────┴────────────┴────────────┴────────────┴────────────┴────────────┴────────────┴────────────┴────────────┘
```

The output shows features extracted for each group (in this case, one group "sp500") with features named using the `column__feature` convention from tsfresh.

## Features

### Current Minimal Feature Set

polars-tsfresh currently implements the "minimal" feature set from tsfresh, providing essential statistical measures:

| Feature                | Description                 | tsfresh Equivalent                              |
| ---------------------- | --------------------------- | ----------------------------------------------- |
| **mean**               | Arithmetic mean of values   | `tsfresh.feature_extraction.mean`               |
| **median**             | Median value                | `tsfresh.feature_extraction.median`             |
| **variance**           | Variance (ddof=0)           | `tsfresh.feature_extraction.variance`           |
| **standard_deviation** | Standard deviation (ddof=0) | `tsfresh.feature_extraction.standard_deviation` |
| **length**             | Number of data points       | `tsfresh.feature_extraction.length`             |
| **maximum**            | Maximum value               | `tsfresh.feature_extraction.maximum`            |
| **minimum**            | Minimum value               | `tsfresh.feature_extraction.minimum`            |
| **absolute_maximum**   | Maximum absolute value      | `tsfresh.feature_extraction.absolute_maximum`   |
| **root_mean_square**   | Root mean square (RMS)      | `tsfresh.feature_extraction.root_mean_square`   |
| **sum_values**         | Sum of all values           | `tsfresh.feature_extraction.sum_values`         |

### Performance Benefits

- **Polars Expressions**: Uses Polars' vectorized operations for optimal performance
- **Grouped Operations**: Efficiently handles multiple time series in a single DataFrame
- **Type Safety**: Full type hints and static analysis support
- **Memory Efficient**: Columnar operations reduce memory overhead

### Roadmap

The project plans to implement comprehensive feature sets including:

- **Distribution Features**: skewness, kurtosis, quantiles, entropy
- **Change & Rate Features**: derivatives, trend analysis
- **Position & Extrema Features**: peak detection, location analysis
- **Frequency Features**: FFT coefficients, spectral analysis
- **Autocorrelation Features**: time series modeling
- **Complexity Features**: entropy measures, complexity estimates

See `plan/tsfresh.yaml` for the complete feature roadmap.

## API Reference

### `extract_features(df, column_id, column_sort)`

Extract features from a time series DataFrame.

**Parameters:**
- `df` (pl.DataFrame): Input DataFrame containing time series data
- `column_id` (str): Name of the column containing group IDs
- `column_sort` (str): Name of the column to sort by (typically time/date)

**Returns:**
- `pl.DataFrame`: DataFrame with extracted features, one row per group

**Example:**
```python
features = extract_features(
    df=my_dataframe,
    column_id="stock_symbol",
    column_sort="timestamp"
)
```

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/lucidfrontier45/polars-tsfresh.git
cd polars-tsfresh

# Install with uv (includes dev dependencies)
uv sync
```

### Testing

```bash
# Run all tests
uv run poe test

# Run specific test
uv run pytest tests/data/test_minimal.py::test_minimal

# Run with coverage
uv run pytest tests/ --cov=polars_tsfresh --cov-report=html
```

### Code Quality

```bash
# Full quality check (linting + type checking)
uv run poe check

# Format code
uv run poe format

# Individual checks
uv run poe ruff_check    # Linting
uv run poe pyrefly_check # Type checking
```

### Project Structure

```
polars-tsfresh/
├── src/polars_tsfresh/
│   ├── __init__.py      # Main API
│   └── features.py      # Feature implementations
├── tests/
│   └── data/
│       ├── test_minimal.py    # Test suite
│       ├── sp500_raw.csv      # Test data
│       └── sp500_tsfresh_features.csv  # Expected results
├── plan/
│   ├── tsfresh.yaml     # Feature definitions
│   └── dev_plan.md      # Development roadmap
└── pyproject.toml       # Project configuration
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all checks pass: `uv run poe check && uv run poe test`
5. Submit a pull request

See `AGENTS.md` for detailed development guidelines and coding standards.

## Related Projects

- **[tsfresh](https://tsfresh.readthedocs.io/)**: The original Python package for time series feature extraction
- **[Polars](https://pola.rs/)**: Fast DataFrame library powering this implementation

## License

Licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Authors

- **杜世橋 Du Shiqiao** - *Initial work* - [lucidfrontier.45@gmail.com](mailto:lucidfrontier.45@gmail.com)