# AGENTS.md
This document provides essential information for coding agents working on the polars-tsfresh project.

## Build, Lint, and Test Commands
This project is managed with `uv`. Always use `uv` to invoke Python and related tools.

### Full Quality Check
```bash
uv run poe check
```
Runs both linting (ruff) and type checking (pyrefly) in sequence.

### Testing

Invoke `pytest` with `uv`.

## Code Style Guidelines

### Function Signatures and Type Hints
- Use comprehensive type hints for all parameters and return values
- Use `pl.Expr` for Polars expressions
- Use `list[pl.Expr]` for collections of expressions

```python
def extract_features(df: pl.DataFrame, column_id: str, column_sort: str) -> pl.DataFrame:
    """Extract features from a Polars DataFrame using tsfresh-like functionality.

    Parameters:
    df (pl.DataFrame): Input Polars DataFrame.
    column_id (str): The name of the column containing the IDs.
    column_sort (str): The name of the column to sort by.

    Returns:
    pl.DataFrame: DataFrame with extracted features.
    """
```

### Documentation
- Use Google-style docstrings with `Args:` and `Returns:` sections
- Document all parameters with types and descriptions
- Document return values with types and descriptions
- Include brief function description

### Polars Patterns
- Use method chaining when possible
- Use `.alias()` for column naming with `column_name__feature_name` pattern
- Prefer expressions over direct column operations
- Use aggregation context for grouped operations

```python
# Good: Expression-based approach
return pl.col(col_name).mean().alias(f"{col_name}__mean")

# Good: Method chaining
grouped = df.sort(column_sort).group_by(column_id)
result = grouped.agg(feature_exprs)
```

### Error Handling
- Use descriptive assertion messages in tests
- Avoid bare exceptions - provide context
- Test floating-point comparisons with tolerance functions

```python
def float_close(a: float, b: float, tol=1e-5) -> bool:
    return abs(a - b) < tol


assert float_close(val, val_true), f"Feature {col} does not match: {val} != {val_true}"
```

### Testing Patterns
- Test files in `tests/` directory
- Use descriptive test function names (e.g., `test_minimal`)
- Load test data from `tests/data/` directory
- Use Path objects for file operations
- Compare floating-point values with tolerance
- Skip irrelevant columns in assertions

```python
def test_minimal():
    x_csv_path = Path(__file__).parent.parent / "data" / "sp500_raw.csv"
    y_csv_path = Path(__file__).parent.parent / "data" / "sp500_tsfresh_features.csv"

    df = pl.read_csv(x_csv_path).with_columns(pl.lit("sp500").alias("kind"))
    features = extract_features(df, column_id="kind", column_sort="date")
    features_true = pl.read_csv(y_csv_path)
    # ... assertions
```

## Development Workflow

1. Make changes to source code
2. Run `uv run poe check` to verify quality
3. Run `uv run poe test` to ensure tests pass

## Project-Specific Notes

- This is a Polars-based reimplementation of tsfresh feature extraction
- Focus on performance and type safety
- Minimal feature set currently includes basic statistical measures
- Uses double underscore (`__`) separator in feature column names
- Designed for time series data grouped by ID columns</content>
<parameter name="filePath">AGENTS.md
