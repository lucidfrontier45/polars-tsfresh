# AGENTS.md - Polars-TSFresh Development Guide

This document provides essential information for coding agents working on the polars-tsfresh project.

## Build, Lint, and Test Commands

### Full Quality Check
```bash
uv run poe check
```
Runs both linting (ruff) and type checking (pyrefly) in sequence.

### Individual Quality Checks
```bash
# Linting only
uv run poe ruff_check

# Type checking only
uv run poe pyrefly_check

# Fix linting issues automatically
uv run poe ruff_fix
```

### Testing
```bash
# Run full test suite with coverage
uv run poe test

# Run single test file
uv run pytest tests/data/test_minimal.py

# Run single test function
uv run pytest tests/data/test_minimal.py::test_minimal

# Run tests with verbose output
uv run pytest tests/ -v

# Run tests with coverage report only
uv run pytest tests/ --cov=polars_tsfresh --cov-report=term
```

### Formatting
```bash
# Format all Python files
uv run poe ruff_format

# Check formatting without changes
uv run poe ruff_check
```

## Code Style Guidelines

### Import Organization
- Standard library imports first
- Third-party imports second (polars, pathlib, etc.)
- Local imports last
- One blank line between import groups
- Import organization is enforced on save in VS Code

```python
import polars as pl
from pathlib import Path

from polars_tsfresh import extract_features
```

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

### Naming Conventions
- Functions: `snake_case` (e.g., `extract_features`, `minimal_feature_set`)
- Variables: `snake_case` (e.g., `feature_columns`, `grouped`)
- Constants: `UPPER_CASE` (none currently used)
- Class names: `PascalCase` (none currently used)

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

### File Organization
- Source code in `src/polars_tsfresh/`
- Main module functions in `__init__.py`
- Feature functions in separate modules (e.g., `features.py`)
- Tests in `tests/` with data in `tests/data/`
- Configuration files in project root

### Linting Rules (Ruff Configuration)
The project uses extensive Ruff linting rules:
- E: pycodestyle errors
- F: Pyflakes
- W: pycodestyle warnings
- I: isort import sorting
- B: flake8-bugbear
- RUF: Ruff-specific rules
- UP: pyupgrade
- N: pep8-naming
- SIM: flake8-simplify
- A: flake8-builtins
- S: flake8-bandit
- DTZ: flake8-datetimez
- PIE: flake8-pie
- PLE: Pylint errors

### Type Checking (Pyrefly)
- Python 3.12 target version
- Strict type checking enabled
- All functions should have complete type annotations

### VS Code Integration
- Format on save enabled
- Import organization on save enabled
- Ruff as default Python formatter
- Pyrefly language server (Pylance disabled)
- Pytest integration enabled
- Coverage gutters extension recommended

## Development Workflow

1. Make changes to source code
2. Run `uv run poe check` to verify quality
3. Run `uv run poe test` to ensure tests pass
4. Use `uv run poe format` if needed
5. Commit changes

## Project-Specific Notes

- This is a Polars-based reimplementation of tsfresh feature extraction
- Focus on performance and type safety
- Minimal feature set currently includes basic statistical measures
- Uses double underscore (`__`) separator in feature column names
- Designed for time series data grouped by ID columns</content>
<parameter name="filePath">AGENTS.md