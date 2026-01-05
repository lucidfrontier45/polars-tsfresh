# Project Overview

## Purpose
This is a Python project called "polars-tsfresh" that aims to implement time series feature extraction functions using Polars expressions. It's based on the tsfresh library concept but implemented with Polars for better performance.

## Tech Stack
- **Python**: 3.13+
- **Core Dependencies**: Polars (>=1.36.1)
- **Development Tools**: 
  - uv for package management
  - ruff for linting and formatting
  - pyrefly for type checking
  - pytest for testing with coverage

## Project Structure
```
polars-tsfresh/
├── src/
│   └── polars_tsfresh/
│       ├── __init__.py (empty)
│       └── features.py (feature extraction functions)
├── plan/
│   ├── dev_plan.md
│   ├── tsfresh.yaml (feature definitions)
│   └── tsfresh.py
├── tests/ (empty)
├── pyproject.toml
└── README.md
```

## Current State
- Basic project setup with Polars dependency
- Feature definitions defined in plan/tsfresh.yaml
- One minimal feature (mean) already implemented in features.py
- Need to implement remaining minimal class features

## Code Style and Conventions
- Target Python 3.13
- Type hints required
- Docstrings follow standard format
- Functions take column name (str) and return pl.Expr
- Naming: snake_case for functions
- Ruff configuration: E, F, W, I, B, RUF, UP, N, SIM, A, S, DTZ, PIE, PLE