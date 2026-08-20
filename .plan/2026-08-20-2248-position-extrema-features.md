# Position and Extrema Features

## Goal

Implement all 13 `position_and_extrema_features` as pure Polars expressions, expose them publicly, and verify them with focused tests.

## Background

Project feature groups return `list[pl.Expr]` for grouped aggregation. New implementation should favor short native expressions over exact tsfresh handling of every null/NaN/empty edge case.

## Approach

1. Add `src/polars_tsfresh/features/position_and_extrema_features.py`.
2. Implement location features with `arg_max()` / `arg_min()` and reversed expressions for last positions.
3. Implement count features with native comparisons, `mean()`, `sum()`, and length normalization.
4. Implement duplicate features with `n_unique()`, extrema comparisons, and counts.
5. Implement longest-strike features using Boolean runs and native Polars run-length operations.
6. Add `position_and_extrema_feature_set()` returning all 13 expressions.
7. Re-export module, functions, and feature set from:
   - `src/polars_tsfresh/features/__init__.py`
   - `src/polars_tsfresh/__init__.py` where module-level exposure applies
8. Add `tests/test_position_and_extrema.py` covering:
   - repeated extrema and tie positions
   - above/below mean counts
   - inclusive threshold behavior
   - general, extrema, and no-duplicate cases
   - consecutive strikes at start, middle, and end
   - grouped extraction through `extract_features`
9. Update `plan/todo.md` to mark 13/13 complete.
10. Run:
    - `uv run poe check`
    - `uv run poe test`

## Trade-offs

- Prefer concise Polars semantics over extensive compatibility branches.
- No NumPy, Python UDFs, or shared abstraction unless native run-length expression requires a tiny helper.
- Basic representative edge tests only; exhaustive tsfresh parity deferred.

## Open questions

None.

## Next step

Implement feature module and smallest focused test suite.
