# Recurring Value Features

## Goal

Implement all 4 `recurring_value_features` as pure Polars expressions in a new module, with tests; update `plan/todo.md` 36/76 → 40/76.

## Background

Group is 0/4 in `plan/todo.md`. Semantics pinned to upstream tsfresh 0.21.2 source (downloaded and verified):

- `percentage_of_reoccurring_values_to_all_values` — #distinct recurring values / #distinct values; NaN for empty series. Normalized to uniques, NOT len(x).
- `percentage_of_reoccurring_datapoints_to_all_datapoints` — #datapoints whose value recurs / len(x); NaN for empty series.
- `sum_of_reoccurring_values` — recurring values summed once each; 0 if none. `[2,2,2,2,1]` → 2.
- `sum_of_reoccurring_data_points` — all recurring occurrences summed. `[2,2,2,2,1]` → 8.

Key Polars primitives (verified against polars 1.43.2 in both select and `group_by().agg()` contexts):

- `x.is_duplicated()` — elementwise "this value occurs >1×".
- `x.is_duplicated() & x.is_first_distinct()` — one flag per distinct recurring value.
- `x.filter(pred).sum()` works inside agg context.

Verified exact vs upstream: `[2,2,2,2,1]` → 0.5, 0.8, 2.0, 8.0; all-unique series → all 0.0; empty series → NaN, NaN, 0.0, 0.0.

All 4 functions admit pure expressions — nothing excluded from the group.

## Approach

1. Create `src/polars_tsfresh/features/recurring_value_features.py`:
   - 4 feature functions + `recurring_value_feature_set(col_name)` returning all 4.
   - `x = pl.col(col_name).cast(pl.Float64)`:
     - `percentage_of_reoccurring_values_to_all_values`: `(x.is_duplicated() & x.is_first_distinct()).sum() / x.n_unique()`
     - `percentage_of_reoccurring_datapoints_to_all_datapoints`: `x.is_duplicated().sum() / pl.len()`
     - `sum_of_reoccurring_values`: `x.filter(x.is_duplicated() & x.is_first_distinct()).sum()`
     - `sum_of_reoccurring_data_points`: `x.filter(x.is_duplicated()).sum()`
   - Percentages return fractions in [0, 1] (upstream returns fraction despite "percentage" name), cast Float64. Sums cast Float64.
   - Column names exact tsfresh naming: `{col}__percentage_of_reoccurring_values_to_all_values` etc.
2. Re-export module + 4 functions + feature set in `src/polars_tsfresh/features/__init__.py` (imports + alphabetical `__all__`). No root `__init__.py` changes — existing groups only expose submodules there.
3. Add `tests/test_recurring.py`, synthetic hand-computed series (pattern of `tests/test_peak_valley.py`):
   - tsfresh docstring example `[2,2,2,2,1]`
   - all-unique series → 0.0 for all four
   - multiple distinct recurring values
   - int column dtype
   - one grouped `extract_features` round-trip
4. Update `plan/todo.md`: header 36/76 → 40/76; group line checked `[x]`, all 4 items checked.
5. Update `README.md` feature tables and `doc/feature-sets.md` with the new group (user decision: update docs this time).
6. `uv run poe check`, `uv run poe test`.

## Trade-offs

- `is_first_distinct & is_duplicated` over `unique().filter()` — `unique()` in agg context returns a list column; the elementwise formulation stays a pure expression. Already verified.
- Fraction (not ×100) for both percentage features — exact tsfresh parity.
- tsfresh not installed in env → hand-computed expectations only (same precedent as peak/crossing group).
- NaN/null input handling not pinned to upstream — test data non-null, consistent with other modules.

## Open questions

None — resolved: fractions (1), docs updated (2), Float64 casts (3).

## Next step

Write module + tests, wire exports, update todo + README + doc, run check/test.
