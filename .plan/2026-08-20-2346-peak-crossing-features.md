# Peak, Valley, and Crossing Features (partial: pure subset)

## Goal

Implement `number_peaks` and `number_crossing_m` as pure Polars expressions in a new `peak_valley_and_crossing_features` module, with tests; `number_cwt_peaks` deliberately excluded.

## Background

Group is 0/3 in `plan/todo.md`. Semantics pinned to upstream tsfresh source:

- `number_peaks(x, n)`: count interior points strictly greater than all `n` neighbors each side. Edge elements never counted (upstream slices `x[n:-n]`); plateaus not counted (strict `>`); series shorter than `2n+1` → 0. Shift nulls at series edges reproduce the interior-only behavior for free.
- `number_crossing_m(x, m)`: count adjacent pairs where one value is strictly below `m` and the other strictly above. Touching `m` exactly is NOT a crossing. Equivalent closed form: `((x-m) * (x.shift(1)-m) < 0).sum()` — exact, unlike boolean XOR which miscounts touches.
- `number_cwt_peaks`: needs scipy `find_peaks_cwt` (Ricker CWT + ridge-line linking). No native Polars equivalent; not part of this task.

## Approach

1. Create `src/polars_tsfresh/features/peak_valley_and_crossing_features.py`:
   - `number_peaks(col_name, n=1)` — build `is_peak` by AND-ing `x > x.shift(j)` / `x > x.shift(-j)` for `j` in `1..n` (Python loop at expression-build time; `n` is static), then `.sum().cast(pl.Int64)`.
   - `number_crossing_m(col_name, m=0.0)` — sign-product form above, `.sum().cast(pl.Int64)`.
   - `peak_valley_and_crossing_feature_set(col_name)` — both with defaults.
2. Re-export module + functions + feature set in `src/polars_tsfresh/features/__init__.py` (imports + alphabetical `__all__`). No top-level `__init__.py` changes — existing groups only expose submodules there.
3. Add `tests/test_peak_valley.py`, synthetic hand-computed series (pattern of `test_change.py`):
   - peaks: support n=1 and n=2; plateau not counted; boundary peaks excluded; short series → 0
   - crossings: basic up/down counts; value exactly `m` on one side → not counted; non-zero `m`; constant series → 0
   - one grouped `extract_features` round-trip
4. Update `plan/todo.md`: header 34/76 → 36/76; group line `[ ]` with 2/3, `number_cwt_peaks` stays unchecked.
5. `uv run poe check`, `uv run poe test`.

## Trade-offs

- Pure expressions over `map_batches` — per AGENTS.md; both functions admit it.
- Sign-product over `(x>m) != (x.shift(1)>m)` — XOR form counts touch-on-`m` as crossing; wrong vs tsfresh.
- tsfresh not installed in env → no upstream-generated reference CSV this time; hand-computed expectations only (upstream semantics already derived from source).

## Open questions

None — defaults n=1, m=0 match tsfresh and existing repo convention (`count_above(t=0.0)`, `quantile(q=0.5)`).

## Next step

Write module + tests, wire exports, update todo, run check/test.
