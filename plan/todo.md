# tsfresh implementation TODO

Source: [`plan/tsfresh.yaml`](tsfresh.yaml)

Status: **16/76 functions implemented**. A checked group means every function in group is implemented in `src/polars_tsfresh/features.py`.

## Feature groups

- [x] `basic_statistics` — 10/10
  - [x] `mean`
  - [x] `median`
  - [x] `variance`
  - [x] `standard_deviation`
  - [x] `length`
  - [x] `maximum`
  - [x] `minimum`
  - [x] `absolute_maximum`
  - [x] `root_mean_square`
  - [x] `sum_values`

- [x] `distribution_features` — 6/6
  - [x] `skewness`
  - [x] `kurtosis`
  - [x] `variation_coefficient`
  - [x] `quantile`
  - [x] `binned_entropy`
  - [x] `benford_correlation`

- [ ] `change_and_rate_features` — 0/5
  - [ ] `mean_abs_change`
  - [ ] `mean_change`
  - [ ] `mean_second_derivative_central`
  - [ ] `absolute_sum_of_changes`
  - [ ] `cid_ce`

- [ ] `position_and_extrema_features` — 0/13
  - [ ] `first_location_of_maximum`
  - [ ] `last_location_of_maximum`
  - [ ] `first_location_of_minimum`
  - [ ] `last_location_of_minimum`
  - [ ] `count_above_mean`
  - [ ] `count_below_mean`
  - [ ] `count_above`
  - [ ] `count_below`
  - [ ] `has_duplicate`
  - [ ] `has_duplicate_max`
  - [ ] `has_duplicate_min`
  - [ ] `longest_strike_above_mean`
  - [ ] `longest_strike_below_mean`

- [ ] `peak_valley_and_crossing_features` — 0/3
  - [ ] `number_peaks`
  - [ ] `number_cwt_peaks`
  - [ ] `number_crossing_m`

- [ ] `recurring_value_features` — 0/4
  - [ ] `percentage_of_reoccurring_values_to_all_values`
  - [ ] `percentage_of_reoccurring_datapoints_to_all_datapoints`
  - [ ] `sum_of_reoccurring_values`
  - [ ] `sum_of_reoccurring_data_points`

- [ ] `frequency_and_wavelet_features` — 0/4
  - [ ] `fft_coefficient`
  - [ ] `fft_aggregated`
  - [ ] `spkt_welch_density`
  - [ ] `cwt_coefficients`

- [ ] `autocorrelation_and_time_series_models` — 0/5
  - [ ] `autocorrelation`
  - [ ] `agg_autocorrelation`
  - [ ] `partial_autocorrelation`
  - [ ] `ar_coefficient`
  - [ ] `augmented_dickey_fuller`

- [ ] `trend_and_regression_features` — 0/3
  - [ ] `linear_trend`
  - [ ] `linear_trend_timewise`
  - [ ] `agg_linear_trend`

- [ ] `entropy_and_complexity_features` — 0/5
  - [ ] `sample_entropy`
  - [ ] `approximate_entropy`
  - [ ] `permutation_entropy`
  - [ ] `lempel_ziv_complexity`
  - [ ] `fourier_entropy`

- [ ] `nonlinearity_and_lag_features` — 0/3
  - [ ] `c3`
  - [ ] `time_reversal_asymmetry_statistic`
  - [ ] `symmetry_looking`

- [ ] `quantile_and_mass_distribution_features` — 0/2
  - [ ] `change_quantiles`
  - [ ] `index_mass_quantile`

- [ ] `specialized_physics_features` — 0/2
  - [ ] `friedrich_coefficients`
  - [ ] `max_langevin_fixed_point`

- [ ] `energy_features` — 0/3
  - [ ] `abs_energy`
  - [ ] `energy_ratio_by_chunks`
  - [ ] `variance_larger_than_standard_deviation`

- [ ] `advanced_matrix_and_similarity_features` — 0/2
  - [ ] `matrix_profile`
  - [ ] `query_similarity_count`

- [ ] `utility_features` — 0/6
  - [ ] `value_count`
  - [ ] `range_count`
  - [ ] `ratio_beyond_r_sigma`
  - [ ] `large_standard_deviation`
  - [ ] `mean_n_absolute_max`
  - [ ] `ratio_value_number_to_time_series_length`

## Update rules

- Check function item after implementing and testing matching function from `plan/tsfresh.yaml`.
- Check group item only when all functions in group are checked.
- Update group and total counts when status changes.
