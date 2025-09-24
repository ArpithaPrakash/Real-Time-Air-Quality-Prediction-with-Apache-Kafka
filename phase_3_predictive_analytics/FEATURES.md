# FEATURES

This document describes the engineered features used by the Phase 3 predictive pipeline and how to regenerate per-feature statistics.

## Overview

The preprocessing pipeline (`fetch_and_preprocess_data` in `preprocess_data.py`) performs the following transformations in order:

- Missing value handling: replaces the dataset sentinel `-200` with NaN, coerces columns to numeric, and fills numeric missing values with the column mean.
- Temporal features: ensures a DatetimeIndex (synthetic hourly index if none available) and adds `hour`, `day`, `month`, and `season` columns.
- Cyclical encodings: transforms periodic time features to sine/cosine pairs to preserve continuity:
  - `hour_sin`, `hour_cos` (24-hour cycle)
  - `dow_sin`, `dow_cos` (7-day cycle from index.dayofweek)
  - `month_sin`, `month_cos` (12-month cycle)

## Rolling / statistical features

For pollutant columns (by default: `CO`, `NOx`, `NO2`, `Benzene`) the pipeline computes rolling-window mean and standard deviation for windows: 3h, 6h, 12h, 24h.
These features are named as:

- `<COL>_roll_mean_<W>` and `<COL>_roll_std_<W>` (e.g. `CO_roll_mean_24`, `NO2_roll_std_3`).

Rationale: short windows (3h) capture transient spikes; medium (6h/12h) capture short-lived events; long (24h) capture diurnal trends.

## Lag features

Lagged values are added for key columns (if present): `CO`, `NOx`, `NO2`, `Benzene`, and `Target` using lags (1, 2, 3, 24) hours.
Naming convention: `<COL>_lag<k>` (e.g. `CO_lag24`).

Rationale: previous values and the previous-day value (lag-24) are often strong predictors for short horizon forecasting.

## Current feature list (as of manifest)

See `train_manifest.json` -> `feature_columns` for an exact list. The manifest also contains per-feature statistics (mean/std/min/max/non_null_count, dtype) computed on the training data used to produce the current models.

## Regenerating feature stats

To recompute per-feature statistics (mean/std/min/max/non-null counts) run the helper script included in the repository root:

```bash
python3 .tmp_compute_feature_stats.py
```

This script imports the pipeline and prints a JSON blob with keys `feature_columns`, `feature_stats`, and `rows`.

## Notes and next steps

Implemented advanced features

- Trend slopes: short-term linear trend estimates over rolling windows are implemented as `<col>_slope_<w>` computed via a rolling linear fit (`add_trend_slope_features` in `preprocess_data.py`).
- Exponential moving averages (EMA): `<col>_ema_<w>` features added via `add_ema_features`.
- Lag-interaction features: pairwise products of lagged variables are computed as `<colA>_lagX__<colB>_lagY` via `add_lag_interaction_features`.
- Block-bootstrap support: time-series-aware CI estimation via `block_bootstrap_ci()` is available in `evaluate.py` and can be enabled with `--use-block-bootstrap` and `--block-size`.

These features are already used by downstream training/tuning/evaluation scripts when available. See `train_manifest.json` for the exact feature list that was used for the most-recent model artifacts.

Next steps (optional hardening)

- Persist the exact `feature_columns` and per-feature training mean/std to disk when saving a trained model so inference can load the schema and compute drift deterministically (`feature_schema_{target}_h{h}.json`).
- Update the inference script to validate incoming message payloads against the saved schema and fail-fast on missing features. This is recommended but optional if you plan to use the current dry-run/drift approach.
