# Data Quality & Preprocessing

This document explains the data quality decisions used in Phase 1 (Air Quality streaming)

1. Sentinel handling

- The UCI Air Quality dataset uses `-200` to represent missing/sensor-failure values.
- We replace `-200` with a pandas `NA` value early in preprocessing: `data.replace(-200, pd.NA)`.

2. Imputation strategy

- By default the pipeline performs simple numeric imputation using column mean after resampling. This is a low-risk baseline suitable for offline training and simple simulation.
- For time-series-aware imputation, consider `ffill`/`bfill` or interpolation that respects seasonality; this can be added as an option to `fetch_and_preprocess_data()`.

3. Outlier & sensor-range validation

- We clip sensor values to reasonable environmental ranges in `validate_sensor_ranges()` (e.g., CO: 0.5-5.0 mg/m3, NOx: 5-100 ppb).
- These are pragmatic values used to avoid unrealistic values in simulation; adjust thresholds if you have domain-specific requirements.

4. Placement of preprocessing

- Preprocessing is implemented in a shared module `phase_1_streaming_infrastructure/preprocess_data.py`. The producer imports and uses it so messages produced to Kafka are cleaned and normalized. Consumers also import the same module for consistent validation.
- This choice favors reproducibility and ensures both producer and consumer share identical transforms.

5. QC reporting

- A QC reporter `export_qc_report(data, out_path)` is available to export CSV summaries with missingness and basic stats. Producers can call with `--qc` to write an on-disk QC report.

6. Monitoring & alerts

- Metrics (processing / failures) are exposed by the consumer on a Prometheus endpoint (default `:8000`).
- Add alerting rules in your monitoring stack to notify when `air_quality_messages_failed_total` or missingness rates exceed thresholds.
