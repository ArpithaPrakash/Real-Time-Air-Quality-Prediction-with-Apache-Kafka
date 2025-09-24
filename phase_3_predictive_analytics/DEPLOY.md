Phase 3 — Deployment & Runbook

This document shows how to run the Phase 3 predictive artifacts in a staging or production-like environment, what the Kafka message payload should look like, and how to scrape Prometheus metrics.

1. Required artifacts

- Trained model files (XGBoost / ARIMA). Example file names produced by tuning/training:
  - `xgboost_{TARGET}_tplus{H}.pkl` (e.g. `xgboost_CO_tplus1_best.pkl`)
  - `arima_{TARGET}.pkl`
- (Optional but recommended) Feature schema & training stats JSON saved next to models: `feature_schema_{TARGET}_h{H}.json` containing:
  - `feature_columns`: ordered list of column names used as model inputs
  - `training_stats`: {"mean": {..}, "std": {..}} used for deterministic drift checks

2. How to run model inference (local)

- Dry-run (load models, start metrics server, no Kafka consumption):

```bash
python3 phase_3_predictive_analytics/model_inference.py --xgb phase_3_predictive_analytics/xgboost_CO_tplus1_best.pkl --arima phase_3_predictive_analytics/arima_CO.pkl --dry-run
```

- Full run (consume from Kafka):

```bash
python3 phase_3_predictive_analytics/model_inference.py --xgb /absolute/path/to/xgboost_CO_tplus1_best.pkl --arima /absolute/path/to/arima_CO.pkl
```

Notes: the script respects env vars `XGBOOST_MODEL_PATH` and `ARIMA_MODEL_PATH` if passed instead of CLI args.

3. Kafka payload (expected JSON schema)

The consumer expects a JSON object in each Kafka message. The exact fields required depend on the `feature_columns` the model was trained with. Example minimal payload (if you trained using the 8-feature ad-hoc vector present in `model_inference.py`):

```json
{
	"CO": 2.1,
	"NOx": 220.5,
	"NO2": 110.3,
	"Benzene": 9.8,
	"hour": 14,
	"day": 23,
	"month": 6,
	"season": 2
}
```

If the model expects many lagged and rolling features (as produced by `prepare_features()`), the payload must contain the named fields used as model inputs (for example: `CO_lag1`, `NOx_lag1`, `CO_roll_mean_24`, ...). To avoid ambiguity, we recommend saving `feature_schema_{TARGET}_h{H}.json` at training time and using it at inference to validate and map incoming payloads.

4. Prometheus metrics

`model_inference.py` starts a Prometheus metrics HTTP server on port 8001 by default. Metrics exposed include:

- `predictions_total` (Counter) — total number of predictions
- `prediction_errors_total` (Counter) — total number of prediction errors (exceptions, invalid inputs)
- `prediction_latency_seconds` (Summary) — latency of prediction calls
- `feature_drift_alert` (Gauge) — set to 1.0 when a simple z-score based drift is detected for incoming sample

Scrape config example for Prometheus (add to `scrape_configs`):

```yaml
- job_name: "air_quality_inference"
  static_configs:
    - targets: ["<INFERENCE_HOST>:8001"]
```

5. Monitoring & alerts

- Create alert for high error rate:
  - If `increase(prediction_errors_total[5m]) / increase(predictions_total[5m]) > 0.05` fire an alert.
- Create alert for drift:
  - If `feature_drift_alert == 1` trending for a configured number of samples, notify data-engineering.

6. Deployment notes

- Use absolute paths for model files in production containers, or set `XGBOOST_MODEL_PATH` / `ARIMA_MODEL_PATH` env vars.
- Ensure the consumer `group.id` is unique per environment.
- Tune `auto.offset.reset` to `latest` in production if you only want new messages; `earliest` is useful in staging for replay.
- Persist `feature_schema_{TARGET}_h{H}.json` at training time (recommended). If the schema is missing, inference falls back to a best-effort feature extraction — this is less deterministic.

7. Troubleshooting

- `Model file not found` warnings: verify CLI paths or env vars.
- `Could not compute training stats for drift detection`: either schema not present or preprocess re-run failed; drift detection will be disabled.
- Prometheus server fails to start: ensure `prometheus_client` package is installed and port 8001 is available.

8. Next improvements

- Save feature schema and training stats at train time and make inference load them and validate incoming messages.
- Add a small wrapper to convert raw Kafka messages (raw sensor stream) into the exact feature set required by the model so inference only accepts validated payloads.

---
