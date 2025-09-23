import argparse
import json
import os
import logging
from confluent_kafka import Consumer
import pickle
import numpy as np
from typing import Optional

# Prometheus and structured logging
from prometheus_client import start_http_server, Counter, Summary, Gauge
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


# Metrics
PRED_COUNT = Counter('predictions_total', 'Total number of predictions')
PRED_ERRORS = Counter('prediction_errors_total', 'Total number of prediction errors')
PRED_LATENCY = Summary('prediction_latency_seconds', 'Prediction latency in seconds')
DRIFT_ALERT = Gauge('feature_drift_alert', 'Simple drift alert (1 if drift detected, 0 otherwise)')


def load_model(model_path):
    """
    Load a trained model from a pickle file.
    """
    if not os.path.exists(model_path):
        logging.warning("Model file not found: %s", model_path)
        return None
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logging.info("Loaded model from %s", model_path)
        return model
    except Exception as e:
        logging.exception("Failed to load model %s: %s", model_path, e)
        return None


def load_training_stats(manifest_path=None):
    """Try to load training summary statistics for simple drift detection.
    If a train_manifest.json exists with paths, attempt to compute or load training means/stds.
    """
    stats = {}
    try:
        # attempt to compute from preprocess_data if available
        from phase_3_predictive_analytics.preprocess_data import fetch_and_preprocess_data
        df = fetch_and_preprocess_data()
        num = df.select_dtypes(include=[float, int])
        stats['mean'] = num.mean().to_dict()
        stats['std'] = num.std().to_dict()
    except Exception:
        logging.info('Could not compute training stats for drift detection; drift disabled')
    return stats


def check_drift(stats, data_point, threshold=3.0):
    """Simple z-score based drift: if any feature z-score > threshold, signal drift.
    data_point: dict-like of feature values
    """
    if not stats:
        return False
    means = stats.get('mean', {})
    stds = stats.get('std', {})
    for k, v in data_point.items():
        if k in means and k in stds:
            try:
                val = float(v)
            except Exception:
                continue
            std = stds.get(k, 0.0) or 0.0
            if std <= 0:
                continue
            z = abs((val - means[k]) / std)
            if z > threshold:
                return True
    return False

def predict_xgboost(model, data):
    """
    Predict using the XGBoost model.
    """
    # The data passed is expected to be a dictionary, we extract features to make a prediction
    # Safely extract features with defaults if missing
    try:
        feat_list = [
            float(data.get('CO', 0.0)),
            float(data.get('NOx', 0.0)),
            float(data.get('NO2', 0.0)),
            float(data.get('Benzene', 0.0)),
            float(data.get('hour', 0.0)),
            float(data.get('day', 0.0)),
            float(data.get('month', 0.0)),
            float(data.get('season', 0.0)),
        ]
    except Exception:
        # fallback: zeros
        feat_list = [0.0] * 8

    features = np.array(feat_list).reshape(1, -1)
    if model is None:
        logging.debug("XGBoost model not available, returning NaN")
        return float('nan')
    try:
        return model.predict(features)[0]
    except Exception:
        logging.exception("XGBoost prediction failed for features: %s", features)
        return float('nan')

def predict_arima(model, data):
    """
    Predict using the ARIMA model.
    The ARIMA model expects only the previous value of the target to predict the next.
    """
    if model is None:
        logging.debug("ARIMA model not available, returning NaN")
        return float('nan')
    try:
        # Many statsmodels ARIMA results support forecast(steps=1) or get_forecast
        if hasattr(model, 'forecast'):
            return float(model.forecast(steps=1)[0])
        elif hasattr(model, 'get_forecast'):
            return float(model.get_forecast(steps=1).predicted_mean.iloc[0])
        else:
            logging.warning("ARIMA model does not support expected forecast API")
            return float('nan')
    except Exception:
        logging.exception("ARIMA prediction failed")
        return float('nan')

def main():
    parser = argparse.ArgumentParser(description='Consume air quality messages and run model inference')
    parser.add_argument('--xgb', help='Path to XGBoost model pickle')
    parser.add_argument('--arima', help='Path to ARIMA model pickle')
    parser.add_argument('--target', default=os.environ.get('MODEL_TARGET', 'CO'), help='Target name used for model filenames (default: CO)')
    parser.add_argument('--horizon', type=int, default=int(os.environ.get('MODEL_HORIZON', '1')), help='Forecast horizon used in model filename suffix (default: 1)')
    parser.add_argument('--dry-run', action='store_true', help='Only load models and exit (no Kafka consumption)')
    args = parser.parse_args()

    # Determine model paths with this precedence:
    # CLI explicit path -> ENV var -> constructed default
    def resolve_path(cli_path, env_var, default):
        if cli_path:
            return cli_path
        env_path = os.environ.get(env_var)
        if env_path:
            return env_path
        return default

    # build default filenames
    target = args.target
    horizon = args.horizon
    suffix = f'_tplus{horizon}' if horizon and horizon > 0 else ''
    default_xgb = f'xgboost_{target}{suffix}.pkl'
    default_arima = f'arima_{target}.pkl'

    xgb_path = resolve_path(args.xgb, 'XGBOOST_MODEL_PATH', default_xgb)
    arima_path = resolve_path(args.arima, 'ARIMA_MODEL_PATH', default_arima)

    # Start metrics server
    try:
        start_http_server(8001)
        logging.info('Prometheus metrics available on :8001')
    except Exception:
        logging.warning('Failed to start Prometheus metrics server; prometheus_client may be missing')

    # Load the trained models
    xgboost_model = load_model(xgb_path)
    arima_model = load_model(arima_path)

    # Load training stats for drift detection
    stats = load_training_stats()

    if args.dry_run:
        logging.info('Dry run complete; exiting after loading models.')
        return

    # Kafka Consumer Configuration
    consumer = Consumer({
        'bootstrap.servers': 'localhost:29092',
        'group.id': 'air_quality_group',
        'auto.offset.reset': 'earliest'
    })

    # Subscribe to the Kafka topic for real-time data
    consumer.subscribe(['air_quality_data'])

    try:
        while True:
            # Poll for a message from Kafka
            msg = consumer.poll(timeout=1.0)

            if msg is None:
                continue  # No message received, continue polling

            if msg.error():
                logging.error("Consumer error: %s", msg.error())
                continue

            # Parse the received message from Kafka (JSON)
            try:
                data = json.loads(msg.value().decode('utf-8'))
            except Exception:
                logging.exception("Failed to decode message payload")
                continue

            # Make predictions using both XGBoost and ARIMA models
            start = time.time()
            xgboost_pred = predict_xgboost(xgboost_model, data)
            arima_pred = predict_arima(arima_model, data)
            latency = time.time() - start
            PRED_LATENCY.observe(latency)
            PRED_COUNT.inc()

            # Check drift and set a metric accordingly
            drift = check_drift(stats, data)
            DRIFT_ALERT.set(1 if drift else 0)

            # Structured JSON logging for downstream parsing
            rec = {
                'topic': msg.topic(),
                'partition': msg.partition(),
                'offset': msg.offset(),
                'data': data,
                'predictions': {'xgboost': xgboost_pred, 'arima': arima_pred},
                'latency_sec': latency,
                'drift': bool(drift),
            }
            logging.info(json.dumps(rec, default=str))

    except KeyboardInterrupt:
        print("Consumer interrupted")
    finally:
        consumer.close()

if __name__ == "__main__":
    main()
