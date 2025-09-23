"""Consolidated training script for phase_3 predictive models.

Usage examples:
  python3.12 train_models.py --target CO --horizon 1 --models xgboost,arima

This script loads the project's `fetch_and_preprocess_data()` function, builds
lagged features for forecasting if horizon>0, trains XGBoost and/or ARIMA, and
saves pickled models to the same directory.
"""
import argparse
import logging
import os
import pickle
from typing import List

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

try:
    from preprocess_data import fetch_and_preprocess_data
except Exception:
    from phase_3_predictive_analytics.preprocess_data import fetch_and_preprocess_data


def build_lagged_features(df: pd.DataFrame, feature_cols: List[str], lags: List[int]) -> pd.DataFrame:
    """Return DataFrame with lagged features for each col in feature_cols.
    lags: list of positive integers, e.g., [1,2,3]
    """
    out = df.copy()
    for lag in lags:
        shifted = df[feature_cols].shift(lag).add_suffix(f'_lag{lag}')
        out = pd.concat([out, shifted], axis=1)
    return out


def prepare_features(df: pd.DataFrame, target: str, horizon: int):
    """Prepare X,y for training.
    If horizon==0, predict current target from available predictors (not recommended).
    If horizon>0, predict target at t+horizon using lagged features up to t.
    """
    df = df.copy()
    if target not in df.columns:
        raise RuntimeError(f"Target column '{target}' not found in data")

    # basic temporal features if present
    temporal = []
    if isinstance(df.index, pd.DatetimeIndex):
        df['hour'] = df.index.hour
        df['day'] = df.index.day
        df['month'] = df.index.month
        df['season'] = ((df.index.month % 12) // 3) + 1
        temporal = ['hour', 'day', 'month', 'season']

    # choose base features: numeric sensor columns excluding the target
    numeric = df.select_dtypes(include=[float, int]).columns.tolist()
    # filter out stray generic 'Target' column and ensure the column has at least one non-NA
    feature_cols = [
        c for c in numeric
        if c != target and c.lower() != 'target' and df[c].notna().any()
    ]
    feature_cols = feature_cols + temporal

    # for forecasting, create lagged features; use lags 1..3 by default
    if horizon > 0:
        # include longer daily lag (24) in addition to short lags
        lags = [1, 2, 3, 24]
        df = build_lagged_features(df, feature_cols, lags)
        # target shifted backward by -horizon to align X (t) -> y (t+horizon)
        y = df[target].shift(-horizon)
        # drop rows with NaN in y or in lagged features
        keep_cols = []
        for col in feature_cols:
            for lag in lags:
                keep_cols.append(f"{col}_lag{lag}")
        X = df[keep_cols]
    else:
        y = df[target]
        X = df[feature_cols]

    X = X.astype(float)
    y = y.astype(float)

    # drop NA rows
    mask = ~(y.isna())
    mask &= ~X.isna().any(axis=1)
    X = X[mask]
    y = y[mask]

    return X, y


def train_xgboost(X, y, out_path):
    from xgboost import XGBRegressor
    m = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    m.fit(X, y)
    with open(out_path, 'wb') as f:
        pickle.dump(m, f)
    logging.info('Saved XGBoost model to %s', out_path)


def train_arima(series: pd.Series, out_path, order=(2, 1, 2)):
    from statsmodels.tsa.arima.model import ARIMA
    s = series.dropna()
    if len(s) < 10:
        raise RuntimeError('Not enough samples to train ARIMA')
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.date_range(end=pd.Timestamp.now(), periods=len(s), freq='h')
    model = ARIMA(s, order=order)
    res = model.fit()
    with open(out_path, 'wb') as f:
        pickle.dump(res, f)
    logging.info('Saved ARIMA model to %s', out_path)


def save_manifest(out_dir: str, info: dict):
    manifest_path = os.path.join(out_dir, 'train_manifest.json')
    with open(manifest_path, 'w') as f:
        import json

        json.dump(info, f, indent=2)
    logging.info('Wrote manifest to %s', manifest_path)


def main():
    parser = argparse.ArgumentParser(description='Train predictive models')
    parser.add_argument('--target', default='CO', help='Target column name')
    parser.add_argument('--horizon', type=int, default=1, help='Forecast horizon (0=current, 1=t+1)')
    parser.add_argument('--models', default='xgboost,arima', help='Comma list: xgboost,arima')
    parser.add_argument('--out-dir', default=os.path.dirname(__file__), help='Output directory for model files')
    args = parser.parse_args()

    logging.info('Loading and preprocessing data...')
    df = fetch_and_preprocess_data()

    X, y = prepare_features(df, args.target, args.horizon)
    logging.info('Prepared features X shape=%s, y length=%s', X.shape, len(y))

    models = [m.strip() for m in args.models.split(',') if m.strip()]
    produced = []

    base_name = args.target
    if args.horizon > 0:
        suffix = f'_tplus{args.horizon}'
    else:
        suffix = ''

    if 'xgboost' in models:
        xgb_path = os.path.join(args.out_dir, f'xgboost_{base_name}{suffix}.pkl')
        logging.info('Training XGBoost -> %s', xgb_path)
        train_xgboost(X.values, y.values, xgb_path)
        produced.append(xgb_path)

    if 'arima' in models:
        arima_path = os.path.join(args.out_dir, f'arima_{base_name}.pkl')
        logging.info('Training ARIMA -> %s', arima_path)
        # ARIMA trains on the full target series (not the X,y split)
        train_arima(df[args.target], arima_path)
        produced.append(arima_path)

    manifest = {
        'models': produced,
        'target': args.target,
        'horizon': args.horizon,
        'feature_columns': X.columns.tolist(),
    }
    save_manifest(args.out_dir, manifest)


if __name__ == '__main__':
    main()
