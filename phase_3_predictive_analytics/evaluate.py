#!/usr/bin/env python3
"""Evaluation utilities for Phase 3 models.

Performs chronological train/test split, trains an XGBoost model on the train portion,
evaluates on test using MAE and RMSE, compares against a naive baseline (previous value),
and computes bootstrap confidence intervals for metric differences.

Usage:
  python evaluate.py --target CO --horizon 1

"""
import argparse
import json
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from phase_3_predictive_analytics.preprocess_data import fetch_and_preprocess_data
    from phase_3_predictive_analytics.train_models import prepare_features
except Exception:
    # allow running from project root
    from preprocess_data import fetch_and_preprocess_data
    from train_models import prepare_features


def chronological_split(X, y, test_size=0.2):
    n = len(X)
    if n == 0:
        raise RuntimeError('Empty feature set')
    split = int(n * (1 - test_size))
    X_train = X.iloc[:split]
    X_test = X.iloc[split:]
    y_train = y.iloc[:split]
    y_test = y.iloc[split:]
    return X_train, X_test, y_train, y_test


def baseline_previous(X_test, y_test, target):
    # Try to use a lagged column if present
    cand = f'{target}_lag1'
    if cand in X_test.columns:
        baseline = X_test[cand].astype(float).values
        # if NaN for the first row, fill with first y_test value or 0
        baseline = np.where(np.isnan(baseline), np.nan, baseline)
        return baseline

    # fallback: previous observed value in y_test (shifted)
    b = y_test.shift(1).to_numpy()
    # first element of b may be NaN — fill with last value from prior (use y_test.iloc[0] as fallback)
    if np.isnan(b[0]):
        b[0] = y_test.iloc[0]
    return b


def bootstrap_ci(diff_values, n_boot=1000, alpha=0.05, seed=42):
    rng = np.random.RandomState(seed)
    n = len(diff_values)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boots[i] = diff_values[idx].mean()
    low = np.percentile(boots, 100 * (alpha / 2))
    high = np.percentile(boots, 100 * (1 - alpha / 2))
    return low, high


def block_bootstrap_ci(diff_values, block_size=24, n_boot=1000, alpha=0.05, seed=42):
    """Block bootstrap for time-series-aware confidence intervals.

    Resamples contiguous blocks of length `block_size` to preserve temporal dependence.
    """
    rng = np.random.RandomState(seed)
    n = len(diff_values)
    if block_size < 1:
        raise ValueError('block_size must be >= 1')
    # number of blocks needed to reach length n (with replacement)
    nb = int(np.ceil(n / block_size))
    boots = np.empty(n_boot)
    for i in range(n_boot):
        sample = []
        for _ in range(nb):
            start = rng.randint(0, max(1, n - block_size + 1))
            sample.append(diff_values[start:start + block_size])
        sample = np.concatenate(sample)[:n]
        boots[i] = sample.mean()
    low = np.percentile(boots, 100 * (alpha / 2))
    high = np.percentile(boots, 100 * (1 - alpha / 2))
    return low, high


def evaluate(args):
    print('Loading and preprocessing data...')
    df = fetch_and_preprocess_data()
    print('Preparing features...')
    X, y = prepare_features(df, args.target, args.horizon)
    print('Total feature rows:', len(X))

    X_train, X_test, y_train, y_test = chronological_split(X, y, test_size=args.test_size)
    print('Train rows:', len(X_train), 'Test rows:', len(X_test))

    # Train a simple XGBoost model
    from xgboost import XGBRegressor

    model = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    model.fit(X_train.values, y_train.values)
    y_pred = model.predict(X_test.values)

    # Baseline predictions
    baseline_pred = baseline_previous(X_test, y_test, args.target)

    # Metrics
    mae_model = mean_absolute_error(y_test.values, y_pred)
    rmse_model = mean_squared_error(y_test.values, y_pred, squared=False)

    mae_baseline = mean_absolute_error(y_test.values, baseline_pred)
    rmse_baseline = mean_squared_error(y_test.values, baseline_pred, squared=False)

    print('\nResults:')
    print(f'XGBoost MAE: {mae_model:.4f}, RMSE: {rmse_model:.4f}')
    print(f'Baseline MAE: {mae_baseline:.4f}, RMSE: {rmse_baseline:.4f}')

    # Paired differences (model - baseline) for MAE (per-sample absolute errors)
    abs_model = np.abs(y_test.values - y_pred)
    abs_base = np.abs(y_test.values - baseline_pred)
    diff = abs_model - abs_base

    if getattr(args, 'use_block_bootstrap', False):
        low, high = block_bootstrap_ci(diff, block_size=args.block_size, n_boot=args.bootstrap, alpha=args.alpha, seed=args.seed)
    else:
        low, high = bootstrap_ci(diff, n_boot=args.bootstrap, alpha=args.alpha, seed=args.seed)
    mean_diff = diff.mean()
    print(f'Bootstrap mean difference (MAE_model - MAE_baseline): {mean_diff:.6f}')
    print(f'{100*(1-args.alpha):.1f}% CI for difference: [{low:.6f}, {high:.6f}]')

    # Save report
    report = {
        'target': args.target,
        'horizon': args.horizon,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'xgboost': {'mae': float(mae_model), 'rmse': float(rmse_model)},
        'baseline': {'mae': float(mae_baseline), 'rmse': float(rmse_baseline)},
        'mae_diff_mean': float(mean_diff),
        'mae_diff_ci': [float(low), float(high)],
    }

    out_path = Path(args.outdir) / f'eval_report_{args.target}_h{args.horizon}.json'
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    print('Wrote evaluation report to', out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', default='CO')
    parser.add_argument('--horizon', type=int, default=1)
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--bootstrap', type=int, default=200, help='Bootstrap iterations (for CI)')
    parser.add_argument('--alpha', type=float, default=0.05, help='Significance level for CI')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--outdir', default=os.path.dirname(__file__))
    parser.add_argument('--use-block-bootstrap', action='store_true', help='Use block bootstrap for CI')
    parser.add_argument('--block-size', type=int, default=24, help='Block size (in samples) for block bootstrap')
    args = parser.parse_args()

    evaluate(args)


if __name__ == '__main__':
    main()
