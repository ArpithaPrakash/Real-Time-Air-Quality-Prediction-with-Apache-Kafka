#!/usr/bin/env python3
"""Hyperparameter tuning with rolling-origin CV and final evaluation.

Small grid search over XGBoost hyperparameters using sklearn TimeSeriesSplit.
Saves best model and evaluation report.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from phase_3_predictive_analytics.preprocess_data import fetch_and_preprocess_data
    from phase_3_predictive_analytics.train_models import prepare_features
except Exception:
    from preprocess_data import fetch_and_preprocess_data
    from train_models import prepare_features


def chronological_holdout_split(X, y, holdout_size=0.2):
    n = len(X)
    split = int(n * (1 - holdout_size))
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


def tune_and_evaluate(args):
    print('Loading data...')
    df = fetch_and_preprocess_data()
    X, y = prepare_features(df, args.target, args.horizon)
    X_train, X_hold, y_train, y_hold = chronological_holdout_split(X, y, holdout_size=args.holdout)

    print('Tuning on train (rolling-origin CV)')
    tscv = TimeSeriesSplit(n_splits=args.n_splits)

    # expanded grid (kept moderate for runtime) - add subsample and colsample
    grid = {
        'max_depth': [3, 6, 9],
        'learning_rate': [0.2, 0.1, 0.05, 0.01],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    import itertools
    best = None
    best_score = float('inf')
    for md, lr, ss, cs in itertools.product(grid['max_depth'], grid['learning_rate'], grid['subsample'], grid['colsample_bytree']):
        scores = []
        for train_idx, val_idx in tscv.split(X_train):
            Xtr, Xval = X_train.iloc[train_idx], X_train.iloc[val_idx]
            ytr, yval = y_train.iloc[train_idx], y_train.iloc[val_idx]
            from xgboost import XGBRegressor
            m = XGBRegressor(n_estimators=150, max_depth=md, learning_rate=lr, subsample=ss, colsample_bytree=cs, random_state=42, verbosity=0)
            m.fit(Xtr.values, ytr.values)
            yp = m.predict(Xval.values)
            scores.append(mean_absolute_error(yval.values, yp))
        avg = float(np.mean(scores))
        print(f'grid max_depth={md} lr={lr} subsample={ss} colsample={cs} mean MAE={avg:.5f}')
        if avg < best_score:
            best_score = avg
            best = {'max_depth': md, 'learning_rate': lr, 'subsample': ss, 'colsample_bytree': cs}

    print('Best params:', best, 'cv_mae=', best_score)

    # Retrain on full train and evaluate on holdout
    from xgboost import XGBRegressor
    best_model = XGBRegressor(n_estimators=300, max_depth=best['max_depth'], learning_rate=best['learning_rate'], subsample=best.get('subsample', 1.0), colsample_bytree=best.get('colsample_bytree', 1.0), random_state=42, verbosity=0)
    best_model.fit(X_train.values, y_train.values)
    y_pred = best_model.predict(X_hold.values)

    mae = mean_absolute_error(y_hold.values, y_pred)
    rmse = mean_squared_error(y_hold.values, y_pred, squared=False)

    # baseline
    # baseline: previous observed value; forward-fill then fallback to mean if still NaN
    baseline_series = y_hold.shift(1).ffill()
    if baseline_series.isna().any():
        baseline_series = baseline_series.fillna(y_hold.mean())
    baseline = baseline_series.values
    mae_base = mean_absolute_error(y_hold.values, baseline)
    rmse_base = mean_squared_error(y_hold.values, baseline, squared=False)

    print('\nHoldout results:')
    print(f'Best XGBoost MAE={mae:.4f} RMSE={rmse:.4f}')
    print(f'Baseline MAE={mae_base:.4f} RMSE={rmse_base:.4f}')

    # Save model and report
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f'xgboost_{args.target}_tplus{args.horizon}_best.pkl'
    import pickle
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)

    report = {
        'best_params': best,
        'cv_mae': best_score,
        'holdout_mae': float(mae),
        'holdout_rmse': float(rmse),
        'baseline_mae': float(mae_base),
        'baseline_rmse': float(rmse_base),
        'n_train': len(X_train),
        'n_holdout': len(X_hold),
    }
    with open(out_dir / f'tune_report_{args.target}_h{args.horizon}.json', 'w') as f:
        json.dump(report, f, indent=2)

    print('Saved model to', model_path)
    print('Saved report to', out_dir / f'tune_report_{args.target}_h{args.horizon}.json')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', default='CO')
    parser.add_argument('--horizon', type=int, default=1)
    parser.add_argument('--outdir', default=os.path.dirname(__file__))
    parser.add_argument('--n_splits', type=int, default=3)
    parser.add_argument('--holdout', type=float, default=0.2)
    args = parser.parse_args()
    tune_and_evaluate(args)


if __name__ == '__main__':
    main()
