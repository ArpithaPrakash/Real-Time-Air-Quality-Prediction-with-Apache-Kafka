import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo


def fetch_and_preprocess_data(rolling_windows=(3, 6, 12, 24), rolling_cols=None,
                              add_ema=True, ema_windows=(3, 6, 12, 24),
                              add_slopes=True, slope_windows=(6, 12, 24),
                              add_interactions=True, interaction_lags=(1, 24)):
    """
    Fetch and preprocess the Air Quality dataset.

    Parameters
    - rolling_windows: iterable of integer window sizes (hours) to compute rolling stats
    - rolling_cols: list of pollutant column names to compute rolling stats for. If None, sensible defaults are used.

    Returns
    - pandas.DataFrame with added temporal, lag, and rolling features
    """
    # Fetch the Air Quality dataset
    air_quality = fetch_ucirepo(id=360)
    X = air_quality.data.features  # Features
    y = air_quality.data.targets   # Target (labels)

    # Convert to pandas DataFrame for easier manipulation
    if isinstance(X, pd.DataFrame):
        data = X.copy()
    else:
        data = pd.DataFrame(X)

    # If targets are provided, add them as a column
    try:
        data['Target'] = y
    except Exception:
        # ignore if y is not aligned or not present
        pass

    # Drop non-numeric/time columns we won't use in numeric processing
    for drop_col in ("Date", "Time"):
        if drop_col in data.columns:
            data.drop(columns=drop_col, inplace=True)

    # Rename columns for ease of use
    rename_map = {
        'CO(GT)': 'CO',
        'C6H6(GT)': 'Benzene',
        'NOx(GT)': 'NOx',
        'NO2(GT)': 'NO2',
        'NMHC(GT)': 'NMHC'
    }
    data.rename(columns=rename_map, inplace=True)

    # Replace -200 with NaN for missing values
    data.replace(-200, pd.NA, inplace=True)

    # Coerce all remaining columns to numeric where possible (non-numeric become NaN)
    for col in data.columns:
        try:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        except Exception:
            pass
    
    # Handle missing values for numeric columns (fill with column mean)
    num_cols = data.select_dtypes(include=['number']).columns
    if len(num_cols) > 0:
        data[num_cols] = data[num_cols].fillna(data[num_cols].mean())
    
    # Feature Engineering: Add temporal features and lagged features
    data = add_temporal_features(data)

    # Rolling/statistical features (configurable)
    data = add_rolling_features(data, cols=rolling_cols, windows=rolling_windows)

    # Add lagged features including a 24-hour (previous-day) lag which is often informative
    data = add_lagged_features(data, lags=(1, 2, 3, 24))

    # Advanced engineered features
    if add_ema:
        data = add_ema_features(data, cols=rolling_cols, windows=ema_windows)
    if add_slopes:
        data = add_trend_slope_features(data, cols=rolling_cols, windows=slope_windows)
    if add_interactions:
        data = add_lag_interaction_features(data, base_cols=rolling_cols, lags=interaction_lags)

    return data


def add_temporal_features(data):
    """
    Add temporal features such as hour, day, month, and seasonal encoding.
    """
    # Ensure the DataFrame has a DatetimeIndex. If not, try to construct one from
    # existing Date/Time columns (if present). Otherwise synthesize a simple hourly
    # DatetimeIndex to allow temporal feature extraction.
    if not isinstance(data.index, pd.DatetimeIndex):
        # Try common timestamp columns
        if 'timestamp' in data.columns:
            try:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data.set_index('timestamp', inplace=True)
            except Exception:
                # fallback to synthetic index below
                pass
        # If still not datetime index, synthesize a regular hourly index
        if not isinstance(data.index, pd.DatetimeIndex):
            # create an hourly index ending now with same length as data
            data.index = pd.date_range(end=pd.Timestamp.now(), periods=len(data), freq='h')

    data['hour'] = data.index.hour
    data['day'] = data.index.day
    data['month'] = data.index.month
    data['season'] = data['month'].apply(lambda x: (x % 12 + 3) // 3)  # 1: Spring, 2: Summer, 3: Fall, 4: Winter

    # Cyclical encodings: hour (24h), day-of-week (7), month (12)
    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)

    # dayofweek uses index.dayofweek (0=Mon .. 6=Sun)
    try:
        dow = data.index.dayofweek
    except Exception:
        dow = (data.index.hour % 7)
    data['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    data['dow_cos'] = np.cos(2 * np.pi * dow / 7)

    data['month_sin'] = np.sin(2 * np.pi * (data['month'] - 1) / 12)
    data['month_cos'] = np.cos(2 * np.pi * (data['month'] - 1) / 12)

    return data


def add_lagged_features(data, lags=(1,)):
    """
    Add lagged features (e.g., previous values).
    """
    cols = [c for c in ('CO', 'NOx', 'NO2', 'Benzene', 'Target') if c in data.columns]
    for lag in lags:
        for col in cols:
            data[f'{col}_lag{lag}'] = data[col].shift(lag)

    # Keep NaNs for lagged features (caller can decide how to treat them). Do not blanket-fill with 0.
    return data


def add_rolling_features(data, cols=None, windows=(3, 6, 24)):
    """Add rolling mean and std features for selected columns.

    Business rationale (short): short windows (3h) capture transient spikes, medium (6h) capture short events, and longer (24h) capture diurnal trends.
    """
    if cols is None:
        # sensible defaults — only include pollutant columns if present
        candidate = ['CO', 'NOx', 'NO2', 'Benzene']
        cols = [c for c in candidate if c in data.columns]

    if not cols:
        return data

    for w in windows:
        rolled = data[cols].rolling(window=w, min_periods=1)
        mean_cols = {c: f'{c}_roll_mean_{w}' for c in cols}
        std_cols = {c: f'{c}_roll_std_{w}' for c in cols}
        data = data.assign(**{mean_cols[c]: rolled[c].mean() for c in cols})
        # fill std NaNs (single observation) with 0
        data = data.assign(**{std_cols[c]: rolled[c].std().fillna(0) for c in cols})

    return data


def add_ema_features(data, cols=None, windows=(3, 6, 12, 24)):
    """Add exponential moving averages (EMA) for selected columns.

    Features are named `<col>_ema_<w>`.
    """
    if cols is None:
        candidate = ['CO', 'NOx', 'NO2', 'Benzene']
        cols = [c for c in candidate if c in data.columns]

    if not cols:
        return data

    for w in windows:
        span = w
        for c in cols:
            try:
                data[f'{c}_ema_{w}'] = data[c].ewm(span=span, adjust=False).mean()
            except Exception:
                data[f'{c}_ema_{w}'] = pd.NA

    return data


def add_trend_slope_features(data, cols=None, windows=(6, 12, 24)):
    """Estimate short-term linear trend (slope) over rolling windows.

    Feature name: `<col>_slope_<w>` where slope is the coefficient of time in a linear fit.
    """
    if cols is None:
        candidate = ['CO', 'NOx', 'NO2', 'Benzene']
        cols = [c for c in candidate if c in data.columns]

    if not cols:
        return data

    for w in windows:
        # rolling apply with a slope estimator
        def slope(x):
            # x is a 1d numpy array
            if np.isnan(x).all() or len(x) < 2:
                return np.nan
            # x may contain NaNs; drop them with corresponding time indices
            idx = np.arange(len(x))
            mask = ~np.isnan(x)
            if mask.sum() < 2:
                return np.nan
            xi = idx[mask]
            yi = x[mask]
            # linear fit yi = a + b*xi -> return b
            try:
                b = np.polyfit(xi, yi, 1)[0]
            except Exception:
                return np.nan
            return float(b)

        for c in cols:
            data[f'{c}_slope_{w}'] = data[c].rolling(window=w, min_periods=2).apply(slope, raw=True)

    return data


def add_lag_interaction_features(data, base_cols=None, lags=(1, 24)):
    """Create interaction features between lagged columns (pairwise products).

    For example, `CO_lag1__NOx_lag1` is the product of CO_lag1 and NOx_lag1.
    """
    if base_cols is None:
        candidate = ['CO', 'NOx', 'NO2', 'Benzene']
        base_cols = [c for c in candidate if c in data.columns]

    if not base_cols:
        return data

    # gather lagged column names that exist
    lagged = []
    for c in base_cols:
        for l in lags:
            name = f'{c}_lag{l}'
            if name in data.columns:
                lagged.append(name)

    # create pairwise interactions (unique pairs)
    from itertools import combinations
    for a, b in combinations(lagged, 2):
        new_name = f'{a}__{b}'
        # product will propagate NaNs where either is NaN
        data[new_name] = data[a] * data[b]

    return data
