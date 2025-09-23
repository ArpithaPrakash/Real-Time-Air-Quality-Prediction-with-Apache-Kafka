import pandas as pd
import json
import random
import logging
import time
import argparse
from ucimlrepo import fetch_ucirepo
from datetime import datetime

# Setup logging configuration
log_file = 'producer.log'
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,  # Log level can be changed to DEBUG, ERROR, etc.
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger()


def fetch_and_preprocess_data():
    """
    Fetch and preprocess the Air Quality dataset.
    - If Date/Time columns are present, create a datetime index and resample hourly.
    - Coerce numeric columns, replace sentinel values, impute, normalize, and validate ranges.
    """
    # Fetch the Air Quality dataset
    air_quality = fetch_ucirepo(id=360)
    X = air_quality.data.features  # Features
    y = air_quality.data.targets   # Target (labels)

    # Convert to pandas DataFrame for easier manipulation
    # fetch_ucirepo may already return a DataFrame in X; handle both cases
    if isinstance(X, pd.DataFrame):
        data = X.copy()
    else:
        data = pd.DataFrame(X)

    # If targets are provided, add them as a column
    try:
        data['Target'] = y
    except Exception:
        pass

    # If Date/Time columns exist, combine into a timestamp and set as index
    if 'Date' in data.columns or 'Time' in data.columns:
        date_series = data['Date'] if 'Date' in data.columns else pd.Series([''] * len(data))
        time_series = data['Time'] if 'Time' in data.columns else pd.Series([''] * len(data))
        combined = date_series.astype(str).str.strip() + ' ' + time_series.astype(str).str.strip()
        data['timestamp'] = pd.to_datetime(combined, dayfirst=True, errors='coerce')
        # If timestamp creation produced at least one valid timestamp, set it as index
        if data['timestamp'].notna().any():
            data.set_index('timestamp', inplace=True)
        # Drop Date/Time columns if present
        for drop_col in ("Date", "Time"):
            if drop_col in data.columns:
                data.drop(columns=drop_col, inplace=True)

    # Rename common columns to short names used by validation
    rename_map = {
        'CO(GT)': 'CO',
        'C6H6(GT)': 'Benzene',
        'NOx(GT)': 'NOx',
        'NO2(GT)': 'NO2',
        'NMHC(GT)': 'NMHC'
    }
    data.rename(columns=rename_map, inplace=True)

    # Replace -200 with NaN for missing values (dataset-specific sentinel)
    data.replace(-200, pd.NA, inplace=True)

    # Coerce all remaining columns to numeric where possible (non-numeric become NaN)
    for col in data.columns:
        try:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        except Exception:
            pass

    # If data has a datetime index, resample to hourly means to regularize time series
    if isinstance(data.index, pd.DatetimeIndex):
        try:
            data = data.resample('h').mean()
        except Exception:
            # If resample fails, continue without resampling
            pass

    # Handle missing values only for numeric columns
    num_cols = data.select_dtypes(include=['number']).columns
    if len(num_cols) > 0:
        # Fill numeric NaNs with column mean (after resampling when applicable)
        data[num_cols] = data[num_cols].fillna(data[num_cols].mean())

        # Normalize numeric columns (Min-Max scaling) with protection against division by zero
        denom = (data[num_cols].max() - data[num_cols].min()).replace({0: 1})
        data[num_cols] = (data[num_cols] - data[num_cols].min()) / denom

    # Add sensor range validation and outlier handling based on environmental thresholds
    data = validate_sensor_ranges(data)

    return data


def validate_sensor_ranges(data):
    """
    Validate sensor readings against known environmental thresholds.
    - CO: 0.5 - 5.0 mg/m³
    - NOx: 5 - 100 ppb
    - NO2: Must fall within a reasonable range, e.g., 0 - 200 µg/m³
    - Benzene: 0.5 - 10.0 µg/m³
    """
    # For safety, only clip columns that actually exist
    if 'CO' in data.columns:
        data['CO'] = data['CO'].clip(lower=0.5, upper=5.0)
    if 'NOx' in data.columns:
        data['NOx'] = data['NOx'].clip(lower=5, upper=100)
    if 'NO2' in data.columns:
        data['NO2'] = data['NO2'].clip(lower=0, upper=200)
    if 'Benzene' in data.columns:
        data['Benzene'] = data['Benzene'].clip(lower=0.5, upper=10.0)

    return data


def qc_summary(data):
    """
    Produce a small QC summary DataFrame with missingness and basic stats.
    Returns a DataFrame suitable for CSV export.
    """
    # Work on a copy so we don't mutate original
    df = data.copy()
    # Ensure numeric coercion for stats
    num_cols = df.select_dtypes(include=['number']).columns

    rows = []
    for col in df.columns:
        col_series = df[col]
        missing = int(col_series.isna().sum())
        total = len(col_series)
        pct_missing = missing / total if total > 0 else 0
        if col in num_cols:
            mean = float(col_series.mean()) if not col_series.dropna().empty else None
            std = float(col_series.std()) if not col_series.dropna().empty else None
            min_v = float(col_series.min()) if not col_series.dropna().empty else None
            max_v = float(col_series.max()) if not col_series.dropna().empty else None
        else:
            mean = std = min_v = max_v = None

        rows.append({
            'column': col,
            'total': total,
            'missing': missing,
            'pct_missing': pct_missing,
            'mean': mean,
            'std': std,
            'min': min_v,
            'max': max_v,
        })

    return pd.DataFrame(rows)


def export_qc_report(data, out_path):
    df = qc_summary(data)
    df.to_csv(out_path, index=False)
    return out_path


def generate_air_quality_data(data):
    """
    Generate random air quality data by selecting a row from the dataset.
    This simulates real-time data streaming.
    """
    # Randomly select a row and convert it to a dictionary
    sample = data.sample(n=1).to_dict(orient="records")[0]

    # Add a timestamp to simulate real-time data streaming
    timestamp = datetime.now().isoformat()
    sample['timestamp'] = timestamp  # Add a timestamp to the sample

    return sample


# delivery_report belongs to producer.py now; keep preprocessing module focused on data preparation


def main():
    """CLI for printing generated samples from the preprocessed dataset.

    This module no longer writes to Kafka. To send messages to Kafka, use
    `phase_1_streaming_infrastructure/producer.py` which imports the
    functions here and handles all Kafka configuration and delivery reporting.
    """
    data = fetch_and_preprocess_data()

    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=10, help='Number of messages to generate/print')
    args = parser.parse_args()

    for _ in range(args.count):
        message = generate_air_quality_data(data)
        message_str = json.dumps(message)
        logger.info(f'Generated message: {message_str}')
        print(message_str)
        time.sleep(1)

if __name__ == "__main__":
    main()
