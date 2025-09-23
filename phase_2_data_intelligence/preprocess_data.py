import pandas as pd
import json
import random
import time
import argparse
from ucimlrepo import fetch_ucirepo
from datetime import datetime
from datetime import datetime

def fetch_and_preprocess_data():
    """
    Fetch and preprocess the Air Quality dataset.
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

    # Parse Date + Time into a proper timestamp if present (preserve temporal fidelity)
    if 'Date' in data.columns and 'Time' in data.columns:
        # Date in the UCI dataset is often day/month/year - use dayfirst=True
        try:
            data['timestamp'] = pd.to_datetime(data['Date'].astype(str) + ' ' + data['Time'].astype(str), dayfirst=True, errors='coerce')
        except Exception:
            data['timestamp'] = pd.to_datetime(data['Date'].astype(str) + ' ' + data['Time'].astype(str), errors='coerce')
        # drop the original Date/Time columns now that we have a parsed timestamp
        data.drop(columns=['Date', 'Time'], inplace=True)
    elif 'timestamp' in data.columns:
        data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')

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
        # don't coerce the timestamp column
        if col == 'timestamp':
            continue
        try:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        except Exception:
            pass

    # If we have a timestamp column, set it as the index and resample to an hourly frequency
    if 'timestamp' in data.columns:
        # drop rows with invalid timestamps
        data = data[~data['timestamp'].isna()].copy()
        data.set_index('timestamp', inplace=True)
        data.sort_index(inplace=True)

        # Resample to hourly averages to obtain consistent cadence for ACF/PACF and decomposition
        try:
            data = data.resample('h').mean()
        except Exception:
            # fallback: if resample fails, keep the original ordering
            pass

        # After resampling, fill remaining numeric NaNs with column mean
        num_cols = data.select_dtypes(include=['number']).columns
        if len(num_cols) > 0:
            data[num_cols] = data[num_cols].fillna(data[num_cols].mean())

        return data

    # Fallback: no timestamp available — fill numeric NaNs with per-column mean and return
    num_cols = data.select_dtypes(include=['number']).columns
    if len(num_cols) > 0:
        data[num_cols] = data[num_cols].fillna(data[num_cols].mean())

    return data

def validate_sensor_ranges(data):
    """
    Validate sensor readings against known environmental thresholds.
    - CO: 0.5 - 5.0 mg/m³
    - NOx: 5 - 100 ppb
    - NO2: Must fall within a reasonable range, e.g., 0 - 200 µg/m³
    - Benzene: 0.5 - 10.0 µg/m³
    """
    # Carbon Monoxide (CO) validation
    data['CO'] = data['CO'].clip(lower=0.5, upper=5.0)

    # Nitrogen Oxides (NOx) validation
    data['NOx'] = data['NOx'].clip(lower=5, upper=100)

    # Nitrogen Dioxide (NO2) validation
    data['NO2'] = data['NO2'].clip(lower=0, upper=200)

    # Benzene validation
    data['Benzene'] = data['Benzene'].clip(lower=0.5, upper=10.0)

    return data

def generate_air_quality_data(data):
    """
    Generate random air quality data by selecting a row from the dataset.
    This simulates real-time data streaming.
    """
    # Randomly select a row and convert it to a dictionary
    row = data.sample(n=1)
    sample = row.to_dict(orient='records')[0]

    # If the dataset has a datetime index, use the sampled index value as the event timestamp
    try:
        if isinstance(data.index, pd.DatetimeIndex):
            sampled_ts = row.index[0]
            sample['timestamp'] = pd.Timestamp(sampled_ts).isoformat()
        else:
            # fallback to current time
            sample['timestamp'] = datetime.now().isoformat()
    except Exception:
        sample['timestamp'] = datetime.now().isoformat()

    return sample

def delivery_report(err, msg):
    if err is not None:
        print(f'Message delivery failed: {err}')
    else:
        print(f'Message delivered to {msg.topic()} [{msg.partition()}] at offset {msg.offset()}')

def main():
    """
    Main function to run the data stream simulation and send data to Kafka.
    """
    # Fetch and preprocess data
    data = fetch_and_preprocess_data()
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=10, help='Number of messages to generate/print')
    args = parser.parse_args()

    # Print generated samples only. Publishing to Kafka is the responsibility of the producer script.
    for _ in range(args.count):
        message = generate_air_quality_data(data)
        message_str = json.dumps(message)
        print(message_str)
        time.sleep(1)

if __name__ == "__main__":
    main()