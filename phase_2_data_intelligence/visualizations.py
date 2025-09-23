import pandas as pd
import matplotlib.pyplot as plt
from preprocess_data import fetch_and_preprocess_data
import os

OUTPUT_DIR = os.path.dirname(__file__) or '.'


def extract_temporal_features(data):
    """
    Extract day of the week and hour from timestamp for pattern analysis.
    If a `timestamp` column is not present, generate a synthetic, evenly-spaced time index
    ending at the current time (minute granularity).
    """
    # Work on a copy to avoid mutating caller's DataFrame
    data = data.copy()

    if 'timestamp' not in data.columns:
        # Generate a synthetic timestamp range ending now
        try:
            now = pd.Timestamp.now()
            # Use minute frequency; if dataset is larger than reasonable, pandas will still create it
            data['timestamp'] = pd.date_range(end=now, periods=len(data), freq='min')
        except Exception:
            # Fallback: set all timestamps to now
            data['timestamp'] = pd.Timestamp.now()
    else:
        # Ensure timestamps are parsed; coerce invalid rows to NaT and fallback if all NaT
        data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
        if data['timestamp'].isna().all():
            data['timestamp'] = pd.date_range(end=pd.Timestamp.now(), periods=len(data), freq='min')

    data['day_of_week'] = data['timestamp'].dt.dayofweek  # Monday=0, Sunday=6
    data['hour_of_day'] = data['timestamp'].dt.hour  # 0-23 hours
    return data


def visualize_daily_weekly_patterns(data):
    """
    Visualize daily and weekly cyclical patterns for pollutants.
    This function is resilient to missing pollutant columns and will save figures as PNG files
    when running in headless environments.
    """
    # Ensure temporal features exist
    if 'hour_of_day' not in data.columns or 'day_of_week' not in data.columns:
        data = extract_temporal_features(data)

    # Choose pollutant columns that actually exist in the DataFrame
    pollutants = [c for c in ['CO', 'NOx', 'NO2', 'Benzene'] if c in data.columns]
    if not pollutants:
        print("No pollutant columns found to visualize. Available columns:", list(data.columns))
        return

    # Aggregate by hour of day
    daily_data = data.groupby('hour_of_day')[pollutants].mean()
    ax = daily_data.plot(figsize=(10, 6))
    ax.set_title("Hourly Average Concentrations (Daily Pattern)")
    ax.set_xlabel("Hour of the Day")
    ax.set_ylabel("Concentration (Normalized)")
    ax.set_xticks(range(0, 24, 1))
    ax.grid(True)
    fig = ax.get_figure()
    fig.tight_layout()
    daily_path = os.path.join(OUTPUT_DIR, 'daily_pattern.png')
    fig.savefig(daily_path)
    plt.close(fig)
    print(f"Saved {daily_path}")

    # Aggregate by day of week
    weekly_data = data.groupby('day_of_week')[pollutants].mean()
    ax = weekly_data.plot(figsize=(10, 6))
    ax.set_title("Average Concentrations by Day of Week (Weekly Pattern)")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Concentration (Normalized)")
    ax.set_xticks(range(0, 7))
    ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    ax.grid(True)
    fig = ax.get_figure()
    fig.tight_layout()
    weekly_path = os.path.join(OUTPUT_DIR, 'weekly_pattern.png')
    fig.savefig(weekly_path)
    plt.close(fig)
    print(f"Saved {weekly_path}")


def main():
    """
    Main function to run the analysis and visualizations.
    """
    data = fetch_and_preprocess_data()
    data = extract_temporal_features(data)
    visualize_daily_weekly_patterns(data)


if __name__ == "__main__":
    main()
