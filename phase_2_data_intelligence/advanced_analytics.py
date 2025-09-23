import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import STL
from preprocess_data import fetch_and_preprocess_data
import os

OUTPUT_DIR = os.path.dirname(__file__) or '.'


def plot_autocorrelation(data):
    """
    Plot the autocorrelation for pollutants and save to disk with improved layout.
    """
    pollutants = [p for p in ['CO', 'NOx', 'NO2', 'Benzene'] if p in data.columns]
    if not pollutants:
        print("No pollutant columns found for autocorrelation plots.")
        return

    # Create a large figure and explicit axes
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True, dpi=150)
    axes = axes.ravel()
    for i, pollutant in enumerate(pollutants):
        ax = axes[i]
        plot_acf(data[pollutant], lags=40, ax=ax)
        ax.set_title(f'{pollutant} Autocorrelation')
    # Save
    out_path = os.path.join(OUTPUT_DIR, 'autocorrelation.png')
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_partial_autocorrelation(data):
    """
    Plot the partial autocorrelation for pollutants and save to disk with improved layout.
    """
    pollutants = [p for p in ['CO', 'NOx', 'NO2', 'Benzene'] if p in data.columns]
    if not pollutants:
        print("No pollutant columns found for partial autocorrelation plots.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True, dpi=150)
    axes = axes.ravel()
    for i, pollutant in enumerate(pollutants):
        ax = axes[i]
        plot_pacf(data[pollutant], lags=40, ax=ax)
        ax.set_title(f'{pollutant} Partial Autocorrelation')
    out_path = os.path.join(OUTPUT_DIR, 'partial_autocorrelation.png')
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def time_series_decomposition(data):
    """
    Decompose the time series into trend, seasonal, and residual components using STL and save plots per pollutant with larger figures.
    """
    pollutants = [p for p in ['CO', 'NOx', 'NO2', 'Benzene'] if p in data.columns]
    for pollutant in pollutants:
        try:
            # Use STL for robust decomposition (works well with seasonal + trend)
            stl = STL(data[pollutant].dropna(), period=24, robust=True)
            res = stl.fit()
            fig = plt.figure(figsize=(12, 9), dpi=150)
            ax1 = fig.add_subplot(411)
            ax1.plot(res.observed)
            ax1.set_title(f'{pollutant} Observed')
            ax2 = fig.add_subplot(412)
            ax2.plot(res.trend)
            ax2.set_title('Trend')
            ax3 = fig.add_subplot(413)
            ax3.plot(res.seasonal)
            ax3.set_title('Seasonal')
            ax4 = fig.add_subplot(414)
            ax4.plot(res.resid)
            ax4.set_title('Residual')
            plt.tight_layout()
            out_path = os.path.join(OUTPUT_DIR, f'{pollutant}_decomposition.png')
            fig.savefig(out_path, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved {out_path}")
        except Exception as e:
            # Fallback to seasonal_decompose for compatibility
            try:
                decomposition = seasonal_decompose(data[pollutant].dropna(), model='additive', period=24)
                fig = decomposition.plot()
                fig.set_size_inches(12, 9)
                fig.set_dpi(150)
                plt.suptitle(f'{pollutant} Time Series Decomposition')
                out_path = os.path.join(OUTPUT_DIR, f'{pollutant}_decomposition.png')
                fig.savefig(out_path, bbox_inches='tight')
                plt.close(fig)
                print(f"Saved {out_path} (fallback)")
            except Exception:
                print(f"Failed to decompose {pollutant}: {e}")


def anomaly_detection(data, threshold=3):
    """
    Detect and visualize anomalies using a simple Z-score method and MAD-based method.
    Export anomalies for downstream use to a CSV file.
    """
    anomalies_rows = []
    pollutants = [p for p in ['CO', 'NOx', 'NO2', 'Benzene'] if p in data.columns]
    for pollutant in pollutants:
        series = data[pollutant].dropna()
        if series.empty:
            continue
        median = np.median(series)
        mad = np.median(np.abs(series - median))
        mean = series.mean()
        std = series.std()

        # Z-score anomalies
        z_scores = (series - mean) / std if std != 0 else (series - mean) * 0.0
        z_anoms = z_scores[ z_scores.abs() > threshold ]

        # MAD-based anomalies (robust)
        if mad == 0:
            mad_z = pd.Series(index=series.index, data=0.0)
        else:
            mad_z = (series - median) / (1.4826 * mad)
        mad_anoms = mad_z[ mad_z.abs() > threshold ]

        # STL residual-based anomalies (seasonal-aware)
        stl_anoms = pd.Series(dtype=float)
        try:
            if isinstance(series.index, pd.DatetimeIndex) and len(series) > 24:
                stl = STL(series, period=24, robust=True)
                stl_res = stl.fit()
                resid = stl_res.resid
                r_median = np.median(resid)
                r_mad = np.median(np.abs(resid - r_median))
                if r_mad == 0:
                    r_mad = 1e-9
                mad_z_resid = (resid - r_median) / (1.4826 * r_mad)
                stl_anoms = mad_z_resid[ mad_z_resid.abs() > threshold ]
        except Exception:
            # if STL fails, leave stl_anoms empty
            stl_anoms = pd.Series(dtype=float)

        # Plot with markers
        fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
        ax.plot(series.index, series.values, label=f'{pollutant} Concentration')
        if not z_anoms.empty:
            ax.scatter(z_anoms.index, series.loc[z_anoms.index], color='red', marker='x', label='Z-score Anomaly')
        if not mad_anoms.empty:
            ax.scatter(mad_anoms.index, series.loc[mad_anoms.index], facecolors='none', edgecolors='orange', label='MAD Anomaly')
        if not stl_anoms.empty:
            ax.scatter(stl_anoms.index, series.loc[stl_anoms.index], color='purple', marker='o', facecolors='none', s=80, label='STL Residual Anomaly')
        ax.set_title(f'{pollutant} Anomalies Detection')
        ax.legend()
        out_path = os.path.join(OUTPUT_DIR, f'{pollutant}_anomalies.png')
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {out_path}")

        # Record anomalies for CSV
        for idx, z in z_anoms.items():
            anomalies_rows.append({'timestamp': idx, 'pollutant': pollutant, 'value': float(series.loc[idx]), 'z_score': float(z), 'mad_z': float(mad_z.loc[idx]), 'method': 'zscore'})
        for idx, mz in mad_anoms.items():
            # avoid duplicate rows if same index triggered both methods
            if idx in z_anoms.index:
                continue
            anomalies_rows.append({'timestamp': idx, 'pollutant': pollutant, 'value': float(series.loc[idx]), 'z_score': float(z_scores.loc[idx]) if idx in z_scores.index else None, 'mad_z': float(mz), 'method': 'mad'})
        for idx, rz in stl_anoms.items():
            # avoid duplicates
            if idx in z_anoms.index or idx in mad_anoms.index:
                continue
            anomalies_rows.append({'timestamp': idx, 'pollutant': pollutant, 'value': float(series.loc[idx]), 'z_score': None, 'mad_z': None, 'method': 'stl_residual'})

    if anomalies_rows:
        anomalies_df = pd.DataFrame(anomalies_rows)
        anomalies_df.sort_values(['timestamp', 'pollutant'], inplace=True)
        csv_path = os.path.join(OUTPUT_DIR, 'anomalies.csv')
        anomalies_df.to_csv(csv_path, index=False)
        print(f"Saved anomalies CSV to: {csv_path}")
    else:
        print("No anomalies detected; anomalies CSV not created.")


def main():
    """
    Main function to run advanced analytics on pollutants.
    """
    data = fetch_and_preprocess_data()

    # Plot Autocorrelation and Partial Autocorrelation
    plot_autocorrelation(data)
    plot_partial_autocorrelation(data)

    # Time Series Decomposition
    time_series_decomposition(data)

    # Anomaly Detection
    anomaly_detection(data)

    # Write a small summary report (top correlations + anomaly counts)
    try:
        write_summary_report()
    except Exception as e:
        print(f"Failed to write summary report: {e}")


def write_summary_report(out_path=None):
    """Create a small markdown summary aggregating correlation and anomaly results."""
    if out_path is None:
        out_path = os.path.join(OUTPUT_DIR, 'phase2_summary.md')

    # Read correlation matrix and p-values if available
    corr_path = os.path.join(OUTPUT_DIR, 'correlation_matrix.csv')
    pval_path = os.path.join(OUTPUT_DIR, 'correlation_matrix_pvalues.csv')
    anomalies_path = os.path.join(OUTPUT_DIR, 'anomalies.csv')

    lines = []
    lines.append('# Phase 2 Summary')
    lines.append('')
    if os.path.exists(corr_path):
        corr = pd.read_csv(corr_path, index_col=0)
        # get top absolute off-diagonal correlations
        pairs = []
        cols = corr.columns.tolist()
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                a = cols[i]; b = cols[j]
                val = corr.loc[a, b]
                pairs.append((abs(val), a, b, val))
        pairs.sort(reverse=True)
        lines.append('## Top correlations')
        for absval, a, b, val in pairs[:5]:
            lines.append(f'- {a} vs {b}: r={val:.3f}')
        lines.append('')
    else:
        lines.append('Correlation matrix not found.')

    # Anomalies summary
    if os.path.exists(anomalies_path):
        anom_df = pd.read_csv(anomalies_path)
        total = len(anom_df)
        counts = anom_df['pollutant'].value_counts().to_dict()
        lines.append('## Anomalies')
        lines.append(f'- Total anomalies detected: {total}')
        for k, v in counts.items():
            lines.append(f'  - {k}: {v}')
        lines.append('')
    else:
        lines.append('Anomalies CSV not found.')

    # write to disk
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f'Wrote summary report to: {out_path}')

if __name__ == "__main__":
    main()
