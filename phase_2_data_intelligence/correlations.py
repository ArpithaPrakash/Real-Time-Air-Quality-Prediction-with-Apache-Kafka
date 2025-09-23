import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from preprocess_data import fetch_and_preprocess_data
from scipy.stats import pearsonr

OUTPUT_DIR = os.path.dirname(__file__) or '.'


def compute_correlation_and_pvalues(data, columns=None):
    """
    Computes the correlation matrix and pairwise Pearson p-values between specified pollutant columns.
    Returns (corr_df, pval_df).
    """
    if columns is None:
        columns = [c for c in ['CO', 'NOx', 'NO2', 'Benzene'] if c in data.columns]
    if not columns:
        raise ValueError("No pollutant columns found in data to compute correlation.")

    corr = pd.DataFrame(index=columns, columns=columns, dtype=float)
    pvals = pd.DataFrame(index=columns, columns=columns, dtype=float)

    for i, a in enumerate(columns):
        for j, b in enumerate(columns):
            # compute only once for symmetric matrix
            if pd.notna(corr.iloc[i, j]):
                continue
            try:
                # drop NA pairs
                valid = data[[a, b]].dropna()
                if len(valid) < 3:
                    r = np.nan
                    p = np.nan
                else:
                    r, p = pearsonr(valid[a], valid[b])
            except Exception:
                r = np.nan
                p = np.nan
            corr.loc[a, b] = r
            corr.loc[b, a] = r
            pvals.loc[a, b] = p
            pvals.loc[b, a] = p

    # Ensure diagonal is exactly 1.0 for correlations and p-values diagonal is NaN
    for c in columns:
        corr.loc[c, c] = 1.0
        pvals.loc[c, c] = np.nan

    return corr.astype(float), pvals.astype(float)


def significance_stars(p):
    """Return significance stars for a p-value."""
    if pd.isna(p):
        return ''
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return ''


def plot_and_save_correlation_matrix(correlation_matrix, pvalues=None, out_prefix='correlation_matrix'):
    """
    Plots and saves the correlation matrix as a PNG with annotations and saves the matrix as CSV.
    If pvalues is provided, annotate cells with significance stars and save pvalues CSV as well.
    """
    # Save CSV for correlation
    csv_path = os.path.join(OUTPUT_DIR, f"{out_prefix}.csv")
    correlation_matrix.to_csv(csv_path)
    print(f"Saved correlation matrix CSV to: {csv_path}")

    if pvalues is not None:
        # Save p-values using scientific formatting so very small values aren't written as 0.0
        p_csv = os.path.join(OUTPUT_DIR, f"{out_prefix}_pvalues.csv")
        # Convert to string-formatted DataFrame with scientific notation, preserving NaN
        # use object dtype to allow string-formatted numbers and blanks without dtype warnings
        p_formatted = pvalues.astype(object).copy()
        for r in p_formatted.index:
            for c in p_formatted.columns:
                val = p_formatted.loc[r, c]
                if pd.isna(val):
                    p_formatted.loc[r, c] = ''
                else:
                    # format with 12 significant digits in exponent form
                    p_formatted.loc[r, c] = f"{val:.12e}"

        p_formatted.to_csv(p_csv)
        print(f"Saved correlation p-values CSV to: {p_csv}")

    # Plot heatmap and save PNG
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    im = ax.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1, interpolation='none')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Axis ticks and labels
    labels = correlation_matrix.columns.tolist()
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)

    # Annotate cells with correlation values and significance
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = correlation_matrix.iloc[i, j]
            pval = None
            if pvalues is not None:
                pval = pvalues.iloc[i, j]
            stars = significance_stars(pval) if pval is not None else ''
            if pd.isna(val):
                text = ''
            else:
                text = f"{val:.2f}{stars}"
            ax.text(j, i, text, ha='center', va='center', color='black')

    ax.set_title('Pollutant Correlation Matrix')
    plt.tight_layout()

    png_path = os.path.join(OUTPUT_DIR, f"{out_prefix}.png")
    plt.savefig(png_path)
    plt.close()
    print(f"Saved correlation matrix PNG to: {png_path}")


def main():
    data = fetch_and_preprocess_data()
    corr, pvals = compute_correlation_and_pvalues(data)
    print("Correlation Matrix:\n", corr)
    print("P-values Matrix:\n", pvals)
    plot_and_save_correlation_matrix(corr, pvalues=pvals)


if __name__ == "__main__":
    main()
