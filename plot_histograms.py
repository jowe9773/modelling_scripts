"""
shapefile_histograms.py
-----------------------
Reads a point shapefile and plots a histogram for each specified attribute.

Edit the variables in the CONFIG section below, then run:
    python shapefile_histograms.py

Requirements:
    pip install geopandas matplotlib seaborn scipy
"""

import sys
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import pandas as pd


# ── CONFIG — edit these ───────────────────────────────────────────────────────

SHAPEFILE = "C:/Users/josie/OneDrive - UCB-O365/Floodplain LW transport modelling/Model Setup and Calibration/DEM_based_model_with_headboxes/results/model_results.gpkg"

COLUMNS = [                            # Columns to plot (add or remove as needed)
    'err_ch-01_fp_005',
    'err_ch-01_fp_00625',
    'err_ch-01_fp_0075',
    'err_ch-01_fp_00875',
    'err_ch-01_fp_01'

]

BINS = 100                              # Number of histogram bins

OUTPUT = "C:/Users/josie/OneDrive - UCB-O365/Floodplain LW transport modelling/Model Setup and Calibration/DEM_based_model_with_headboxes/results/plots/best_model_histogram.png"
                                       # Set to e.g. "histograms.png" to save,
                                       # or leave as None to show interactively

# ─────────────────────────────────────────────────────────────────────────────


def plot_histograms(gdf, columns, output_path=None, bins=30):
    """Plot a histogram for each column in a tidy grid."""
    n = len(columns)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    sns.set_theme(style="whitegrid", palette="muted")
    palette = sns.color_palette("husl", n)

    # ── Compute shared axis limits ────────────────────────────────────────────
    all_data = pd.concat([gdf[col].dropna() for col in columns])
    x_min, x_max = all_data.min(), all_data.max()

    # Pre-compute counts to find shared y-limit
    y_max = 0
    for col in columns:
        counts, _ = np.histogram(gdf[col].dropna(), bins=bins,
                                 range=(x_min, x_max))
        y_max = max(y_max, counts.max())
    y_max *= 1.1  # 10% headroom

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    axes = np.array(axes).flatten()

    for i, col in enumerate(columns):
        ax = axes[i]
        data = gdf[col].dropna()

        mean, std = data.mean(), data.std()
        rmse = np.sqrt((data ** 2).mean())

        ax.hist(data, bins=bins, range=(x_min, x_max),
                color=palette[i], edgecolor="white",
                linewidth=0.5, alpha=0.85)

        # ±1σ and ±2σ vertical lines
        for nsig, (ls, lw, label) in enumerate(
                [("--", 1.2, "±1σ"), ("-.", 1.0, "±2σ")], start=1):
            for sign in (-1, 1):
                ax.axvline(mean + sign * nsig * std,
                           color="black", linestyle=ls, linewidth=lw,
                           alpha=0.6,
                           label=label if sign == 1 else None)

        ax.axvline(mean, color="black", linestyle="-", linewidth=1.4,
                   alpha=0.8, label="mean")

        ax.legend(fontsize=7, loc="upper left")

        pct1 = 100 * ((data >= mean - std)   & (data <= mean + std)).mean()
        pct2 = 100 * ((data >= mean - 2*std) & (data <= mean + 2*std)).mean()

        stats_text = (f"n = {len(data):,}\n"
                      f"mean = {mean:.3g}\n"
                      f"median = {data.median():.3g}\n"
                      f"std = {std:.3g}\n"
                      f"RMSE = {rmse:.3g}\n"
                      f"±1σ: [{mean-std:.3g}, {mean+std:.3g}]\n"
                      f"±2σ: [{mean-2*std:.3g}, {mean+2*std:.3g}]")

        ax.text(0.97, 0.95, stats_text, transform=ax.transAxes,
                fontsize=8, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

        ax.set_title(col, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel(col, fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.tick_params(labelsize=8)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, y_max)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Attribute Histograms", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"✅  Saved plot → {output_path}")
        plt.show()
    else:
        plt.show()


# ── Run ───────────────────────────────────────────────────────────────────────

print(f"📂  Loading shapefile: {SHAPEFILE}")
gdf = gpd.read_file(SHAPEFILE)
print(f"    {len(gdf):,} features loaded | CRS: {gdf.crs}")
print(f"    Available columns: {[c for c in gdf.columns if c != 'geometry']}")

# ── Filter to bounding box ────────────────────────────────────────────────────
from shapely.geometry import box

bbox = box(250, -2000, 9500, 2000)
gdf = gdf[gdf.geometry.within(bbox)]
print(f"    {len(gdf):,} features after bounding box filter")

# Validate columns
missing = [c for c in COLUMNS if c not in gdf.columns]
if missing:
    sys.exit(f"❌  Column(s) not found in shapefile: {missing}\n"
             f"    Check the 'Available columns' list above.")

print(f"📊  Plotting histograms for: {COLUMNS}")
plot_histograms(gdf, COLUMNS, output_path=OUTPUT, bins=BINS)