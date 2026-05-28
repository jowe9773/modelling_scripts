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


# ── CONFIG — edit these ───────────────────────────────────────────────────────

SHAPEFILE = "C:/Users/josie/OneDrive - UCB-O365/Floodplain LW transport modelling/Model Setup and Calibration/QGIS/data/fp_calibration_model_outputs/fp_comparison_refinement_1.shp"   # Path to your .shp file

COLUMNS = [                            # Columns to plot (add or remove as needed)
    "fp_005_err",
    "fp_008_err",
    "fp_01_err",
    "fp_012_err"
]

BINS = 100                              # Number of histogram bins

OUTPUT = None                          # Set to e.g. "histograms.png" to save,
                                       # or leave as None to show interactively

# ─────────────────────────────────────────────────────────────────────────────


def plot_histograms(gdf, columns, output_path=None, bins=30):
    """Plot a histogram for each column in a tidy grid."""
    n = len(columns)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    sns.set_theme(style="whitegrid", palette="muted")
    palette = sns.color_palette("husl", n)

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    axes = np.array(axes).flatten()    # always a 1-D list of axes

    for i, col in enumerate(columns):
        ax = axes[i]
        data = gdf[col].dropna()

        ax.hist(data, bins=bins, color=palette[i], edgecolor="white",
                linewidth=0.5, alpha=0.85)

        # Overlay a KDE curve
        if data.std() > 0:
            kde_x = np.linspace(data.min(), data.max(), 300)
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data)
            ax2 = ax.twinx()
            ax2.plot(kde_x, kde(kde_x), color=palette[i], linewidth=2,
                     linestyle="--", alpha=0.7)
            ax2.set_ylabel("Density", fontsize=9, color="grey")
            ax2.tick_params(axis="y", labelcolor="grey", labelsize=8)
            ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

        # Stats annotation
        stats_text = (f"n = {len(data):,}/n"
                      f"mean = {data.mean():.3g}/n"
                      f"median = {data.median():.3g}/n"
                      f"std = {data.std():.3g}")
        ax.text(0.97, 0.95, stats_text, transform=ax.transAxes,
                fontsize=8, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

        ax.set_title(col, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel(col, fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.tick_params(labelsize=8)

    # Hide any unused subplot panels
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Attribute Histograms", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"✅  Saved plot → {output_path}")
    else:
        plt.show()


# ── Run ───────────────────────────────────────────────────────────────────────

print(f"📂  Loading shapefile: {SHAPEFILE}")
gdf = gpd.read_file(SHAPEFILE)
print(f"    {len(gdf):,} features loaded | CRS: {gdf.crs}")
print(f"    Available columns: {[c for c in gdf.columns if c != 'geometry']}")

# Validate columns
missing = [c for c in COLUMNS if c not in gdf.columns]
if missing:
    sys.exit(f"❌  Column(s) not found in shapefile: {missing}/n"
             f"    Check the 'Available columns' list above.")

print(f"📊  Plotting histograms for: {COLUMNS}")
plot_histograms(gdf, COLUMNS, output_path=OUTPUT, bins=BINS)