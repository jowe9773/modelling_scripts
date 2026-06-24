""" shapefile_histograms.py
-----------------------
Reads a point shapefile and plots a histogram for each specified attribute.
Edit the variables in the CONFIG section below, then run:
    python shapefile_histograms.py

Requirements: pip install geopandas matplotlib seaborn scipy
"""

import sys
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import pandas as pd

# ── CONFIG — edit these ─────────────────────────────────────────────────────────
SHAPEFILE = "C:/Users/josie/OneDrive - UCB-O365/Floodplain LW transport modelling/testing_each_rfsd/nowood/model_results.gpkg"
COLUMNS = [
    'err_0.25x_fp0075_ch01125',
    'err_0.5x_fp0075_ch01125',
    'err_1.0x_fp0075_ch01125'
]
BINS    = 100
OUTPUT  = "C:/Users/josie/OneDrive - UCB-O365/Floodplain LW transport modelling/testing_each_rfsd/nowood/RMSE_histograms.png"

# Outlier removal: drop rows where ANY error column falls outside ±N std devs.
# Set to None to disable.
OUTLIER_STD_THRESHOLD = 2.0
# ────────────────────────────────────────────────────────────────────────────────


def remove_outliers_by_std(gdf, columns, n_std):
    mask = pd.Series(True, index=gdf.index)
    removed_counts = {}

    for col in columns:
        data = (
            gdf[col]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        mean = data.mean()
        std = data.std()

        lower = mean - n_std * std
        upper = mean + n_std * std

        col_mask = (
            gdf[col].isna()
            |
            ((gdf[col] >= lower) & (gdf[col] <= upper))
        )

        removed_counts[col] = (~col_mask & gdf[col].notna()).sum()

        mask &= col_mask

    return gdf[mask], removed_counts


def calculate_relative_RMSE(gdf, columns, depth_column="water_dept"):
    """
    Calculate Relative RMSE = RMSE / mean(depth_column)
    Returns
    -------
    dict  {column_name: relative_rmse}
    """
    mean_depth = gdf[depth_column].dropna().mean()
    if mean_depth == 0:
        raise ValueError(
            f"Mean of '{depth_column}' is zero; cannot compute relative RMSE."
        )
    relative_rmse = {}
    for col in columns:
        rmse = np.sqrt((gdf[col].dropna() ** 2).mean())
        relative_rmse[col] = rmse / mean_depth
    return relative_rmse


def plot_histograms(gdf, columns, output_path=None, bins=30):
    """Plot a histogram for each column in a tidy grid."""
    n     = len(columns)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    sns.set_theme(style="whitegrid", palette="muted")
    palette = sns.color_palette("husl", n)

    # Shared x limits
    all_data = pd.concat([gdf[col].dropna() for col in columns])
    x_min, x_max = all_data.min(), all_data.max()

    # Shared y limits
    y_max = 0
    for col in columns:
        counts, _ = np.histogram(gdf[col].dropna(), bins=bins, range=(x_min, x_max))
        y_max = max(y_max, counts.max())
    y_max *= 1.1

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    axes = np.array(axes).flatten()
    used_axes = set()

    for rank, col in enumerate(columns):
        row    = rank % nrows
        col_idx = rank // nrows
        ax_idx = row * ncols + col_idx
        used_axes.add(ax_idx)
        ax = axes[ax_idx]

        data   = gdf[col].dropna()
        mean   = data.mean()
        std    = data.std()
        rmse   = np.sqrt((data ** 2).mean())

        ax.hist(
            data, bins=bins, range=(x_min, x_max),
            color=palette[rank], edgecolor="white", linewidth=0.5, alpha=0.85
        )

        # ±1σ and ±2σ lines
        for nsig, (ls, lw, label) in enumerate(
            [("--", 1.2, "±1σ"), ("-.", 1.0, "±2σ")], start=1
        ):
            for sign in (-1, 1):
                ax.axvline(
                    mean + sign * nsig * std,
                    color="black", linestyle=ls, linewidth=lw, alpha=0.6,
                    label=label if sign == 1 else None
                )

        ax.axvline(mean, color="black", linestyle="-", linewidth=1.4, alpha=0.8, label="mean")

        stats_text = (
            f"n = {len(data):,}\n"
            f"mean = {mean:.3g}\n"
            f"median = {data.median():.3g}\n"
            f"std = {std:.3g}\n"
            f"RMSE = {rmse:.3g}\n"
            f"±1σ: [{mean-std:.3g}, {mean+std:.3g}]\n"
            f"±2σ: [{mean-2*std:.3g}, {mean+2*std:.3g}]"
        )
        ax.text(
            0.97, 0.95, stats_text, transform=ax.transAxes,
            fontsize=8, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
        )

        ax.set_title(f"#{rank+1} | RMSE={rmse:.3g}\n{col}", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel(col, fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, y_max)
        ax.legend(fontsize=7, loc="upper left")

    for i, ax in enumerate(axes):
        if i not in used_axes:
            ax.set_visible(False)

    outlier_note = (
        f"Outlier filter: ±{OUTLIER_STD_THRESHOLD}σ" if OUTLIER_STD_THRESHOLD else "No outlier filter"
    )
    fig.suptitle(
        f"Attribute Histograms (sorted by RMSE) | {outlier_note}",
        fontsize=14, fontweight="bold", y=1.01
    )
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"✅ Saved plot → {output_path}")
    plt.show()


# ── Run ──────────────────────────────────────────────────────────────────────────
print(f"📂 Loading shapefile: {SHAPEFILE}")
gdf = gpd.read_file(SHAPEFILE)
print(f"   {len(gdf):,} features loaded | CRS: {gdf.crs}")
print(f"   Available columns: {[c for c in gdf.columns if c != 'geometry']}")

# ── Filter to bounding box ────────────────────────────────────────────────────────
from shapely.geometry import box
bbox = box(250, -2000, 9500, 2000)
gdf  = gdf[gdf.geometry.within(bbox)]
print(f"   {len(gdf):,} features after bounding box filter")

# ── Outlier removal ───────────────────────────────────────────────────────────────
if OUTLIER_STD_THRESHOLD is not None:
    n_before = len(gdf)
    gdf, removed_counts = remove_outliers_by_std(gdf, COLUMNS, OUTLIER_STD_THRESHOLD)
    n_after = len(gdf)
    print(f"\n🔍 Outlier removal (±{OUTLIER_STD_THRESHOLD}σ):")
    for col, n_removed in removed_counts.items():
        print(f"   {col}: {n_removed:,} outliers removed")
    print(f"   Total rows: {n_before:,} → {n_after:,} ({n_before - n_after:,} removed)")
else:
    print("\n⚠️  Outlier removal disabled (OUTLIER_STD_THRESHOLD = None)")

relative_rmse = calculate_relative_RMSE(gdf, COLUMNS)
print("\nRelative RMSE values:")
for col, rrmse in relative_rmse.items():
    print(f"   {col}: Relative RMSE = {rrmse:.4f}")

# Sort columns by RMSE (ascending)
column_rmse = {col: np.sqrt((gdf[col].dropna() ** 2).mean()) for col in COLUMNS}
COLUMNS = sorted(COLUMNS, key=lambda c: column_rmse[c])
print("\nColumns sorted by RMSE:")
for col in COLUMNS:
    print(f"   {col}: RMSE={column_rmse[col]:.4f}")

# Validate columns
missing = [c for c in COLUMNS if c not in gdf.columns]
if missing:
    sys.exit(f"❌ Column(s) not found in shapefile: {missing}\n"
             f"   Check the 'Available columns' list above.")

print(f"\n📊 Plotting histograms for: {COLUMNS}")
plot_histograms(gdf, COLUMNS, output_path=OUTPUT, bins=BINS)