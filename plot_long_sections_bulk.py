# Iterate Through Pass Values and Save Scatter Plots

"""
Shapefile → GeoPandas → Scatter Plot Export
===========================================

This script:
    1. Loads a shapefile into a GeoDataFrame
    2. Extracts X/Y coordinates from geometry
    3. Iterates through Pass values (1–6)
    4. Creates a scatter plot for each pass
    5. Saves each plot as a PNG file

Dependencies:
    pip install geopandas matplotlib shapely
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────
# STEP 1 – Load shapefile
# ─────────────────────────────────────────────────────────────

sample_path = (
    "C:/Users/josie/OneDrive - UCB-O365/Floodplain LW transport modelling/Model Setup and Calibration/true_shape_flume/test_comparison00701.shp"
)

shapefile_path = sample_path

gdf = gpd.read_file(shapefile_path)

print("── GeoDataFrame overview ──")
print(f"CRS       : {gdf.crs}")
print(f"Geometry  : {gdf.geom_type.unique()}")
print(f"Rows      : {len(gdf)}")
print(f"Columns   : {list(gdf.columns)}")
print(gdf.head())


# ─────────────────────────────────────────────────────────────
# STEP 2 – Extract coordinates
# ─────────────────────────────────────────────────────────────

gdf["x_coord"] = gdf.geometry.x
gdf["y_coord"] = gdf.geometry.y


# ─────────────────────────────────────────────────────────────
# STEP 3 – Plot settings
# ─────────────────────────────────────────────────────────────

PLOT_ATTRIBUTE = [
    "depth",
    "CAD",
    "TIF",
    "TIF_007"
]

ATTRIBUTE_COLORS = [
    "black",
    "green",
    "blue",
    "purple"
]

FILTER_COLUMN = "Pass"


# ─────────────────────────────────────────────────────────────
# STEP 4 – Output folder
# ─────────────────────────────────────────────────────────────

OUTPUT_FOLDER = (
    "C:/Users/josie/OneDrive - UCB-O365/Floodplain LW transport modelling/Model Setup and Calibration/true_shape_flume"
)

# create folder if it doesn't exist
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# STEP 5 – Iterate through Pass values and save plots
# ─────────────────────────────────────────────────────────────

for pass_value in range(1, 7):

    # subset data
    subset = gdf[gdf[FILTER_COLUMN] == pass_value].copy()

    print(f"\n── Pass = {pass_value} ──")
    print(f"Rows after filter: {len(subset)}")

    # skip empty subsets
    if len(subset) == 0:
        print("No rows found, skipping.")
        continue

    # create figure
    fig, ax = plt.subplots(figsize=(9, 5))

    # plot each attribute
    for attr, color in zip(PLOT_ATTRIBUTE, ATTRIBUTE_COLORS):

        # skip missing columns
        if attr not in subset.columns:
            print(f"Column not found: {attr}")
            continue

        ax.scatter(
            subset["x_coord"],
            subset[attr],
            c=color,
            label=attr,
            alpha=0.75,
            edgecolors="white",
            linewidths=0.4,
            s=60,
        )

    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Value")
    ax.set_title(f"X Coordinate vs Attributes — Pass {pass_value}")

    ax.legend()

    plt.tight_layout()

    # output filename
    out_file = f"{OUTPUT_FOLDER}/scatter_pass_{pass_value}.png"

    # save figure
    plt.savefig(out_file, dpi=300)

    print(f"Saved: {out_file}")

    # close figure to avoid memory buildup
    plt.close(fig)


print("\nFinished generating plots.")

