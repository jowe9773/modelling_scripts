"""
Shapefile → GeoPandas → Scatter Plot
=====================================
Workflow:
  1. Create a sample point shapefile (replace with your own file path)
  2. Load it into a GeoDataFrame
  3. Subset rows by an attribute value
  4. Plot X coordinate vs. a chosen attribute as a scatter plot

Dependencies:
  pip install geopandas matplotlib shapely
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point

sample_path = "C:/Users/josie/OneDrive - UCB-O365/Floodplain LW transport modelling/Model Setup and Calibration/QGIS/data/fp_calibration_model_outputs/fp_comparison_coarse.shp"
# ─────────────────────────────────────────────────────────────
# STEP 1 – Load your shapefile
# ─────────────────────────────────────────────────────────────
shapefile_path = sample_path          # ← replace with your actual path

gdf = gpd.read_file(shapefile_path)

print("── GeoDataFrame overview ──")
print(f"CRS       : {gdf.crs}")
print(f"Geometry  : {gdf.geom_type.unique()}")
print(f"Rows      : {len(gdf)}")
print(f"Columns   : {list(gdf.columns)}")
print(gdf.head())


# ─────────────────────────────────────────────────────────────
# STEP 2 – Extract X coordinate into its own column
# ─────────────────────────────────────────────────────────────
gdf["x_coord"] = gdf.geometry.x   # longitude (or easting for projected CRS)
gdf["y_coord"] = gdf.geometry.y   # latitude  (or northing)


# ─────────────────────────────────────────────────────────────
# STEP 3 – Subset by an attribute value
#           Change FILTER_COLUMN and FILTER_VALUE to match your data
# ─────────────────────────────────────────────────────────────
FILTER_COLUMN = "Pass"   # column to filter on
FILTER_VALUE  = 1          # keep only rows where column == this value

subset = gdf[gdf[FILTER_COLUMN] == FILTER_VALUE].copy()

print(f"── Subset: {FILTER_COLUMN} == '{FILTER_VALUE}' ──")
print(f"Rows after filter: {len(subset)}")


# ─────────────────────────────────────────────────────────────
# STEP 4 – Scatter plot: X coordinate vs. chosen attributes
# ─────────────────────────────────────────────────────────────

PLOT_ATTRIBUTE = ["depth", "fp01_1", "fp02_1", "fp03_1", "fp04_1", "fp05_1"]
ATTRIBUTE_COLORS = ["black", "green", "blue", "purple", "pink", "red"]

fig, ax = plt.subplots(figsize=(9, 5))

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
ax.set_title("X Coordinate vs Attributes")

ax.legend()
plt.tight_layout()
plt.show()