import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree
from pathlib import Path

# --- Configuration ---
INPUT_FILE  = "C:/Users/josie/OneDrive - UCB-O365/Floodplain LW transport modelling/Model Setup and Calibration/DEM_based_model_with_headboxes/results/model_results.gpkg"
OUTPUT_DIR  = "C:/Users/josie/OneDrive - UCB-O365/Floodplain LW transport modelling/Model Setup and Calibration/DEM_based_model_with_headboxes/results"
PASS_COL    = "Pass"
WINDOW_M    = 100.0
OUTLIER_THR = 20
# ---------------------

gdf = gpd.read_file(INPUT_FILE)

if gdf.crs.is_geographic:
    gdf = gdf.to_crs(gdf.estimate_utm_crs())

def local_medians(coords, values, window_m):
    tree = cKDTree(coords)
    medians = []
    for pt in coords:
        neighbours = tree.query_ball_point(pt, r=window_m)
        valid = values[neighbours][~np.isnan(values[neighbours])]
        medians.append(np.median(valid) if len(valid) > 0 else np.nan)
    return np.array(medians)

gdf["depth_avg"] = np.nan
outlier_indices = []  # collect outlier row indices across all passes

for pass_id, group in gdf.groupby(PASS_COL):
    coords = np.column_stack([group.geometry.x, group.geometry.y])
    values = group["water_dept"].values

    # Pass 1: preliminary median to identify outliers
    prelim_medians = local_medians(coords, values, WINDOW_M)
    is_outlier = np.abs(values - prelim_medians) > OUTLIER_THR

    # Record the actual GeoDataFrame indices of outlier rows
    outlier_indices.extend(group.index[is_outlier].tolist())

    # Pass 2: recompute median with outliers masked
    clean_values = values.copy()
    clean_values[is_outlier] = np.nan
    final_medians = local_medians(coords, clean_values, WINDOW_M)

    gdf.loc[group.index, "depth_avg"] = final_medians

    n_out = is_outlier.sum()
    print(f"Pass {pass_id}: {n_out} outlier(s) removed ({n_out/len(values)*100:.1f}%)")

# Drop outlier rows before saving
gdf_clean = gdf.drop(index=outlier_indices)
print(f"\nTotal rows removed: {len(outlier_indices)}")
print(f"Rows remaining: {len(gdf_clean)}")

output_path = Path(OUTPUT_DIR) / (Path(INPUT_FILE).stem + "_averaged.gpkg")
gdf_clean.to_file(output_path, driver="GPKG")
print(f"Saved to {output_path}")