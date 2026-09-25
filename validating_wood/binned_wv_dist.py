import numpy as np
import rasterio
from rasterio.features import geometry_mask
import geopandas as gpd
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# Inputs
# -------------------------------------------------------------------

#FOR FLUME RESULTS
experiment_name = "20240614_exp1"
raster_path = (rf"C:\Users\josie\OneDrive - UCB-O365\Flume Data\processed_cart_data\{experiment_name}\{experiment_name}_difference.tif")
mask_path = (rf"C:\Users\josie\OneDrive - UCB-O365\Flume Data\processed_cart_data\{experiment_name}\{experiment_name}_true_wood.shp")

# FOR MODEL RESULTS
raster_path = r"C:\Users\josie\OneDrive - UCB-O365\Floodplain LW transport modelling\testing_each_rfsd\1.0x_wood_volume_raster_scaled.tif"
mask_path = r"C:\Users\josie\OneDrive - UCB-O365\Floodplain LW transport modelling\Data\3d Model Data\DEM based models\model_clip.shp"

# -------------------------------------------------------------------
# Bin size (in raster coordinate units)
# -------------------------------------------------------------------

bin_dx = 10   # longitudinal bin width  — set to None to use 1 raster cell
bin_dy = 10   # transverse  bin height  — set to None to use 1 raster cell

# -------------------------------------------------------------------
# Plot limits
# -------------------------------------------------------------------

pltxmin_long,  pltxmax_long  = 0,     10000
pltxmin_trans, pltxmax_trans = -2000, 2000

# -------------------------------------------------------------------
# Read wood polygons
# -------------------------------------------------------------------
gdf = gpd.read_file(mask_path)

# -------------------------------------------------------------------
# Read raster and apply mask
# -------------------------------------------------------------------

with rasterio.open(raster_path) as src:

    if gdf.crs != src.crs:
        print(f"Reprojecting mask from {gdf.crs} to {src.crs}")
        gdf = gdf.to_crs(src.crs)

    depth = src.read(1)

    if src.nodata is not None:
        depth = np.where(depth == src.nodata, 0, depth)

    depth = np.nan_to_num(depth, nan=0)

    mask = geometry_mask(
        gdf.geometry,
        transform=src.transform,
        invert=True,
        out_shape=depth.shape
    )

    depth = np.where(mask, depth, 0)

    dx = src.transform.a
    dy = abs(src.transform.e)
    cell_area = dx * dy

    cell_volumes = depth * cell_area * 1e-9   # mm³ -> m³

    bin_dx = bin_dx if bin_dx is not None else dx
    bin_dy = bin_dy if bin_dy is not None else dy

    cells_per_bin_x = max(1, round(bin_dx / dx))
    cells_per_bin_y = max(1, round(bin_dy / dy))

    # -------------------------------------------------------------------
    # Longitudinal distribution
    # -------------------------------------------------------------------

    per_col = np.sum(cell_volumes, axis=0)
    ncols = per_col.shape[0]
    n_bins_x = ncols // cells_per_bin_x

    per_col_trimmed = per_col[: n_bins_x * cells_per_bin_x]
    longitudinal_volume = per_col_trimmed.reshape(n_bins_x, cells_per_bin_x).sum(axis=1)

    x_left = src.bounds.left + dx / 2
    col_centres = x_left + np.arange(ncols) * dx
    col_centres_trimmed = col_centres[: n_bins_x * cells_per_bin_x]
    x_bin_centres = col_centres_trimmed.reshape(n_bins_x, cells_per_bin_x).mean(axis=1)

    # -------------------------------------------------------------------
    # Transverse distribution
    # -------------------------------------------------------------------

    per_row = np.sum(cell_volumes, axis=1)
    nrows = per_row.shape[0]
    n_bins_y = nrows // cells_per_bin_y

    per_row_trimmed = per_row[: n_bins_y * cells_per_bin_y]
    transverse_volume = per_row_trimmed.reshape(n_bins_y, cells_per_bin_y).sum(axis=1)

    y_top = src.bounds.top - dy / 2
    row_centres = y_top - np.arange(nrows) * dy
    row_centres_trimmed = row_centres[: n_bins_y * cells_per_bin_y]
    y_bin_centres = row_centres_trimmed.reshape(n_bins_y, cells_per_bin_y).mean(axis=1)

    transverse_volume = transverse_volume[::-1]
    y_bin_centres     = y_bin_centres[::-1]

# -------------------------------------------------------------------
# Compute CDFs
# -------------------------------------------------------------------

total_volume = np.sum(cell_volumes)

long_cdf  = np.cumsum(longitudinal_volume) / total_volume
trans_cdf = np.cumsum(transverse_volume)   / total_volume

# -------------------------------------------------------------------
# Print totals
# -------------------------------------------------------------------

print(f"Total stored volume = {total_volume:.6e} m³")
print(f"Longitudinal bins: {n_bins_x}  ({cells_per_bin_x} cells each, bin width  = {cells_per_bin_x * dx:.1f} units)")
print(f"Transverse bins:   {n_bins_y}  ({cells_per_bin_y} cells each, bin height = {cells_per_bin_y * dy:.1f} units)")

# -------------------------------------------------------------------
# Plot — 2 rows x 2 cols: left = PDF-style, right = CDF
# -------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)

ax_long_pdf, ax_long_cdf   = axes[0]
ax_tran_pdf, ax_tran_cdf   = axes[1]

bin_width_x = cells_per_bin_x * dx
bin_width_y = cells_per_bin_y * dy

# -- Longitudinal PDF --------------------------------------------------
ax_long_pdf.plot(x_bin_centres, longitudinal_volume, linewidth=2, marker='o', markersize=3)
ax_long_pdf.set_xlabel("Downstream distance")
ax_long_pdf.set_ylabel("Stored volume (m³)")
ax_long_pdf.set_title(f"Longitudinal distribution  [bin = {bin_width_x:.0f} units]")
ax_long_pdf.set_xlim(pltxmin_long, pltxmax_long)
ax_long_pdf.grid(True)

# -- Longitudinal CDF --------------------------------------------------
ax_long_cdf.plot(x_bin_centres, long_cdf, linewidth=2, color='C1')
ax_long_cdf.set_xlabel("Downstream distance")
ax_long_cdf.set_ylabel("Cumulative fraction of total volume")
ax_long_cdf.set_title(f"Longitudinal CDF  [bin = {bin_width_x:.0f} units]")
ax_long_cdf.set_xlim(pltxmin_long, pltxmax_long)
ax_long_cdf.set_ylim(0, 1)
ax_long_cdf.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax_long_cdf.grid(True)

# -- Transverse PDF ----------------------------------------------------
ax_tran_pdf.plot(y_bin_centres, transverse_volume, linewidth=2, marker='o', markersize=3)
ax_tran_pdf.set_xlabel("Transverse distance")
ax_tran_pdf.set_ylabel("Stored volume (m³)")
ax_tran_pdf.set_title(f"Transverse distribution  [bin = {bin_width_y:.0f} units]")
ax_tran_pdf.set_xlim(pltxmin_trans, pltxmax_trans)
ax_tran_pdf.grid(True)

# -- Transverse CDF ----------------------------------------------------
ax_tran_cdf.plot(y_bin_centres, trans_cdf, linewidth=2, color='C1')
ax_tran_cdf.set_xlabel("Transverse distance")
ax_tran_cdf.set_ylabel("Cumulative fraction of total volume")
ax_tran_cdf.set_title(f"Transverse CDF  [bin = {bin_width_y:.0f} units]")
ax_tran_cdf.set_xlim(pltxmin_trans, pltxmax_trans)
ax_tran_cdf.set_ylim(0, 1)
ax_tran_cdf.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax_tran_cdf.grid(True)

plt.suptitle("Wood volume spatial distribution", fontsize=13, fontweight='bold')
plt.show()