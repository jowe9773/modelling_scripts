import numpy as np
import rasterio
from rasterio.features import geometry_mask
import geopandas as gpd
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# Inputs
# -------------------------------------------------------------------

#FOR FLUME RESULTS
experiment_name = "20240529_exp2"
raster_path = (rf"C:\Users\josie\OneDrive - UCB-O365\Flume Data\processed_cart_data\{experiment_name}\{experiment_name}_difference.tif")
mask_path = (rf"C:\Users\josie\OneDrive - UCB-O365\Flume Data\processed_cart_data\{experiment_name}\{experiment_name}_true_wood.shp")

# FOR MODEL RESULTS
raster_path = r"C:\Users\josie\OneDrive - UCB-O365\Floodplain LW transport modelling\testing_each_rfsd\0.25x_wood_volume_raster_scaled.tif"
mask_path = r"C:\Users\josie\OneDrive - UCB-O365\Floodplain LW transport modelling\Data\3d Model Data\DEM based models\model_clip.shp"

# -------------------------------------------------------------------
# Read wood polygons
# -------------------------------------------------------------------
gdf = gpd.read_file(mask_path)

# -------------------------------------------------------------------
# Read raster and apply mask
# -------------------------------------------------------------------

with rasterio.open(raster_path) as src:
    # ---------------------------------------------------------------
    # Reproject polygons if needed
    # ---------------------------------------------------------------

    if gdf.crs != src.crs:
        print(f"Reprojecting mask from {gdf.crs} to {src.crs}")
        gdf = gdf.to_crs(src.crs)

    # ---------------------------------------------------------------
    # Read raster
    # ---------------------------------------------------------------
    depth = src.read(1)

    # Replace NoData with 0
    if src.nodata is not None:
        depth = np.where(depth == src.nodata, 0, depth)

    # Replace NaNs with 0
    depth = np.nan_to_num(depth, nan=0)

    # ---------------------------------------------------------------
    # Apply wood mask
    # ---------------------------------------------------------------

    mask = geometry_mask(
        gdf.geometry,
        transform=src.transform,
        invert=True,          # True inside polygons
        out_shape=depth.shape
    )

    depth = np.where(mask, depth, 0)

    # ---------------------------------------------------------------
    # Cell dimensions
    # ---------------------------------------------------------------

    dx = src.transform.a
    dy = abs(src.transform.e)

    cell_area = dx * dy

    # ---------------------------------------------------------------
    # Convert raster depth to cell volume
    # ---------------------------------------------------------------

    cell_volumes = depth * cell_area

    # Convert mm³ -> m³
    cell_volumes *= 1e-9

    # ---------------------------------------------------------------
    # Longitudinal distribution
    # (sum along rows -> one value per column)
    # ---------------------------------------------------------------

    longitudinal_volume = np.sum(cell_volumes, axis=0)

    x_coords = (
        src.bounds.left
        + dx / 2
        + np.arange(src.width) * dx
    )

    # ---------------------------------------------------------------
    # Transverse distribution
    # (sum along columns -> one value per row)
    # ---------------------------------------------------------------

    transverse_volume = np.sum(cell_volumes, axis=1)

    y_coords = (
        src.bounds.top
        - dy / 2
        - np.arange(src.height) * dy
    )

    # Flip arrays so Y increases upward
    transverse_volume = transverse_volume[::-1]
    y_coords = y_coords[::-1]

# -------------------------------------------------------------------
# Print totals for checking
# -------------------------------------------------------------------

print(f"Total stored volume = {np.sum(cell_volumes):.6e} m³")

# -------------------------------------------------------------------
# Plot distributions
# -------------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(10, 8),
    constrained_layout=True
)

pltxmin_long= 0
pltxmax_long = 10000
pltxmin_trans = -2000
pltxmax_trans = 2000

pltymin_long= 0
pltymax_long = 5 *10**-5
pltymin_trans = 0
pltymax_trans = 8 *10**-5


# ---------------------------------------------------------------
# Longitudinal distribution
# ---------------------------------------------------------------

ax1.plot(x_coords, longitudinal_volume, linewidth=2)

ax1.set_xlabel("Downstream distance")
ax1.set_ylabel("Stored volume (m³)")
ax1.set_title("Longitudinal distribution of stored wood volume")
ax1.set_xlim(pltxmin_long, pltxmax_long)
ax1.set_ylim(pltymin_long, pltymax_long)

ax1.grid(True)

# ---------------------------------------------------------------
# Transverse distribution
# ---------------------------------------------------------------

ax2.plot(y_coords, transverse_volume, linewidth=2)

ax2.set_xlabel("Transverse distance")
ax2.set_ylabel("Stored volume (m³)")
ax2.set_title("Transverse distribution of stored wood volume")
ax2.set_xlim(pltxmin_trans, pltxmax_trans)
ax2.set_ylim(pltymin_trans, pltymax_trans)

ax2.grid(True)

plt.show()