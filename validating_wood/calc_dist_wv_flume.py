import numpy as np
import rasterio
from rasterio.features import geometry_mask
import geopandas as gpd
from matplotlib import pyplot as plt

# -------------------------------------------------------------------
# Inputs
# -------------------------------------------------------------------

experiment_name = "20240808_exp1"

raster_path = (
    rf"C:\Users\josie\OneDrive - UCB-O365\Flume Data\processed_cart_data\{experiment_name}\{experiment_name}_difference.tif"
)

mask_path = (
    rf"C:\Users\josie\OneDrive - UCB-O365\Flume Data\processed_cart_data\{experiment_name}\{experiment_name}_true_wood.shp"
)

# -------------------------------------------------------------------
# Read mask polygons
# -------------------------------------------------------------------

gdf = gpd.read_file(mask_path)

# -------------------------------------------------------------------
# Read raster and apply mask
# -------------------------------------------------------------------

with rasterio.open(raster_path) as src:

    # Reproject polygons if necessary
    if gdf.crs != src.crs:
        print(f"Reprojecting mask from {gdf.crs} to {src.crs}")
        gdf = gdf.to_crs(src.crs)

    # Read raster
    depth = src.read(1)

    # Replace NoData with 0
    if src.nodata is not None:
        depth = np.where(depth == src.nodata, 0, depth)

    # Replace NaNs with 0
    depth = np.nan_to_num(depth, nan=0)

    # Create mask:
    # True inside polygons, False outside
    mask = geometry_mask(
        gdf.geometry,
        transform=src.transform,
        invert=True,              # makes inside polygons = True
        out_shape=depth.shape
    )

    # Keep values only inside polygons
    depth = np.where(mask, depth, 0)

    # Cell dimensions
    dx = src.transform.a
    dy = abs(src.transform.e)

    cell_area = dx * dy

    # Volume per downstream slice (one value per column)
    volumes = np.sum(depth * cell_area, axis=0)

    # Convert mm³ → m³
    volumes = volumes * 1e-9

    # Downstream distance corresponding to each column
    distances = np.arange(src.width) * dx

print(volumes)

# -------------------------------------------------------------------
# Plot
# -------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(distances, volumes, linewidth=2)

ax.set_xlabel("Distance downstream (mm)")
ax.set_ylabel("Volume in slice (m³)")
ax.set_title("Longitudinal distribution of stored volume")

ax.grid(True)

plt.tight_layout()
plt.show()