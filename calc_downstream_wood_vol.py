import numpy as np
import rasterio
from matplotlib import pyplot as plt

raster_path = r"C:\Users\josie\OneDrive - UCB-O365\Floodplain LW transport modelling\Data\Wood Data\test_for_integrating.tif"




with rasterio.open(raster_path) as src:

    # Read raster
    depth = src.read(1)

    # Replace NoData with 0
    if src.nodata is not None:
        depth = np.where(depth == src.nodata, 0, depth)

    # Replace any NaNs with 0
    depth = np.nan_to_num(depth, nan=0)

    # Cell dimensions
    dx = src.transform.a          # cell width
    dy = abs(src.transform.e)     # cell height

    cell_area = dx * dy

    # Volume in each downstream slice (one value per column)
    volumes = np.sum(depth * cell_area, axis=0)

    # Convert volume from mm to meters
    volumes = volumes*1e-9

    # Downstream distance corresponding to each column
    distances = np.arange(src.width) * dx

print(volumes)

# Plot
fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(distances, volumes, linewidth=2)

ax.set_xlabel("Distance downstream (mm)")
ax.set_ylabel("Volume in slice (m³)")
ax.set_title("Longitudinal distribution of stored volume")

ax.grid(True)

plt.tight_layout()
plt.show()