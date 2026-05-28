import numpy as np
import rasterio
from rasterio.transform import from_origin

# -------------------------------------------------
# Raster parameters
# -------------------------------------------------

xmin, xmax = 0, 9759
ymin, ymax = -2000, 2000

cell_size = 1

# Number of columns and rows
width = xmax - xmin
height = ymax - ymin

# -------------------------------------------------
# Create x-coordinate values
# -------------------------------------------------

# Pixel-center x coordinates
x_coords = np.arange(xmin + 0.5, xmax + 0.5, cell_size)

# Plane equation:
# z = x * 0.01 + 350
z_values = x_coords * -0.01 + 350

# Repeat across all rows
raster = np.tile(z_values, (height, 1)).astype(np.float32)

# -------------------------------------------------
# Define georeferencing transform
# -------------------------------------------------

transform = from_origin(
    xmin,   # west
    ymax,   # north
    cell_size,
    cell_size
)

# -------------------------------------------------
# Write GeoTIFF
# -------------------------------------------------
output_location = "C:/Users/josie/OneDrive - UCB-O365/Floodplain LW transport modelling/Model Setup and Calibration/QGIS/data/detrending_data"

output_file =  output_location + "/x_plane_raster.tif"

with rasterio.open(
    output_file,
    "w",
    driver="GTiff",
    height=height,
    width=width,
    count=1,
    dtype="float32",
    crs=32615,
    transform=transform,
) as dst:

    dst.write(raster, 1)

print(f"Saved: {output_file}")