from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.features import geometry_mask
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter


# ------------------------------------------------------------------
# USER INPUTS
# ------------------------------------------------------------------
loc = "rl"
POINTS_FILE = f"C:/Users/josie/OneDrive - UCB-O365/Floodplain LW transport modelling/Model Setup and Calibration/data/interpolation_points/channel_points.shp"
POLYGON_FILE = f"C:/Users/josie/OneDrive - UCB-O365/Floodplain LW transport modelling/Model Setup and Calibration/data/flume_geometry/channel_bottom_boundary.shp"

VALUE_FIELD = "SAMPLE_1"

OUTPUT_RASTER = f"C:/Users/josie/OneDrive - UCB-O365/Floodplain LW transport modelling/Model Setup and Calibration/data/elevation_rasters/channel_bottom_surface_elev_smoothed.tif"

CELL_SIZE = 1         # raster resolution
POWER = 2             # IDW power
SEARCH_RADIUS = None  # e.g. 5.0, or None for unlimited
N_NEIGHBOURS = 12

# ------------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------------

points = gpd.read_file(POINTS_FILE)
polygon = gpd.read_file(POLYGON_FILE)

# Ensure same CRS
polygon = polygon.to_crs(points.crs)

# ------------------------------------------------------------------
# EXTRACT POINT DATA
# ------------------------------------------------------------------

x = points.geometry.x.values
y = points.geometry.y.values
z = points[VALUE_FIELD].values

coords = np.column_stack([x, y])

# ------------------------------------------------------------------
# CREATE OUTPUT GRID
# ------------------------------------------------------------------

xmin, ymin, xmax, ymax = polygon.total_bounds

cols = int(np.ceil((xmax - xmin) / CELL_SIZE))
rows = int(np.ceil((ymax - ymin) / CELL_SIZE))

x_grid = xmin + (np.arange(cols) + 0.5) * CELL_SIZE
y_grid = ymax - (np.arange(rows) + 0.5) * CELL_SIZE

xx, yy = np.meshgrid(x_grid, y_grid)

grid_points = np.column_stack([xx.ravel(), yy.ravel()])

print(f"Rows: {rows:,}")
print(f"Cols: {cols:,}")
print(f"Cells: {rows * cols:,}")

# ------------------------------------------------------------------
# BUILD KD-TREE
# ------------------------------------------------------------------

tree = cKDTree(coords)

distances, indices = tree.query(
    grid_points,
    k=N_NEIGHBOURS
)

# Handle single-neighbour case
if N_NEIGHBOURS == 1:
    distances = distances[:, np.newaxis]
    indices = indices[:, np.newaxis]

# ------------------------------------------------------------------
# OPTIONAL SEARCH RADIUS
# ------------------------------------------------------------------

if SEARCH_RADIUS is not None:
    distances[distances > SEARCH_RADIUS] = np.nan

# ------------------------------------------------------------------
# IDW INTERPOLATION
# ------------------------------------------------------------------

# Exact point locations
exact_match = distances[:, 0] == 0

weights = np.where(
    distances > 0,
    1.0 / (distances ** POWER),
    0
)

weight_sum = np.nansum(weights, axis=1)

idw_values = (
    np.nansum(weights * z[indices], axis=1)
    / weight_sum
)

# Use exact values where grid falls on a point
idw_values[exact_match] = z[indices[exact_match, 0]]

grid = idw_values.reshape(rows, cols)

# ------------------------------------------------------------------
# MASK TO POLYGON
# ------------------------------------------------------------------

transform = from_origin(
    xmin,
    ymax,
    CELL_SIZE,
    CELL_SIZE
)

mask = geometry_mask(
    polygon.geometry,
    transform=transform,
    invert=True,
    out_shape=(rows, cols)
)

grid[~mask] = np.nan

SIGMA = 2.0

valid = np.isfinite(grid)

# Replace NaNs with 0
data = np.where(valid, grid, 0)

# Blur values
blurred_data = gaussian_filter(data, sigma=SIGMA)

# Blur weights
weights = gaussian_filter(valid.astype(float), sigma=SIGMA)

# Normalize
grid_smoothed = blurred_data / weights

# Restore outside polygon
grid_smoothed[~valid] = np.nan

# ------------------------------------------------------------------
# SAVE GEOTIFF
# ------------------------------------------------------------------

with rasterio.open(
    OUTPUT_RASTER,
    "w",
    driver="GTiff",
    height=rows,
    width=cols,
    count=1,
    dtype="float32",
    crs=points.crs,
    transform=transform,
    nodata=np.nan,
) as dst:
    dst.write(grid.astype("float32"), 1)

print(f"Saved: {OUTPUT_RASTER}")