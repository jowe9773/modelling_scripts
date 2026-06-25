import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio
from rasterio.transform import from_origin

# -------------------------------------------------------------------
# Inputs
# -------------------------------------------------------------------

database_path = r"C:\Users\josie\OneDrive - UCB-O365\Floodplain LW transport modelling\testing_each_rfsd\wood\0.25x_fp0075_ch01125.gid\Wood\Wood.rep"

output_raster_path = r"C:\Users\josie\OneDrive - UCB-O365\Floodplain LW transport modelling\testing_each_rfsd\0.25x_wood_volume_raster.tif"

# Coarse output raster cell size (m)
cell_size = 0.001

# Fine raster cell size (m) — used to resolve each log's footprint
fine_cell_size = 0.0001

# Flume domain bounds (m)
flume_xmin = 0.0
flume_xmax = 9.759
flume_ymin = -2.0
flume_ymax = 2.0

# -------------------------------------------------------------------
# Read and filter to final timestep
# -------------------------------------------------------------------

df = pd.read_csv(database_path, sep=';')
df.columns = df.columns.str.strip()

df['Time(s)'] = pd.to_numeric(df['Time(s)'], errors='coerce')
df = df.dropna(subset=['Time(s)'])

final_time = df['Time(s)'].max()
df = df[df['Time(s)'] == final_time].copy()

print(f"Final timestep = {final_time}")
print(f"Number of logs  = {len(df)}")

# -------------------------------------------------------------------
# Build coarse raster grid
# -------------------------------------------------------------------

nx_coarse = int(np.ceil((flume_xmax - flume_xmin) / cell_size))
ny_coarse = int(np.ceil((flume_ymax - flume_ymin) / cell_size))

volume_grid = np.zeros((ny_coarse, nx_coarse), dtype=np.float64)

print(f"Coarse grid : {nx_coarse} x {ny_coarse}  ({cell_size} m cells)")

# -------------------------------------------------------------------
# Process each log
# -------------------------------------------------------------------

for log_idx, row in df.iterrows():

    x0    = row['X']
    y0    = row['Y']
    L     = row['Length(m)']
    D     = row['Diameter(m)']
    R     = D / 2.0
    theta = row['Angle']   # radians

    V_log = np.pi * R**2 * L   # total cylindrical volume

    # ------------------------------------------------------------------
    # 1. Build a local fine raster covering this log's bounding box
    # ------------------------------------------------------------------
    half_x = 0.5 * (L * abs(np.cos(theta)) + D * abs(np.sin(theta)))
    half_y = 0.5 * (L * abs(np.sin(theta)) + D * abs(np.cos(theta)))

    bbox_xmin = x0 - half_x - fine_cell_size
    bbox_xmax = x0 + half_x + fine_cell_size
    bbox_ymin = y0 - half_y - fine_cell_size
    bbox_ymax = y0 + half_y + fine_cell_size

    nfx = int(np.ceil((bbox_xmax - bbox_xmin) / fine_cell_size))
    nfy = int(np.ceil((bbox_ymax - bbox_ymin) / fine_cell_size))

    fx_centres = bbox_xmin + (np.arange(nfx) + 0.5) * fine_cell_size
    fy_centres = bbox_ymin + (np.arange(nfy) + 0.5) * fine_cell_size

    gx, gy = np.meshgrid(fx_centres, fy_centres)

    # ------------------------------------------------------------------
    # 2. Log-local coordinates (s = along axis, w = perpendicular)
    # ------------------------------------------------------------------
    dx_g = gx - x0
    dy_g = gy - y0

    s_coord =  dx_g * np.cos(theta) + dy_g * np.sin(theta)
    w_coord = -dx_g * np.sin(theta) + dy_g * np.cos(theta)

    # ------------------------------------------------------------------
    # 3. Mask to log footprint
    # ------------------------------------------------------------------
    inside = (np.abs(s_coord) <= L / 2.0) & (np.abs(w_coord) <= R)

    if not inside.any():
        continue

    # ------------------------------------------------------------------
    # 4. Local wood thickness at each fine cell = chord length at that
    #    perpendicular offset: 2 * sqrt(R² - w²)
    #    No normalisation — each fine cell gets the true local depth (m).
    # ------------------------------------------------------------------
    w_vals = w_coord[inside]
    fine_volumes = 2.0 * np.sqrt(np.maximum(R**2 - w_vals**2, 0.0))

    # ------------------------------------------------------------------
    # 5. Map fine cells → coarse grid and accumulate
    # ------------------------------------------------------------------
    px = gx[inside]
    py = gy[inside]

    coarse_col = np.floor((px - flume_xmin) / cell_size).astype(int)
    coarse_row = np.floor((py - flume_ymin) / cell_size).astype(int)

    # Validate coarse_row BEFORE flipping — negative values would wrap
    # to the top of the grid after the flip, creating false artifacts.
    valid = (
        (coarse_col >= 0) & (coarse_col < nx_coarse) &
        (coarse_row >= 0) & (coarse_row < ny_coarse)
    )

    coarse_row_f = (ny_coarse - 1) - coarse_row[valid]

    np.add.at(
        volume_grid,
        (coarse_row_f, coarse_col[valid]),
        fine_volumes[valid]
    )

    print(f"  Log {log_idx:>4d}: V_log={V_log:.4e}  "
          f"assigned={fine_volumes[valid].sum():.4e}  "
          f"fine cells={inside.sum()}")

print(f"\nTotal accumulated thickness in raster : {volume_grid.sum():.4e} m")

# -------------------------------------------------------------------
# Save coarse raster as GeoTIFF
# -------------------------------------------------------------------

transform = from_origin(
    west=flume_xmin,
    north=flume_ymax,
    xsize=cell_size,
    ysize=cell_size
)

with rasterio.open(
    output_raster_path, 'w',
    driver='GTiff',
    height=ny_coarse, width=nx_coarse,
    count=1,
    dtype=volume_grid.dtype,
    crs='EPSG:32615',
    transform=transform,
    nodata=0.0,
    # Note: cell values are accumulated chord thickness (m)
) as dst:
    dst.write(volume_grid, 1)

print(f"Raster saved → {output_raster_path}")

# -------------------------------------------------------------------
# Plot
# -------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

ax = axes[0]
nonzero = volume_grid[volume_grid > 0]
im = ax.imshow(
    volume_grid,
    origin='lower',
    extent=[flume_xmin, flume_xmax, flume_ymin, flume_ymax],
    cmap='YlOrRd',
    norm=mcolors.LogNorm(vmin=nonzero.min() if nonzero.size else 1e-12,
                         vmax=volume_grid.max())
)
plt.colorbar(im, ax=ax, label='Accumulated wood thickness per cell (m)')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title(f'Wood volume raster — {cell_size} m cells  (t = {final_time:.1f} s)')
ax.set_aspect('equal')
ax.scatter(df['X'], df['Y'], s=4, c='white', alpha=0.6, zorder=3, label='Centroids')
ax.legend(fontsize=8)

ax2 = axes[1]
for _, row in df.iterrows():
    x0, y0 = row['X'], row['Y']
    L, theta = row['Length(m)'], row['Angle']
    x1 = x0 - (L/2)*np.cos(theta);  x2 = x0 + (L/2)*np.cos(theta)
    y1 = y0 - (L/2)*np.sin(theta);  y2 = y0 + (L/2)*np.sin(theta)
    ax2.plot([x1, x2], [y1, y2], linewidth=0.8, alpha=0.7)
ax2.scatter(df['X'], df['Y'], s=5)
ax2.set_xlabel('X (m)');  ax2.set_ylabel('Y (m)')
ax2.set_title('Reconstructed log positions')
ax2.set_xlim(flume_xmin, flume_xmax);  ax2.set_ylim(flume_ymin, flume_ymax)
ax2.set_aspect('equal');  ax2.grid(True)

plt.show()