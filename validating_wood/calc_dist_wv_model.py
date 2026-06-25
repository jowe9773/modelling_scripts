import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# Inputs
# -------------------------------------------------------------------

database_path = r"C:\Users\josie\OneDrive - UCB-O365\Floodplain LW transport modelling\testing_each_rfsd\wood\1.0x_fp0075_ch01125.gid\Wood\Wood.rep"

# Width of downstream bins (m)
dx = 0.001
dy = 0.001

# Number of points used to discretize each log
n_segments = 500

# -------------------------------------------------------------------
# Read database
# -------------------------------------------------------------------

df = pd.read_csv(database_path, sep=';')

# Remove whitespace from column names
df.columns = df.columns.str.strip()


print(len(df.columns))

with open(database_path, 'r') as f:
    print(len(f.readline().strip().split(';')))
    print(len(f.readline().strip().split(';')))

# -------------------------------------------------------------------
# Keep only final timestep
# -------------------------------------------------------------------

df['Time(s)'] = pd.to_numeric(df['Time(s)'], errors='coerce')
df = df.dropna(subset=['Time(s)'])

final_time = df['Time(s)'].max()

df = df[df['Time(s)'] == final_time].copy()




print(f"Final timestep = {final_time}")
print(f"Number of logs = {len(df)}")

# -------------------------------------------------------------------
# Define downstream domain
# -------------------------------------------------------------------

xmin = (
    df['X']
    - 0.5 * df['Length(m)']
).min()

xmax = (
    df['X']
    + 0.5 * df['Length(m)']
).max()

x_edges = np.arange(xmin, xmax + dx, dx)
x_distances = x_edges[:-1] + dx / 2

x_stored_volume = np.zeros(len(x_distances))

ymin = df['Y'].min() - df['Length(m)'].max()/2
ymax = df['Y'].max() + df['Length(m)'].max()/2

y_edges = np.arange(ymin, ymax + dy, dy)
y_distances = y_edges[:-1] + dy/2

stored_volume_y = np.zeros(len(y_distances))

# -------------------------------------------------------------------
# Process each log
# -------------------------------------------------------------------

for _, row in df.iterrows():

    x0 = row['X']
    y0 = row['Y']

    L = row['Length(m)']
    D = row['Diameter(m)']
    theta = row['Angle']

    # Total volume of the log
    Vlog = np.pi * (D / 2)**2 * L

    # Coordinates along the log centreline
    s = np.linspace(-L/2, L/2, n_segments)

    xs = x0 + s * np.cos(theta)
    ys = y0 + s * np.sin(theta)

    # Volume represented by each segment
    dV = Vlog / n_segments

    # Determine transverse bin for each segment
    y_inds = np.floor((ys - ymin) / dy).astype(int)

    valid = (y_inds >= 0) & (y_inds < len(stored_volume_y))

    np.add.at(stored_volume_y, y_inds[valid], dV)

    # Determine downstream bin for each segment
    inds = np.floor((xs - xmin) / dx).astype(int)

    valid = (inds >= 0) & (inds < len(x_stored_volume))

    np.add.at(x_stored_volume, inds[valid], dV)

# -------------------------------------------------------------------
# Setup Plot
# -------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(10, 8),
    constrained_layout=True
)

pltxmin_long= 0
pltxmax_long = 10
pltxmin_trans = -2
pltxmax_trans = 2

pltymin_long= 0
pltymax_long = 5 *10**-5
pltymin_trans = 0
pltymax_trans = 8 *10**-5


# ---------------------------------------------------------------
# Plot longitudinal distribution
# ---------------------------------------------------------------

ax1.plot(x_distances, x_stored_volume, linewidth=2)

ax1.set_xlabel("Downstream distance")
ax1.set_ylabel("Stored volume (m³)")
ax1.set_title("Longitudinal distribution of stored wood volume")
ax1.set_xlim(pltxmin_long, pltxmax_long)
ax1.set_ylim(pltymin_long, pltymax_long)

ax1.grid(True)

# ---------------------------------------------------------------
# Plot transverse distribution
# ---------------------------------------------------------------

ax2.plot(y_distances, stored_volume_y, linewidth=2)

ax2.set_xlabel("Transverse distance")
ax2.set_ylabel("Stored volume (m³)")
ax2.set_title("Transverse distribution of stored wood volume")
ax2.set_xlim(pltxmin_trans, pltxmax_trans)
ax2.set_ylim(pltymin_trans, pltymax_trans)

ax2.grid(True)


# -------------------------------------------------------------------
# Plot reconstructed logs in X-Y space
# -------------------------------------------------------------------

fig3, ax3 = plt.subplots(figsize=(12, 4))

for _, row in df.iterrows():

    x0 = row['X']
    y0 = row['Y']

    L = row['Length(m)']
    theta = row['Angle']

    # Uncomment if angles are actually degrees
    # theta = np.deg2rad(theta)

    x1 = x0 - (L / 2) * np.cos(theta)
    x2 = x0 + (L / 2) * np.cos(theta)

    y1 = y0 - (L / 2) * np.sin(theta)
    y2 = y0 + (L / 2) * np.sin(theta)

    ax3.plot([x1, x2], [y1, y2], linewidth=1)

# Optional: plot log centroids
ax3.scatter(df['X'], df['Y'], s=5)

ax3.set_xlabel('X (m)')
ax3.set_ylabel('Y (m)')
ax3.set_title('Reconstructed wood positions at final timestep')
ax3.set_xlim(pltxmin_long, pltxmax_long)
ax3.set_ylim(pltxmin_trans, pltxmax_trans)

# Make distances look realistic
ax3.set_aspect('equal')

ax3.grid(True)

plt.tight_layout()

plt.show()