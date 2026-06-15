import pandas as pd
import numpy as np

# =====================================================
# USER SETTINGS
# =====================================================

input_file = "C:/Users/josie/OneDrive - UCB-O365/Floodplain LW transport modelling/Data/Wood Data/dropping_schedule_uncongested.xlsx"
output_file = "C:/Users/josie/OneDrive - UCB-O365/Floodplain LW transport modelling/Data/Wood Data/dropping_schedule_uncongested_random.xlsx"

# Random variation ranges
# Values will be added to the existing values.

x_range = (-0.05, 0.05)          # meters
y_range = (-0.05, 0.05)          # meters
angle_range = (-0.02, 0.02)      # radians
density_range = (-50, 50)        # kg/m³

# Optional: make results reproducible
np.random.seed(42)

# =====================================================
# LOAD FILE
# =====================================================

df = pd.read_excel(input_file)

# =====================================================
# APPLY RANDOM PERTURBATIONS
# =====================================================

df["X [m]"] += np.random.uniform(
    x_range[0], x_range[1], len(df)
)

df["Y [m]"] += np.random.uniform(
    y_range[0], y_range[1], len(df)
)

df["Angle [rad]"] += np.random.uniform(
    angle_range[0], angle_range[1], len(df)
)

df["Density [kg/m3]"] += np.random.uniform(
    density_range[0], density_range[1], len(df)
)

# =====================================================
# SAVE
# =====================================================

df.to_excel(output_file, index=False)

print(f"Saved randomized file to: {output_file}")