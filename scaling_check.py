import geopandas as gpd
import pandas as pd
from pathlib import Path

# -----------------------------
# INPUT SHAPEFILE
# -----------------------------
# Replace with your shapefile path
shapefile_path = r"C:/Users/josie/OneDrive - UCB-O365/Masters Work/Masters Data and Analyses/20250429_cleaned_cart_data/20240603_exp1/20240603_exp1_nowood(MAS).CSV/20240603_exp1_nowood(MAS).shp"
# -----------------------------
# LOAD SHAPEFILE
# -----------------------------
gdf = gpd.read_file(shapefile_path)

# -----------------------------
# DEFINE CONDITIONS
# -----------------------------
conditions = {
    "water_dept > 5": gdf["water_dept"] > 5,
    "froude < 1": gdf["froude"] < 1,
    "Re > 5000": gdf["Re"] > 5000,
    "Weber > 11": gdf["Weber"] > 11,
}

# -----------------------------
# CALCULATE FRACTIONS
# -----------------------------
total_points = len(gdf)

results = []

for name, condition in conditions.items():
    count = condition.sum()
    fraction = count / total_points

    results.append({
        "Condition": name,
        "Count": count,
        "Total Points": total_points,
        "Fraction": fraction
    })

# -----------------------------
# DISPLAY RESULTS
# -----------------------------
results_df = pd.DataFrame(results)

print("/nCondition Summary:/n")
print(results_df)

# Optional: save results to CSV
output_csv = Path(shapefile_path).with_name("condition_summary.csv")
results_df.to_csv(output_csv, index=False)

print(f"/nResults saved to:/n{output_csv}")