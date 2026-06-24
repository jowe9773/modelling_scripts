import geopandas as gpd
from shapely.affinity import affine_transform

# ------------------------------------------------------------------
# INPUTS
# ------------------------------------------------------------------
value = "2.0_dowel_locations"

input_shp  = f"C:/Users/josie/OneDrive - UCB-O365/Floodplain LW transport modelling/3D model of flume/dowel_hole_locatinos/{value}.shp"
output_shp = f"C:/Users/josie/OneDrive - UCB-O365/Floodplain LW transport modelling/3D model of flume/dowel_hole_locatinos/true_scale_{value}.shp"

# ------------------------------------------------------------------
# SCALING PARAMETERS  (mirrors the raster script)
# ------------------------------------------------------------------
SCALE = 0.001  # spatial units mm → m

# ------------------------------------------------------------------
# LOAD
# ------------------------------------------------------------------
gdf = gpd.read_file(input_shp)

# ------------------------------------------------------------------
# RESCALE COORDINATES, keeping (0, 0) fixed
#   x_new = x * SCALE + 0
#   y_new = y * SCALE + 0
# The offset terms (x_off, y_off) are 0, so if a vertex was at
# the origin it stays there — identical logic to the raster script.
# ------------------------------------------------------------------
matrix = [
    SCALE, 0.0,    # a, b  (x-scale, x-shear)
    0.0,   SCALE,  # d, e  (y-shear, y-scale)
    0.0,   0.0     # x_off, y_off → origin unchanged
]

gdf["geometry"] = gdf["geometry"].apply(
    lambda geom: affine_transform(geom, matrix) if geom is not None else geom
)

# ------------------------------------------------------------------
# WRITE OUTPUT
# ------------------------------------------------------------------
gdf.to_file(output_shp)

print("Done.")
print(f"Output written to: {output_shp}")