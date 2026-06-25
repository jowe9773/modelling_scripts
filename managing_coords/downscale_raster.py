import rasterio
from rasterio.transform import Affine
import numpy as np

# ------------------------------------------------------------------
# INPUTS
# ------------------------------------------------------------------
input_tif = r"C:\Users\josie\OneDrive - UCB-O365\Floodplain LW transport modelling\testing_each_rfsd\0.25x_wood_volume_raster.tif"
output_tif = r"C:\Users\josie\OneDrive - UCB-O365\Floodplain LW transport modelling\testing_each_rfsd\0.25x_wood_volume_raster_scaled.tif"

scale_data = True
SCALE = 1000  # mm → m conversion factor

# ------------------------------------------------------------------
# OPEN INPUT
# ------------------------------------------------------------------
with rasterio.open(input_tif) as src:
    data = src.read().astype(np.float32)
    nodata = src.nodata
    old_transform = src.transform

    if scale_data == True:
        # Convert elevation values from mm → m
        if nodata is not None:
            valid = data != nodata
            data[valid] *= SCALE/100
        else:
            data *= SCALE/100

    # Scale pixel size by the same factor, keeping origin (0, 0) fixed.
    # old_transform.c and old_transform.f are the upper-left X and Y.
    # We scale those too so the origin stays at (0, 0) regardless of
    # what the input origin was.
    new_transform = Affine(
        old_transform.a * SCALE,   # scaled pixel width
        old_transform.b,           # rotation (usually 0)
        old_transform.c * SCALE,   # scaled upper-left X → stays 0 if input was 0
        old_transform.d,           # rotation (usually 0)
        old_transform.e * SCALE,   # scaled pixel height (negative for north-up)
        old_transform.f * SCALE    # scaled upper-left Y → stays 0 if input was 0
    )

    profile = src.profile.copy()
    profile.update(
        dtype="float32",
        transform=new_transform,
        nodata=nodata
    )

# ------------------------------------------------------------------
# WRITE OUTPUT
# ------------------------------------------------------------------
with rasterio.open(output_tif, "w", **profile) as dst:
    dst.write(data)

print("Done.")
print(f"New transform:\n{new_transform}")
print(f"Output written to: {output_tif}")