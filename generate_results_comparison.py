import os
import glob
import re
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin
import geopandas as gpd
from scipy.spatial import cKDTree


# --- INPUTS ---
points_shapefile = r"C:/Users/josie/OneDrive - UCB-O365/Flume Data/processed_cart_data/20240606_exp1/20240606_exp1_nowood(MAS).CSV/20240606_exp1_nowood(MAS).shp"
parent_directory = r"C:/Users/josie/OneDrive - UCB-O365/Floodplain LW transport modelling/Model Setup and Calibration/DEM_based_model_with_headboxes"
output_directory = r"C:/Users/josie/OneDrive - UCB-O365/Floodplain LW transport modelling/Model Setup and Calibration/DEM_based_model_with_headboxes/results"

CRS = "EPSG:32615"
COORD_SCALE = 1000.0
DATA_SCALE  = 1000.0

PASS_COL    = "Pass"
WINDOW_M    = 20
OUTLIER_THR = 10


# ------------------------------------------------------------------ #
# ASC -> GeoTIFF FUNCTIONS                                            #
# ------------------------------------------------------------------ #

def read_ascii_header(f) -> dict:
    header = {}
    for _ in range(6):
        parts = f.readline().split()
        if len(parts) != 2:
            raise ValueError(f"Malformed header line: {' '.join(parts)!r}")
        header[parts[0].lower()] = float(parts[1])
    return header


def parse_origin(header: dict, cellsize: float) -> tuple[float, float]:
    nrows = int(header["nrows"])
    if "xllcorner" in header and "yllcorner" in header:
        x_ul = header["xllcorner"]
        y_ul = header["yllcorner"] + nrows * cellsize
    elif "xllcenter" in header and "yllcenter" in header:
        xll = header["xllcenter"]
        yll = header["yllcenter"]
        x_ul = xll - 0.5 * cellsize
        y_ul = yll + (nrows - 0.5) * cellsize
    else:
        raise ValueError(
            "ASCII header must contain either 'xllcorner'/'yllcorner' "
            "or 'xllcenter'/'yllcenter'."
        )
    return x_ul, y_ul


def ascii_to_geotiff(
    input_ascii: str | Path,
    output_tif: str | Path,
    coord_scale: float = 1.0,
    data_scale: float = 1.0,
    crs: str | None = None,
) -> Path:
    input_ascii = Path(input_ascii)
    output_tif  = Path(output_tif)

    if not input_ascii.is_file():
        raise FileNotFoundError(f"Input file not found: {input_ascii}")

    output_tif.parent.mkdir(parents=True, exist_ok=True)

    with open(input_ascii, "r") as f:
        header = read_ascii_header(f)
        data   = np.loadtxt(f)
        if data.ndim == 1:
            data = data.reshape(1, -1)

    ncols  = int(header["ncols"])
    nrows  = int(header["nrows"])
    nodata = header["nodata_value"]

    if data.shape != (nrows, ncols):
        raise ValueError(
            f"Data shape {data.shape} does not match header "
            f"(nrows={nrows}, ncols={ncols})."
        )

    if data_scale != 1.0:
        data = data.astype(float)
        data[data != nodata] *= data_scale

    cellsize = header["cellsize"] * coord_scale
    scaled_header = {
        **header,
        **{k: header[k] * coord_scale
           for k in ("xllcorner", "yllcorner", "xllcenter", "yllcenter")
           if k in header}
    }

    x_ul, y_ul = parse_origin(scaled_header, cellsize)
    transform  = from_origin(x_ul, y_ul, cellsize, cellsize)

    with rasterio.open(
        output_tif, "w",
        driver="GTiff",
        height=nrows,
        width=ncols,
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)

    print(f"  Saved GeoTIFF: {output_tif}")
    return output_tif


def sample_raster_at_points(tif_path: Path, gdf: gpd.GeoDataFrame, nodata: float) -> list:
    coords = []
    for geom in gdf.geometry:
        if geom.geom_type == "MultiPoint":
            coords.append((geom.geoms[0].x, geom.geoms[0].y))
        else:
            coords.append((geom.x, geom.y))

    with rasterio.open(tif_path) as src:
        sampled = [val[0] for val in src.sample(coords)]

    return [np.nan if v == nodata else v for v in sampled]


# ------------------------------------------------------------------ #
# SPATIAL MEDIAN / OUTLIER FUNCTIONS                                  #
# ------------------------------------------------------------------ #

def local_medians(coords, values, window_m):
    tree    = cKDTree(coords)
    medians = []
    for pt in coords:
        neighbours = tree.query_ball_point(pt, r=window_m)
        valid = values[neighbours][~np.isnan(values[neighbours])]
        medians.append(np.median(valid) if len(valid) > 0 else np.nan)
    return np.array(medians)


def compute_depth_avg(gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, list]:
    """
    Compute per-pass outlier-cleaned spatial median of water_dept.
    Returns the gdf with a 'depth_avg' column added, and a list of outlier indices.
    """
    if gdf.crs.is_geographic:
        gdf = gdf.to_crs(gdf.estimate_utm_crs())

    gdf["depth_avg"] = np.nan
    outlier_indices  = []

    for pass_id, group in gdf.groupby(PASS_COL):
        coords = np.column_stack([group.geometry.x, group.geometry.y])
        values = group["water_dept"].values

        prelim_medians = local_medians(coords, values, WINDOW_M)
        is_outlier     = np.abs(values - prelim_medians) > OUTLIER_THR

        outlier_indices.extend(group.index[is_outlier].tolist())

        clean_values             = values.copy()
        clean_values[is_outlier] = np.nan
        final_medians            = local_medians(coords, clean_values, WINDOW_M)

        gdf.loc[group.index, "depth_avg"] = final_medians

        n_out = is_outlier.sum()
        print(f"  Pass {pass_id}: {n_out} outlier(s) removed ({n_out/len(values)*100:.1f}%)")

    return gdf, outlier_indices


# ------------------------------------------------------------------ #
# MAIN                                                                 #
# ------------------------------------------------------------------ #

# --- LOAD SHAPEFILE ---
print("Loading points shapefile...")
gdf = gpd.read_file(points_shapefile)
print(f"  Loaded {len(gdf)} points | CRS: {gdf.crs}\n")

# --- COMPUTE SMOOTHED OBSERVED DEPTHS ---
print("Computing spatially smoothed observed depths (depth_avg)...")
gdf, outlier_indices = compute_depth_avg(gdf)
gdf_clean = gdf.drop(index=outlier_indices)
print(f"  Total outliers removed: {len(outlier_indices)} | Rows remaining: {len(gdf_clean)}\n")

# Drop duplicate point locations, keeping first occurrence
gdf_clean["_x"] = gdf_clean.geometry.x
gdf_clean["_y"] = gdf_clean.geometry.y
gdf_clean = gdf_clean.drop_duplicates(subset=["_x", "_y"]).drop(columns=["_x", "_y"])
print(f"  After deduplication: {len(gdf_clean)} rows remaining\n")

# --- FILE DISCOVERY & PROCESSING ---
for model_dir in sorted(os.listdir(parent_directory)):
    model_dir_path = os.path.join(parent_directory, model_dir)

    if not os.path.isdir(model_dir_path):
        continue

    col_name = Path(model_dir_path).stem

    hydraulic_path = os.path.join(model_dir_path, "Rasters", "Hydraulic")
    if not os.path.isdir(hydraulic_path):
        print(f"  [{model_dir}] WARNING: Hydraulic folder not found, skipping.")
        continue

    depth_files = glob.glob(os.path.join(hydraulic_path, "Depth*.asc"))
    if len(depth_files) == 1:
        depth_file = depth_files[0]
    elif len(depth_files) == 0:
        print(f"  [{model_dir}] WARNING: No depth file found, skipping.")
        continue
    else:
        print(f"  [{model_dir}] WARNING: Multiple depth files found, skipping.")
        continue

    output_tif = Path(output_directory) / f"{col_name}.tif"
    tif_path   = ascii_to_geotiff(
        input_ascii=depth_file,
        output_tif=output_tif,
        coord_scale=COORD_SCALE,
        data_scale=DATA_SCALE,
        crs=CRS,
    )

    nodata = float(re.search(r'nodata_value\s+([\d\-\.]+)',
                   open(depth_file).read(), re.IGNORECASE).group(1))
    gdf_clean[col_name] = sample_raster_at_points(tif_path, gdf_clean, nodata)

    # Error = modelled depth - smoothed observed depth
    err_col = f"err_{col_name}"
    gdf_clean[err_col] = gdf_clean[col_name] - gdf_clean["depth_avg"]
    print(f"  Calculated error column: {err_col}\n")

# --- SAVE OUTPUT ---
output_gpkg = Path(output_directory) / "model_results.gpkg"
gdf_clean.to_file(output_gpkg, driver="GPKG")
print(f"\nDone! Output saved to: {output_gpkg}")
print(f"Columns: {list(gdf_clean.columns)}")