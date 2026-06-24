from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin
 
 
def read_ascii_header(f) -> dict:
    """Read the 6-line ESRI ASCII raster header and return a dict of lowercase keys."""
    header = {}
    for _ in range(6):
        parts = f.readline().split()
        if len(parts) != 2:
            raise ValueError(f"Malformed header line: {' '.join(parts)!r}")
        header[parts[0].lower()] = float(parts[1])
    return header
 
 
def parse_origin(header: dict, cellsize: float) -> tuple[float, float]:
    """
    Return the upper-left corner (x_ul, y_ul) of the raster in map units.
    Handles both xllcorner/yllcorner and xllcenter/yllcenter conventions.
    cellsize must already be in the output map units.
    """
    nrows = int(header["nrows"])
 
    if "xllcorner" in header and "yllcorner" in header:
        xll = header["xllcorner"]
        yll = header["yllcorner"]
        x_ul = xll
        y_ul = yll + nrows * cellsize
 
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
) -> None:
    """
    Convert an ESRI ASCII raster to a GeoTIFF.
 
    Parameters
    ----------
    input_ascii : str or Path
        Path to the input .asc file.
    output_tif : str or Path
        Path for the output .tif file.
    coord_scale : float
        Multiplier applied to all coordinates AND cell size from the header.
        Use this when the header units differ from your target map units.
        e.g. 1000.0 if the header is in km and you want metres.
    data_scale : float
        Multiplier applied to all valid (non-nodata) pixel values.
        e.g. 1000.0 if pixel values are in km and you want metres.
        Set to 1.0 to leave values unchanged.
    crs : str or None
        Optional CRS string, e.g. "EPSG:32615".
    """
    input_ascii = Path(input_ascii)
    output_tif = Path(output_tif)
 
    if not input_ascii.is_file():
        raise FileNotFoundError(f"Input file not found: {input_ascii}")
 
    output_tif.parent.mkdir(parents=True, exist_ok=True)
 
    # ------------------------------------------------------------------ #
    # 1. Read header + data                                                #
    # ------------------------------------------------------------------ #
    with open(input_ascii, "r") as f:
        header = read_ascii_header(f)
        data = np.loadtxt(f)
 
    ncols = int(header["ncols"])
    nrows = int(header["nrows"])
    nodata = header["nodata_value"]
 
    if data.shape != (nrows, ncols):
        raise ValueError(
            f"Data shape {data.shape} does not match header "
            f"(nrows={nrows}, ncols={ncols})."
        )
 
    # ------------------------------------------------------------------ #
    # 2. Scale pixel values                                                #
    # ------------------------------------------------------------------ #
    if data_scale != 1.0:
        data = data.astype(float)
        data[data != nodata] *= data_scale
 
    # ------------------------------------------------------------------ #
    # 3. Build the affine transform                                        #
    # ------------------------------------------------------------------ #
    # Scale coordinates and cell size from header units to map units
    cellsize = header["cellsize"] * coord_scale
    scaled_header = {
        **header,
        **{k: header[k] * coord_scale
           for k in ("xllcorner", "yllcorner", "xllcenter", "yllcenter")
           if k in header}
    }
 
    x_ul, y_ul = parse_origin(scaled_header, cellsize)
    transform = from_origin(x_ul, y_ul, cellsize, cellsize)
 
    # ------------------------------------------------------------------ #
    # 4. Write GeoTIFF                                                     #
    # ------------------------------------------------------------------ #
    with rasterio.open(
        output_tif,
        "w",
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
 
    x_lr = x_ul + ncols * cellsize
    y_lr = y_ul - nrows * cellsize
    print(f"Saved: {output_tif}")
    print(f"  CRS        : {crs}")
    print(f"  Cell size  : {cellsize} m  (header {header['cellsize']} × {coord_scale})")
    print(f"  Upper-left : ({x_ul:.2f}, {y_ul:.2f})")
    print(f"  Lower-right: ({x_lr:.2f}, {y_lr:.2f})")
    print(f"  Size       : {ncols} cols × {nrows} rows")
 
 
# --------------------------------------------------------------------------- #
# Example usage                                                                #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    n = "05"
    raster = "Depth"

    input_path = Path(
        f"C:/Users/josie/OneDrive - UCB-O365/Floodplain LW transport modelling/Model Setup and Calibration/true_shape_flume/test.gid/Rasters/Hydraulic/Depth____100.0010729600599.asc"
    )

    output_path = Path(f"C:/Users/josie/OneDrive - UCB-O365/Floodplain LW transport modelling/Model Setup and Calibration/true_shape_flume/updated_geometry00701").with_suffix(".tif")

    print(output_path)
 
    ascii_to_geotiff(
        input_ascii=input_path,
        output_tif=output_path,
        coord_scale=1000.0,  # header is in km → metres
        data_scale=1000.0,   # pixel values are in km depth → metres
        crs="EPSG:32615",
    )