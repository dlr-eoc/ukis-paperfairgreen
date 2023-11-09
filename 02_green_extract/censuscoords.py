from pathlib import Path

import geopandas as gp
import numpy as np
import pandas as pd
import rasterio as rio


def census_coords_from_raster(
    census_path: Path,
    raster_path: Path,
    window_size: int = 101,
) -> np.ndarray:
    census_pts = gp.read_file(census_path)
    with rio.open(str(raster_path)) as lcc:
        census_xs = [x for x in census_pts.geometry.x]
        census_ys = [y for y in census_pts.geometry.y]

        rows, cols = rio.transform.rowcol(
            lcc.transform,
            census_xs,
            census_ys,
        )

    xs = np.array(cols)
    ys = np.array(rows)
    xys = np.array((xs, ys)).T

    offset = (window_size - 1) // 2
    ul_xys = xys - offset

    ul_xys_pd = pd.DataFrame(ul_xys)

    ul_xys_pd.sort_values(by=[0, 1], inplace=True)

    return np.array(ul_xys_pd)
