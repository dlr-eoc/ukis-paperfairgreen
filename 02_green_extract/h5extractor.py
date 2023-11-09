"""Extract image and OSM data into h5 file."""
from multiprocessing import Pool, cpu_count
from os import remove
from os.path import exists, getsize
from time import sleep, time

import numpy as np
from censuscoords import census_coords_from_raster
from GreenluDB import GreenluDBhandler
from h5py import File as h5_file
from numpy import array
from numpy import max as np_max
from numpy import mean, median
from numpy import min as np_min
from numpy import stack
from rasterio import open as rio_open
from rasterio import windows as rio_windows
from tqdm import tqdm

# some globals
CENSUS = True
# data base configuration
DB_CONFIG_FILE = "database.ini"
# raster files -- link all files to ./data!
LCC_FILE = "data/landcover_DE_2018_v1_3035.tif"
R_FILE = "data/b4_3035_uc.tif"
G_FILE = "data/b3_3035_uc.tif"
B_FILE = "data/b2_3035_uc.tif"
NDVI_FILE = "data/ndvi_p50_3035_uc.tif"
EUA_PROX_FILE = "data/eua_urban_prox.tif"
CENSUS_FILE = "data/census.gpkg"
# h5 ouput file
H5_FILE = "output/extract_v8_census.h5" if CENSUS else "output/extracts_v8.h5"

# Size of each tile
TILE_SIZE_X = 101  # 64
TILE_SIZE_Y = 101  # 64
# offset between tiles, allows for overlay
TILE_OFFSET_X = TILE_SIZE_X
TILE_OFFSET_Y = TILE_SIZE_Y
# chunk sizes for iterators, 9 is arbitrary but empirically works well
CHUNK_SIZE_X = 9 * TILE_OFFSET_X
CHUNK_SIZE_Y = 9 * TILE_OFFSET_Y


def main():
    t_start = time()

    if CENSUS:
        print("Create census iterable")
        census_ul_array = census_coords_from_raster(CENSUS_FILE, LCC_FILE)
        chunk_size = 10
        census_ul_chunks = np.split(
            census_ul_array, np.arange(chunk_size, len(census_ul_array), chunk_size)
        )
        with Pool(cpu_count()) as pool:
            _ = [
                x
                for x in tqdm(
                    pool.imap_unordered(extract_census_chunks, census_ul_chunks),
                    total=len(census_ul_chunks),
                )
            ]
        return 0

    # create iterable with chunk_ul_[x|y]
    with rio_open(LCC_FILE) as f:
        lcc_shape = f.shape
        chunk_ul_iterable = (
            (ulx, uly)
            for ulx in range(0, lcc_shape[1] - CHUNK_SIZE_X, CHUNK_SIZE_X)
            for uly in range(0, lcc_shape[0] - CHUNK_SIZE_Y, CHUNK_SIZE_Y)
        )
        num_of_chunks = sum(
            1
            for x in range(0, lcc_shape[1] - CHUNK_SIZE_X, CHUNK_SIZE_X)
            for y in range(0, lcc_shape[0] - CHUNK_SIZE_Y, CHUNK_SIZE_Y)
        )

    # iterate over chunks
    with Pool(cpu_count()) as pool:
        _ = [
            x
            for x in tqdm(
                pool.imap_unordered(mp_extract_chunk, chunk_ul_iterable), total=num_of_chunks
            )
        ]

    print(f"Finished sucessfully\nRuntime {time() - t_start:.2f} seconds")
    return 0


def chunk_to_h5(chunk_dict, filename):
    # convert dict to individual arrays
    try:
        img_arr = array([tile["img"] for tile in chunk_dict.values() if tile])
        osm_arr = stack([list(tile["osm"].values()) for tile in chunk_dict.values() if tile])
        eua_arr = stack([list(tile["eua"].values()) for tile in chunk_dict.values() if tile])
        urb_arr = stack([tile["urb"] for tile in chunk_dict.values() if tile])
        ulc_arr = stack([coords for coords, tile in chunk_dict.items() if tile])
    except ValueError:
        return None

    # repeat with sleep when file is blocked
    kwargs = dict(compression="gzip", compression_opts=4)
    while True:
        try:
            with h5_file(filename, "a") as f:
                if "/img" not in f:
                    # set up datasets if they don't exist
                    f.create_dataset(
                        "img", data=img_arr, maxshape=(None, *img_arr.shape[1:]), **kwargs
                    )
                    f.create_dataset(
                        "osm", data=osm_arr, maxshape=(None, *osm_arr.shape[1:]), **kwargs
                    )
                    f.create_dataset(
                        "eua", data=eua_arr, maxshape=(None, *eua_arr.shape[1:]), **kwargs
                    )
                    f.create_dataset(
                        "urb", data=urb_arr, maxshape=(None, *urb_arr.shape[1:]), **kwargs
                    )
                    f.create_dataset(
                        "ulc", data=ulc_arr, maxshape=(None, *ulc_arr.shape[1:]), **kwargs
                    )
                else:
                    # resize datasets
                    f["img"].resize(f["img"].shape[0] + img_arr.shape[0], axis=0)
                    f["osm"].resize(f["osm"].shape[0] + osm_arr.shape[0], axis=0)
                    f["eua"].resize(f["eua"].shape[0] + eua_arr.shape[0], axis=0)
                    f["urb"].resize(f["urb"].shape[0] + urb_arr.shape[0], axis=0)
                    f["ulc"].resize(f["ulc"].shape[0] + ulc_arr.shape[0], axis=0)
                    # append datasets
                    f["img"][-img_arr.shape[0] :] = img_arr
                    f["osm"][-osm_arr.shape[0] :] = osm_arr
                    f["eua"][-eua_arr.shape[0] :] = eua_arr
                    f["urb"][-urb_arr.shape[0] :] = urb_arr
                    f["ulc"][-ulc_arr.shape[0] :] = ulc_arr
            # break when write successful
            break
        except FileExistsError:
            pass
        except BlockingIOError:
            pass
        except OSError:
            pass
        except ValueError:
            raise
        # sleep and retry when file is blocked
        sleep(0.01)


def extract_chunk(chunk_ul_x, chunk_ul_y):
    # read raster datasets
    lcc_dataset = rio_open(LCC_FILE, "r")
    rgbndvi_dataset = [rio_open(f, "r") for f in [R_FILE, G_FILE, B_FILE, NDVI_FILE]]
    urb_prox_dataset = rio_open(EUA_PROX_FILE, "r")

    win_chunk = rio_windows.Window(
        chunk_ul_x, chunk_ul_y, CHUNK_SIZE_X + TILE_SIZE_X, CHUNK_SIZE_Y + TILE_SIZE_Y
    )
    chunk_extent = lcc_dataset.window_bounds(win_chunk)  # extent in geographic coodinates

    # connect to database
    db = GreenluDBhandler(DB_CONFIG_FILE, chunk_extent)

    chunk_dict = {
        (chunk_ul_x + tile_ul_x, chunk_ul_y + tile_ul_y): extract_window(
            chunk_ul_x + tile_ul_x,
            chunk_ul_y + tile_ul_y,
            db,
            rgbndvi_dataset,
            lcc_dataset,
            urb_prox_dataset,
        )
        for tile_ul_x in range(0, CHUNK_SIZE_X, TILE_OFFSET_X)
        for tile_ul_y in range(0, CHUNK_SIZE_Y, TILE_OFFSET_Y)
    }

    chunk_to_h5(chunk_dict, H5_FILE)
    lcc_dataset.close()
    rgbndvi_dataset.close()
    urb_prox_dataset.close()


def extract_window(tile_ul_x, tile_ul_y, db, rgbndvi_dataset, lcc_dataset, urb_prox_dataset):
    """Extract a window of data."""
    # construct current window
    win = rio_windows.Window(tile_ul_x, tile_ul_y, TILE_SIZE_X, TILE_SIZE_Y)
    # extent in geographic coodinates
    win_ext = lcc_dataset.window_bounds(win)

    # read landcover data
    w = lcc_dataset.read(1, window=win)

    # if window is outside of Germany, a.k.a. only NoData (i.e. 0) in lcc,
    # skip extraction of this tile
    if np_max(w) == 0:
        return None

    # read raster data
    w_rgbndvi = [ds.read(1, window=win) for ds in rgbndvi_dataset]
    # clip and aggregate tile from chunk
    aggregates = db.aggregate_features_in_window(win_ext)

    # aggregate urban proximit raster for current chunk
    w_urb = urb_prox_dataset.read(1, window=win)
    urban_prox_stats = array((np_min(w_urb), mean(w_urb), median(w_urb), np_max(w_urb)))

    return {
        "img": stack((*w_rgbndvi, w), axis=2),
        "osm": aggregates[0],
        "eua": aggregates[1],
        "urb": urban_prox_stats,
    }


def mp_extract_chunk(iterable):
    return extract_chunk(*iterable)


def extract_census_chunks(census_ul_chunk: np.ndarray):
    """Extract census iterables."""
    # read raster datasets
    lcc_dataset = rio_open(LCC_FILE, "r")
    rgbndvi_dataset = [rio_open(f, "r") for f in [R_FILE, G_FILE, B_FILE, NDVI_FILE]]
    urb_prox_dataset = rio_open(EUA_PROX_FILE, "r")

    window_ul_x = np.min(census_ul_chunk[:, 0])
    window_ul_y = np.min(census_ul_chunk[:, 1])
    window_ur_x = np.max(census_ul_chunk[:, 0])
    window_ur_y = np.max(census_ul_chunk[:, 1])
    chunk_size_x = window_ur_x - window_ul_x
    chunk_size_y = window_ur_y - window_ul_y

    win_chunk = rio_windows.Window(
        window_ul_x, window_ul_y, chunk_size_x + TILE_SIZE_X, chunk_size_y + TILE_SIZE_Y
    )
    chunk_extent = lcc_dataset.window_bounds(win_chunk)  # extent in geographic coodinates

    # connect to database
    db = GreenluDBhandler(DB_CONFIG_FILE, chunk_extent)

    chunk_dict = {
        (tile_ul_x, tile_ul_y): extract_window(
            tile_ul_x, tile_ul_y, db, rgbndvi_dataset, lcc_dataset, urb_prox_dataset
        )
        for tile_ul_x, tile_ul_y in census_ul_chunk
    }

    chunk_to_h5(chunk_dict, H5_FILE)
    lcc_dataset.close()
    for ds in rgbndvi_dataset:
        ds.close()
    urb_prox_dataset.close()
    del db


if __name__ == "__main__":
    raise SystemExit(main())
