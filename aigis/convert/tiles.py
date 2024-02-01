# -*- coding: utf-8 -*-
import glob
import itertools
import logging
import os

import rasterio as rio
import rasterio.windows as riow

"""A collection of functions for manipulating raster tiles."""


log = logging.getLogger(__name__)


def get_tiles(
    geotiff: rio.DatasetReader,
    tile_width: int = 2000,
    tile_height: int = 2000,
    map_units: bool = False,
    offset: float = 0.0,
):

    """Defines a set of tiles over a raster layer based on user specified
    dimensions.

    Args:
        raster (rio.DatasetReader): Rasterio raster object
        tile_width (int, optional): Width of tile. Defaults to 2000.
        tile_height (int, optional): Height of tile. Defaults to 2000.
        map_units (bool, optional): If True, tile_width and tile_height are in map units. Defaults to False.
        offset (float, optional): Padding/offset/overlap in percentage of tile. Defaults to 0.0.

    Yields:
        window (rio.windows.Window): Rasterio window object
        transform (Affine): Rasterio affine transform object
    """

    if map_units:
        if geotiff.transform.b == geotiff.transform.d == 0:
            # Get pixel size (x is width) (https://gis.stackexchange.com/questions/379005/using-raster-transform-function-of-rasterio-in-python)
            cell_x, _ = geotiff.transform.a, -geotiff.transform.e
            tile_width, tile_height = int(tile_width / cell_x + 0.5), int(
                tile_height / cell_x + 0.5
            )
        else:
            log.error("ValueError: Coefficient a from raster.transform.a is not width.")
            raise ValueError("Coefficient a from raster.transform.a is not width.")

    ncols, nrows = geotiff.meta["width"], geotiff.meta["height"]

    corners = itertools.product(
        range(0, ncols, tile_width), range(0, nrows, tile_height)
    )
    big_window = riow.Window(col_off=0, row_off=0, width=ncols, height=nrows)

    offset_w = int(tile_width * offset / 100)
    offset_h = int(tile_height * offset / 100)

    TILE_WIDTH = min(tile_width + (offset_w * 2), ncols)
    TILE_HEIGHT = min(tile_height + (offset_h * 2), nrows)

    for col_corner, row_corner in corners:

        if col_corner == 0:
            tile_width = min(TILE_WIDTH + offset_w, ncols)
        else:
            tile_width = min(TILE_WIDTH + (offset_w * 2), ncols)

        if row_corner == 0:
            tile_height = min(TILE_HEIGHT + offset_h, nrows)
        else:
            tile_height = min(TILE_HEIGHT + (offset_h * 2), nrows)

        window = riow.Window(
            col_off=max(0, col_corner - offset_w),
            row_off=max(0, row_corner - offset_h),
            width=tile_width,
            height=tile_height,
        ).intersection(big_window)

        transform = riow.transform(window, geotiff.transform)
        yield window, transform


def save_tiles(
    geotiff: rio.DatasetReader,
    out_path: str,
    tile_size: int = 2000,
    tile_template: str = "tile_{}-{}.tif",
    offset: float = 0.0,
    map_units: bool = True,
):
    """Save tiles from a raster file.

    Args:
        raster (rio.DatasetReader): Rasterio raster object
        out_path (str): Path to save tiles to.
        tile_size (int): Size of tiles.
        tile_template (str): Template for tile names. Should contain two {} placeholders for the x and y coordinates of the tile.
        offset (float, optional): Padding/offset/overlap in percentage of tile. Defaults to 0.0.
        map_units (bool, optional): If True, tile_width and tile_height are in map units. Defaults to True.

    Returns:
        None
    """

    if not os.path.exists(out_path):
        os.makedirs(out_path)
        log.info(f"created '{out_path}'")
    else:
        log.info(f"'{out_path}' already exists")

    # with rio.open(raster_geotiffpath) as geotiff:
    tile_width, tile_height = tile_size, tile_size
    meta = geotiff.meta.copy()
    for window, transform in get_tiles(
        geotiff, tile_width, tile_height, map_units=map_units, offset=offset
    ):
        meta["transform"] = transform
        meta["width"], meta["height"] = window.width, window.height
        tile_path = os.path.join(
            out_path, tile_template.format(int(window.col_off), int(window.row_off))
        )
        with rio.open(tile_path, "w", **meta) as outds:
            outds.write(geotiff.read(window=window))
    # Close the big raster now that we are done with it.
    # geotiff.close()


def get_tiles_list_from_dir(tiles_dir: str, extension: str = "tif"):
    """Get a list of tiles from a directory.

    Args:
        tiles_dir (str): Path to tiles directory.
        extension (str, optional): Extension of tiles. Defaults to "tif".

    Returns:
        tiles_list (list): List of tiles.
    """

    tiles_list = glob.glob(os.path.join(tiles_dir, "*." + extension))
    return tiles_list


def load_tiles_from_list(tiles_list: list):
    """Load tiles from a list.

    Args:
        tiles_list (list): List of rasterio raster objects.

    Returns:
        tiles (list): List of rasterio raster objects.
    """

    tiles = []
    for tile in tiles_list:
        with rio.open(tile) as geotiff:
            tiles.append(geotiff)
    return tiles


def load_tiles_from_dir(tiles_dir: str, extension: str = "tif"):
    """Load tiles from a directory.

    Args:
        tiles_dir (str): Path to tiles directory.

    Returns:
        tiles (list): List of rasterio raster objects.
    """
    tile_list = get_tiles_list_from_dir(tiles_dir, extension)
    tiles = []
    for tile in tile_list:
        with rio.open(tile) as geotiff:
            tiles.append(geotiff)
    return tiles


def tile_neighbourhood_list(tiles: list) -> dict:
    """Get a dictionary of tile neighbourhoods.

    Using the naming schema of tiles, find the neighbouring tiles for each tile, and store them in a dictionary.

    It is important to check if the tile names are in the following format: "tile_{}-{}.tif", where the {} are the x and y coordinates of the tile.

    Args:
        tiles (list): List of rasterio raster objects.

    Returns:
        neighbourhood_dict (dict): Dictionary of tile neighbourhoods.
    """
    # Find all x and y corner coordinates
    # tiles = list(set(tiles_df.raster_path.values.tolist()))
    all_x = []
    all_y = []
    for tile in tiles:
        tile_name = os.path.basename(tile)
        tile_name = tile_name.split(".")[0]
        tile_name = tile_name.split("_")[1]
        tile_name = tile_name.split("-")
        x, y = int(tile_name[0]), int(tile_name[1])
        all_x.append(x)
        all_y.append(y)

    # Sort coordinates as corner coordinates
    all_x = sorted(list(set(all_x)))
    all_y = sorted(list(set(all_y)))

    # Create a dictionary of tile neighbourhoods
    neighbourhood_dict = {}
    for tile in tiles:
        # Get tile coordinates
        tile_name = os.path.basename(tile)
        tile_name_main = tile_name.split(".")[0]
        tile_name = tile_name_main.split("_")[1]
        tile_name = tile_name.split("-")
        x, y = int(tile_name[0]), int(tile_name[1])
        x_index = all_x.index(x)
        y_index = all_y.index(y)

        # Get tile neighbourhoods in all 8 corners
        neighbourhood_dict[tile] = {
            "tile": tile,
            "tile_name": tile_name_main,
            "neighbour_tiles": [],
            "neighbour_tiles_names": [],
        }

        for tile_n in tiles:
            if tile_n == tile:
                continue
            # Get tile coordinates
            tile_name_n = os.path.basename(tile_n)
            tile_name_main_n = tile_name_n.split(".")[0]
            tile_name_n = tile_name_main_n.split("_")[1]
            tile_name_n = tile_name_n.split("-")
            x_n, y_n = int(tile_name_n[0]), int(tile_name_n[1])

            # Check if tile is in neighbourhood
            if x_n in [
                all_x[max(x_index - 1, 0)],
                x,
                all_x[min(x_index + 1, len(all_x) - 1)],
            ] and y_n in [
                all_y[max(y_index - 1, 0)],
                y,
                all_y[min(y_index + 1, len(all_y) - 1)],
            ]:
                neighbourhood_dict[tile]["neighbour_tiles"].append(tile_n)
                neighbourhood_dict[tile]["neighbour_tiles_names"].append(
                    tile_name_main_n
                )

    # neighbourhood_dict_df = pd.DataFrame(neighbourhood_dict).T.reset_index(drop=True)
    return neighbourhood_dict
