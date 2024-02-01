# -*- coding: utf-8 -*-
import logging

import fiona
import geopandas as gpd
import pandas as pd
import rasterio as rio
from shapely.geometry import MultiPoint, Polygon, box
from tqdm import tqdm

"""A collection of functions for converting between spatial and pixel
coordinates.

This module contains functions for converting between spatial and pixel coordinates, as well as functions for creating
pixel polygons for a list of raster tiles. The pixel polygons are used to create COCO annotations for each polygon.

Also contains functions for creating polygons from coco annotations.
"""


log = logging.getLogger(__name__)


def wkt_parser(wkt_str: str, wkt_keyword="LOCAL_CS["):
    """Parses a WKT string to extract the local coordinate system.

    Args:
        wkt_str (str): WKT string

    Returns:
        str: Local coordinate system
    """
    # TODO: Make this tool smarter. Right now it just looks for the first LOCAL_CS[ and returns everything after that.

    wkt = wkt_str.split('"')
    if wkt_keyword in wkt:
        return wkt[wkt.index(wkt_keyword) + 1]
    return wkt_str


def read_crs_from_raster(raster_file: str):
    """Reads the coordinate system from a raster file.

    Args:
        raster_file (str): Path to raster file

    Returns:
        str: Coordinate system
    """
    with rio.open(raster_file) as src:
        user_crs = src.crs.to_wkt()
        user_crs = wkt_parser(user_crs)
        return user_crs


def reproject_coords(src_crs, dst_crs, coords):
    """Reprojects a list of coordinates from one coordinate system to another.

    Args:
        src_crs (str): Source coordinate system
        dst_crs (str): Destination coordinate system
        coords (list): List of coordinates to reproject

    Returns:
        list: List of reprojected coordinates
    """

    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    xs, ys = fiona.transform.transform(src_crs, dst_crs, xs, ys)
    return [[x, y] for x, y in zip(xs, ys)]


def pixel_to_spatial_rio(geotiff, row_index, col_index):
    """Converts pixel coordinates to spatial coordinates using rasterio. More
    information here: https://stackoverflow.com/questions/52443906/pixel-array-
    position-to-lat-long-gdal-python.

    Args:
        geotiff (rio.DatasetReader): Rasterio raster object (do not read, just open via rasterio.open(raster_path))
        row_index (int): pixel row
        col_index (int): pixel column

    Returns:
        tuple: (x,y) spatial coordinates
    """

    return geotiff.xy(row_index, col_index)  # px, py


def pixel_segmentation_to_spatial_rio(geotiff, segmentation):
    """Converts pixel segmentation to spatial segmentation using rasterio.

    Args:
        geotiff (rio.DatasetReader): Rasterio raster object for reference
        segmentation (list): List of pixel coordinates defining the polygon in coco segmentation format

    Returns:
        converted_coords (list): List of spatial coordinates defining the polygon
    """
    converted_coords = []
    segmentation = [
        (segmentation[i + 1], segmentation[i]) for i in range(0, len(segmentation), 2)
    ]
    for point in segmentation:
        log.debug(f"Converting {point} to spatial coordinates in raster {geotiff}")
        x, y = pixel_to_spatial_rio(geotiff, point[0], point[1])
        spatial_point = [x, y]
        converted_coords.append(spatial_point)

    polygon = Polygon(converted_coords)
    return polygon


def pixel_bbox_to_spatial_rio(geotiff, bbox):
    """Converts pixel bbox to spatial bbox using rasterio.

    Args:
        geotiff (rio.DatasetReader): Rasterio raster object for reference
        bbox (list): List of pixel coordinates defining the bbox in coco bbox format

    Returns:
        converted_coords (list): List of spatial coordinates defining the bbox
    """
    converted_coords = []
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
    for point in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]:
        log.debug(f"Converting {point} to spatial coordinates in raster {geotiff}")
        x, y = pixel_to_spatial_rio(geotiff, point[0], point[1])
        spatial_point = x, y
        converted_coords.append(spatial_point)
    return converted_coords


def spatial_to_pixel_rio(geotiff, x, y):
    """Converts spatial coordinates to pixel coordinates using rasterio.

    Args:
        geotiff (rio.DatasetReader): Rasterio raster object
        x (float): longitudinal coordinate in spatial units
        y (float): latitudinal coordinate in spatial units

    Returns:
        tuple: (row_index,col_index) pixel coordinates
    """

    row_index, col_index = geotiff.index(x, y)  # lon,lat
    return row_index, col_index


def spatial_polygon_to_pixel_rio(geotiff, polygon) -> list:
    """Converts spatial polygon to pixel polygon using rasterio.

    Args:
        geotiff (rio.DatasetReader): Rasterio raster object
        polygon (shapely.geometry.Polygon): Polygon in spatial coordinates

    Returns:
        converted_coords (list): List of pixel coordinates defining the polygon
    """
    converted_coords = []
    for point in list(MultiPoint(polygon.exterior.coords).geoms):
        log.debug(f"Converting {point} to pixel coordinates in raster {geotiff}")
        y, x = spatial_to_pixel_rio(geotiff, point.x, point.y)
        pixel_point = x, y
        converted_coords.append(pixel_point)
    return converted_coords


def get_tile_polygons(raster_tile: str, geojson: gpd.GeoDataFrame, filter: int = 0):
    """Create polygons from a geosjon for an individual raster tile.

    Args:
        raster_tile: (str) a file name referring to the raster tile to be loaded
        geojson: (gpd.GeoDataFrame) a geodataframe with polygons
        filter: (int) an integer to filter out polygons with area less than the filter value

    Returns:
        tile_polygon: geodataframe with polygons within the raster's extent
    """
    # print(geojson.shape)
    # Load raster tile
    raster_tile = rio.open(raster_tile)
    raster_extent = gpd.GeoDataFrame(
        {"id": 1, "geometry": [box(*raster_tile.bounds)]}, crs=geojson.crs
    )
    # geojson = geojson.to_crs(geojson)
    tile_polygons = geojson.clip(raster_extent)

    # Split multipolygon
    tile_polygons = tile_polygons.explode(index_parts=False)
    tile_polygons = tile_polygons.reset_index(drop=True)

    # Filter out zero area polygons by re-projecting to the best UTM CRS for area calculation
    target_crs = geojson.estimate_utm_crs()
    tile_polygons = tile_polygons.to_crs(target_crs)
    tile_polygons = tile_polygons[tile_polygons.geometry.area > filter]
    tile_polygons = tile_polygons.to_crs(geojson.crs)
    # if filter is True:
    #     tile_polygons = tile_polygons[tile_polygons.geometry.area > 5000]
    tile_polygons = tile_polygons.reset_index(drop=True)
    # print(tile_polygons.shape)
    return tile_polygons


def pixel_polygons_for_raster_tiles(
    raster_file_list: list, geojson: gpd.GeoDataFrame, verbose=1
):
    """Create pixel polygons for a list of raster tiles.

    Args:
        raster_file_list (list): List of raster files
        geojson (gpd.GeoDataFrame): GeoDataFrame containing polygons
        verbose (int): Verbosity level

    Returns:
        pixel_df (pd.DataFrame): DataFrame containing pixel polygons
    """
    tmp_list = []
    log.info(f"Creating pixel polygons for {len(raster_file_list)} tiles")
    for index, file in enumerate(raster_file_list):
        tmp = get_tile_polygons(file, geojson, 0)
        tmp["raster_tile"] = rio.open(file)
        tmp["image_id"] = index
        tmp_list.append(tmp)

    log.info(f"Concatenating {len(tmp_list)} GeoDataFrames")
    pixel_df = pd.concat(tmp_list).reset_index()
    pixel_df = pixel_df.drop(columns=["index"])
    log.info(f"Creating pixel polygons for {pixel_df.shape[0]} polygons")
    if verbose > 0:
        tqdm.pandas()
        pixel_df["pixel_polygon"] = pixel_df.progress_apply(
            lambda row: spatial_polygon_to_pixel_rio(
                row["raster_tile"], row["geometry"]
            ),
            axis=1,
        )
    else:
        pixel_df["pixel_polygon"] = pixel_df.apply(
            lambda row: spatial_polygon_to_pixel_rio(
                row["raster_tile"], row["geometry"]
            ),
            axis=1,
        )
    pixel_df["annot_id"] = range(0, 0 + len(pixel_df))
    log.info(f"Pixel polygons created for {pixel_df.shape[0]} polygons")
    return pixel_df
