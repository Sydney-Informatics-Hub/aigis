# -*- coding: utf-8 -*-
import argparse
import logging

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon
from tqdm import tqdm

# set up logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

tqdm.pandas()


def storey_averager(annotation, storey_column="storeys"):
    """This function will get the average number of storeys of buildings in the
    annotation geodataframe.

    Args:
        annotation (geodataframe): A geodataframe of annotations.

    Returns:
        average_storeys (int): The average number of storeys of buildings in the annotation geodataframe.
    """

    for i, row in annotation.iterrows():
        try:
            if (
                row[storey_column] == "None"
                or row[storey_column] == "0"
                or row[storey_column] == 0
            ):
                annotation_d = annotation.drop(i)
        except KeyError:
            pass
    try:
        average_storeys = annotation_d[storey_column].mean()
    except KeyError:
        average_storeys = 1
        logger.warning(
            "No storeys column found in the annotation geodataframe. Will assume that all buildings have 1 storey."
        )

    return average_storeys


def density_estimate_combined_area(
    annotation,
    crs=None,
    average_storeys=None,
    footprint_ratio: float = 0.5,
    storey_column: str = "storeys",
    area: float = None,
) -> float:
    """This function uses the density_estimate_area_area and
    density_estimate_number_area functions to create a combined density
    estimate. It uses the footprint_ratio to determine the ratio of the two
    density estimates.

    Args:
        annotation (geodataframe): A geodataframe of annotations.
        crs (str): The crs of the annotation geodataframe to calculate the area. If none is given, will rely on UTM crs. If fails, will use 'EPSG:3857' as fallback. Defaults to None.
        average_storeys (int): The average number of storeys of buildings in the annotation geodataframe. If None, will not calculate the average number of storeys using the meta data. Defaults to None.
        footprint_ratio (float): The ratio of the footprint-area-based density to number-based density calculations. It should be a number between 0 and 1. 0 means the footprint area density won't be considered and 1 means number density won't be considered. Defaults to 0.5.

    Returns:
        density (float): The combined density estimate.
    """

    assert (
        footprint_ratio >= 0 and footprint_ratio <= 1
    ), "footprint_ratio must be between 0 and 1"

    density_area = density_estimate_area_area(
        annotation, crs, average_storeys, storey_column, area
    )

    density_number = density_estimate_number_area(
        annotation, crs, average_storeys, storey_column, area
    )

    density = (
        density_area * footprint_ratio + density_number * (1 - footprint_ratio)
    ) / 2

    return density


def density_estimate_number_area(
    annotation,
    crs=None,
    average_storeys=None,
    storey_column: str = "storeys",
    area=None,
) -> float:
    """This function will get the area of annotation geodataframe, and also get
    the number of geometries in the annotation geodataframe, and return a
    number that is the number of geometries in the annotation geodataframe
    divided by area of the annotation geodataframe.

    Args:
        annotation (geodataframe): A geodataframe of annotations.
        crs (str): The crs of the annotation geodataframe to calculate the area. If none is given, will rely on UTM crs. If fails, will use 'EPSG:3857' as fallback. Defaults to None.
        average_storeys (int): The average number of storeys of buildings in the annotation geodataframe. If None, will not calculate the average number of storeys using the meta data. Defaults to None.

    Returns:
        density (float): The number of geometries in the annotation geodataframe divided by the area of the annotation geodataframe.
    """

    if crs is None:
        try:
            crs = annotation.estimalte_utm_crs()
        except Exception as e:
            crs = "EPSG:3857"
            logger.warning(
                f"Could not estimate the UTM crs of the annotation geodataframe. Will use {crs} as fallback."
            )
            logger.warning(e)

    if annotation.crs != crs:
        annotation = annotation.to_crs(crs)

    if average_storeys is None:
        average_storeys = storey_averager(annotation, storey_column)
        logger.info(
            f"Average storeys is {average_storeys} for the given extent {annotation.bounds}"
        )
    elif average_storeys == 0:
        average_storeys = 1
        logger.warning(
            "Average storeys cannot be 0. Will assume that all buildings have 1 storey."
        )
    else:
        average_storeys = int(average_storeys)
        logger.info(
            f"Average storeys is {average_storeys} for the given extent {annotation.bounds}"
        )
    # bounds = annotation.total_bounds
    # width = abs(bounds[2] - bounds[0])
    # height = abs(bounds[3] - bounds[1])
    # area = width * height
    # print("area", area)
    number = annotation.shape[0]
    # print("number", number)
    # print("average_storeys", average_storeys)
    density = (number * average_storeys) / area
    # print("number density", density)

    return density


def density_estimate_area_area(
    annotation,
    crs=None,
    average_storeys=None,
    storey_column: str = "storeys",
    area: float = None,
) -> float:
    """This function will get the area of annotation geodataframe, and also get
    the number of geometries in the annotation geodataframe, and return a
    number that is the total area of geometries in the annotation geodataframe
    divided by the area of the annotation geodataframe.

    Args:
        annotation (geodataframe): A geodataframe of annotations.
        crs (str): The crs of the annotation geodataframe to calculate the area. If none is given, will rely on UTM crs. If fails, will use 'EPSG:3857' as fallback. Defaults to None.
        average_storeys (int): The average number of storeys of buildings in the annotation geodataframe. If None, will not calculate the average number of storeys using the meta data. Defaults to None.

    Returns:
        density (float): The area of geometries in the annotation geodataframe divided by the area of annotation.
    """
    if crs is None:
        try:
            crs = annotation.estimalte_utm_crs()
        except Exception as e:
            crs = "EPSG:3857"
            logger.warning(
                f"Could not estimate the UTM crs of the annotation geodataframe. Will use {crs} as fallback."
            )
            logger.warning(e)

    if annotation.crs != crs:
        annotation = annotation.to_crs(crs)

    if average_storeys is None:
        average_storeys = storey_averager(annotation, storey_column)
        logger.info(
            f"Average storeys is {average_storeys} for the given extent {annotation.bounds}"
        )
    elif average_storeys == 0:
        average_storeys = 1
        logger.warning(
            "Average storeys cannot be 0. Will assume that all buildings have 1 storey."
        )
    else:
        average_storeys = int(average_storeys)
        logger.info(
            f"Average storeys is {average_storeys} for the given extent {annotation.bounds}"
        )

    # bounds = annotation.total_bounds
    # width = abs(bounds[2] - bounds[0])
    # height = abs(bounds[3] - bounds[1])
    # area = width * height
    # print("area", area)
    footprint_area = 0
    for _, row in annotation.iterrows():
        footprint_area += row["geometry"].area
    # print("footprint_area", footprint_area)
    # print("average_storeys", average_storeys)
    density = (footprint_area * average_storeys) / area
    # print("area density", density)

    return density


def density_map_maker(
    gdf: gpd.GeoDataFrame,
    average_storeys: int = None,
    footprint_ratio: float = 0.5,
    tile_size: int = 100,
    size_unit: str = None,
    area_unit: str = "utm",
    storey_column: str = "storeys",
) -> gpd.GeoDataFrame:
    """This function will use the density_estimate_combined_area function to
    create a density map, by tiling the geojson.

    Args:
        gdf (geodataframe): A geodataframe of annotations.
        average_storeys (int): The average number of storeys of buildings in the annotation geodataframe. If None, will not calculate the average number of storeys using the meta data. Defaults to None.
        footprint_ratio (float): The ratio of the footprint-area-based density to number-based density calculations. It should be a number between 0 and 1. 0 means the footprint area density won't be considered and 1 means number density won't be considered. Defaults to 0.5.
        tile_size (int): The size of the tile. Defaults to 10.
        size_unit (str): The unit of the tile size. If is None, will use the unit of the crs of the gdf or from 'area_unit'. If set to 'percent', will use the percentage of the width of the gdf bounds. Defaults to percent. Overall, this can be ignored as long as percentage of width is the preferred window size.
        area_unit (str): The unit of the area. Defaults to "utm".

    Returns:
        grid (geodataframe): A geodataframe of the density map.
    """

    # Prepare the gdf
    if area_unit == "meter":
        crs = 3857
        gdf = gdf.to_crs(epsg=crs)
    elif area_unit == "utm":
        crs = gdf.estimate_utm_crs()
        gdf = gdf.to_crs(crs)
    elif area_unit is None:
        crs = None
    else:
        logger.warning("area_unit must be 'meter', 'utm', or None.")

    # Get the bounds of the gdf
    bounds = gdf.total_bounds
    width = abs(bounds[2] - bounds[0])
    area = width * width
    x_min, y_min, x_max, y_max = bounds

    # Get the tile size
    if size_unit == "percent":
        tile_size = width * tile_size / 100
    elif size_unit is None:
        pass
    else:
        logger.warning("size_unit must be 'percent' or None. Will assume None.")

    assert tile_size > 0, "tile_size must be greater than 0."

    # Create the grid
    x_coords = np.arange(x_min, x_max, tile_size)
    y_coords = np.arange(y_min, y_max, tile_size)
    polygons = []
    for x in x_coords:
        for y in y_coords:
            polygons.append(
                Polygon(
                    [
                        (x, y),
                        (x + tile_size, y),
                        (x + tile_size, y + tile_size),
                        (x, y + tile_size),
                    ]
                )
            )
    grid = gpd.GeoDataFrame({"geometry": polygons}, crs=gdf.crs)
    gdf = gdf[gdf.geometry.area > 0]
    gdf = gdf.reset_index(drop=True)

    gdf["geometry"] = gdf["geometry"].buffer(0)

    logger.info(
        f"Created a grid of {grid.shape[0]} tiles. Now getting the grid densities..."
    )
    # Get the density for each tile
    density_map = []
    for extent in tqdm(grid.geometry, total=grid.shape[0]):
        raster_extent = gpd.GeoDataFrame({"id": 1, "geometry": [extent]}, crs=gdf.crs)

        # save polygon extent to shp
        # raster_extent_cp = raster_extent.copy()
        # raster_extent_cp["geometry"] = raster_extent_cp["geometry"].buffer(0)
        # set crs
        # raster_extent_cp.crs = gdf.crs
        # raster_extent_cp.to_file("raster_extent.shp")

        raster_extent = raster_extent.buffer(0)

        tile_polygons = gdf.clip(raster_extent)

        # Split multipolygon
        tile_polygons = tile_polygons.explode(index_parts=False)
        tile_polygons = tile_polygons.reset_index(drop=True)

        # get density for each tile
        density = density_estimate_combined_area(
            tile_polygons,
            crs=crs,
            average_storeys=average_storeys,
            storey_column=storey_column,
            footprint_ratio=footprint_ratio,
            area=area,
        )
        # print("grid density:", density, "\n=======")
        density_map.append(density)

    # Create the density map
    grid["density"] = density_map

    return grid


def density_maker_geojson(
    input_path: str,
    average_storeys: int = None,
    footprint_ratio: float = 0.5,
    tile_size: int = 100,
    size_unit: str = None,
    area_unit: str = "utm",
    output_path: str = None,
    storey_column: str = "storeys",
) -> gpd.GeoDataFrame:
    """This function is a wrapper function for density_map_maker calculating
    and saving the density map as a geojson.

    Args:
        gdf (str): A path to a geojson file of annotations.
        average_storeys (int): The average number of storeys of buildings in the annotation geodataframe. If None, will not calculate the average number of storeys using the meta data. Defaults to None.
        footprint_ratio (float): The ratio of the footprint-area-based density to number-based density calculations. It should be a number between 0 and 1. 0 means the footprint area density won't be considered and 1 means number density won't be considered. Defaults to 0.5.
        tile_size (int): The size of the tile. Defaults to 10.
        size_unit (str): The unit of the tile size. If is None, will use the unit of the crs of the gdf or from 'area_unit'. If set to 'percent', will use the percentage of the width of the gdf bounds. Defaults to percent. Overall, this can be ignored as long as percentage of width is the preferred window size.
        area_unit (str): The unit of the area. Defaults to "utm".
        output_path (str): The path to save the output geojson. If None, will not save the output geojson. Defaults to None.

    Returns:
        grid (geodataframe): A geodataframe of the density map.
    """
    # read the geojson
    gdf = gpd.read_file(input_path)

    # print("Geodataframe:\n",gdf)

    # create the density map
    grid = density_map_maker(
        gdf,
        average_storeys=average_storeys,
        footprint_ratio=footprint_ratio,
        tile_size=tile_size,
        size_unit=size_unit,
        area_unit=area_unit,
        storey_column=storey_column,
    )

    print("Grid:\n", grid)

    # save the density map
    if output_path is not None:
        grid.to_file(output_path, driver="GeoJSON")
    else:
        output_path = input_path.replace(".geojson", "_density.geojson")
        grid.to_file(output_path, driver="GeoJSON")

    logger.info(f"Saved the density map to {output_path}")

    return grid


def create_parser():
    parser = argparse.ArgumentParser(
        description="Create a density map from a geojson of annotations."
    )
    parser.add_argument(
        "--input-path",
        "-i",
        type=str,
        required=True,
        help="Path to a geojson file of annotations.",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        type=str,
        default=None,
        help="The path to save the output geojson. If None, will not save the output geojson. Defaults to None.",
    )
    parser.add_argument(
        "--average-storeys",
        "-a",
        type=int,
        default=None,
        help="The average number of storeys of buildings in the annotation geodataframe. If None, will not calculate the average number of storeys using the meta data. Defaults to None.",
    )
    parser.add_argument(
        "--storey-column",
        "-s",
        type=str,
        default="storeys",
        help="The column name of the storey information. Defaults to 'storeys'.",
    )
    parser.add_argument(
        "--footprint-ratio",
        "-f",
        type=float,
        default=0.5,
        help="The ratio of the footprint-area-based density to number-based density calculations. It should be a number between 0 and 1. 0 means the footprint area density won't be considered and 1 means number density won't be considered. Defaults to 0.5.",
    )
    parser.add_argument(
        "--tile-size",
        "-t",
        type=int,
        default=200,
        help="The size of the tile. Defaults to 200.",
    )
    parser.add_argument(
        "--size-unit",
        "-u",
        type=str,
        default=None,
        help="The unit of the tile size. If is None, will use the unit of the crs of the gdf or from 'area_unit'. If set to 'percent', will use the percentage of the width of the gdf bounds. Defaults to percent. Overall, this can be ignored as long as percentage of width is the preferred window size.",
    )
    parser.add_argument(
        "--area-unit",
        "-r",
        type=str,
        default="utm",
        help="The unit of the area. Defaults to 'utm'.",
    )
    return parser


def main(args=None):
    parser = create_parser()
    args = parser.parse_args(args)

    density_maker_geojson(
        args.input_path,
        average_storeys=args.average_storeys,
        footprint_ratio=args.footprint_ratio,
        tile_size=args.tile_size,
        size_unit=args.size_unit,
        area_unit=args.area_unit,
        output_path=args.output_path,
        storey_column=args.storey_column,
    )


if main:
    main()
