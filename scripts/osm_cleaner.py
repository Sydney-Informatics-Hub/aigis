# -*- coding: utf-8 -*-
import argparse
import glob
import logging
import os

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from tqdm import tqdm

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

tqdm.pandas()

# Helper functions


def to_int(x):
    """Convert a string to int.

    Args:
        x (str): The string to convert.

    Returns:
        int: The converted int.
    """
    try:
        return int(x)
    except Exception as e:
        logger.info(f"Skipped converting {x} to int. Will return 0.\n: {e}")
        return 0


def cleaner_function(x):
    """Clean the level column in the OSM data.

    Args:
        x (str): The string to clean.

    Returns:
        int: The cleaned int.
    """
    if x == ">1" or x == "1.5" or x == 1.5:
        return 1
    elif x == 0 or x == "0":
        return 1
    elif str(x).lower() == "kiosk":
        return 1
    else:
        return to_int(x)


def level_average(intersecting_gdf, column, default_na=0):
    """Get the average level of the buildings intersecting a block

    Args:
        intersecting_gdf (gpd.GeoDataFrame): The GeoDataFrame of intersecting buildings.
        column (str): The column to get the average of.

    Returns:
        float: The average level.
    """
    # drop rows with columns that are None or zero before averaging
    intersecting_gdf = intersecting_gdf[intersecting_gdf[column].notnull()]
    intersecting_gdf = intersecting_gdf[intersecting_gdf[column] > 0]
    if intersecting_gdf.shape[0] > 0:
        return intersecting_gdf[column].mean()
    else:
        logger.info(
            f"No intersecting buildings found. The block will return a {default_na} value for average."
        )
        return default_na


def level_std_average(intersecting_gdf, column, default_na=0):
    """Get the average level of the intersecting buildings within its standard
    deviation.

    Args:
        intersecting_gdf (gpd.GeoDataFrame): The GeoDataFrame of intersecting buildings.
        column (str): The column to get the average of.

    Returns:
        float: The average level.
    """
    # drop rows with columns that are None or zero before averaging
    intersecting_gdf = intersecting_gdf[intersecting_gdf[column].notnull()]
    intersecting_gdf = intersecting_gdf[intersecting_gdf[column] > 0]

    mean = intersecting_gdf[column].mean()
    std = intersecting_gdf[column].std()

    # Find the values in between the mean +- std
    intersecting_gdf = intersecting_gdf[
        intersecting_gdf[column].between(mean - std, mean + std)
    ]

    if intersecting_gdf.shape[0] > 0:
        return intersecting_gdf[column].mean()
    else:
        logger.info(
            f"No intersecting buildings found. The block will return a {default_na} value for average."
        )
        return default_na


def replacer(row, column_1, column_2, fallback_value):
    """Replace the value in column_1 with the value in column_2 if the value in
    column_1 is None or zero.

    Args:
        row (pd.Series): The row to replace.
        column_1 (str): The column to replace.
        column_2 (str): The column to replace with.

    Returns:
        int: The replaced value.
    """
    # if there is no value in column_1, or if the value is zero, null, or none, return the value in column_2
    # else return the value in column_1

    if (
        row[column_1] is None
        or row[column_1] == 0
        or row[column_1] == "0"
        or row[column_1] == np.nan
        or row[column_1] == "nan"
        or row[column_1] == "None"
        or row[column_1] == "none"
        or row[column_1] == "Null"
        or row[column_1] == "null"
        or row[column_1] == "NULL"
    ):
        return row[column_2]
    else:
        # print(row[column_1],row[column_2])
        if fallback_value is not None:
            if row[column_1] is None or row[column_1] == 0:
                return fallback_value
        return int(row[column_1])


def level_bracketing(x, other=None):
    """Categorise the OSM data based on the number of levels.

    Args:
        x (int): The number of levels.

    Returns:
        str: The level category.
    """
    if x <= 3:
        return "low"
    elif x <= 9:
        return "mid"
    elif x > 9:
        return "high"
    else:
        return other


# Main functions


def merge_osm_blocks(
    osm_path: str = "/home/sahand/Data/GIS2COCO/osm_building_annotations_by_10_percent_grid/",
    save: bool = True,
    ignored_files: list = ["merged.geojson", "merged_filtered.geojson"],
):
    """Merge all the OSM files in the osm_path into one GeoDataFrame.

    Args:
        osm_path (str): Path to the OSM files.
        save (bool, optional): Whether to save the merged GeoDataFrame. Defaults to True.
        ignored_files (list, optional): List of files to ignore. Defaults to ["merged.geojson", "merged_filtered.geojson"].

    Returns:
        gpd.GeoDataFrame: The merged GeoDataFrame.
    """
    # Read in the buildings from all files in the osm directory
    osm_files = glob.glob(os.path.join(osm_path, "*.geojson"))
    osm_files = [
        file for file in osm_files if os.path.basename(file) not in ignored_files
    ]
    # Merge all the files into one dataframe
    # Initialize a list to hold GeoDataFrames
    gdfs = []
    first_gdf = gpd.read_file(osm_files[0])
    crs = first_gdf.crs

    input(f"crs will be set to {crs}. \nPress Enter to continue...")

    # Read each file into a GeoDataFrame and add it to the list
    for file in tqdm(osm_files):
        try:
            gdf = gpd.read_file(file)
            gdf.geometry = gdf.geometry.to_crs(crs)
            gdfs.append(gdf)
        except Exception as e:
            logger.error(f"Error reading {file}: {e}")
            input("Error reading file. Press Enter to continue...")
    # Concatenate all GeoDataFrames in the list into one GeoDataFrame
    osm = pd.concat(gdfs)
    gdf_osm = gpd.GeoDataFrame(osm, crs=crs, geometry=osm.geometry)

    if save:
        gdf_osm.to_file(
            os.path.join(os.path.dirname(osm_path), "merged.geojson"), driver="GeoJSON"
        )

    return gdf_osm


def filter_osm_columns(
    osm_path: str = "/home/sahand/Data/GIS2COCO/osm_building_annotations_by_10_percent_grid/",
    columns: str = "/home/sahand/Data/GIS2COCO/osm_columns.csv",
    save: bool = True,
):
    """Filter out the columns we don't need from the OSM data.

    Args:
        osm_path (str, optional): Path to the OSM file. Defaults to "/home/sahand/Data/GIS2COCO/osm_building_annotations_by_10_percent_grid/".
        columns (str, optional): Path to the columns csv file. Defaults to "/home/sahand/Data/GIS2COCO/osm_columns.csv".
        save (bool, optional): Whether to save the filtered OSM data. Defaults to True.

    Returns:
        gpd.GeoDataFrame: The filtered OSM data.
    """

    # Read in the OSM data
    if os.path.isdir(osm_path):
        osm = merge_osm_blocks(osm_path=osm_path, save=False)
    else:
        osm = gpd.read_file(osm_path)

    # Read in the columns to keep
    columns = pd.read_csv(columns).potentially_good.values.tolist()

    # Filter out the columns we don't need
    osm = osm[columns]

    # Save the filtered OSM data
    if save:
        osm.to_file(
            os.path.join(os.path.dirname(osm_path), "merged_filtered.geojson"),
            driver="GeoJSON",
        )

    return osm


def osm_level_cleaner(
    osm_path: str = "/home/sahand/Data/GIS2COCO/osm_building_annotations_by_10_percent_grid/merged_filtered.geojson",
    column: str = "building:levels",
    save: bool = True,
    clean=cleaner_function,
):
    """Clean the level column in the OSM data.

    Args:
        osm_path (str, optional): Path to the OSM file. Defaults to "/home/sahand/Data/GIS2COCO/osm_building_annotations_by_10_percent_grid/merged_filtered.geojson".
        column (str, optional): The column to clean. Defaults to "building:levels".
        save (bool, optional): Whether to save the cleaned OSM data. Defaults to True.
        clean (function, optional): The function to clean the column. Defaults to cleaner_function.

    Returns:
        gpd.GeoDataFrame: The cleaned OSM data.
    """

    # Read in the OSM data
    if os.path.isdir(osm_path):
        annotations = merge_osm_blocks(osm_path=osm_path, save=False)
    else:
        annotations = gpd.read_file(osm_path)

    # Clean the level column
    annotations[column] = annotations[column].progress_apply(lambda x: clean(x))

    # Save the cleaned OSM data
    if save:
        out_path = os.path.join(os.path.dirname(osm_path), "merged_cleaned.geojson")
        annotations.to_file(out_path, driver="GeoJSON")

    return annotations


def level_interpolation(
    osm_path: str = "/home/sahand/Data/GIS2COCO/osm_building_annotations_by_10_percent_grid/merged_cleaned.geojson",
    column: str = "building:levels",
    save: bool = True,
    out_name: str = "merged_interpolated.geojson",
    area_unit: str = "utm",
    size_unit: str = None,
    tile_size: float = 500,
    average_function=level_std_average,
    total_average: float = None,
    save_average_grid=True,
):
    """Interpolate the level column in the OSM data.

    Args:
        osm_path (str, optional): Path to the OSM file. Defaults to "/home/sahand/Data/GIS2COCO/osm_building_annotations_by_10_percent_grid/merged_cleaned.geojson".
        column (str, optional): The column to interpolate. Defaults to "building:levels".
        save (bool, optional): Whether to save the interpolated OSM data. Defaults to True.
        out_name (str, optional): The name of the output file. Defaults to "merged_interpolated.geojson".
        area_unit (str, optional): The unit of the area. Defaults to "utm".
        size_unit (str, optional): The unit of the tile size. Defaults to None.
        tile_size (float, optional): The size of the tiles. This is effectively the averaging spatial window size. Defaults to 500.
        average_function (function, optional): The function to get the average. Defaults to level_std_average.

    Returns:
        gpd.GeoDataFrame: The interpolated OSM data.
    """

    # Read in the OSM data
    if os.path.isdir(osm_path):
        gdf = merge_osm_blocks(osm_path=osm_path, save=False)
    else:
        gdf = gpd.read_file(osm_path)

    # Interpolate the level column

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
        f"Created a grid of {grid.shape[0]} tiles. Now getting the grid levels..."
    )

    # Get level average for each block
    logger.info("Getting the level average for each block...")
    grid["level_average"] = grid.geometry.progress_apply(
        lambda x: average_function(gdf[gdf.intersects(x)], column)
    )

    if total_average is None:
        # Get the total average
        logger.info("Getting the total average...")
        total_average = average_function(gdf, column)
    else:
        logger.info(
            "The total_average is given. Will use that instead of calculating it."
        )

    # Set the level_average to the total_average if the level_average is None
    logger.info(
        f"Setting the level_average to the total_average {total_average} if the level_average is None..."
    )
    grid["level_average"] = grid["level_average"].progress_apply(
        lambda x: total_average if x is None or x == 0 else x
    )

    if save_average_grid:
        logger.info("Saving the average grid...")
        grid.to_file(
            os.path.join(os.path.dirname(osm_path), "average_grid.geojson"),
            driver="GeoJSON",
        )

    # Set level_average in gdf to the grid level_average
    logger.info("Setting the level_average in gdf to the grid level_average...")
    gdf["level_average"] = gdf.geometry.progress_apply(
        lambda x: grid[grid.intersects(x)]["level_average"].values[0]
    )

    # Replace the empty level column rows with average values
    logger.info("Interpolating the level column...")
    gdf["interpolated_level"] = gdf.progress_apply(
        lambda x: replacer(x, column, "level_average", total_average), axis=1
    )

    # Save the interpolated OSM data
    if save:
        logger.info("Saving the interpolated OSM data...")
        out_path = os.path.join(os.path.dirname(osm_path), out_name)
        gdf.to_file(out_path, driver="GeoJSON")

    return gdf


def osm_level_categorise(
    osm_path: str = "/home/sahand/Data/GIS2COCO/osm_building_annotations_by_10_percent_grid/merged_interpolated.geojson",
    column: str = "interpolated_level",
    save: bool = True,
    categorise=level_bracketing,
    out_name: str = "merged_categorised.geojson",
):
    """Categorise the OSM data based on the number of levels.

    Args:
        osm_path (str, optional): Path to the OSM file. Defaults to "/home/sahand/Data/GIS2COCO/osm_building_annotations_by_10_percent_grid/merged_filtered.geojson".
        column (str, optional): The column to categorise. Defaults to "interpolated_level".
        save (bool, optional): Whether to save the categorised OSM data. Defaults to True.
        categorise (function, optional): The function to categorise the level column. Defaults to level_bracketing.
        out_name (str, optional): The name of the output file. Defaults to "merged_categorised.geojson".

    Returns:
        gpd.GeoDataFrame: The categorised OSM data.
    """

    # Read in the OSM data
    if os.path.isdir(osm_path):
        annotations = merge_osm_blocks(osm_path=osm_path, save=False)
    else:
        annotations = gpd.read_file(osm_path)

    # Categorise `column` column and add the vategories based on level category: 1-3 | 4-9 | 10+ to a new column of `level_categories`
    annotations["level_categories"] = annotations[column].progress_apply(
        lambda x: categorise(x)
    )

    # Save the filtered OSM data
    if save:
        annotations.to_file(
            os.path.join(os.path.dirname(osm_path), out_name),
            driver="GeoJSON",
        )

    return annotations


def osm_landuse_concat():
    raise NotImplementedError


def argparser():
    parser = argparse.ArgumentParser(
        description="OSM data cleaner and building level categoriser."
    )
    parser.add_argument(
        "--osm_path",
        type=str,
        default="osm_building_annotations_by_10_percent_grid/",
        help="Path to the OSM files.",
    )
    parser.add_argument(
        "--columns",
        type=str,
        default="osm_columns.csv",
        help="Path to the columns filter csv file. The listed columns will be used.",
    )
    parser.add_argument(
        "--column",
        type=str,
        default="building:levels",
        help="The column to use for the building height.",
    )
    parser.add_argument(
        "--out_name",
        type=str,
        default="merged_categorised.geojson",
        help="The name of the output merged and height categorised file.",
    )
    parser.add_argument(
        "--area_unit",
        type=str,
        default="utm",
        help="The unit of the area.",
    )
    parser.add_argument(
        "--size_unit",
        type=str,
        default=None,
        help="The unit of the tile size.",
    )
    parser.add_argument(
        "--tile_size",
        type=float,
        default=500,
        help="The size of the tiles. This is effectively the averaging spatial window size. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--average_function",
        type=str,
        default="level_std_average",
        help="The function to get the average. If 'level_std_average', the average will be within the standard deviation, ignoring the outliers. If 'level_average', the average will be the mean of all the values, including the outliers. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--categorise",
        type=str,
        default="level_bracketing",
        help="The function to categorise the level column. If 'level_bracketing', the levels will be categorised into low, mid, and high. Defaults to %(default)s",
    )
    parser.add_argument(
        "--cleaner_in_path",
        type=str,
        default=None,
        help="The path to the cleaned OSM file. If given, means we have a data with filtered columns, but we want to clean up the column of interest. If None, the path will be set to the path of the merged and filtered OSM file, instructing the script to run the previous levels as well.",
    )
    parser.add_argument(
        "--interpolate_in_path",
        type=str,
        default=None,
        help="The path to the interpolated OSM file. If given, means we have a cleaned data and we only need to continue from interpolation. If None, the path will be set to the path of the merged and cleaner OSM file, instructing the script to run the previous levels as well.",
    )
    parser.add_argument(
        "--total_average",
        type=float,
        default=None,
        help="The total average level. If None, will be calculated from the data.",
    )

    return parser.parse_args()


def main(args):
    if args.average_function == "level_std_average":
        average_function = level_std_average
    else:
        average_function = level_average

    if args.categorise == "level_bracketing":
        categorise = level_bracketing
    else:
        raise NotImplementedError

    if args.cleaner_in_path is not None:
        cleaner_in_path = args.cleaner_in_path
        logger.info(
            f"Assuming the data is filtered and is readey to be cleaned. The cleaner_in_path is set to {cleaner_in_path}."
        )
    else:
        cleaner_in_path = os.path.join(
            os.path.dirname(args.osm_path), "merged_filtered.geojson"
        )

        print("Running filter_osm_columns")
        filter_osm_columns(osm_path=args.osm_path, columns=args.columns, save=True)
        print("Done.")

    if args.interpolate_in_path is not None:
        interpolate_in_path = args.interpolate_in_path
        logger.info(
            f"Assuming the data is cleaned and is readey to be interpolated. The interpolate_in_path is set to {interpolate_in_path}."
        )
    else:
        interpolate_in_path = os.path.join(
            os.path.dirname(args.osm_path), "merged_cleaned.geojson"
        )

        print("Running osm_level_cleaner")
        osm_level_cleaner(
            osm_path=cleaner_in_path,
            column=args.column,
            save=True,
        )
        print("Done.")

    print("Running osm_level_interpolation")
    level_interpolation(
        osm_path=interpolate_in_path,
        column=args.column,
        save=True,
        out_name="merged_interpolated.geojson",
        area_unit=args.area_unit,
        size_unit=args.size_unit,
        tile_size=args.tile_size,
        average_function=average_function,
        total_average=args.total_average,
    )
    print("Done.")
    print("Running osm_level_categorise")
    osm_level_categorise(
        osm_path=os.path.join(
            os.path.dirname(args.osm_path), "merged_interpolated.geojson"
        ),
        save=True,
        categorise=categorise,
        out_name=args.out_name,
    )
    print("Done.")
    print("All operations completed successfully.")


if __name__ == "__main__":
    args = argparser()
    main(args)
