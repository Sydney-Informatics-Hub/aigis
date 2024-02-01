# -*- coding: utf-8 -*-
"""This script supports converting a COCO dataset into a georeferenced geojson
file."""
import argparse

# import glob
import logging
import os
import warnings
from datetime import datetime

# import traceback
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
from shapely.ops import unary_union
from tqdm import tqdm

from aerial_conversion.coco import (
    coco_annotation_per_image_df,
    coco_categories_dict,
    polygon_prep,
)
from aerial_conversion.coordinates import (
    pixel_segmentation_to_spatial_rio,
    read_crs_from_raster,
)
from aerial_conversion.tiles import get_tiles_list_from_dir, load_tiles_from_list

# import rasterio as rio
warnings.simplefilter(action="ignore", category=FutureWarning)


log = logging.getLogger(__name__)

tqdm.pandas()


def merge_class_polygons_geopandas(tiles_df_zone_groups, crs, keep_geom_type):
    """Merge overlapping polygons in each class/zone.

    This method uses geopandas overlay to merge overlapping polygons in each class/zone.

    Args:
        tiles_df_zone_groups (list): List of GeoDataFrames, one per zone (building/annotation typle/class)
        crs (str): Coordinate system
        keep_geom_type (bool): If not set, return only geometries of the same geometry type as df1 has, otherwise, return all resulting geometries. It it advised not to set this parameter. It is known to casue polygon matching issues.

    Returns:
        polygons_df (GeoDataFrame): GeoDataFrame of merged polygons
    """
    print("Using the overlay method to merge overlapping polygons in each class/zone.")
    # Create a list of GeoDataFrames, one per zone (building/annotation typle/class)
    polygons_df_zone_groups = list()
    for tiles_df_zone in tiles_df_zone_groups:
        tiles_df_zone = tiles_df_zone.reset_index(drop=True)

        # Convert segmentations to polygons
        tiles_df_zone["geometry"] = tiles_df_zone.apply(
            lambda x: pixel_segmentation_to_spatial_rio(
                x["geotiff"], x["segmentation"]
            ),
            axis=1,
        )

        for i, row in tqdm(tiles_df_zone.iterrows(), total=tiles_df_zone.shape[0]):
            # Create a GeoDataFrame with the polygons
            polygons_df_tmp = gpd.GeoDataFrame(crs=crs, geometry=[row["geometry"]])
            # polygons_df_tmp["zone_code"] = row["zone_code"]
            # polygons_df_tmp["zone_name"] = row["zone_name"]
            # polygons_df_tmp["tile"] = row["tile_name"]
            if not polygons_df_tmp.empty:
                if i == 0:
                    polygons_df_zone = polygons_df_tmp.copy()
                else:
                    # Merge the GeoDataFrames and combine overlapping polygons
                    if row["marginal"] is True:
                        polygons_df_zone = gpd.overlay(
                            polygons_df_zone,
                            polygons_df_tmp,
                            how="union",
                            keep_geom_type=keep_geom_type,
                        )  # .reset_index(drop=True)
                        # TODO: Fix the following warning, or be careful abou the versions. (pandas==2.1.0, geopandas==0.13.2):
                        # FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.
                        #   In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes.
                        #   To retain the old behavior, exclude the relevant entries before the concat operation
                    else:
                        polygons_df_zone = pd.concat(
                            [polygons_df_zone, polygons_df_tmp], ignore_index=True
                        )

            # print(polygons_df_zone)
        polygons_df_zone["zone_code"] = row["zone_code"]
        polygons_df_zone["zone_name"] = row["zone_name"]
        polygons_df_zone_groups.append(polygons_df_zone)

    polygons_df = pd.concat(polygons_df_zone_groups, ignore_index=True)
    return polygons_df


def merge_class_polygons_shapely(tiles_df_zone_groups, crs):
    """Merge overlapping polygons in each class/zone.

    This method uses shapely unary_union to merge overlapping polygons in each class/zone.

    Args:
        tiles_df_zone_groups (list): List of GeoDataFrames, one per zone (building/annotation typle/class)
        crs (str): Coordinate system

    Returns:
        polygons_df (GeoDataFrame): GeoDataFrame of merged polygons
    """
    print(
        "Using the unary_union method to merge overlapping polygons in each class/zone."
    )
    # polygons_df_zone_groups = []
    for index, tiles_df_zone in tqdm(
        enumerate(tiles_df_zone_groups), total=len(tiles_df_zone_groups)
    ):
        tiles_df_zone = tiles_df_zone.reset_index(drop=True)

        # Convert segmentations to polygons
        tiles_df_zone["geometry"] = tiles_df_zone.apply(
            lambda x: pixel_segmentation_to_spatial_rio(
                x["geotiff"], x["segmentation"]
            ),
            axis=1,
        )

        # Merge overlapping polygons in each class/zone
        zone_name = tiles_df_zone["zone_name"][0]
        zone_code = tiles_df_zone["zone_code"][0]
        multipolygon = unary_union(tiles_df_zone["geometry"])
        # polygons = list(multipolygon.geoms)

        if index == 0:
            polygons_df = gpd.GeoDataFrame(crs=crs, geometry=[multipolygon])
            polygons_df["zone_code"] = zone_code
            polygons_df["zone_name"] = zone_name
        else:
            polygons_df_tmp = gpd.GeoDataFrame(crs=crs, geometry=[multipolygon])
            polygons_df_tmp["zone_code"] = zone_code
            polygons_df_tmp["zone_name"] = zone_name
            polygons_df = pd.concat([polygons_df, polygons_df_tmp], ignore_index=True)

    polygons_df = polygons_df.explode("geometry").reset_index(drop=True)

    # for poly in tqdm(polygons[1:]):
    #     polygons_df_tmp = gpd.GeoDataFrame(crs=crs, geometry=[poly])
    #     polygons_df_tmp["zone_code"] = zone_code
    #     polygons_df_tmp["zone_name"] = zone_name
    #     polygons_df = pd.concat([polygons_df, polygons_df_tmp], ignore_index=True)
    return polygons_df


def shape_regulariser(
    polygon, simplify_tolerance, minimum_rotated_rectangle, orthogonalisation
):
    """Regularise the shape of a polygon.

    Args:
        polygon (shapely.geometry.Polygon): Polygon to regularise
        simplify_tolerance (float): Tolerance for simplifying polygons. Accepts values between 0.0 and 1.0.
        minimum_rotated_rectangle (bool): If true, will return the minimum rotated rectangle of the polygon. If set, simplification will be ignored.
        orthogonalise (bool): If true, will orthogonalise the polygon. This does not work with minimum-rotated-rectangle. It only works with simplification.

    Returns:
        shapely.geometry.Polygon: Regularised polygon
    """
    polygon_point_tuples = list(polygon.exterior.coords)
    polygon = polygon_prep(
        polygon_point_tuples,
        simplify_tolerance,
        minimum_rotated_rectangle,
        orthogonalisation,
    )
    polygon = Polygon(polygon)

    return polygon


#%% Command-line driver


def main(args=None):
    """Command-line driver."""
    test_data_path = "/home/sahand/Data/GIS2COCO/chatswood/big_tiles_200_b/"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "tiledir",
        type=Path,
        default=test_data_path,
        help="Path to the input tiles directory with rasters. PNG files are not required.",
    )
    ap.add_argument(
        "cocojson",
        default=os.path.join(test_data_path, "coco-out-tol_0.4-b.json"),
        type=Path,
        help="Path to the input coco json file.",
    )
    ap.add_argument(
        "--tile-extension",
        default="tif",
        type=str,
        help="Extension of tiles. Defaults to %(default)s. Do not include a period before the extension.",
    )
    ap.add_argument(
        "--geojson-output",
        "-o",
        # required=True,
        default=os.path.join(
            test_data_path,
            f"coco_2_geojson_{datetime.today().strftime('%Y-%m-%d')}.geojson",
        ),
        type=Path,
        help="Path to output geojson file.",
    )
    ap.add_argument(
        "--tile-search-margin",
        default=0,
        type=int,
        help="-- Deprecation Warning -- Int percentage of tile size to use as a search margin for finding overlapping polygons while joining raster. This function only reports the marginal tiles and is not functional in any other way. Defaults to %(default)s.",
    )
    ap.add_argument(
        "--not-keep-geom-type",
        action=argparse.BooleanOptionalAction,
        help="-- Deprecated Argument -- If not set, return only geometries of the same geometry type as df1 has, otherwise, return all resulting geometries. It it advised not to set this parameter. It is known to casue polygon matching issues.",
    )
    ap.add_argument(
        "--simplify-tolerance",
        type=float,
        default=0.9,
        help="Tolerance for simplifying polygons. Accepts values between 0.0 and 1.0. Defaults to %(default)s.",
    )
    ap.add_argument(
        "--minimum-rotated-rectangle",
        action=argparse.BooleanOptionalAction,
        help="If true, will return the minimum rotated rectangle of the polygon. If set, simplification will be ignored.",
    )
    ap.add_argument(
        "--orthogonalisation",
        action=argparse.BooleanOptionalAction,
        help="If true, will orthogonalise the polygon. This does not work with minimum-rotated-rectangle. It only works with simplification. You can manually set a simplification tolerance, but a default value will be used if not set. If the shape is pre-simplified in the `aerial_annotation` package, then this will not be necessary to simplify again and you can set simplify-tolerance to a very low value or zero.",
    )
    ap.add_argument(
        "--meta-name",
        type=str,
        default="Aerial Segmentation Predictions",
        help="Name of the prediction.",
    )
    ap.add_argument(
        "--meta-type",
        type=str,
        default="FeatureCollection",
        help="Type of the output geojson file.",
    )
    ap.add_argument(
        "--properties-json",
        type=Path,
        default=None,
        help="Path to a JSON file containing common properties to add to the geojson file for each annotation.",
    )
    ap.add_argument(
        "--coordinates-z",
        type=float,
        default=None,
        help="A common Z coordinate to use for all annotations. If not set, will not add Z coordinates.",
    )
    ap.add_argument(
        "--license",
        type=Path,
        help="Path to a license description in COCO JSON format. If not supplied, will default to MIT license.",
    )
    args = ap.parse_args(args)

    """
    Read tiles and COCO JSON, and convert to GeoJSON.
    """
    geojson_path = args.geojson_output
    tile_dir = args.tiledir
    meta_name = args.meta_name
    coco_json_path = args.cocojson
    tile_extension = args.tile_extension
    simplify_tolerance = args.simplify_tolerance
    minimum_rotated_rectangle = args.minimum_rotated_rectangle
    orthogonalisation = args.orthogonalisation

    # keep_geom_type = (
    #     not args.not_keep_geom_type
    # )  # should be True # only meaningful when using geopandas overlay method
    tile_search_margin = args.tile_search_margin

    # Read tiles
    log.info("Reading tiles from %s" % tile_dir)
    tiles_list = get_tiles_list_from_dir(tile_dir, tile_extension)
    geotifs = load_tiles_from_list(tiles_list=tiles_list)
    crs = read_crs_from_raster(tiles_list[0])

    # Read COCO JSON and extract annotations per reference image
    tiles_df = pd.DataFrame(
        zip(tiles_list, geotifs), columns=["raster_path", "geotiff"]
    )
    tiles_df["tile_name"] = tiles_df["raster_path"].apply(
        lambda x: os.path.basename(x).split(".")[0]
    )

    coco_images_df = coco_annotation_per_image_df(coco_json_path, tile_search_margin)
    coco_categories = coco_categories_dict(coco_json_path)
    # coco_images_df["annotations"][0]["bbox"]

    # Merge COCO JSON with tiles
    tiles_df = pd.merge(tiles_df, coco_images_df, on="tile_name", how="outer")

    # Unpack annotations to segmentation column, yielding longer dataframe
    tiles_df.dropna(subset=["annotations"], inplace=True)
    tiles_df["segmentation"] = tiles_df["annotations"].apply(
        lambda x: x["segmentation"]
    )
    tiles_df["bbox"] = tiles_df["annotations"].apply(lambda x: x["bbox"])
    tiles_df["zone_code"] = tiles_df["annotations"].apply(lambda x: x["category_id"])
    tiles_df["zone_name"] = tiles_df["zone_code"].apply(
        lambda x: coco_categories[x]["name"]
    )
    tiles_df["marginal"] = tiles_df["annotations"].apply(lambda x: x["marginal"])
    tiles_df = tiles_df.drop(columns=["annotations"])

    # tiles_df[tiles_df["marginal"]==False]

    # Group by zone ID, extract polygons, and merge overlapping polygons in the same zone
    tiles_df_grouped = tiles_df.groupby(["zone_code"]).groups
    tiles_df_zone_groups = list()
    for zone in tiles_df_grouped:
        tiles_df_zone_groups.append(tiles_df.loc[tiles_df_grouped[zone]])

    polygons_df = merge_class_polygons_shapely(tiles_df_zone_groups, crs)
    # polygons_df = merge_class_polygons_geopandas(tiles_df_zone_groups,crs,keep_geom_type) # geopandas overlay method -- slow

    # change crs of the gpd and its geometries to "epsg:4326"
    original_crs = polygons_df.crs
    polygons_df = polygons_df.to_crs("epsg:4326")

    print("Regularising the shape of the polygons.")
    # print(polygons_df)
    polygons_df["geometry"] = polygons_df["geometry"].progress_apply(
        lambda x: shape_regulariser(
            x, simplify_tolerance, minimum_rotated_rectangle, orthogonalisation
        )
    )

    polygons_df = polygons_df.to_crs(original_crs)

    # print(polygons_df)

    try:
        polygons_df.Name = meta_name
    except Exception as e:
        log.error(f"Could not set Name property of geojson. Error message: {e}")
        print("FIX this code!")
    # Save to geojson
    polygons_df.to_file(geojson_path, driver="GeoJSON")


if __name__ == "__main__":
    main()
