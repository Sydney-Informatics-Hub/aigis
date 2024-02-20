# -*- coding: utf-8 -*-
"""This script supports converting a vector shapefile and raster file into a
COCO format dataset."""
# -*- coding: utf-8 -*-
import argparse
import glob
import logging
import os
import os.path
import traceback
from pathlib import Path

import geopandas as gpd
import rasterio as rio

from aerial_conversion.coco import (
    coco_image_annotations,
    coco_json,
    coco_polygon_annotations,
    make_category_object,
)
from aerial_conversion.coordinates import pixel_polygons_for_raster_tiles, wkt_parser
from aerial_conversion.tiles import save_tiles

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)


def assemble_coco_json(
    raster_file_list, geojson, license_json, info_json, categories_json, colour
):

    pixel_poly_df = pixel_polygons_for_raster_tiles(raster_file_list, geojson)
    # pixel_poly_df.to_csv(json_name[:-5]+"_df.csv")
    # pixel_poly_df_sample = pixel_poly_df.sample(1)
    # pixel_poly_df_sample.to_csv(json_name[:-5]+".csv")

    coco = coco_json()
    coco.images = coco_image_annotations(raster_file_list, colour).images
    coco.annotations = coco_polygon_annotations(pixel_poly_df)
    coco.license = license_json
    coco.categories = categories_json
    coco.info = info_json
    coco.type = "instances"

    return coco


#%% Command-line driver


def main(args=None):
    """Command-line driver."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--polygon-file",
        required=True,
        default=".",
        type=Path,
        help="Path to a georeferenced shapefile polygon vector file (e.g. geoJSON) that contains annotations for the raster file.",
    )
    ap.add_argument(
        "--raster-file",
        required=True,
        type=Path,
        help="Path to the raster file (e.g. geoTIFF).",
    )
    ap.add_argument(
        "--tile-size",
        default=1000,
        type=float,
        help="Int length in meters of square tiles to generate from raster. Defaults to 1000 meters.",
    )
    ap.add_argument(
        "--tile-dir",
        required=True,
        type=Path,
        help="Path to where the cut raster tiles should be stored.",
    )
    ap.add_argument(
        "--class-column",
        required=True,
        type=str,
        help="Column name in GeoJSON where classes are stored.",
    )
    ap.add_argument(
        "--json-name",
        default="coco_from_gis.json",
        type=Path,
        help="Path to the output COCO JSON file.",
    )
    ap.add_argument(
        "--crs", type=str, default=None, help="Specifiy the project crs to use."
    )
    ap.add_argument(
        "--trim-class",
        default=0,
        type=int,
        help="Characters to trim of the start of each class name. A clummsy solution, set to 0 by default which leaves class names as is.",
    )
    ap.add_argument(
        "--cleanup",
        action=argparse.BooleanOptionalAction,
        help="If set, will purge *.tif tiles from the directory. Default to false.",
    )
    ap.add_argument(
        "--save-gdf",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="If set, will save a GeoDataFrame that you can use to reconstruct a spatial version of the dataset.",
    )
    ap.add_argument(
        "--short-file-name",
        action=argparse.BooleanOptionalAction,
        help="If set, saves a short file name in the COCO for images.",
    )
    ap.add_argument(
        "--grayscale",
        action=argparse.BooleanOptionalAction,
        help="If set, will generate grayscale images.",
    )
    ap.add_argument(
        "--offset",
        default=0.0,
        type=float,
        help="Padding/offset/overlap in percentage of tile. Defaults to 0.0.",
    )
    ap.add_argument(
        "--license",
        type=Path,
        help="Path to a license description in COCO JSON format. If not supplied, will default to MIT license.",
    )
    ap.add_argument(
        "--info",
        required=True,
        type=Path,
        help="Path to info description in COCO JSON format. This can be an empty file.",
    )
    args = ap.parse_args(args)

    """
    Create tiles from raster and convert to COCO JSON format.
    """
    # root_dir = "/home/sahand/Data/GIS2COCO/"
    # raster_path = os.path.join(root_dir, "chatswood/chatswood_hd.tif")
    # geojson_path = os.path.join(root_dir, "chatswood/chatswood.geojson")
    # out_path = os.path.join(root_dir, "chatswood/tiles/")
    # tile_size = 500
    # user_crs = None
    # class_column = "zone_name"  # "zone_name" # "zone_code"
    # trim_class = 0
    # license = None
    # info = os.path.join(root_dir, "chatswood/info.json")
    # json_name = os.path.join(root_dir, "chatswood/coco_from_gis.json")
    # colour = True
    # offset = 0.0

    raster_path = args.raster_file
    geojson_path = args.polygon_file
    out_path = args.tile_dir
    tile_size = args.tile_size
    # change tile size to float from string
    try:
        tile_size = float(tile_size)
    except ValueError:
        pass
    user_crs = args.crs
    class_column = args.class_column
    trim_class = args.trim_class
    license = args.license
    info = args.info
    json_name = args.json_name
    colour = not args.grayscale
    offset = args.offset

    log.info(f"Creating {tile_size} m*m tiles from {raster_path}")

    # Read input files
    geotiff = rio.open(raster_path)
    geojson = gpd.read_file(geojson_path)
    # geojson.crs
    # Reproject geojson on geotiff
    if user_crs is None:
        user_crs = geotiff.crs.to_wkt()
        user_crs = wkt_parser(user_crs)

    try:
        geojson = geojson.to_crs(user_crs)
    except Exception as e:
        log.error(f"CRS not recognized, please specify a valid CRS. Error message: {e}")
        traceback.print_exc()
        raise e

    # Create raster tiles
    save_tiles(
        geotiff, out_path, tile_size, tile_template="tile_{}-{}.tif", offset=offset
    )
    geotiff.close()

    # Read the created raster tiles into a list.
    raster_file_list = []
    for filename in glob.iglob(os.path.join(f"{out_path}", "*.tif")):
        raster_file_list.append(filename)

    log.info(f"{len(raster_file_list)} raster tiles created")

    # Create class_id for category mapping
    # Check if the specified class column exists
    # TODO: make less hacky
    if args.class_column not in geojson.columns:
        # If it doesn't exist, create a new column with the specified name and fill it with the string value of class-column argument
        log.error(
            f"Class column {args.class_column} not found in GeoJSON. Will create one instead..."
        )
        geojson[args.class_column] = args.class_column
    geojson["class_id"] = geojson[class_column].factorize()[0]
    log.debug("Class column is: %s", class_column)
    log.debug("Class id is: %s", geojson["class_id"])
    log.debug("Trim class is: %s", trim_class)
    categories_json = make_category_object(geojson, class_column, trim_class)
    
    # Make sure geojson class_column is string type
    geojson[class_column] = geojson[class_column].astype(str)

    # If license is not supplied, use MIT by default
    if license is None:
        license_json = {
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License",
        }
    else:
        # Read user supplied license
        # TODO: incorporate different licenses depending on images: this feature is almost never used but would be nice to support.
        license_json = open(license, "r")

    info_json = open(info, "r")

    log.info("Converting to COCO")
    # We are now ready to make the COCO JSON.
    spatial_coco = assemble_coco_json(
        raster_file_list, geojson, license_json, info_json, categories_json, colour
    )

    # Write COCO JSON to file.
    with open(json_name, "w") as f:
        f.write(spatial_coco.toJSON())
    log.info(f"COCO JSON saved to {json_name}")

    # if args.save_gdf:
    #     pixel_poly_df['raster_tile_name'] = pixel_poly_df.apply(lambda row: os.path.basename(row['raster_tile']), axis = 1)
    #     with open ("gdf_output.csv", "w") as f:
    #         f.write(pixel_poly_df)


if __name__ == "__main__":
    main()
