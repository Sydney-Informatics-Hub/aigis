#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import glob
import json
import logging
import os

# import geopandas as gpd
import rasterio as rio
from aerial_conversion.coco import raster_to_coco
from aerial_conversion.tiles import save_tiles
from detectron2.config import get_cfg

# from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor

from aerialseg.utils import assemble_coco_json, extract_all_annotations_df

# import traceback


log = logging.getLogger(__name__)


def create_parser():
    parser = argparse.ArgumentParser(
        description="Run Detectron2 prediction on aerial imagery and show image with results."
    )
    parser.add_argument(
        "--raster-file",
        "-r",
        type=str,
        help="Path to a raster file. Will split the raster into tiles and run prediction on each tile, before merging them back. ",
    )
    parser.add_argument(
        "--tile-size",
        "-z",
        type=float,
        default=1000,
        help="Tile size in degrees. Default: %(default)s.",
    )
    parser.add_argument(
        "--overlap",
        "-l",
        type=int,
        default=10,
        help="Overlap size in percent. Default: %(default)s.",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Model configuration YAML file from Detectron2.",
    )
    parser.add_argument(
        "--weights",
        "-w",
        type=str,
        required=True,
        help="Path to a weights .pth file output from Detectron2 training.",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.7,
        help="Detection threshold. Default: %(default)s.",
    )
    parser.add_argument(
        "--simplify-tolerance",
        "-s",
        type=float,
        default=0.9,
        help="Tolerance for simplifying polygons. Accepts values between 0.0 and 1.0. Default: %(default)s.",
    )
    parser.add_argument(
        "--minimum-rotated-rectangle",
        "-m",
        action=argparse.BooleanOptionalAction,
        help="If set, will return the minimum rotated rectangle of the polygons.",
    )
    parser.add_argument(
        "--coco",
        type=str,
        default=None,
        help="Path to a COCO JSON containg the annotation categories. "
        "These will be printed onto the image if provided.",
    )
    parser.add_argument(
        "--force-cpu",
        action=argparse.BooleanOptionalAction,
        help="If set, will force CPU inference.",
    )
    parser.add_argument(
        "--coco-out",
        "-o",
        type=str,
        default=None,
        help="Path to a COCO JSON file to save the predictions to. By default will save to the same directory as the input image with 'coco-out.json' name.",
    )
    parser.add_argument(
        "--temp-dir",
        "-d",
        type=str,
        default=None,
        help="Path to a temporary directory to store the raster tiles. By default will use the system temp directory.",
    )

    return parser

    # Debugging toy inputs:
    # config_file = "tests/data/config.yml"
    # weights_file = "tests/data/model_final.pth"
    # images = glob.glob(os.path.join("/home/sahand/Data/GIS2COCO/chatswood/big_tiles_200_b/", "*.png"))
    # coco = "/home/sahand/Data/GIS2COCO/chatswood/big_tiles_200_b/coco_from_gis_hd_200.json"
    # out = "/home/sahand/Data/GIS2COCO/chatswood/big_tiles_200_b/coco-out-tol_0.9-b.json"
    # out = "/home/sahand/Data/GIS2COCO/chatswood/big_tiles_200_b/coco-out-mrr.json"


def main(args=None):
    parser = create_parser()
    args = parser.parse_args()

    raster_path = args.raster_file
    tile_size = args.tile_size
    config_file = args.config
    weights_file = args.weights
    offset = args.overlap

    # Create a temporary directory to store the raster tiles.
    if args.temp_dir is None:
        out_path = os.path.join(".", ".tmp", "tiles")

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Preapare raster and tiles
    log.info(f"Creating {tile_size} m*m tiles from {raster_path}")

    # Read input files
    geotiff = rio.open(raster_path)

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

    # Make png images from the tiles
    images = []
    for filename in raster_file_list:
        raster_to_coco(filename, 0, "png")
        images.append(filename.replace(".tif", ".png"))

    log.info(f"{len(images)} png tiles created")

    # Prepare model
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = weights_file
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.threshold

    # If we have the COCO JSON then we can set up the class names for the prediction.
    if args.coco is not None:
        with open(args.coco, "r") as f:
            coco = json.load(f)
        categories = coco["categories"]
        categories_keyed = {
            c["id"]: {"name": c["name"], "supercategory": c["supercategory"]}
            for c in categories
        }
        # thing_classes = [c["name"] for c in categories] # These 3 steps are not required for inference if not visualising
        # meta = MetadataCatalog.get("predict")
        # meta.thing_classes = thing_classes
    else:
        categories_keyed = None

    assert (
        len(images) > 0
    ), f"No images found in the input directory given the pattern {args.in_pattern}."

    # If not many images are given, this should be okay to go with CPU inference.
    if args.force_cpu:
        cfg.MODEL.DEVICE = "cpu"

    predictor = DefaultPredictor(cfg)

    all_annotations = extract_all_annotations_df(
        images,
        predictor,
        simplify_tolerance=args.simplify_tolerance,
        minimum_rotated_rectangle=args.minimum_rotated_rectangle,
    )
    coco_json = assemble_coco_json(
        all_annotations,
        images,
        categories=categories_keyed,
        license="",
        info="",
        type="instances",
    )
    if args.coco_out is None:
        if args.minimum_rotated_rectangle:
            args.coco_out = os.path.join(args.indir, "coco-out-mrr.json")
        else:
            args.coco_out = os.path.join(
                args.indir, f"coco-out-tol_{str(args.simplify_tolerance)}.json"
            )

    coco_json.write_to_file(args.coco_out)


if __name__ == "__main__":
    main()
