#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import glob
import json
import os

from detectron2.config import get_cfg

# from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor

from aerialseg.utils import assemble_coco_json, extract_all_annotations_df


def create_parser():
    parser = argparse.ArgumentParser(
        description="Run Detectron2 prediction on aerial imagery and show image with results."
    )
    parser.add_argument(
        "--indir",
        "-i",
        type=str,
        required=True,
        help="Path to a directory containing the input images.",
    )
    parser.add_argument(
        "--in-pattern",
        "-p",
        type=str,
        default="*.png",
        help="Glob pattern to match the input images. Default: %(default)s.",
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

    config_file = args.config
    weights_file = args.weights

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

    # scan the input directory for images based on the pattern given
    images = glob.glob(os.path.join(args.indir, args.in_pattern))
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
