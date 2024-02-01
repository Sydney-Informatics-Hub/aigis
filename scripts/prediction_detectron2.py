#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor

from aerialseg.utils import visualize_or_save_image


def create_parser():
    parser = argparse.ArgumentParser(
        description="Run Detectron2 prediction on aerial imagery and show image with results."
    )
    parser.add_argument("image", type=str, help="Image on which to run prediction.")
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
        "--coco",
        type=str,
        default=None,
        help="Path to a COCO JSON containg the annotation categories. "
        "These will be printed onto the image if provided.",
    )
    parser.add_argument(
        "--png-out",
        type=str,
        default=None,
        help="PNG filename to write out the annotated image. "
        "Default is to display image on the screen.",
    )
    return parser


def main(args=None):
    parser = create_parser()
    args = parser.parse_args()
    cfg = get_cfg()

    config_file = args.config
    weights_file = args.weights
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = weights_file
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.threshold

    meta = MetadataCatalog.get("predict")
    # If we have the COCO JSON then we can set up the class names for the prediction.
    if args.coco is not None:
        with open(args.coco, "r") as f:
            coco = json.load(f)
        categories = coco["categories"]
        thing_classes = [c["name"] for c in categories]
        meta.thing_classes = thing_classes

    # Just need the CPU for a single image
    cfg.MODEL.DEVICE = "cpu"
    predictor = DefaultPredictor(cfg)

    visualize_or_save_image(
        image=args.image, predictor=predictor, meta=meta, png_out=args.png_out
    )


if __name__ == "__main__":
    main()
