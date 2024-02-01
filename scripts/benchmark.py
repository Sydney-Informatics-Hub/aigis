# -*- coding: utf-8 -*-
"""Example Use:

python scripts/benchmark.py --input-dir demo_data --output-dir
output_data --config-yaml weights/poc_cfg.yaml --model-weights
weights/model_final.pth --roi-score-thresh 0.2
"""
import argparse
import os
import time

import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from PIL import Image


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Segment buildings in images using Detectron2"
    )
    parser.add_argument(
        "--input-dir", required=True, help="Input directory containing images"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory to save annotated images"
    )
    parser.add_argument(
        "--config-yaml", required=True, help="Path to the Detectron2 config YAML file"
    )
    parser.add_argument(
        "--model-weights", required=True, help="Path to the model weights file"
    )
    parser.add_argument(
        "--roi-score-thresh",
        type=float,
        default=0.5,
        help="ROI score threshold for detection",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Set up Detectron2 configuration and model
    cfg = get_cfg()
    cfg.merge_from_file(args.config_yaml)
    cfg.MODEL.WEIGHTS = args.model_weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.roi_score_thresh
    # cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8
    cfg.MODEL.DEVICE = "cpu"
    predictor = DefaultPredictor(cfg)

    os.makedirs(args.output_dir, exist_ok=True)
    iteration_time_sum = 0.0
    io_time_sum = 0.0
    predict_time_sum = 0.0
    total_images = 0

    for filename in os.listdir(args.input_dir):
        if (
            filename.endswith(".jpg")
            or filename.endswith(".png")
            or filename.endswith(".tif")
        ):
            input_path = os.path.join(args.input_dir, filename)
            output_path = os.path.join(args.output_dir, filename)

            start_time = time.time()
            im = cv2.imread(input_path)
            after_read_time = time.time()
            annotated_im = segment_buildings(im, predictor)
            after_predict_time = time.time()
            annotated_im.save(output_path)
            end_time = time.time()

            read_time = after_read_time - start_time
            predict_time = after_predict_time - after_read_time
            write_time = end_time - after_predict_time
            iteration_time = end_time - start_time
            iteration_time_sum += iteration_time
            io_time_sum += read_time + write_time
            predict_time_sum += predict_time
            print(
                f"Processed: {filename} | "
                f"Read+Write time: {read_time + write_time:.2f} s | "
                f"Predict time: {predict_time:.2f} s | "
                f"Total time: {iteration_time:.2f} s"
            )
            total_images += 1

    # Calculate and print performance benchmarks
    print("\nPerformance Benchmarks:")
    print(f"Total Images Processed: {total_images}")
    print(f"Average Read/Write Time: {io_time_sum / total_images:.2f} s")
    print(f"Average Predict Time: {predict_time_sum / total_images:.2f} s")
    print(f"Average Iteration Time: {iteration_time_sum / total_images:.2f} s")


def segment_buildings(im, predictor):
    im = np.array(im)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1], scale=0.5, instance_mode=ColorMode.IMAGE_BW)
    print(len(outputs["instances"]), "buildings detected.")
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    return Image.fromarray(out.get_image()[:, :, ::-1])


if __name__ == "__main__":
    main()
