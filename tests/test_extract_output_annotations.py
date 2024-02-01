# -*- coding: utf-8 -*-
import os

from aerialseg.utils import extract_output_annotations


def test_output_dims():
    """Test extract_output_annotations function."""
    import cv2
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor

    TEST_DIR = os.path.dirname(os.path.abspath(__file__))

    # read sample image
    image = cv2.imread(os.path.join(TEST_DIR, "data/test1.png"))

    # load model
    cfg = get_cfg()
    config_file = os.path.join(TEST_DIR, "data/config.yml")
    if not os.path.exists(config_file):
        raise Exception(f"Config file not found at {config_file}")

    cfg.MODEL.DEVICE = "cpu"
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = os.path.join(TEST_DIR, "data/model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    predictor = DefaultPredictor(cfg)

    # predict
    mask_arrays, polygons, bbox, labels = extract_output_annotations(predictor(image))

    # test
    print(mask_arrays)
    print(polygons)
    print(bbox)
    print(labels)

    assert len(mask_arrays) >= 1
    assert len(polygons) >= 1
    assert len(bbox) >= 1
    assert len(labels) >= 1
