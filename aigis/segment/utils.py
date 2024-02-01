# -*- coding: utf-8 -*-
import os
import warnings

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from aerial_conversion import coco
from detectron2.utils.visualizer import Visualizer
from matplotlib import pylab as plt
from PIL import Image
from shapely.geometry import Polygon
from tqdm import tqdm

"""
Plotting and visualisation utilities
"""

def save_images_as_gif(input_folder, output_gif_path, duration=100):
    """Save a folder of images as an animated GIF.

    Args:
        input_folder (str): Path to the folder containing image files (e.g., JPEG or PNG).
        output_gif_path (str): Path to save the animated GIF file.
        duration (int, optional): Duration (in milliseconds) for each frame in the GIF. Default is 100ms.

    Returns:
        None
    """
    image_files = [
        f
        for f in os.listdir(input_folder)
        if f.endswith((".jpg", ".png", ".jpeg", ".gif"))
    ]

    if not image_files:
        print("No image files found in the input folder.")
        return

    images = []
    for image_file in sorted(image_files):
        image_path = os.path.join(input_folder, image_file)
        img = Image.open(image_path)
        images.append(img)

    # Save the animated GIF
    images[0].save(
        output_gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
    )


def plot_polygons(polygons):

  # Reshape the coordinates to separate x and y values
  olygons = [np.array(polygons[i]).reshape(-1, 2) for i in range(len(polygons))]

  # Create a plot
  fig, ax = plt.subplots()

  # Plot each polygon
  for polygon in polygons:
      ax.plot(polygon[:, 0], polygon[:, 1], marker='o', linestyle='-')

  # Set labels and title
  ax.set_xlabel('X-axis')
  ax.set_ylabel('Y-axis')
  ax.set_title('Polygon Predictions')
  plt.gca().invert_yaxis()

  # Show the plot
  plt.show()


def polygon_prep(
    polygon, simplify_tolerance: float = 0.0, minimum_rotated_rectangle: bool = False
):
    """Prepares a polygon for export.

    Args:
        polygon (list): A list of coordinates
        simplify_tolerance (float, optional): Tolerance for simplifying polygons. Accepts values between 0.0 and 1.0. Defaults to 0.0. If simplify_tolerance > 0, will simplify the polygon, without minimum rotated rectangle.
        minimum_rotated_rectangle (bool, optional): If true, will return the minimum rotated rectangle of the polygon. Defaults to False. If simplify_tolerance > 0, will simplify the polygon without minimum rotated rectangle.

    Returns:
        polygon (list): A list of coordinates
    """
    if len(polygon) < 3:
        warnings.warn(
            f"The polygon has less than 3 points! This is not an actual polygon, and can be a line or point(s). Polygon: {polygon}."
        )
    polygon = Polygon(polygon)
    if minimum_rotated_rectangle:
        polygon = polygon.minimum_rotated_rectangle
    else:
        if simplify_tolerance > 0:
            polygon = polygon.simplify(simplify_tolerance)
    polygon = np.array(polygon.exterior.coords)

    return polygon


def extract_output_annotations(
    output,
    flatten: bool = False,
    simplify_tolerance: float = 0.0,
    minimum_rotated_rectangle: bool = False,
):
    """Extracts polygons, bounding boxes, and binary masks from prediction
    ouputs.

    Args:
        output: Detectron2 prediction output
        flatten (bool): If true, will flatten polygons, as such used in coco segmentations.
        simplify_tolerance (float, optional): Tolerance for simplifying polygons. Accepts values between 0.0 and 1.0. Defaults to 0.0.
        minimum_rotated_rectangle (bool, optional): If true, will return the minimum rotated rectangle of the polygon. Defaults to False.

    Returns:
        mask_arrays (list): A list of binary masks
        polygons (list): A list of polygons
        bbox (list): A list of bounding boxes
        labels (list): A list of labels
    """
    mask_array = output["instances"].pred_masks.to("cpu").numpy()
    num_instances = mask_array.shape[0]
    # scores = output['instances'].scores.to("cpu").numpy()
    labels = output["instances"].pred_classes.to("cpu").numpy()
    bbox = output["instances"].pred_boxes.to("cpu").tensor.numpy()
    # print(mask_array.shape)
    mask_array = np.moveaxis(mask_array, 0, -1)
    mask_arrays = []
    polygons = []
    labels_list = []
    bbox_list = []

    for i in range(num_instances):
        # img = np.zeros_like(image)
        mask_array_instance = mask_array[:, :, i : (i + 1)]

        # img = np.where(mask_array_instance[i] == True, 255, img)
        polygon_sv = sv.mask_to_polygons(mask_array_instance)

        if len(polygon_sv) > 0:  # if there is at least one polygon
            for polygon in polygon_sv:
                polygon = polygon_prep(
                    polygon,
                    simplify_tolerance=simplify_tolerance,
                    minimum_rotated_rectangle=minimum_rotated_rectangle,
                )
                mask_arrays.append(mask_array_instance)
                labels_list.append(labels[i])
                bbox_list.append(bbox[i])

                if flatten:
                    polygons.append(polygon.flatten().tolist())
                else:
                    polygons.append(polygon.tolist())

        else:
            warnings.warn(f"Polygon {i} is empty! Skipping polygon.")

    return mask_arrays, polygons, bbox_list, labels_list


def extract_tile_annotations_df(
    image_path,
    image_id,
    predictor,
    simplify_tolerance: float = 0.0,
    minimum_rotated_rectangle: bool = False,
):
    """Reads through tiles, predicts, and extracts annnotations as a dataframe.

    Args:
        image_path (str): path to the tile png file
        image_id (int): an id for the image tile. Usually a unique int
        predictor: Detectron2 predictor object
        simplify_tolerance (float, optional): Tolerance for simplifying polygons. Accepts values between 0.0 and 1.0. Defaults to 0.0.
        minimum_rotated_rectangle (bool, optional): If true, will return the minimum rotated rectangle of the polygon. Defaults to False.

    Returns:
        Pandas.DataFrame: A dataframe of annotations
    """
    image = cv2.imread(image_path)
    output = predictor(image)
    _, polygons, _, labels = extract_output_annotations(
        output,
        simplify_tolerance=simplify_tolerance,
        minimum_rotated_rectangle=minimum_rotated_rectangle,
    )
    annotations = pd.DataFrame(
        {"pixel_polygon": polygons, "image_id": image_id, "class_id": labels}
    )  # "annot_id" should be added later

    return annotations


def extract_all_annotations_df(
    images_list: list,
    predictor,
    simplify_tolerance: float = 0.0,
    minimum_rotated_rectangle: bool = False,
):
    """Extract and combine tile annotations into a single dataframe.

    Args:
        images_list (list): A list of image paths
        predictor: Detectron2 predictor object
        simplify_tolerance (float, optional): Tolerance for simplifying polygons. Accepts values between 0.0 and 1.0. Defaults to 0.0.
        minimum_rotated_rectangle (bool, optional): If true, will return the minimum rotated rectangle of the polygon. Defaults to False.

    Returns:
        Pandas.DataFrame: A dataframe of annotations
    """

    all_annotations = []
    for image_index, image in tqdm(enumerate(images_list), total=len(images_list)):
        all_annotations.append(
            extract_tile_annotations_df(
                image,
                image_index,
                predictor,
                simplify_tolerance=simplify_tolerance,
                minimum_rotated_rectangle=minimum_rotated_rectangle,
            )
        )

    all_annotations = pd.concat(all_annotations)
    all_annotations = all_annotations.reset_index(drop=True)
    all_annotations = all_annotations.reset_index()
    all_annotations.columns = ["annot_id", "pixel_polygon", "image_id", "class_id"]

    return all_annotations


def assemble_coco_json(
    annotations,
    images,
    categories: dict = None,
    license: str = "",
    info: str = "",
    type: str = "instances",
):
    """Generate a coco json object.

    Args:
        annotations (Pandas.DataFrame): a dataframe of annotations, usually generated via extract_all_annotations_df function.
        images (list): a list of image paths
        license (str): license of the dataset
        info (str): info of the dataset
        type (str, optional): type of the segmentation. Defaults to "instances"

    Returns:
        coco_json: a coco json object
    """
    coco_json = coco.coco_json()
    coco_json.images = coco.create_coco_images_object_png(images).images
    coco_json.annotations = coco.coco_polygon_annotations(
        annotations
    )  # [tmp2]#[annots_tmp[0]]#
    coco_json.license = license
    coco_json.type = type
    coco_json.info = info
    if categories is not None:
        coco_json.categories = [
            coco.make_category(
                class_name=str(categories[cat]["name"]),
                class_id=cat,
                supercategory=categories[cat]["supercategory"],
            )
            for cat in annotations.groupby("class_id").groups.keys()
        ]
    else:
        coco_json.categories = [
            coco.make_category(class_name=str(cat), class_id=cat)
            for cat in annotations.groupby("class_id").groups.keys()
        ]

    return coco_json


def visualize_or_save_image(image: str, predictor, meta=None, png_out: str = ""):
    """Process an image for object instance detection, visualize the results,
    and optionally save them as a PNG.

    Args:
        image (str): The input image file path.
        predictor: The object instance detection model.
        meta: The metadata catalog for the model.
        png_out (str, optional): If provided, save the visualization as a PNG at this file path.

    Returns:
        None
    """
    im = cv2.imread(image)
    # Could serialise the outputs to a file
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1], metadata=meta)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    plt.figure(figsize=(8, 8), tight_layout=True)
    plt.imshow(out.get_image())
    plt.axis("off")
    if png_out:
        plt.savefig(png_out)
    else:
        plt.show()
