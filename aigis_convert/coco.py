# -*- coding: utf-8 -*-
"""A collectio of functions and structures for reading, writing, and creating
coco annotations."""
import json
import logging
import os
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
from PIL import Image
from shapely.geometry import Polygon

from .orthogonalise import orthogonalise

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)


class coco_json:
    """Class to hold the coco json format.

    Attributes:
        coco_image (coco_image): coco_image object. The class is intended to hold the meta data, instead of actual image arrays.
        coco_images (coco_images): coco_images object containing a list of coco_image objects.
        coco_poly_ann (coco_poly_ann): coco_poly_ann object containing a single polygon annotation.
        coco_poly_anns (coco_poly_anns): coco_poly_anns object containing a list of coco_poly_ann objects.
    """

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, indent=4)

    def write_to_file(self, filepath):
        with open(filepath, "w") as f:
            f.write(self.toJSON())

    class coco_image:
        """Class to hold a single coco image object."""

        pass

    class coco_images:
        """Class to hold a list of coco_image objects."""

        pass

    class coco_poly_ann:
        """Class to hold a single coco polygon annotation."""

        pass

    class coco_poly_anns:
        """Class to hold a list of ccoco_poly_ann."""

        pass


def make_category(
    class_name: str, class_id: int, supercategory: str = "landuse", trim=0
):
    """Function to build an individual COCO category.

    Args:
        class_name (str): Name of class
        class_id (int): ID of class
        supercategory (str, optional): Supercategory of class. Defaults to "landuse".
        trim (int, optional): Number of characters to trim from class name. Defaults to 0.

    Returns:
        category (dict): COCO category object
    """
    log.debug(f"Making category for {class_name} with ID {class_id}.")
    category = {
        "supercategory": supercategory,
        "id": int(class_id),
        "name": class_name[trim:],
    }
    return category


def make_category_object(
    geojson: gpd.GeoDataFrame,
    class_column: str,
    trim: int,
    supercategory_default: str = "landuse",
):
    """Function to build a COCO categories object.

    Args:
        geojson (gpd.GeoDataFrame): GeoDataFrame containing class data
        class_column (str): Name of column containing class names
        trim (int): Number of characters to trim from class name
        supercategory_default (str, optional): Default supercategory of classes. Defaults to "landuse".

    Returns:
        categories_json (list): List of COCO category objects
    """

    # TODO: Implement way to read supercategory data.

    log.debug(f"Making category object from {class_column} column.")

    classes = pd.DataFrame(geojson[class_column].unique(), columns=["class"])
    classes["class_id"] = classes.index
    categories_json = []

    log.info(f"Making category object with {len(classes)} classes.")
    log.info(f"The classes df is {str(classes)}")

    for _, row in classes.iterrows():
        categories_json.append(
            make_category(row["class"], row["class_id"], supercategory_default, trim)
        )

    return categories_json


def raster_to_coco(
    raster_path: str,
    index: int,
    extension: str = "png",
    bands: int = 3,
    colour: bool = True,
):
    """Function to convert a raster to a COCO image object.

    Args:
        raster_path (str): Path to raster
        index (int): Index of raster
        extension (str, optional): Extension of image. Defaults to "png".
        bands (int): The number of bands to save. Default is 3 (for R-G-B).
        colour (bool, optional): If True, save image in colour. Defaults to True.

    Returns:
        image (coco_image): COCO image object
    """

    geotiff = rio.open(raster_path)
    raster = geotiff.read()
    # print(raster[:3].shape)
    if bands > 1:
        raster = raster[:bands]
        if colour is False:
            # take the average of raster bands
            raster = raster.mean(axis=0)
            bands = 1
    # print(raster.shape)
    raster_name = os.path.splitext(raster_path)[0]
    image_name = f"{raster_name}.{extension}"

    with rio.Env():
        with rio.open(
            image_name,
            "w",
            driver=extension.upper(),
            height=geotiff.shape[0],
            width=geotiff.shape[1],
            count=bands,
            dtype=geotiff.dtypes[0],
            nodata=0,
            compress="deflate",
            transform=geotiff.profile["transform"],
        ) as dst:
            if colour is False:
                dst.write(raster, 1)
            else:
                dst.write(raster)

    # Create each individual image object
    image = coco_json.coco_image()
    image.license = 1
    image.file_name = os.path.basename(image_name)
    image.height = geotiff.shape[0]
    image.width = geotiff.shape[1]
    image.id = index

    return image


def create_coco_image_object_png(image_path: str, index: int):
    """Function to create a COCO image (coco_json.coco_image) object.

    Args:
        image_path (str): Path to image
        index (int): Index of image

    Returns:
        image (coco_image): COCO image object
    """
    im = Image.open(image_path)

    image = coco_json.coco_image()
    image.license = 1
    image.file_name = os.path.basename(image_path)
    image.width, image.height = im.size
    image.id = index

    return image


def create_coco_images_object_png(image_path_list: list):
    """Function to create a COCO images (coco_json.coco_images) object.

    Args:
        image_path_list (list): List of image paths

    Returns:
        images (coco_images): coco_images object
    """
    images = coco_json.coco_images()
    images.images = [
        create_coco_image_object_png(image_path, index)
        for index, image_path in enumerate(image_path_list)
    ]

    return images


def coco_bbox(polygon):
    """Generate a COCO format bounding box from a Polygon.

    Based on code from:
    #https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch/#coco-dataset-format

    Args:
        polygon (Polygon): Polygon object

    Returns:
        cc_bbox (list): COCO format bounding box
    """

    bounds = polygon.bounds
    top_left_x = bounds[0]
    top_left_y = bounds[1]  # lowest y val, cause it's from top down.
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    cc_bbox = [top_left_x, top_left_y, width, height]

    return cc_bbox


def coco_polygon_annotation(pixel_polygon: list, image_id, annot_id, class_id):
    """Function to convert a polygon to a COCO annotation object.

    Args:
        pixel_polygon (list): List of pixel coordinates generated via coordinates.spatial_polygon_to_pixel_rio()
        image_id (int): ID of image
        annot_id (int): ID of annotation
        class_id (int): ID of class

    Returns:
        annot (dict): COCO annotation object
    """

    annot = {
        "segmentation": [item for sublist in pixel_polygon for item in sublist],
        "area": Polygon(pixel_polygon).area,
        "iscrowd": 0,
        "image_id": image_id,
        "bbox": coco_bbox(Polygon(pixel_polygon)),
        "category_id": class_id,
        "id": annot_id,
    }

    return annot


def coco_polygon_annotations(polygon_df):
    """Function to convert a GeoDataFrame of polygons to a list of COCO
    annotation objects.

    Args:
        polygon_df (gpd.GeoDataFrame): GeoDataFrame containing polygon data

    Returns:
        annotations_tmp (list): List of COCO annotation objects
    """

    annotations_tmp = []
    for _, row in polygon_df.iterrows():
        annotations_tmp.append(
            coco_polygon_annotation(
                row["pixel_polygon"], row["image_id"], row["annot_id"], row["class_id"]
            )
        )

    return annotations_tmp


def coco_image_annotations(raster_file_list, colour):
    """Function to convert a list of rasters to a list of COCO image objects.

    Args:
        raster_file_list (list): List of raster files
        colour (bool, optional): If True, save image in colour.

    Returns:
        images (coco_images): coco_images object
    """

    # TODO: Make this function more efficcient by parllizing it. Multiple instances of raster_to_coco can be run in parallel.

    images = coco_json.coco_images()
    images.images = [
        raster_to_coco(raster_file, index, "png", 3, colour)
        for index, raster_file in enumerate(raster_file_list)
    ]

    return images


def coco_reader(coco_json: str):
    """Function to read a COCO JSON file.

    Args:
        coco_json (str): Path to COCO JSON file

    Returns:
        coco (dict): COCO JSON file as a dict
    """

    with open(coco_json, "r") as f:
        coco = json.load(f)

    return coco


def coco_annotation_per_image(coco_json: str, tile_search_margin: int = 5):
    """Function to get a list of annotations per image. This function is
    necessary in order to get the tile name from the COCO JSON file, as a
    reference for spatial coordinate conversion.

    Args:
        coco_json (str): Path to COCO JSON file
        tile_search_margin (int, optional): Percentage of tile size to use as a search margin for finding overlapping polygons while joining raster. Defaults to 5 percent.

    Returns:
            annotations_per_image (list): List of annotations per image
    """
    coco_data = coco_reader(coco_json)
    annotations_per_image = {}
    for image in coco_data["images"]:
        image_id = image["id"]
        image_annotations = []
        margine_h = image["height"] * tile_search_margin / 100
        margine_w = image["width"] * tile_search_margin / 100
        margine_h_min = image["height"] - margine_h
        margine_w_min = image["width"] - margine_w
        margine_h_max = image["height"] + margine_h
        margine_w_max = image["width"] + margine_w
        for annotation in coco_data["annotations"]:
            if annotation["image_id"] == image_id:
                marginal = False
                x_min, y_min, x_max, y_max = annotation["bbox"]
                x_max = x_max + x_min
                y_max = y_max + y_min
                if tile_search_margin > 0:
                    if (
                        x_min < margine_w_min
                        or x_max > margine_w_max
                        or y_min < margine_h_min
                        or y_max > margine_h_max
                    ):
                        marginal = True  # print(f"new tile_width: {tile_width}, tile_height: {tile_height}")

                annotation["marginal"] = marginal
                image_annotations.append(annotation)
        annotations_per_image[image_id] = {
            "tile_name": image["file_name"].split(".")[0],
            "annotations": image_annotations,
        }
    return annotations_per_image


def coco_annotation_per_image_df(coco_json: str, tile_search_margin: int = 10):
    """Function to get a list of annotations per image. This function is
    necessary in order to get the tile name from the COCO JSON file, as a
    reference for spatial coordinate conversion.

    Args:
        coco_json (str): Path to COCO JSON file
        tile_search_margin (int, optional): Percentage of tile size to use as a search margin for finding overlapping polygons while joining raster. Defaults to 10.

    Returns:
            annotations_per_image (pd.DataFrame): Data frame of annotations with images image
    """
    coco_images_df = pd.DataFrame(
        coco_annotation_per_image(coco_json, tile_search_margin)
    ).T
    coco_images_df = coco_images_df.explode("annotations").reset_index(drop=True)
    return coco_images_df


def coco_categories_dict(coco_json: str):
    """Function to get a list of categories from a COCO JSON file.

    Args:
        coco_json (str): Path to COCO JSON file

    Returns:
            categories (list): List of categories
    """
    coco_data = coco_reader(coco_json)
    categories = coco_data["categories"]
    categories_dict = {}
    for category in categories:
        categories_dict[category["id"]] = {
            "name": category["name"],
            "supercategory": category["supercategory"],
        }
    return categories_dict


def polygon_prep(
    polygon,
    simplify_tolerance: float = 0.0,
    minimum_rotated_rectangle: bool = False,
    orthogonalisation: bool = False,
):
    """Prepares a polygon for export.

    Orthoginalisation is based on the orthogonalisation script version 1.0.4 written by Martin Machyna. Full credits at orthogoalise submodule.

    Args:
        polygon (list): A list of coordinates
        simplify_tolerance (float, optional): Tolerance for simplifying polygons. Accepts values between 0.0 and 1.0. Defaults to 0.0. If simplify_tolerance > 0, will simplify the polygon, without minimum rotated rectangle.
        minimum_rotated_rectangle (bool, optional): If true, will return the minimum rotated rectangle of the polygon. Defaults to False. If simplify_tolerance > 0, will simplify the polygon without minimum rotated rectangle.
        orthogonalise (bool, optional): If true, will return the orthogonalised polygon. Defaults to False. If simplify_tolerance > 0, will simplify the polygon without minimum rotated rectangle.

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
        if orthogonalisation:
            polygon = orthogonalise.orthogonalise_polygon(polygon)
    polygon = np.array(polygon.exterior.coords)

    return polygon
