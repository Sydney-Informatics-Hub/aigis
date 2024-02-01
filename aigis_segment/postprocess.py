# -*- coding: utf-8 -*-
import os
import warnings

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from aerialseg import utils
from aerial_conversion import coco
from aerial_conversion import coordinates
from detectron2.utils.visualizer import Visualizer
from matplotlib import pylab as plt
from PIL import Image
from shapely.geometry import Polygon
from tqdm import tqdm

import rasterio
import rasterio.transform
from shapely.geometry import Polygon
import geopandas as gpd
from shapely.validation import make_valid


def detectron2_to_polygons(outputs, prediction_simplification=1):

  mask_array = outputs["instances"].pred_masks.to("cpu").numpy()
  num_instances = mask_array.shape[0]
  # scores = output['instances'].scores.to("cpu").numpy()
  labels = outputs["instances"].pred_classes.to("cpu").numpy()
  bbox = outputs["instances"].pred_boxes.to("cpu").tensor.numpy()
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
                  simplify_tolerance=prediction_simplification,
                  minimum_rotated_rectangle=False,
                )
              mask_arrays.append(mask_array_instance)
              labels_list.append(labels[i])
              bbox_list.append(bbox[i])
              polygons.append(polygon.flatten().tolist())

      else:
          warnings.warn(f"Polygon {i} is empty! Skipping polygon.")

  return polygons

def convert_polygons_to_geospatial(polygons, tif_file):
    with rasterio.open(tif_file) as src:
        raster_transform = src.transform

    # make polygons into an np array
    polygons = [np.array(polygons[i]).reshape(-1, 2) for i in range(len(polygons))]

    geospatial_polygons = []  
    for polygon_vertices in polygons:
        # Convert polygon vertices to geospatial coordinates
        geo_polygon = []
        for vertex in polygon_vertices:
            x, y = rasterio.transform.xy(raster_transform, vertex[1], vertex[0])
            geo_polygon.append((x, y))

        # Ensure the right-hand rule for polygon winding order
        if geo_polygon[0] != geo_polygon[-1]:
            geo_polygon.append(geo_polygon[0])

        # Create a Shapely polygon
        shapely_polygon = Polygon(geo_polygon)

        geospatial_polygons.append(shapely_polygon)

    # Create a GeoDataFrame from the Shapely polygons
    gdf = gpd.GeoDataFrame({'geometry': geospatial_polygons})
    #give the gdf the raster's crs
    gdf = geo_dataframe.set_crs(src.crs)

    # Example saving geojson
    #geo_dataframe.to_file("output.geojson", driver="GeoJSON")
  
    return gdf
