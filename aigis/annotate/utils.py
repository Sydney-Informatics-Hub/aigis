# -*- coding: utf-8 -*-
# aerialannotation/utils.py

import json
import geopandas as gpd
from pprint import pprint
from matplotlib import pylab as plt
import pandas as pd
from shapely.geometry import Polygon, box
from pyproj import Proj, transform


def create_grid_geojson(bbox, tile_size):
    min_lon, min_lat, max_lon, max_lat = bbox

    # Transform the coordinates to EPSG:3857 (Web Mercator) for easier calculations
    in_proj = Proj(init='epsg:4326')
    out_proj = Proj(init='epsg:3857')
    min_lon, min_lat = transform(in_proj, out_proj, min_lon, min_lat)
    max_lon, max_lat = transform(in_proj, out_proj, max_lon, max_lat)

    # Calculate the number of tiles in x and y directions
    num_tiles_x = int((max_lon - min_lon) / tile_size)
    num_tiles_y = int((max_lat - min_lat) / tile_size)

    features = []

    for i in range(num_tiles_x):
        for j in range(num_tiles_y):
            # Calculate the coordinates of the current tile
            tile_min_lon = min_lon + i * tile_size
            tile_max_lon = min_lon + (i + 1) * tile_size
            tile_min_lat = min_lat + j * tile_size
            tile_max_lat = min_lat + (j + 1) * tile_size

            # Convert the coordinates back to EPSG:4326
            tile_min_lon, tile_min_lat = transform(out_proj, in_proj, tile_min_lon, tile_min_lat)
            tile_max_lon, tile_max_lat = transform(out_proj, in_proj, tile_max_lon, tile_max_lat)

            # Create a polygon for the current tile
            tile_polygon = box(tile_min_lon, tile_min_lat, tile_max_lon, tile_max_lat)

            # Create a GeoJSON feature for the current tile
            feature = {
                "type": "Feature",
                "properties": {
                    "id": i * num_tiles_y + j,
                    "left": tile_min_lon,
                    "top": tile_max_lat,
                    "right": tile_max_lon,
                    "bottom": tile_min_lat,
                    "row_index": i,
                    "col_index": j
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [list(tile_polygon.exterior.coords)]
                }
            }

            features.append(feature)

    # Create a GeoJSON feature collection
    feature_collection = {
        "type": "FeatureCollection",
        "name": "GSU_grid_1",
        "crs": {
            "type": "name",
            "properties": {
                "name": "urn:ogc:def:crs:EPSG::3857"
            }
        },
        "features": features
    }

    return json.dumps(feature_collection)


def show_mask(
    image, mask, alpha=0.1, cmap="viridis", edges=True, edge_colour="green", output=None
):
    """Plot a mask overlaid onto an image, with highlighted edges if required.

    Inputs
    ======
    image (np.ndarray): An input image array - ideally in colour
    mask (np.ndarray): An input mask array - two values (0=masked, 255=unmasked)

    alpha (float, optional): Transparency of mask when overlaid onto image
    cmap (str, optional): Colourmap to use for mask image
    edges (bool, optional): determine the edges of the mask and draw a solid line from these
    edge_colour (str, optional): colour of the edge highlight
    output (str, optional): filename to output figure to (if None, plot on the screen)
    """
    fig = plt.figure(figsize=(20, 20))
    plt.imshow(image)
    plt.axis("off")
    mask_arr = mask[:, :, 0]
    mask_arr = np.ma.masked_where(mask_arr == 0, mask_arr)
    plt.imshow(mask_arr, alpha=alpha, cmap=cmap, vmin=0, vmax=255)
    if edges:
        plt.contour(mask[:, :, 0], [0], colors=[edge_colour])
    if output:
        plt.savefig(output)
    else:
        plt.show()
    plt.clf()
    plt.close(fig)


def geojson_csv_filter(geojson_path, csv_path):

    """
    Filter a GeoJSON loaded in as a GeoDataFrame based on matching feature IDs in a CSV file.

    Parameters:
    - geojson_path (str): Path to the GeoJSON file.
    - csv_path (str): Path to the CSV file containing feature IDs. CSV file should have one column, named 'id'

    Returns:
    - filtered_gdf (geopandas.GeoDataFrame): Filtered GeoDataFrame based on matching IDs.
    """
    # Load GeoJSON file
    gdf = gpd.read_file(geojson_path)

    # Load CSV file with feature IDs
    # TODO: Make this more robust and accept a list of ids instead.
    ids_to_keep = set(pd.read_csv(csv_path)['id'])

    # Filter GeoDataFrame based on matching IDs
    filtered_gdf = gdf[gdf['id'].isin(ids_to_keep)]
    
    return filtered_gdf
