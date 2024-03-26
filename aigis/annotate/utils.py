# -*- coding: utf-8 -*-
# aigis.annotate.utils.py

import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib import pylab as plt


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
    """Filter a GeoJSON loaded in as a GeoDataFrame based on matching feature
    IDs in a CSV file.

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
    ids_to_keep = set(pd.read_csv(csv_path)["id"])

    # Filter GeoDataFrame based on matching IDs
    filtered_gdf = gdf[gdf["id"].isin(ids_to_keep)]

    return filtered_gdf


def read_boundary_file(file_path):
    try:
        boundary_data = gpd.read_file(file_path)
        boundary_data = boundary_data.to_crs(epsg=3857)  # Convert to EPSG 3857
        boundary_data = boundary_data.dissolve(by='geometry').boundary
        return boundary_data
    except Exception as e:
        print(f"Error reading boundary file: {e}")
        return None

def create_grid(boundary_data, grid_size):
    try:
        # Calculate the bounding box of the boundary data
        bbox = boundary_data.total_bounds
        
        # Calculate the number of grid cells in each dimension
        num_cells_x = int((bbox[2] - bbox[0]) / grid_size)
        num_cells_y = int((bbox[3] - bbox[1]) / grid_size)
        
        # Create a grid of polygons
        grid = gpd.GeoDataFrame(geometry=gpd.GeoSeries())
        
        # Iterate over the grid cells and create polygons
        for i in range(num_cells_x):
            for j in range(num_cells_y):
                # Calculate the coordinates of the grid cell
                minx = bbox[0] + i * grid_size
                miny = bbox[1] + j * grid_size
                maxx = minx + grid_size
                maxy = miny + grid_size
                
                # Create a polygon for the grid cell
                polygon = gpd.GeoSeries([Polygon([(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)])])
                
                # Add the polygon to the grid
                grid = grid.append({'geometry': polygon[0]}, ignore_index=True)
        
        # Create the GeoJSON structure
        features = []
        for index, row in grid.iterrows():
            feature = {
                "type": "Feature",
                "properties": {
                    "id": index,
                    "left": row.geometry.bounds[0],
                    "top": row.geometry.bounds[1],
                    "right": row.geometry.bounds[2],
                    "bottom": row.geometry.bounds[3],
                    "row_index": int(index / num_cells_y),
                    "col_index": index % num_cells_y
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [list(row.geometry.exterior.coords)]
                }
            }
            features.append(feature)
        
        geojson = {
            "type": "FeatureCollection",
            "name": "GSU_grid_1",
            "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::3857"}},
            "features": features
        }
        
        # Save the grid as a GeoJSON file
        grid.to_file("grid.geojson", driver="GeoJSON")
        
        return geojson
    except Exception as e:
        print(f"Error creating grid: {e}")
        return None