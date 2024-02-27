# -*- coding: utf-8 -*-
# aerialannotation/utils.py

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
