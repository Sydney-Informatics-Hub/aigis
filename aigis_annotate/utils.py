# -*- coding: utf-8 -*-
# aerialannotation/utils.py

import json
import geopandas as gpd
from pprint import pprint
import pandas as pd

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
