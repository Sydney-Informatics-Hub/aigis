# -*- coding: utf-8 -*-

import geopandas as gpd
import matplotlib.pyplot as plt

def calculate_coverage(boundary_file, building_file, tree_file):
    # Load boundary file
    boundary_data = gpd.read_file(boundary_file)

    # Load building outline geoparquet
    building_data = gpd.read_file(building_file)

    # Load tree outline geoparquet
    tree_data = gpd.read_file(tree_file)

    # Calculate coverage percentage for buildings
    building_coverage = (building_data.geometry.area.sum() / boundary_data.geometry.area.sum()) * 100

    # Calculate coverage percentage for trees
    tree_coverage = (tree_data.geometry.area.sum() / boundary_data.geometry.area.sum()) * 100

    # Generate histogram for building sizes
    building_data['area'] = building_data.geometry.area
    building_data['area'].plot.hist(bins=10)
    plt.xlabel('Building Area')
    plt.ylabel('Frequency')
    plt.title('Distribution of Building Sizes')
    plt.show()

    # Generate histogram for tree sizes
    tree_data['area'] = tree_data.geometry.area
    tree_data['area'].plot.hist(bins=10)
    plt.xlabel('Tree Area')
    plt.ylabel('Frequency')
    plt.title('Distribution of Tree Sizes')
    plt.show()

    return building_coverage, tree_coverage

# Usage example
boundary_file = 'path/to/boundary_file.shp'
building_file = 'path/to/building_outline.parquet'
tree_file = 'path/to/tree_outline.parquet'

building_coverage, tree_coverage = calculate_coverage(boundary_file, building_file, tree_file)
print(f"Building coverage: {building_coverage}%")
print(f"Tree coverage: {tree_coverage}%")