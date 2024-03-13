#!/usr/bin/env python
# coding: utf-8

import os
import geopandas as gpd
import pandas as pd
from shapely.geometry import mapping
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import os
import os
import geopandas as gpd
import pandas as pd
import seaborn as sns

def merge_geojson(directory):
    # Create an empty geodataframe to store the merged data
    merged_gdf = gpd.GeoDataFrame()

    # Iterate over each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".geojson"):
            # Read the geojson file into a geodataframe
            filepath = os.path.join(directory, filename)
            gdf = gpd.read_file(filepath)

            # Convert the GeoDataFrame to a GeoSeries
            gdf = gdf.geometry

            # Calculate the total area of the extent
            extent_area = gdf.total_bounds.area

            # Calculate the number, total area, and average area for each feature within the "buildings" and "trees" layers
            for layer in ["buildings", "trees"]:
                layer_count = len(gdf[gdf["layer"] == layer])
                layer_area = gdf[gdf["layer"] == layer].area.sum()
                layer_avg_area = layer_area / layer_count

                # Print the results for each layer
                print(f"Geojson: {filename}, Layer: {layer}")
                print(f"Number of features: {layer_count}")
                print(f"Total area: {layer_area}")
                print(f"Average area: {layer_avg_area}")

            # Merge the current geodataframe with the merged geodataframe
            merged_gdf = merged_gdf.append(gdf, ignore_index=True)

    return merged_gdf


# Calculate the total area of the GeoDataFrame extent
total_area = gdf.total_bounds
total_extent_area = (total_area[2] - total_area[0]) * (total_area[3] - total_area[1])

# Count the features in the layers 'buildings' and 'trees'
buildings_count = len(gdf[gdf['layer'] == 'buildings'])
trees_count = len(gdf[gdf['layer'] == 'trees'])

# Calculate the average area and total area for features in the 'buildings' and 'trees' layers
buildings_avg_area = gdf[gdf['layer'] == 'buildings'].geometry.area.mean()
buildings_total_area = gdf[gdf['layer'] == 'buildings'].geometry.area.sum()

trees_avg_area = gdf[gdf['layer'] == 'trees'].geometry.area.mean()
trees_total_area = gdf[gdf['layer'] == 'trees'].geometry.area.sum()

# Print the summary
print(f"Total area of the GeoDataFrame extent: {total_extent_area}")
print(f"Number of features in 'buildings' layer: {buildings_count}")
print(f"Number of features in 'trees' layer: {trees_count}")
print(f"Average area of features in 'buildings' layer: {buildings_avg_area}")
print(f"Total area of features in 'buildings' layer: {buildings_total_area}")
print(f"Average area of features in 'trees' layer: {trees_avg_area}")
print(f"Total area of features in 'trees' layer: {trees_total_area}")

# Function to calculate statistics for each GeoJSON file
def calculate_statistics(file_path):
    # Read GeoJSON file
    gdf = gpd.read_file(file_path)
    
    # Calculate total area of the GeoDataFrame extent
    total_extent_area = gdf.total_bounds
    total_extent_area = (total_extent_area[2] - total_extent_area[0]) * (total_extent_area[3] - total_extent_area[1])
    
    # Count the features in the layers 'buildings' and 'trees'
    buildings_count = len(gdf[gdf['layer'] == 'buildings'])
    trees_count = len(gdf[gdf['layer'] == 'trees'])
    
    # Calculate the average area and total area for features in the 'buildings' and 'trees' layers
    buildings_avg_area = gdf[gdf['layer'] == 'buildings'].geometry.area.mean()
    buildings_total_area = gdf[gdf['layer'] == 'buildings'].geometry.area.sum()
    
    trees_avg_area = gdf[gdf['layer'] == 'trees'].geometry.area.mean()
    trees_total_area = gdf[gdf['layer'] == 'trees'].geometry.area.sum()
    
    # Calculate the area that is not classified as building or tree
    not_building_or_tree_area = total_extent_area - buildings_total_area - trees_total_area
    
    # Create a summary GeoDataFrame for this GeoJSON file
    summary_df = gpd.GeoDataFrame({
        'location': os.path.splitext(os.path.basename(file_path))[0],
        'total_extent_area': total_extent_area,
        'buildings_count': buildings_count,
        'trees_count': trees_count,
        'buildings_avg_area': buildings_avg_area,
        'buildings_total_area': buildings_total_area,
        'trees_avg_area': trees_avg_area,
        'trees_total_area': trees_total_area,
        'not_building_or_tree_area': not_building_or_tree_area
    }, index=[0])
    
    return summary_df

# Directory containing GeoJSON files
directory = '/Users/henrylydecker/Desktop/lcz_demo_imgs/demo_results/'

# List to store individual summary GeoDataFrames
summary_dfs = []

# Iterate through each GeoJSON file in the directory
for filename in os.listdir(directory):
    if filename.endswith('.geojson'):
        file_path = os.path.join(directory, filename)
        summary_df = calculate_statistics(file_path)
        summary_dfs.append(summary_df)

# Merge all summary GeoDataFrames into one
merged_df = gpd.GeoDataFrame(pd.concat(summary_dfs, ignore_index=True))

# Print the merged GeoDataFrame
print(merged_df)

# Function to calculate area for each feature
def calculate_feature_area(geojson_path):
    # Read GeoJSON file
    gdf = gpd.read_file(geojson_path)
    
    # Calculate area for each feature
    gdf['area'] = gdf.geometry.area
    
    return gdf

# Function to calculate statistics for each layer
def calculate_layer_statistics(gdf, layer_name):
    # Filter features by layer name
    layer_gdf = gdf[gdf['layer'] == layer_name]
    
    # Calculate statistics
    count = len(layer_gdf)
    average_area = layer_gdf['area'].mean()
    total_area = layer_gdf['area'].sum()
    
    return count, average_area, total_area


# Initialize an empty list to store DataFrames
dfs = []

# Iterate over each GeoJSON file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".geojson"):
        # Extract location name from filename
        location = os.path.splitext(filename)[0]
        
        # Read GeoJSON file and calculate feature area
        geojson_path = os.path.join(directory, filename)
        gdf = calculate_feature_area(geojson_path)
        
        # Calculate statistics for buildings layer
        buildings_count, buildings_avg_area, buildings_total_area = calculate_layer_statistics(gdf, 'buildings')
        
        # Calculate statistics for trees layer
        trees_count, trees_avg_area, trees_total_area = calculate_layer_statistics(gdf, 'trees')
        
        # Create DataFrames for each layer
        buildings_df = pd.DataFrame({'location': location, 'layer': 'buildings', 'count': buildings_count,
                                     'average_area': buildings_avg_area, 'total_area': buildings_total_area}, 
                                    index=[0])
        trees_df = pd.DataFrame({'location': location, 'layer': 'trees', 'count': trees_count,
                                 'average_area': trees_avg_area, 'total_area': trees_total_area}, 
                                index=[0])
        
        # Append DataFrames to list
        dfs.append(buildings_df)
        dfs.append(trees_df)

# Concatenate DataFrames into final DataFrame
stats_df = pd.concat(dfs, ignore_index=True)

# Display the final DataFrame with all statistics
stats_df

# Function to calculate statistics for each feature
def calculate_feature_statistics(feature):
    # Calculate area for the feature
    area = feature.geometry.area
    
    # Get feature properties
    properties = feature.drop(columns='geometry').to_dict()
    
    # Create a dictionary with statistics
    statistics = {
        'area': area,
        **properties
    }
    
    return statistics


# Initialize an empty list to store dictionaries
all_statistics = []

# Iterate over each GeoJSON file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".geojson"):
        # Read GeoJSON file
        geojson_path = os.path.join(directory, filename)
        gdf = gpd.read_file(geojson_path)
        
        # Iterate over each feature in the GeoDataFrame
        for index, feature in gdf.iterrows():
            # Calculate statistics for the feature
            statistics = calculate_feature_statistics(feature)
            
            # Add location information
            statistics['location'] = os.path.splitext(filename)[0]
            
            # Append statistics to the list
            all_statistics.append(statistics)

# Create DataFrame from the list of dictionaries
stats_df = pd.DataFrame(all_statistics)

# Display the final DataFrame with all statistics
stats_df

# Assuming you have already calculated the stats_df DataFrame as described in the previous solution

# Filter the DataFrame for buildings and trees
buildings_df = stats_df[stats_df['layer'] == 'buildings']
#trees_df = stats_df[stats_df['layer'] == 'trees']

# Create a figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Create boxplots for building area and tree area
sns.boxplot(x='location', y='area', data=buildings_df, ax=ax, color='skyblue', width=0.3)
#sns.boxplot(x='location', y='area', data=trees_df, ax=ax, color='lightgreen', width=0.3)

# Set labels and title
ax.set_xlabel('Location', fontsize=14)
ax.set_ylabel('Area (m²)', fontsize=14)
ax.set_title('Building Area Distribution by Location', fontsize=16)

# Add a legend
#ax.legend(['Building Area', 'Tree Area'])

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Show plot
plt.show()


# Filter the DataFrame for buildings
buildings_df = stats_df[stats_df['layer'] == 'buildings']

# Get unique locations
locations = buildings_df['location'].unique()

# Define fixed bin width for histograms
bin_width = 500

# Determine common x-axis limits
max_area = buildings_df['area'].max()
min_area = 0  # Set to 0 or any other desired minimum value

# Create individual histograms for each location
for location in locations:
    # Filter data for the current location
    data = buildings_df[buildings_df['location'] == location]
    
    # Create histogram with fixed bin width and common x-axis limits
    plt.figure(figsize=(8, 6))
    sns.histplot(data=data, x='area', color='skyblue', bins=range(0, int(max_area) + bin_width, bin_width))
    
    # Set common x-axis limits
    plt.xlim(min_area, max_area)
    
    # Set title and labels
    plt.title(f'Building Size Distribution - {location}', fontsize=14)
    plt.xlabel('Building Area (m²)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    
    # Save the plot as a separate image file
    plt.savefig(f'{location}_building_size_distribution.png', dpi=300, bbox_inches='tight')
    
    # Show the plot
    plt.show()

# Assuming you have already calculated the stats_df DataFrame as described in the previous solution

# Filter the DataFrame for buildings
trees_df = stats_df[stats_df['layer'] == 'trees']

# Get unique locations
locations = trees_df['location'].unique()

# Define fixed bin width for histograms
bin_width = 100

# Determine common x-axis limits
max_area = buildings_df['area'].max()
min_area = 0  # Set to 0 or any other desired minimum value

# Create individual histograms for each location
for location in locations:
    # Filter data for the current location
    data = trees_df[trees_df['location'] == location]
    
    # Create histogram with fixed bin width and common x-axis limits
    plt.figure(figsize=(8, 6))
    sns.histplot(data=data, x='area', color='skyblue', bins=range(0, int(max_area) + bin_width, bin_width))
    
    # Set common x-axis limits
    plt.xlim(min_area, max_area)
    
    # Set title and labels
    plt.title(f'Tree Patch Size Distribution - {location}', fontsize=14)
    plt.xlabel('Tree Patch Area (m²)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    
    # Save the plot as a separate image file
    plt.savefig(f'{location}_building_size_distribution.png', dpi=300, bbox_inches='tight')
    
    # Show the plot
    plt.show()

# Filter the DataFrame for buildings and trees
buildings_df = stats_df[stats_df['layer'] == 'buildings']
trees_df = stats_df[stats_df['layer'] == 'trees']

# Get unique locations
locations = buildings_df['location'].unique()

# Determine common x-axis limits
max_area_buildings = buildings_df['area'].max()
max_area_trees = trees_df['area'].max()
max_area = max(max_area_buildings, max_area_trees)
min_area = 0  # Set to 0 or any other desired minimum value

# Create individual density plots for each location
for location in locations:
    # Filter data for the current location
    building_data = buildings_df[buildings_df['location'] == location]['area']
    tree_data = trees_df[trees_df['location'] == location]['area']
    
    # Create density plots for building and tree areas
    plt.figure(figsize=(8, 6))
    sns.kdeplot(building_data, color='blue', label='Building Area')
    sns.kdeplot(tree_data, color='green', label='Tree Patch Area')
    
    # Set common x-axis limits
    plt.xlim(min_area, max_area)
    
    # Set title and labels
    plt.title(f'Building and Tree Patch Size Distribution - {location}', fontsize=14)
    plt.xlabel('Area (m²)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    
    # Add legend
    plt.legend()
    
    # Save the plot as a separate image file
    plt.savefig(f'{location}_building_tree_size_distribution.png', dpi=300, bbox_inches='tight')
    
    # Show the plot
    plt.show()

# Function to calculate statistics for each GeoJSON file
def calculate_statistics(file_path):
    # Read GeoJSON file
    gdf = gpd.read_file(file_path)
    
    # Calculate total area of the GeoDataFrame extent
    total_extent_area = gdf.total_bounds
    total_extent_area = (total_extent_area[2] - total_extent_area[0]) * (total_extent_area[3] - total_extent_area[1])
    
    # Count the features in the layers 'buildings' and 'trees'
    buildings_count = len(gdf[gdf['layer'] == 'buildings'])
    trees_count = len(gdf[gdf['layer'] == 'trees'])
    
    # Calculate the average area and total area for features in the 'buildings' and 'trees' layers

    # Function to calculate statistics for each GeoJSON file
    def calculate_statistics(file_path):
        # Read GeoJSON file
        gdf = gpd.read_file(file_path)
        
        # Calculate total area of the GeoDataFrame extent
        total_extent_area = gdf.total_bounds
        total_extent_area = (total_extent_area[2] - total_extent_area[0]) * (total_extent_area[3] - total_extent_area[1])
        
        # Count the features in the layers 'buildings' and 'trees'
        buildings_count = len(gdf[gdf['layer'] == 'buildings'])
        trees_count = len(gdf[gdf['layer'] == 'trees'])
        
        # Calculate the average area and total area for features in the 'buildings' and 'trees' layers
        buildings_avg_area = gdf[gdf['layer'] == 'buildings'].geometry.area.mean()
        buildings_total_area = gdf[gdf['layer'] == 'buildings'].geometry.area.sum()
        
        trees_avg_area = gdf[gdf['layer'] == 'trees'].geometry.area.mean()
        trees_total_area = gdf[gdf['layer'] == 'trees'].geometry.area.sum()
        
        # Calculate the area that is not classified as building or tree
        not_building_or_tree_area = total_extent_area - buildings_total_area - trees_total_area
        
        # Create a summary GeoDataFrame for this GeoJSON file
        summary_df = gpd.GeoDataFrame({
            'location': os.path.splitext(os.path.basename(file_path))[0],
            'total_extent_area': total_extent_area,
            'buildings_count': buildings_count,
            'trees_count': trees_count,
            'buildings_avg_area': buildings_avg_area,
            'buildings_total_area': buildings_total_area,
            'trees_avg_area': trees_avg_area,
            'trees_total_area': trees_total_area,
            'not_building_or_tree_area': not_building_or_tree_area
        }, index=[0])
        
        return summary_df

    # Directory containing GeoJSON files
    directory = '/Users/henrylydecker/Desktop/lcz_demo_imgs/demo_results/'

    # List to store individual summary GeoDataFrames
    summary_dfs = []

    # Iterate through each GeoJSON file in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.geojson'):
            file_path = os.path.join(directory, filename)
            summary_df = calculate_statistics(file_path)
            summary_dfs.append(summary_df)

    # Merge all summary GeoDataFrames into one
    merged_df = gpd.GeoDataFrame(pd.concat(summary_dfs, ignore_index=True))

    # Calculate the percentage area covered by buildings, trees, and other for each location
    merged_df['buildings_percentage'] = (merged_df['buildings_total_area'] / merged_df['total_extent_area']) * 100
    merged_df['trees_percentage'] = (merged_df['trees_total_area'] / merged_df['total_extent_area']) * 100
    merged_df['other_percentage'] = ((merged_df['total_extent_area'] - merged_df['buildings_total_area'] - merged_df['trees_total_area']) / merged_df['total_extent_area']) * 100

    # Sort the DataFrame by the "trees_percentage" column in ascending order
    merged_df = merged_df.sort_values(by='trees_percentage')

    # Plot pie charts for each location
    for index, row in merged_df.iterrows():
        labels = ['Building %', 'Tree %', 'Other %']
        sizes = [row['buildings_percentage'], row['trees_percentage'], row['other_percentage']]
        colors = ['lightgrey', 'lightgreen', 'black']
        
        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
        plt.title(row['location'], fontsize=16)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.show()
