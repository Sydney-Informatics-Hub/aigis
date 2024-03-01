# -*- coding: utf-8 -*-
"""Utility functions for the aigis.convert module."""

import json
import os
import shutil
import pandas as pd


__version__ = "1.0.0"


def condense_csv(filename):
    """
    Process Roboflow multiclass image classification data stored in .csv format.

    Args:
        filename (str): The path to the .csv file.

    Returns:
        pandas.DataFrame: A condensed DataFrame containing the filename and class columns.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(filename)
    
    # Create a new DataFrame to store the condensed data
    condensed_df = pd.DataFrame(columns=['filename', 'class'])
    
    # Iterate over each row in the original DataFrame
    for index, row in df.iterrows():
        # Get the filename
        filename = row['filename']
        
        # Iterate over each class column
        for column in df.columns[1:]:
            # Check if the class is present (indicated by 1)
            if row[column] == 1:
                # Create a new row with filename and class
                new_row = pd.DataFrame({'filename': [filename], 'class': [column]})
                # Concatenate the new row with the condensed DataFrame
                condensed_df = pd.concat([condensed_df, new_row], ignore_index=True)
    
    return condensed_df


def copy_files_to_folders(df, directory_path):
    """
    Copy files to folders based on their class.

    Args:
        df (pandas.DataFrame): The DataFrame containing the filename and class columns.
        directory_path (str): The directory path to look for the files in.

    Returns:
        None
    """
    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Get the filename and class
        filename = row['filename']
        class_name = row['class']
        
        # Create a folder for the class if it doesn't exist
        class_folder = os.path.join(directory_path, class_name)
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)
        
        # Copy the file to the class folder
        source = os.path.join(directory_path, filename)
        destination = os.path.join(class_folder, filename)
        shutil.copy(source, destination)



def recode_file_names(coco_json_path: str, image_directory: str, output_json_path: str) -> None:
    """
    Recodes the file names in a COCO JSON file.
    Parameters:
    - coco_json_path (str): The path to the original COCO JSON file.
    - image_directory (str): The path to the directory containing the images.
    - output_json_path (str): The path to save the recoded COCO JSON file.

    Returns:
    None
    """
    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)

    for image_info in coco_data["images"]:
        file_name = image_info["file_name"]
        file_id = file_name.split("/")[0]
        new_file_name = f"{file_id}.jpg"
        image_info["file_name"] = new_file_name

    # Update the images path if necessary
    if image_directory:
        for image_info in coco_data["images"]:
            file_name = image_info["file_name"]
            new_file_path = os.path.join(file_name)
            image_info["file_name"] = new_file_path

    with open(output_json_path, "w") as f:
        json.dump(coco_data, f, indent=4)


if __name__ == "__main__":
    # Code to execute when the module is run as a script
    pass
