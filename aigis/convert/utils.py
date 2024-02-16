"""
Utility functions for the aigis.convert module.
"""

import json
import os

__version__ = "1.0.0"

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
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    for image_info in coco_data['images']:
        file_name = image_info['file_name']
        file_id = file_name.split('/')[0]
        new_file_name = f"{file_id}.jpg"
        image_info['file_name'] = new_file_name

    # Update the images path if necessary
    if image_directory:
        for image_info in coco_data['images']:
            file_name = image_info['file_name']
            new_file_path = os.path.join(file_name)
            image_info['file_name'] = new_file_path

    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f, indent=4)

if __name__ == "__main__":
    # Code to execute when the module is run as a script
    pass
