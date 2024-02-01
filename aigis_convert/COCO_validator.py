# -*- coding: utf-8 -*-
import json

# Credits to: https://gist.github.com/srishilesh/6c953ff1d7ee006b412be7674d1542cb


def assertions(key, values, required_keys, unique_key=None):
    unique_key_id_mapper = {}
    for value in values:
        if unique_key is not None:
            unique_key_id_mapper[value["id"]] = value[unique_key]
        for required_key in required_keys:
            assert (
                required_key in value
            ), "'{}' does not contain the required key '{}'".format(key, required_key)
    return unique_key_id_mapper


def annotation_assertions(key, annotations, image_map, category_map):
    required_keys = [
        "area",
        "iscrowd",
        "bbox",
        "category_id",
        # "ignore",
        "segmentation",
        "image_id",
        "id",
    ]
    assertions("annotations", coco_data["annotations"], required_keys, None)
    for annotation in annotations:
        assert (
            len(annotation["bbox"]) == 4
        ), "'{}' key in 'annotations' does not match the expected format".format("bbox")
        assert (
            annotation["category_id"] in category_map
        ), "'{}' is not present in the 'categories' mapping".format("category_id")
        assert (
            annotation["image_id"] in image_map
        ), "'{}' is not present in the 'images' mapping".format("image_id")
        assert annotation["area"] == (
            annotation["bbox"][2] * annotation["bbox"][3]
        ), "Mismatch of values in '{}' and '{}'".format("area", "bbox")
        assert (
            len(annotation["segmentation"][0]) == 8
            or len(annotation["segmentation"]) == 0
        ), "'{}' must either be an empty list or contain a list of 8 values".format(
            "segmentation"
        )
        assert (
            annotation["iscrowd"] == 0 or annotation["iscrowd"] == 1
        ), "'{}' must either be 0 or 1. {} is invalid".format(
            "iscrowd", annotation["iscrowd"]
        )


def main(coco_data):
    required_keys = ["images", "type", "annotations", "categories"]
    for required_key in required_keys:
        assert (
            required_key in coco_data.keys()
        ), "Required key '{}' not found in the COCO dataset".format(required_key)
        assert (
            len(coco_data[required_key]) > 0
        ), "Required key '{}' does not contain values".format(required_key)

    image_map = assertions(
        "images",
        coco_data["images"],
        ["file_name", "height", "width", "id"],
        "file_name",
    )
    category_map = assertions(
        "categories", coco_data["categories"], ["id", "name", "supercategory"], "name"
    )
    annotation_assertions(
        "annotations", coco_data["annotations"], image_map, category_map
    )
    print("The dataset format is COCO!")


if __name__ == "__main__":
    coco_file = r"/home/sahand/Data/GIS2COCO/chatswood/coco_from_gis.json"
    with open(coco_file) as json_file:
        coco_data = json.load(json_file)
    main(coco_data)
