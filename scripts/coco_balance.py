# -*- coding: utf-8 -*-
# from pycocotools.coco import COCO
import argparse
import datetime
import json
import random

import pandas as pd
from tqdm import tqdm

# import os


def stats(json_data):
    # categories = json_data["categories"]
    annotations = json_data["annotations"]
    # images = json_data["images"]
    # info = json_data["info"]
    # licenses = json_data["licenses"]

    annotation_list = []
    for annot in annotations:
        annotation_list.append(
            {"category_id": annot["category_id"], "image_id": annot["image_id"]}
        )
    annotation_df = pd.DataFrame(annotation_list)

    cats_per_image = annotation_df.groupby("image_id").count()
    cats_unique_per_image = annotation_df.groupby("image_id").nunique()
    cats_stat_per_image = annotation_df.groupby("category_id").count()

    print(cats_per_image)
    print(cats_unique_per_image)
    print(cats_stat_per_image)

    cats_unique_per_image = cats_unique_per_image.reset_index()
    cats_unique_per_image = cats_unique_per_image.groupby("category_id").count()
    cats_unique_per_image = cats_unique_per_image.reset_index()
    cats_unique_per_image.columns = ["cats", "images"]
    print("stats for the diversity of cats in images")
    print(cats_unique_per_image)
    return cats_per_image, cats_unique_per_image, cats_stat_per_image


def class_crop(json_data):
    """Find the images that have only the high frequency category and mark a
    subsample that is makes the frequency larger than the mid-frequency classes
    for removal.

    Args:
        json_data (dict): the json data of the coco dataset

    Returns:
        set: the ids of the images to remove

    Example:
    >>> to_remove = class_crop(json_data)
    """
    annotation_list = []
    annotations = json_data["annotations"]
    for annot in annotations:
        annotation_list.append(
            {
                "category_id": annot["category_id"],
                "image_id": annot["image_id"],
                "annotation_id": annot["id"],
            }
        )
    annotation_df = pd.DataFrame(annotation_list)

    # Group by 'image_id' and count unique 'category_id' for each group
    unique_categories_per_image = annotation_df.groupby("image_id")[
        "category_id"
    ].nunique()
    # Convert the Series to a DataFrame for better readability
    unique_categories_df = unique_categories_per_image.reset_index()
    unique_categories_df.columns = ["image_id", "unique_categories_count"]
    # print(unique_categories_df)

    # Group by 'image_id' and 'category_id', and count the occurrences
    category_counts = (
        annotation_df.groupby(["image_id", "category_id"])
        .size()
        .reset_index(name="count")
    )
    # print(category_counts)
    # Create a pivot table with 'image_id' as rows, 'category_id' as columns, and 'count' as values
    pivot_df = category_counts.pivot_table(
        index="image_id", columns="category_id", values="count", fill_value=0
    )
    # Set values to int
    pivot_df = pivot_df.astype(int).reset_index()
    pivot_df.columns = [
        f"cat_{x}" if x != "image_id" else "image_id" for x in pivot_df.columns
    ]
    print("\nannotation categories per image")
    print(pivot_df)

    # Find the frequencies in each class
    class_frequencies = annotation_df.groupby("category_id").count().reset_index()
    class_frequencies = class_frequencies[["category_id", "image_id"]]
    class_frequencies.columns = ["category_id", "frequency"]
    # Sort the frequencies
    class_frequencies = class_frequencies.sort_values(by="frequency", ascending=True)
    print("\nsorted class frequencies before balancing")
    print(class_frequencies)

    # List the images per each category
    images_per_category = (
        annotation_df.groupby("category_id")["image_id"].apply(list).reset_index()
    )
    images_per_category["image_id"] = images_per_category["image_id"].apply(
        lambda x: list(set(x))
    )
    print("\nimage per category")
    print(images_per_category)

    # the number of images with only the middle category
    mid_freq_cat = class_frequencies.iloc[len(class_frequencies) // 2]["category_id"]
    low_freq_cat = class_frequencies.iloc[0]["category_id"]
    high_freq_cat = class_frequencies.iloc[-1]["category_id"]

    print("mid_cat is ", mid_freq_cat)

    mid_freq_cat_image_ids = set(
        images_per_category[images_per_category["category_id"] == mid_freq_cat][
            "image_id"
        ].values[0]
    )
    high_freq_cat_image_ids = set(
        images_per_category[images_per_category["category_id"] == high_freq_cat][
            "image_id"
        ].values[0]
    )
    low_freq_cat_image_ids = set(
        images_per_category[images_per_category["category_id"] == low_freq_cat][
            "image_id"
        ].values[0]
    )

    # find the mid_cat that are not in high and low cats
    uniques_in_high_freq_cat = high_freq_cat_image_ids - (
        mid_freq_cat_image_ids | low_freq_cat_image_ids
    )
    uniques_in_mid_freq_cat = mid_freq_cat_image_ids - (
        high_freq_cat_image_ids | low_freq_cat_image_ids
    )
    uniques_in_low_freq_cat = low_freq_cat_image_ids - (
        high_freq_cat_image_ids | mid_freq_cat_image_ids
    )

    print("high freq cat unique len:", len(uniques_in_high_freq_cat))
    print("mid freq cat unique len:", len(uniques_in_mid_freq_cat))
    print("low freq cat unique len:", len(uniques_in_low_freq_cat))

    # sample fom the high freq cat with the number of mid images only
    uniques_in_high_freq_cat_resampled = random.sample(
        uniques_in_high_freq_cat, len(uniques_in_mid_freq_cat)
    )
    to_remove = uniques_in_high_freq_cat - set(uniques_in_high_freq_cat_resampled)
    print("keeping", len(uniques_in_high_freq_cat_resampled))
    print("removing", len(to_remove))

    return to_remove


def class_balance(json_data):
    """Balance the dataset by removing a subsample of images with the high
    frequency category.

    Args:
        json_data (dict): the json data of the coco dataset

    Returns:
        dict: the json data of the balanced coco dataset
    """
    to_remove = class_crop(json_data)
    # remove iamges with the given ids
    images = json_data["images"]
    new_images = []
    for image in tqdm(images):
        if image["id"] not in to_remove:
            new_images.append(image)
    print("Removed {} images".format(len(images) - len(new_images)))
    json_data["images"] = new_images

    # remove annotations with the removed images
    annotations = json_data["annotations"]
    new_annotations = []
    for annot in tqdm(annotations):
        if annot["image_id"] not in to_remove:
            new_annotations.append(annot)
    print("Removed {} annotations".format(len(annotations) - len(new_annotations)))
    json_data["annotations"] = new_annotations

    return json_data


def isolate_cat(json_data: dict, cat_ids: list):
    """Isolate the given categories from the dataset.

    Args:
        json_data (dict): the json data of the coco dataset
        cat_ids (list): the ids of the categories to isolate

    Returns:
        dict: the json data of the isolated coco dataset
    """
    # Limit the annotations to the given categories
    print("Limiting the annotations to the given categories")
    annotations = json_data["annotations"]
    new_annotations = []
    for annot in tqdm(annotations):
        if annot["category_id"] in cat_ids:
            new_annotations.append(annot)
    print("Removed {} annotations".format(len(annotations) - len(new_annotations)))
    assert len(new_annotations) > 0, "No annotations left after filtering"

    json_data["annotations"] = new_annotations

    # Limit the categories to the given categories
    print("Limiting the categories to the given categories")
    categories = json_data["categories"]
    new_categories = []
    for cat in tqdm(categories):
        if cat["id"] in cat_ids:
            new_categories.append(cat)
    print("Removed {} categories".format(len(categories) - len(new_categories)))
    json_data["categories"] = new_categories

    # Remove the images if not referenced in annotations
    print("Removing the images if not referenced in annotations")
    image_ids = []
    for annot in annotations:
        image_ids.append(annot["image_id"])
    image_ids = list(set(image_ids))
    images = json_data["images"]
    new_images = []
    for image in tqdm(images):
        if image["id"] in image_ids:
            new_images.append(image)
    print("Removed {} images".format(len(images) - len(new_images)))
    json_data["images"] = new_images

    return json_data


def main(args=None):
    if args is None:
        args = parse_arguments(args)

    # Load the COCO json file
    print("Loading the COCO json file")
    with open(args.json_path, "r") as f:
        json_data = json.load(f)

    # stats(json_data)

    if args.balance_cats:
        json_data = class_balance(json_data)

    if args.isolate_cat:
        cats = args.isolate_cat.split(",")
        if args.int_cats:
            cats = [int(cat) for cat in cats]
        json_data = isolate_cat(json_data, cats)

    # Save the COCO json file
    if args.output_path is None:
        response = input("Finish without an output file? ([y]/n)")
        if response == "y":
            print("Finishing without saving the COCO json file")
        elif response == "n":
            date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output_path = args.json_path.replace(".json", f"_balanced_{date}.json")
            print("Saving to default output path", args.output_path)

    if args.output_path is not None:
        print("Saving the COCO json file")
        with open(args.output_path, "w") as f:
            json.dump(json_data, f, indent=2)


def parse_arguments(args):
    parser = argparse.ArgumentParser(description="Balance a COCO dataset")
    parser.add_argument(
        "--json_path", "-i", type=str, help="Path to your COCO json file"
    )
    parser.add_argument(
        "--output_path",
        "-o",
        type=str,
        default=None,
        help="Path to your output COCO json file",
    )
    parser.add_argument(
        "--isolate_cat",
        "-c",
        type=str,
        help="Comma separated list of category ids to isolate from to main dataset.",
    )
    parser.add_argument(
        "--int_cats",
        action=argparse.BooleanOptionalAction,
        help="Set this flag if the categores are integers in isolate cats function.",
    )
    parser.add_argument(
        "--balance_cats",
        action=argparse.BooleanOptionalAction,
        help="Smart balance the dataset. Will try to balance the dataset as much as possible by removing a subsample of the high-frequncy-class-only images.",
    )

    return parser.parse_args(args)


if __name__ == "__main__":
    main()
