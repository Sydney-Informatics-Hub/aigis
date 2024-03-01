import os
import csv
import torch
import torch.nn.functional as F
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import argparse
import rasterio
from shapely.geometry import Polygon
from aigis.convert.coordinates import wkt_parser
import geopandas as gpd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

processor = ViTImageProcessor.from_pretrained("ViT_LCZs_v2", local_files_only=True)
model = ViTForImageClassification.from_pretrained("ViT_LCZs_v2", local_files_only=True).to(device)

def predict(image_path):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_prob = F.softmax(logits, dim=-1).detach().cpu().numpy().max()
    predicted_class_idx = logits.argmax(-1).item()
    label = model.config.id2label[predicted_class_idx].split(",")[0]
    return label, float(predicted_class_prob)

def predict_images(input_dir, output_name,user_crs=None):
    """
    Predicts the labels and confidences for a set of images in the given input directory,
    and writes the results to a CSV file specified by the output_csv parameter.

    Args:
        input_dir (str): The path to the directory containing the input images.
        output_csv (str): The path to the output CSV file.

    Returns:
        None
    """
    image_files = os.listdir(input_dir)
    predictions = []
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        # get the image bounds form the geotiff
        with rasterio.open(image_path) as src:
            bounds = src.bounds
            if user_crs is None:
                user_crs = src.crs.to_wkt()
                user_crs = wkt_parser(user_crs)
        # make a polygon out of bounds
        polygon = Polygon([(bounds.left, bounds.bottom), (bounds.right, bounds.bottom), (bounds.right, bounds.top), (bounds.left, bounds.top)])
        

        label, confidence = predict(image_path)
        predictions.append((image_file, label, confidence, polygon))

    with open(output_name+".csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Image Filename', 'Prediction', 'Confidence'])
        writer.writerows(predictions)

    # create a geodataframe from the predictions
    gdf = gpd.GeoDataFrame(predictions, columns=['Image Filename', 'Prediction', 'Confidence', 'geometry'])
    gdf.set_crs(user_crs, inplace=True)
    gdf.to_parquet(output_name+".geoparquet")
    gdf.to_file(output_name+".geojson", driver="GeoJSON")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ViT LCZ Classification')
    parser.add_argument('--input_dir', type=str, help='Path to input directory containing images')
    parser.add_argument('--output_name', type=str, help='Path to output csv, geojson, and geparquet files')
    parser.add_argument('--user_crs', type=str, help='User defined crs')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_name = args.output_name
    user_crs = args.user_crs

    predict_images(input_dir, output_name, user_crs)
