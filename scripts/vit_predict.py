import os
import csv
import torch
import torch.nn.functional as F
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import argparse

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

def predict_images(input_dir, output_csv):
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
        label, confidence = predict(image_path)
        predictions.append((image_file, label, confidence))

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Image Filename', 'Prediction', 'Confidence'])
        writer.writerows(predictions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ViT LCZ Classification')
    parser.add_argument('--input_dir', type=str, help='Path to input directory containing images')
    parser.add_argument('--output_csv', type=str, help='Path to output CSV file')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_csv = args.output_csv

    predict_images(input_dir, output_csv)
