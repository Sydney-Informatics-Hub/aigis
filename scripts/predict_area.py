import argparse
import subprocess
from aigis.annotate.utils import read_boundary_file, create_grid
from aigis.segment.models import download_detectron2_model_weights
from aigis.utils.analysis import calculate_coverage

def main(args):
    boundary_file = read_boundary_file(args.boundary)
    grid = create_grid(boundary_file, args.grid_size)

    download_detectron2_model_weights("trees")
    download_detectron2_model_weights("buildings")

    subprocess.run(["python", "aigis/scripts/get_raster_jpeg.py", "grid.geojson"])
    subprocess.run(["python", "aigis/scripts/jpeg2tiff.py", "output_tiles", "grid.geojson"])

    # Predict on the raster tif you downloaded with the bbox selection in the leafmap
    # Trees
    subprocess.run(["python", "aigis/scripts/prediction_batch_detectron2.py", "--indir", "tiff_tiles/", "-p", "*.tif", "-c", "treev3_tms_sixmaps_cfg.yaml", "-w", "treev3model_0012499.pth", "-t", "0.3", "--coco-out", "trees.json", "-s", "0.0"])
    subprocess.run(["python", "aigis/scripts/coco2geojson.py", "tiff_tiles/", "trees.json", "--simplify-tolerance", "3.0", "--geoparquet-output", "trees.geoparquet", "--geojson-output", "trees.geojson"])

    # Predict on the raster tif you downloaded with the bbox selection in the leafmap
    # Buildings
    subprocess.run(["python", "aigis/scripts/prediction_batch_detectron2.py", "--indir", "tiff_tiles/", "-p", "*.tif", "-c", "buildings_poc_cfg.yml", "-w", "model_final.pth", "-t", "0.1", "--coco-out", "buildings.json", "-s", "0.0"])
    subprocess.run(["python", "aigis/scripts/coco2geojson.py", "tiff_tiles/", "buildings.json", "--simplify-tolerance", "0.1", "--geoparquet-output", "buildings.geoparquet", "--geojson-output", "buildings.geojson"])

    building_coverage, tree_coverage = calculate_coverage(boundary_file, "buildings.geoparquet", "trees.geoparquet")
    print(f"Building coverage: {building_coverage}%")
    print(f"Tree coverage: {tree_coverage}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--boundary", help="Path to boundary file")
    parser.add_argument("--grid-size", type=int, help="Grid size")
    args = parser.parse_args()

    main(args)