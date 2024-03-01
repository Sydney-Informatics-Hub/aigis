import rasterio
from rasterio import windows
import os
import math
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def split_geotiff(input_geotiff, output_dir, num_tiles=16):
    """
    Split a GeoTIFF into smaller tiles.

    Parameters:
    - input_geotiff: Path to the input GeoTIFF file.
    - output_dir: Directory where the split tiles will be saved.
    - num_tiles: The total number of tiles to split into. Must be a perfect square.
    """
    # Open the input GeoTIFF
    with rasterio.open(input_geotiff) as src:
        width, height = src.width, src.height
        # Calculate number of splits on each dimension
        # Assuming num_tiles is a perfect square (e.g., 16 = 4x4)
        num_splits = int(math.sqrt(num_tiles))
        tile_width = width // num_splits
        tile_height = height // num_splits
        
        # Loop through the number of splits to generate tiles
        for i in range(num_splits):
            for j in range(num_splits):
                # Calculate window boundaries (left, bottom, right, top)
                window = windows.Window(i * tile_width, j * tile_height, tile_width, tile_height)
                
                # Read the data from the window
                window_data = src.read(window=window)
                
                # Define the transform for the new tile
                transform = src.window_transform(window)
                
                # Define the output path for the tile
                output_path = os.path.join(output_dir, f"{os.path.basename(input_geotiff)}_{i}_{j}.tif")
                
                # Save the tile as a new GeoTIFF
                with rasterio.open(
                    output_path, 'w',
                    driver='GTiff',
                    height=window_data.shape[1],
                    width=window_data.shape[2],
                    count=src.count,
                    dtype=window_data.dtype,
                    crs=src.crs,
                    transform=transform,
                ) as dest:
                    dest.write(window_data)



# main block
if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Split a directory of GeoTIFFs into smaller tiles.')
    parser.add_argument('--input_geotiff', '-i', type=str, help='Path to the input GeoTIFF file.', default=None)
    parser.add_argument('--output_dir', '-o', type=str, help='Directory where the split tiles will be saved.', default=None)
    parser.add_argument('--num_tiles', '-n', type=int, default=16, help='The total number of tiles to split into. Must be a perfect square.')
    parser.add_argument('--workers', '-w', type=int, default=4, help='Number of workers to use for processing.')
    args = parser.parse_args()

    if args.input_geotiff is None:
        args.input_geotiff = '.'

    if args.output_dir is None:
        # use the root directory of the input geotiff as the output directory
        # find the upper directory of the input geotiff
        upper_dir = os.path.dirname(os.getcwd())
        args.output_dir = os.path.join(upper_dir, f'split_tiles_{args.num_tiles}')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for file in tqdm(os.listdir(args.input_geotiff)):
            if file.endswith('.tif'):
                file_path = os.path.join(args.input_geotiff, file)
                futures.append(executor.submit(split_geotiff, file_path, args.output_dir, args.num_tiles))

        # # Optionally, you can wait for all futures to complete and handle exceptions
        # for future in tqdm(futures):
        #     future.result()
