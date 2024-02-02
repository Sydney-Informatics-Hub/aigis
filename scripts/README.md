# Scripts

This folder stores useful scripts demonstrating functions of aerial-annotation.


## Standalone Scripts and Tools

### `batch_geojson2coco.py`
Batch processing of geojson files to coco format.

### `coco_balance.py`
Utilities for balancing Coco classes by removing unwanted categories, oversampling underrepresented categories, and undersampling overrepresented categories.

### `coco_split.py`
Utilities for splitting Coco datasets into train, validation, and test sets.

### `coco2geojson.py`
Script to convert Coco annotations to geojson format.

### `denisty_map.py`
Utilities for generating density maps from satellite imagery based on building heights and counts.

### `download_raster.py`
Utilities for downloading satellite imagery from Google Earth Engine.

### `fine_tuning_detectron2.py`
Script for fine tuning detectron2 models on custom datasets.

### `geojson2coco.py`
Script to convert geojson annotations to Coco format.

### `get_raster_jpeg.py`
Script to extract JPEG images from satellite imagery.

### `make_mask.py``
Script to generate binary masks from satellite imagery.

### `osm_cleaner.py`
Utilities for cleaning and parsing OSM building data. Useful for building training data sets for model fine tuning.

### `poc_gradio.py` 
Minimal gradio application for inference demos of fine tuned models built with `aigis`

### `sa1.py` 
Utilities for working with ABS SA1 boundaries to perform data collection.