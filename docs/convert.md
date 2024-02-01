# WIP: Conversion from previous standalone package to submodule underway, this documentation is old and will be changing.

# aerial-conversion
Open source tools enabling interchange between computer vision annotation and GIS data formats. Part of https://github.com/Sydney-Informatics-Hub/PIPE-3956-aerial-segmentation

Module documentation is available at [documentation](https://sydney-informatics-hub.github.io/aerial-conversion/).

Source code available at [github](https://github.com/Sydney-Informatics-Hub/aerial-conversion).

Releases available at [github](https://github.com/Sydney-Informatics-Hub/aerial-conversion/releases).

---

## Input and Output Data Formats

The repository can convert between the following formats:

* Images and annotations in COCO JSON format. 

* Georeferenced shapefile polygon vector files, with a readme file linking to the original web map server or aerial imagery source to be rendered.

---

## Setup

```
conda create -n aerial-conversion-dev python=3.9

conda activate aerial-conversion-dev

pip install 'git+https://github.com/Sydney-Informatics-Hub/aerial-conversion.git'
```



## Dataset

A toy dataset has been uploaded to Roboflow. It is a small subset, containing Chatswood region, available [here](https://universe.roboflow.com/sih-vpfnf/gis-hd-200x200).

There are multiple versions of this dataset. Please ignore the first version. Version 2 and later versions are the ones that are being used. The main difference of version 2 and 3 is that [version 2](https://universe.roboflow.com/sih-vpfnf/gis-hd-200x200/2) contains 90 degree augmentaions, while [version 3](https://universe.roboflow.com/sih-vpfnf/gis-hd-200x200/3) does not.

For implementing this in your code, you can use the following code snippet:

```python
from roboflow import Roboflow
 
rf = Roboflow(api_key= 'your_roboflow_api_key' )
workspace_name = "sih-vpfnf" 
dataset_version = 3 
project_name = "gis-hd-200x200" 
dataset_download_name = "coco-segmentation" 

project = rf.workspace(workspace_name).project(project_name)
dataset = project.version(dataset_version).download(dataset_download_name)
```
<!-- 
# Register the dataset
from detectron2.data.datasets import register_coco_instances
dataset_name = "chatswood-dataset" #@param {type:"string"}
dataset_folder = "gis-hd-200x200" #@param {type:"string"}
register_coco_instances(f"{dataset_name}_train", {}, f"{dataset_folder}/train/_annotations.coco.json", f"/content/{dataset_folder}/train/")
register_coco_instances(f"{dataset_name}_val", {}, f"{dataset_folder}/valid/_annotations.coco.json", f"/content/{dataset_folder}/valid/")
register_coco_instances(f"{dataset_name}_test", {}, f"{dataset_folder}/test/_annotations.coco.json", f"/content/{dataset_folder}/test/")

# Use the dataset
from detectron2.config import get_cfg

cfg = get_cfg()
cfg.DATASETS.TRAIN = (f"{dataset_name}_train",)
cfg.DATASETS.TEST = (f"{dataset_name}_test",)
# then do the other configs

``` -->



## Usage

To create tiles from a raster file, use the following command:


```bash
python -m aerial_conversion.scripts.geojson2coco \
                --raster-file /path/to/data/chatswood_hd.tif \
                --polygon-file /path/to/data/chatswood.geojson \
                --tile-dir /path/to/data/big_tiles \
                --json-name /path/to/data/coco_from_gis_hd.json \
                --info /path/to/data/info.json \
                --class-column zone_name 
```

To merge multiple COCO JSON files, and yield a geojson file for the input raster, use the following command:

```bash
python -m aerial_conversion.scripts.coco2geojson \
                /path/to/data/raster_tiles/dir \
                /path/to/data/predictions-coco.json \
                --tile-extension tif \
                --geojson-output /path/to/data/output.geojson \
                --meta-name <name_of_the_dataset>
                --minimum-rotated-rectangle 
```

To do a batch conversion, when the conversion should be carried on multiple input images, use the following command:

```bash
python -m aerial_conversion.scripts.batch_geojson2coco \
                --raster-dir /path/to/data/rasters/ \
                --vector-dir /path/to/data/geojsons/ \
                --output-dir /path/to/data/outputs/ \
                --tile-size <size_of_the_tiles>
                --class-column <class_column_in_geojsons_to_look_for>
                --overlap <overlap_in_percentage_between_tiles> \
                --pattern <a_pattern_to_ensure_raster_names_matches_the_geojsons>
                --concatenate <whether_to_concatenate_the_output_geojsons_into_one_big_json>
                --info /path/to/data/info.json \
                --resume <whether_to_resume_the_process_in_case_new_images_are_added>
```

`--raster-dir` argument is the path to the raster directory. This is a necessary argument. Rasters are expected to be in this directory.

`--vector-dir` argument is the path to the vector directory. This is a necessary argument. geojsons are expected to be in this directory, with matching names with the rasters.

Please ensure the rasters in the raster directory are named similarly as the geojsons. If they only differ in a prefix, you can use the `--pattern` argument to specify the prefix. For example, if the rasters are named `osm_1.tif`, `osm_2.tif`, and the geojsons are named `1.geojson`, `2.geojson`, you can use `--pattern osm_` to ensure the rasters are matched with the geojsons.

`--output-dir` argument is the path to the output directory. This is a necessary argument. 

`--class-column` argument is also a necessary argument that should be provided. If the provided argument is wrong or does not exist, the code will create the column with default values.

`--overlap` argument is the percentage of overlap between the tiles. For example, if the tile size is 200, and the overlap is 0.5, the overlap between the tiles will be 100 pixels.

`--concatenate` argument is a store-true argument. If it is set, the output geojsons will be concatenated into one big geojson.

`--resume` argument is a store-true argument. If it is set, the code will resume the process from the last image that was processed. This is useful when new images are added to the raster directory, and the process should be resumed from the last image that was processed.

`--info` argument is the path to the info json file. This is a necessary argument. If the argument is not provided, the code will create the info json file with default values.


`--tile-size` argument is the size of the tiles in meters.


## Other Scrpts

### Splitting Dataset

Splitting dataset to train, test, and validation sets can be achieved using the following script:

```bash
python -m aerial_conversion.scripts.coco_split -s 0.7 /path/to/concatenated_coco.json /path/to/save/output/train.json /path/to/save/output/test_valid.json

python -m aerial_conversion.scripts.coco_split -s 0.667 /path/to/test_valid.json /path/to/save/output/test.json /path/to/save/output/valid.json
```


### Balancing dataset

To tinker with the dataset and balance it, the following scrips can be used. 

To isolate the categories:
    
```bash
python -m aerial_conversion.scripts.coco_balance -i /path/to/input/coco.json -o /path/to/output/coco-catlimited.json -c '<category 1>,<category 2>,...' --int_cats
```

`--int_cats` argument is a store-true argument. If it is set, the categories will be interpreted as integers. Otherwise, they will be interpreted as strings.

`-c` argument is the categories to be isolated. They should be comma separated.


To balance the dataset by removing a subsample of the images which have only a single category (the biggest category):

```bash
python -m aerial_conversion.scripts.coco_balance -i /path/to/input/coco.json -o /path/to/output/coco-balanced.json --balance_cats
```

`--balance_cats` argument is a store-true argument. If it is set, the dataset will be balanced by removing a subsample of the images which have only a single category (the biggest category).



<!-- ---

## Documentation

The documentation for the project is provided in the [documentation](https://github.com/Sydney-Informatics-Hub/aerial-conversion/tree/main/docs/_build/html/index.html) file.
Please read the [documentation](https://github.com/Sydney-Informatics-Hub/aerial-conversion/tree/main/docs/_build/html/index.html) for further informationon the project, modules, and dependencies. -->

---
k
## Contributing to the Project

Please make sure to install all the required libraries in the [requirements.txt](https://github.com/Sydney-Informatics-Hub/aerial-conversion/tree/main/requirements.txt) file for development.


### Commit rules:

In this project, `pre-commit` is being used. Hence, please make sure you have it in your env by `pip install pre-commit`.

Make sure to run pre-commit on each run. You can run it before commit on all files to check your code by typing `pre-commit run --all-files --verbose`, without staging your pre-commit config.
Otherwise, you can run it via `pre-commit run` only, or just envoking it while committing (and failing, and committing again).

Alternatively, to add the hook, after installing pre-commit, run:

```
pre-commit install
```
<!-- 
### Documentation update:

* To update the documentation, navigate to the [docs](https://github.com/Sydney-Informatics-Hub/aerial-conversion/tree/main/docs/) directory.
* Remove the old `rst` files from the the docs directory, except `index.rst`.
* Navigate to the upper directory: `cd ..`.
* Input `sphinx-apidoc -o docs .` to regenerate the `rst` files.
* Navigate back to the `docs` directory.
* Update the `index.rst` file to include the new `rst` files, if required. Usually not needed. (You don't have to include the submodules.)
* Then input `make html` for updating the html file. -->
