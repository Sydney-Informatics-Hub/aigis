# aerial-annotation
Open source annotations tools for aerial imagery. Part of https://github.com/Sydney-Informatics-Hub/PIPE-3956-aerial-segmentation


## Input Data Format

Download the SA1 file from [Australian Bureau of Statistics](https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files) or similar.

## Output Data Format

Images and annotations stored in COCO JSON format. 



## Installation Instructions

To prepare the environment, run the following commands:

```
conda create --name aerial-annotation python=3.9

conda activate aerial-annotation

pip install -r requirements.txt

```

## Annotating Trees

The script `make_mask.py` in the `scripts` directory can be used to create tree annotations for a set of `.tiff` format aerial images. It uses a modified form of the
`LangSAM` model that is part of the [segment-geospatial](https://samgeo.gishub.org/) package. This leverages both GroundingDINO and the Segment Anything Model with a
text prompt `tree` to detect trees in aerial imagery: see [here](https://samgeo.gishub.org/examples/text_prompts/) for an example of using segment-geospatial to detect
trees.

The method used here has some important modifications from the vanilla segment-geospatial package:

- It adds a new box size threshold called `box-reject` that rejects GroundingDINO boxes that are larger than a given fraction of the input image.

- For boxes larger than `box-reject` a secondary `box-threshold` is used to allow for large boxes containg trees at high confidence.

- Tree annotation masks from overlapping tiles are merged without a call to `gdal.warp`. Instead they are merged by accepting only pixels in the merged mask that are annotated in more than 50% of the individual tiles.

#### Example usage:

```
make_mask.py /path/to/tiff/images \
--output /path/to/output/directory \
--tile-size=600 \
--tile-overlap=30 \
--box-reject=0.9 \
--high-box-threshold=0.35 \
--box-threshold=0.23
```

In the above example the options are:

`--tile-size=600`: The size in pixels of tiles to split the input images into.

`--tile-overlap=30`: Overlap size (as a percentage of `tile-size`) and padding of the tiles. With the options `--tile-size=600` and `--tile-overlap=30` roughly 30 tiles of size 1400x1400 are created for an input image of 3000x3000 pixels.

`--box-reject=0.9`: Reject GroundingDINO boxes larger than 90% of the tile size with value below `--high-box-threshold`.

`--high-box-threshold=0.35`: Box threshold for rejecting GroundingDINO boxes larger than box-reject.

`--box-threshold=0.23`: Box threshold for boxes smaller the `box-reject=0.9`.

### Toy Dataset

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

### Data Cleaning and Preparation

Building data from OSM can be cleaned and prepared for level categorisation using the following code snippet:

```
python scripts/osm_cleaner.py --osm_path /path/to/tiles/osm_building_annotations_by_10_percent_grid/ --columns /data/osm_columns.csv 
```

Where `osm_columns.csv` is a CSV file containing the columns we are interested to keep from the OSM data. OSM columns of interest are located in `data/osm_columns.csv`.



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

### Commit rules:

In this project, `pre-commit` is being used. Hence, please make sure you have it in your
environment by installing it with `pip install pre-commit`.

Make sure to run pre-commit on each commit. You can run it before commit on all files in the
repository using `pre-commit run --all-files`. Otherwise, you can run it via `pre-commit run`
which will only run it on files staged for commit.

Alternatively, to add the hook, after installing pre-commit, run:

```
pre-commit install
```

this will run the pre-commit hooks every time you commit changes to the repository.

