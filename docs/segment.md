# aigis segment
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/SIH/building-segmentation)

Open source aerial imagery segmentation model fine tuning, evaluation, and prediction tools. Part of https://github.com/Sydney-Informatics-Hub/PIPE-3956-aerial-segmentation

# WIP: Conversion from previous standalone package to submodule underway, this documentation is old and will be changing.

## Setup

### System Prep

Before installing detectron2, we should make sure our system has the following requirements installed

- g++ needs to be installed for detectron2: `sudo apt-get install g++`
- Install cv2 requirements: `sudo apt-get update && apt-get install ffmpeg libsm6 libxext6 -y`
- Finally, make sure NVIDIA drivers are installed, follow instructions [here](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html#ubuntu-lts). This has been tested for tesla T4 GPUs. Note: no need to run post installation steps in case CUDA is installed on your conda env. 

Full example:

```bash
sudo apt-get install g++
sudo apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
sudo apt-get install linux-headers-$(uname -r)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID | sed -e 's/\.//g')
wget https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-drivers
```

### Local (or interactive VM)

```bash
conda create -n aerial-segmentation python==3.9

conda activate aerial-segmentation

pip install 'git+https://github.com/Sydney-Informatics-Hub/aerial-segmentation.git'

pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

## Input Data Format

Images and annotations in COCO JSON.

## Output Data Format

Images and annotations in COCO JSON.

## Example notebooks

Jupyter notebooks demonstrating use of a detectron2 model for segmentation are in the `notebooks` dir.

The google colab notebook can be run [here](https://colab.research.google.com/github/Sydney-Informatics-Hub/aerial-segmentation/blob/main/notebooks/detectron2_fine_tuning_colab.ipynb)


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

### Fine-tuning

Run on RONIN with a defined COCO JSON dataset in subdirectories.

```{bash}
conda create -n training python=3.9
conda activate training
```

#### 1. Get raster & geojson annotations using aerial annotation
Refer to examples in [aerial-annotation](https://github.com/Sydney-Informatics-Hub/aerial-annotation) for what the data needs to look like, e.g. directory structure of raster and annotation files.

#### 2. Convert the geojson annotations into COCO JSON files, and concatenate the converted COCO JSONs into one file using aerial conversion

Refer to examples in [aerial-conversion](https://github.com/Sydney-Informatics-Hub/aerial-conversion) for how to convert and concatenate geojson files into one COCO JSON file.


#### 3. Split the concatenated COCO JSON into train and test (and valid) using cocosplit.

Example usage to split into train (70%), test (20%) and valid (10%) data sets:
```
git clone https://github.com/akarazniewicz/cocosplit

# Modify requirements.txt to replace sklearn with scikit-learn for python3.9
pip install -r requirements.txt

cd cocosplit

python cocosplit.py -s 0.7 /path/to/concatenated_coco.json /path/to/save/output/train.json /path/to/save/output/test_valid.json

python cocosplit.py -s 0.667 /path/to/test_valid.json /path/to/save/output/test.json /path/to/save/output/valid.json
```


#### 4. Run the fine tuning script `fine_tuning_detectron2`

Run `fine_tuning_detectron2 -h` to display the help message that describes all the available command-line options and arguments for model fine tuning.

Example usage:
```
fine_tuning_detectron2 \
--train-json /path/to/train.json \
--test-json /path/to/test.json \
--eval-json /path/to/valid.json --evaluate-model \
--image-root path/to/rasters/ \
--max-iter=20000 --batch-size=8 --device=cuda \
--dataset-name=my_dataset \
--output-dir path/to/save/model/output/ \
--use-wandb --wandb-key samplekey5d6f65e625c
```


### Prediction

The following code snippet can be used to predict on a directory of tiles (batch prediction):


```bash
prediction_batch_detectron2 -i "path/to/tiles" -c "path/to/config.yml" -w "path/to/weights/model.pth" --coco "path/to/coco.json" --simplify-tolerance 0.3  --threshold 0.7 --force-cpu 

```

For getting the minimum rotated rectangles in tile level, you can use the following script:

```bash
prediction_batch_detectron2 -i "path/to/tiles" -c "path/to/config.yml" -w "path/to/weights/model.pth" --coco "path/to/coco.json" --minimum-rotated-rectangle --threshold 0.7 --force-cpu 

```

For more information about the batch script, you may run:

```bash
prediction_batch_detectron2  --help

```

For prediction and visualisation on a single image, you can use the following script:

```bash
prediction_detectron2 --image "path/to/image" --config "path/to/config.yml" --weights "path/to/weights/model.pth" --threshold 0.7 --coco "path/to/coco.json"

```

For more information about the single image script, you may run:

```bash
prediction_detectron2  --help

```

For prediction and yielding a COCO JSON from a raster, you can use the following script:

```bash

prediction_raster_detectron2 --raster-file "path/to/raster.tif"  --tile-size 0.002 --config "path/to/config.yml" --weights "path/to/weights/model.pth" --threshold 0.7 --coco-out "path/to/output/coco.json" --temp-dir "path/to/tile/storage/" --simplify-tolerance 0.95

```

### Density Estimation and Mapping

The repository also contains a script for density estimation. The script can be used as follows:

```bash
python -m scripts.density_map --input-path /path/to/annotation.geojson --average-storeys 1 --footprint-ratio 0.5 --tile-size 200 --area-unit utm

```

Where: 

- `input-path` is the path to the geojson file containing the annotations, 
- `average-storeys` is the average number of storeys in the buildings. If None is given, the script will look for the `storeys` property in the geojson file (or any column that has been specified in the `storeys-column` argument), 
- `footprint-ratio` the ratio of area-based density to number-based density. The script calculates density in two ways: area-based and number-based. The area-based density is calculated by dividing the area of the building by the area of the tile. The number-based density is calculated by dividing the number of buildings in the tile by the area of the tile. The area-based density is then multiplied by the `footprint-ratio` and the number-based density is multiplied by `1 - footprint-ratio`. The default value is 0.5,
- `tile-size` is the size of the tile in pixels for making a map. This basically acts as the resolution parameter, and 
- `area-unit` is the unit of the area of the building. The area unit can be either `utm` or `meter`. `meter` will use `3857` as the EPSG code, while `utm` will use the UTM zone of the centroid of the building. The default value is `utm` for better accuracy.




## Contributing to the Project

Please make sure to install all the required libraries in the [requirements.txt](https://github.com/Sydney-Informatics-Hub/aerial-segmentation/tree/main/requirements.txt) file for development.


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

## Deploy benchmark script to Jetson Nano

There is a Docker image available with a GPU enabled version of PyTorch and Detectron2 compiled on a
Jetson Nano using Python 3.8. This can be used to deploy scripts from this repository on these devices.
The image can be retrieved to the local Docker repository on a Jetson Nano using:

```
sudo docker pull sydneyinformaticshub/aerialseg:det2-py38-jetson
```

In order to run the Docker image first make sure the Jetson Nano has the lastest compatible version of
Jetpack installed. You will need at least 4.6 to run the image, 5.xand greater are not compatible. Check
[here](https://developer.nvidia.com/embedded/jetpack-archive) for a list of available Jetpack versions
compatible with the Nano.

Once Jetpack is updated you can log into the Docker image and mount the `aerial-segmentation` repository
from the current directory inside it using:

```
sudo docker run -v ${PWD}/aerial-segmentation:/root/aerial-segmentation -it --runtime nvidia sydneyinformaticshub/aerialseg:det2-py38-jetson /bin/bash
```

The `sydneyinformaticshub/aerialseg:det2-py38-jetson` image can also be used as a base image to install
further packages and scripts as required.
