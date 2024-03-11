AIGIS.segment package
=====================


Input Data Format
-----------------

Images and annotations in COCO JSON.

Output Data Format
------------------

Images and annotations in GeoJson, Geoparquet, or COCO JSON.

Example notebooks
-----------------

Jupyter notebooks demonstrating use of a detectron2 model for segmentation are in the ``notebooks`` dir.

The google colab notebook can be run `here <https://colab.research.google.com/github/Sydney-Informatics-Hub/aerial-segmentation/blob/main/notebooks/detectron2_fine_tuning_colab.ipynb>`_.

Usage
-----

Fine-tuning
~~~~~~~~~~~

The following steps are required to fine-tune a model:

1. Get raster & geojson annotations using aerial annotation tool (:doc:`aigis.annotate<aigis.annotate>`)

2. Convert the geojson annotations into COCO JSON files, and concatenate the converted COCO JSONs into one file using aerial conversion (:doc:`aigis.convert<aigis.convert>`)

3. Split the concatenated COCO JSON into train and test (and valid) using :doc:`coco-split<aigis.other>`

4. Run the fine-tuning script ``fine_tuning_detectron2.py``

   Run ``python aigis/scripts/fine_tuning_detectron2.py -h`` to display the help message that describes all the available command-line options and arguments for model fine-tuning.

   Example usage:

.. code-block:: bash

      python aigis/scripts/fine_tuning_detectron2.py \
      --train-json /path/to/train.json \
      --test-json /path/to/test.json \
      --eval-json /path/to/valid.json --evaluate-model \
      --image-root path/to/rasters/ \
      --max-iter=20000 --batch-size=8 --device=cuda \
      --dataset-name=my_dataset \
      --output-dir path/to/save/model/output/ \
      --use-wandb --wandb-key samplekey5d6f65e625c

Prediction
~~~~~~~~~~

The following code snippet can be used to predict on a directory of tiles (batch prediction):

.. code-block:: bash

    python aigis/scripts/prediction_batch_detectron2.py -i "path/to/tiles" -c "path/to/config.yml" -w "path/to/weights/model.pth" --coco "path/to/coco.json" --simplify-tolerance 0.3  --threshold 0.7 --force-cpu 

For getting the minimum rotated rectangles in tile level, you can use the following script:

.. code-block:: bash

    python aigis/scripts/prediction_batch_detectron2.py -i "path/to/tiles" -c "path/to/config.yml" -w "path/to/weights/model.pth" --coco "path/to/coco.json" --minimum-rotated-rectangle --threshold 0.7 --force-cpu 

For more information about the batch script, you may run:

.. code-block:: bash

    python aigis/scripts/prediction_batch_detectron2.py  --help

For prediction and visualisation on a single image, you can use the following script:

.. code-block:: bash

    python prediction_detectron2.py --image "path/to/image" --config "path/to/config.yml" --weights "path/to/weights/model.pth" --threshold 0.7 --coco "path/to/coco.json"

For more information about the single image script, you may run:

.. code-block:: bash

    python aigis/scripts/prediction_detectron2.py  --help

For prediction and yielding a COCO JSON from a raster, you can use the following script:

.. code-block:: bash

    python aigis/scripts/prediction_raster_detectron2.py --raster-file "path/to/raster.tif"  --tile-size 0.002 --config "path/to/config.yml" --weights "path/to/weights/model.pth"


----------------


Submodules
----------

aigis.segment.eval module
-------------------------

.. automodule:: aigis.segment.eval
   :members:
   :undoc-members:
   :show-inheritance:

aigis.segment.postprocess module
--------------------------------

.. automodule:: aigis.segment.postprocess
   :members:
   :undoc-members:
   :show-inheritance:

aigis.segment.utils module
--------------------------

.. automodule:: aigis.segment.utils
   :members:
   :undoc-members:
   :show-inheritance:

Module contents
---------------

.. automodule:: aigis.segment
   :members:
   :undoc-members:
   :show-inheritance:
