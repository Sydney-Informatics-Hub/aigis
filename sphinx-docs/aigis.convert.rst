AIGIS.convert package
=====================

Subpackages
-----------

.. toctree::
   :maxdepth: 4

   aigis.convert.orthogonalise

Using The Convert Package
---------------------

The convert package is used to convert between different coordinate systems and formats. The main use case is to convert between the COCO format and the geodataframe. 
The COCO format is a JSON file that contains the coordinates of the objects in the image. The geodataframe is a pandas dataframe that contains the coordinates of the objects in the image, which can be either stored as geojson or geoparquet.

o create tiles from a raster file, use the following command:

.. code-block:: bash

   python aigis/scripts/geojson2coco.py \
                   --raster-file /path/to/data/chatswood_hd.tif \
                   --polygon-file /path/to/data/chatswood.geojson \
                   --tile-dir /path/to/data/big_tiles \
                   --json-name /path/to/data/coco_from_gis_hd.json \
                   --info /path/to/data/info.json \
                   --class-column zone_name 

To merge multiple COCO JSON files, and yield a geojson file for the
input raster, use the following command:

.. code-block:: bash

   python -m aigis/scripts/geojson2coco.py \
                   /path/to/data/raster_tiles/dir \
                   /path/to/data/predictions-coco.json \
                   --tile-extension tif \
                   --geojson-output /path/to/data/output.geojson \
                   --meta-name <name_of_the_dataset>
                   --minimum-rotated-rectangle 



To do a batch conversion, when the conversion should be carried on
multiple input images, use the following command:

.. code-block:: bash

   python -m aigis/scripts/geojson2coco.py \
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

``--raster-dir`` argument is the path to the raster directory. This is a
necessary argument. Rasters are expected to be in this directory.

``--vector-dir`` argument is the path to the vector directory. This is a
necessary argument. geojsons are expected to be in this directory, with
matching names with the rasters.

Please ensure the rasters in the raster directory are named similarly as
the geojsons. If they only differ in a prefix, you can use the
``--pattern`` argument to specify the prefix. For example, if the
rasters are named ``osm_1.tif``, ``osm_2.tif``, and the geojsons are
named ``1.geojson``, ``2.geojson``, you can use ``--pattern osm_`` to
ensure the rasters are matched with the geojsons.

``--output-dir`` argument is the path to the output directory. This is a
necessary argument.

``--class-column`` argument is also a necessary argument that should be
provided. If the provided argument is wrong or does not exist, the code
will create the column with default values.

``--overlap`` argument is the percentage of overlap between the tiles.
For example, if the tile size is 200, and the overlap is 0.5, the
overlap between the tiles will be 100 pixels.

``--concatenate`` argument is a store-true argument. If it is set, the
output geojsons will be concatenated into one big geojson.

``--resume`` argument is a store-true argument. If it is set, the code
will resume the process from the last image that was processed. This is
useful when new images are added to the raster directory, and the
process should be resumed from the last image that was processed.

``--info`` argument is the path to the info json file. This is a
necessary argument. If the argument is not provided, the code will
create the info json file with default values.

``--tile-size`` argument is the size of the tiles in meters.


Submodules
----------

aigis.convert.COCO\_validator module
------------------------------------

.. automodule:: aigis.convert.COCO_validator
   :members:
   :undoc-members:
   :show-inheritance:

aigis.convert.coco module
-------------------------

.. automodule:: aigis.convert.coco
   :members:
   :undoc-members:
   :show-inheritance:

aigis.convert.coordinates module
--------------------------------

.. automodule:: aigis.convert.coordinates
   :members:
   :undoc-members:
   :show-inheritance:

aigis.convert.tiles module
--------------------------

.. automodule:: aigis.convert.tiles
   :members:
   :undoc-members:
   :show-inheritance:

aigis.convert.utils module
--------------------------

.. automodule:: aigis.convert.utils
   :members:
   :undoc-members:
   :show-inheritance:

Module contents
---------------

.. automodule:: aigis.convert
   :members:
   :undoc-members:
   :show-inheritance:
