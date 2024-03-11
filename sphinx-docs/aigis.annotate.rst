AIGIS.annotate package
======================


Using The Annotate Package
--------------------------

The ``aigis.annotate`` package provides a set of tools for annotating images and creating masks for use in training machine learning models. The package is designed to be used with aerial imagery, but can be used with any image data.
Segment geospatial is used to detect trees in aerial imagery. 


Annotating Trees
~~~~~~~~~~~~~~~~

The script ``make_mask.py`` in the ``scripts`` directory can be used to create tree annotations for a set of ``.tiff`` format aerial images. It uses a modified form of the
`LangSAM` model that is part of the `segment-geospatial <https://samgeo.gishub.org/>`_ package. This leverages both GroundingDINO and the Segment Anything Model with a
text prompt ``tree`` to detect trees in aerial imagery: see `here <https://samgeo.gishub.org/examples/text_prompts/>`_ for an example of using segment-geospatial to detect
trees.

The method used here has some important modifications from the vanilla segment-geospatial package:

- It adds a new box size threshold called ``box-reject`` that rejects GroundingDINO boxes that are larger than a given fraction of the input image.

- For boxes larger than ``box-reject`` a secondary ``box-threshold`` is used to allow for large boxes containg trees at high confidence.

- Tree annotation masks from overlapping tiles are merged without a call to ``gdal.warp``. Instead, they are merged by accepting only pixels in the merged mask that are annotated in more than 50% of the individual tiles.

Example usage
~~~~~~~~~~~~~

.. code-block:: bash

    python aigis/scripts/make_mask.py /path/to/tiff/images \
    --output /path/to/output/directory \
    --tile-size=600 \
    --tile-overlap=30 \
    --box-reject=0.9 \
    --high-box-threshold=0.35 \
    --box-threshold=0.23

In the above example, the options are:

``--tile-size=600``: The size in pixels of tiles to split the input images into.

``--tile-overlap=30``: Overlap size (as a percentage of ``tile-size``) and padding of the tiles. With the options ``--tile-size=600`` and ``--tile-overlap=30`` roughly 30 tiles of size 1400x1400 are created for an input image of 3000x3000 pixels.

``--box-reject=0.9``: Reject GroundingDINO boxes larger than 90% of the tile size with value below ``--high-box-threshold``.

``--high-box-threshold=0.35``: Box threshold for rejecting GroundingDINO boxes larger than box-reject.

``--box-threshold=0.23``: Box threshold for boxes smaller than ``box-reject=0.9``.


--------------


Submodules
----------

aigis.annotate.utils module
---------------------------

.. automodule:: aigis.annotate.utils
   :members:
   :undoc-members:
   :show-inheritance:

Module contents
---------------

.. automodule:: aigis.annotate
   :members:
   :undoc-members:
   :show-inheritance:
