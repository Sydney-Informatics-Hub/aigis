.. SoilWaterNow documentation master file, created by
   sphinx-quickstart on Fri Feb 17 16:31:15 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to AI annotation, segmentation, and conversion tools for GIS imagery documentation
===========================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:


aigis is a comprehensive toolkit for aerial and satellite imagery acquisition, processing, annotation, and analysis using artificial intelligence. This repository contains three main components:

annotate: Scripts for annotating aerial imagery data. Detailed usage instructions can be found in the aerial_annotation directory.

convert: Tools for converting aerial imagery data to various formats. For detailed instructions, refer to the aerial_conversion directory.

segment: Scripts for segmenting aerial imagery using deep learning models. Refer to the aerial_segmentation directory for more details.


The source code for AIGIS is available on `GitHub <https://github.com/Sydney-Informatics-Hub/aigis>`__.



Table of contetns
=================

-  `Input and Output Data Formats <#input-and-output-data-formats>`__

-  `Setup <#setup>`__

-  `Dataset <#dataset>`__

-  `Usage <#usage>`__

-  `Contributing to the Project <#contributing-to-the-project>`__


--------------


Basic Usage
===========


Input and Output Data Formats
-----------------------------

The repository can convert between the following formats:

-  Images and annotations in COCO JSON format.

-  Georeferenced shapefile polygon vector files, with a readme file
   linking to the original web map server or aerial imagery source to be
   rendered.

--------------

Setup
-----

::
   # Create a new conda environment
   conda create -n aigis python==3.10
   # Activate the environment
   conda activate aigis
   # Clone the code from the repository
   git clone https://github.com/Sydney-Informatics-Hub/aigis
   # Install torch and torchvision
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   # Install the required libraries (including detectron2)
   pip install 'git+https://github.com/facebookresearch/detectron2.git'
   pip install segment-geospatial groundingdino-py leafmap localtileserver
   pip install -e aigis


Dataset
-------

A toy dataset has been uploaded to Roboflow. It is a small subset,
containing Chatswood region, available
`here <https://universe.roboflow.com/sih-vpfnf/gis-hd-200x200>`__.

There are multiple versions of this dataset. Please ignore the first
version. Version 2 and later versions are the ones that are being used.
The main difference of version 2 and 3 is that `version
2 <https://universe.roboflow.com/sih-vpfnf/gis-hd-200x200/2>`__ contains
90 degree augmentaions, while `version
3 <https://universe.roboflow.com/sih-vpfnf/gis-hd-200x200/3>`__ does
not.

For implementing this in your code, you can use the following code
snippet:

.. code:: python

   from roboflow import Roboflow
    
   rf = Roboflow(api_key= 'your_roboflow_api_key' )
   workspace_name = "sih-vpfnf" 
   dataset_version = 3 
   project_name = "gis-hd-200x200" 
   dataset_download_name = "coco-segmentation" 

   project = rf.workspace(workspace_name).project(project_name)
   dataset = project.version(dataset_version).download(dataset_download_name)



Usage
-----
The project has many usages, in 3 categories: annotation, conversion, and segmentation.
For each category, please visit the relevant section/link below.


TBC


--------------

Contributing to the Project
---------------------------

Please make sure to install all the required libraries in the
`requirements.txt <https://github.com/Sydney-Informatics-Hub/aerial-conversion/tree/main/requirements.txt>`__
file for development.

Commit rules
~~~~~~~~~~~~~

In this project, ``pre-commit`` is being used. Hence, please make sure
you have it in your env by ``pip install pre-commit``.

Make sure to run pre-commit on each run. You can run it before commit on
all files to check your code by typing
``pre-commit run --all-files --verbose``, without staging your
pre-commit config. Otherwise, you can run it via ``pre-commit run``
only, or just envoking it while committing (and failing, and committing
again).

Alternatively, to add the hook, after installing pre-commit, run:

::

   pre-commit install



Documentation manual update
---------------------

In case of new modules being added, the following update procedure can be followed:

-  To update the documentation, navigate to the
   `sphinx-docs <https://github.com/Sydney-Informatics-Hub/aigis/tree/main/sphinx-docs/>`__
   directory.
-  Remove the old ``rst`` files from the the docs directory, except
   ``index.rst``. 
-  Make sure ``index.rst`` is not empty.
-  Make sure there is a ``config.py`` file in the ``docs`` directory, containing the root directory, version, and other project details.
-  Navigate to the upper directory: ``cd ..``.
-  Input ``sphinx-apidoc -o doc .`` to regenerate the ``rst`` files.

The next steps are not required, since they are automatically done by github actions. However, they are included here for completeness.

-  Navigate back to the ``doc`` directory.
-  Update the ``index.rst`` file to include the new ``rst`` files, if
   required. Usually not needed. (You don’t have to include the submodules.)
-  Then input ``make html`` for updating the html file.



Indices and tables
==================

For detailed explanation of modules and functions, please see the following pages:

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
