AIGIS Other Scripts
===================


Using Other Scripts
---------------------

The following scripts are used to manipulate the dataset in various ways. They are used to split the dataset, balance the dataset, and to convert the dataset to other formats.

Splitting Dataset
-----------------

Splitting datasets to train, test, and validate sets can be achieved using the following script:

.. code-block:: bash

    python -m aigis/scripts/coco_split.py -s 0.7 /path/to/concatenated_coco.json /path/to/save/output/train.json /path/to/save/output/test_valid.json

    python -m aigis/scripts/coco_split.py -s 0.667 /path/to/test_valid.json /path/to/save/output/test.json /path/to/save/output/valid.json


Balancing dataset
-----------------

To tinker with the dataset and balance it, the following scrips can be used.

To isolate the categories:

.. code-block:: bash

    python -m aigis/scripts/coco_balance.py -i /path/to/input/coco.json -o /path/to/output/coco-catlimited.json -c '<category 1>,<category 2>,...' --int_cats

``--int_cats`` argument is a store-true argument. If it is set, the categories will be interpreted as integers. Otherwise, they will be interpreted as strings.

``-c`` argument is the categories to be isolated. They should be comma-separated.

To balance the dataset by removing a subsample of the images that have only a single category (the biggest category):

.. code-block:: bash

    python -m aigis/scripts/coco_balance.py -i /path/to/input/coco.json -o /path/to/output/coco-balanced.json --balance_cats

``--balance_cats`` argument is a store-true argument. If it is set, the dataset will be balanced by removing a subsample of the images that have only a single category (the biggest category).

