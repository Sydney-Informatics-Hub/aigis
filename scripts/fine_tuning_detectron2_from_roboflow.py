#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os

import wandb
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.data.datasets import load_coco_json, register_coco_instances
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.logger import setup_logger
from roboflow import Roboflow

setup_logger()


def create_parser():
    parser = argparse.ArgumentParser(
        description="Fine-tune Detectron2 weights using annotated Roboflow dataset."
    )
    roboflow_group = parser.add_argument_group("Roboflow options")
    roboflow_group.add_argument(
        "--workspace",
        "-w",
        type=str,
        help="Roboflow workspace name.",
        required=True,
    )
    roboflow_group.add_argument(
        "--project",
        "-p",
        type=str,
        help="Roboflow project name.",
        required=True,
    )
    roboflow_group.add_argument(
        "--project-version",
        type=int,
        default=0,
        help="Roboflow project version number to download. (Default: Highest available).",
    )
    roboflow_group.add_argument(
        "--roboflow-api-key", "-a", type=str, help="Roboflow API key.", required=True
    )
    wandb_group = parser.add_argument_group("Weights & Biases options")
    wandb_group.add_argument(
        "--use-wandb",
        action="store_true",
        help="Initialise Weights & Biases (wandb) for this run.",
    )
    wandb_group.add_argument(
        "--wandb-project",
        type=str,
        default="gis-segmentation",
        help="The name of the wandb project being sent the new run. (Default: %(default)s)",
    )
    wandb_group.add_argument(
        "--wandb-entity",
        type=str,
        default="sih",
        help="wandb username or team name to whose UI the run will be sent.",
    )
    det_group = parser.add_argument_group("Detectron2 options")
    det_group.add_argument(
        "--det2-model",
        type=str,
        default=os.path.join(
            "COCO-InstanceSegmentation", "mask_rcnn_R_101_FPN_3x.yaml"
        ),
        help="Detectron2 model configuration YAML to use. "
        "See https://github.com/facebookresearch/detectron2/tree/main/configs for what is available. "
        "(Default: %(default)s)",
    )
    det_group.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Number of images per batch in training. (Default: %(default)s)",
    )
    det_group.add_argument(
        "--max-iter",
        type=int,
        default=3000,
        help="Number of iterations in training. (Default: %(default)s)",
    )
    det_group.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(".", "output"),
        help="Directory to save output configuration yaml and pre-trained weights. (Default: %(default)s)",
    )
    det_group.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device (cuda or cpu) on which to run the fine tuning. (Default=%(default)s)",
    )
    det_group.add_argument(
        "--evaluate-model",
        action="store_true",
        help="Evaluate the model on the validation data after fine tuning.",
    )
    return parser


def get_roboflow_dataset(
    api_key: str, workspace: str, project: str, version_number: int = 0
):
    """Download a dataset from the Roboflow server.

    Parameters
    ----------
    api_key : str
        A Roboflow API key required to access the data.
    workspace : str
        Roboflow workspace name.
    project : str
        Roboflow project name.
    version_number : int, optional
        The version number of the Roboflow project.
        Values < 1 mean find the highest available version.

    Returns
    -------
    :class:`Dataset`
        A Dataset instance pointing to the data downloaded from Roboflow
    """

    # This script only supports 'coco-segmentation' format.
    # I'm making it explicit here in case we generalise it in future.
    roboflow_dataset_format = "coco-segmentation"

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project)
    version_list = project.versions()
    all_versions = [os.path.basename(version.id) for version in version_list]
    # if version_number is 0 get the highest available otherwise
    # try to pick the specified version
    if version_number < 1:
        version_number = max(all_versions)
    version = project.version(version_number)
    dataset = version.download(roboflow_dataset_format)
    return dataset


def register_coco_json_from_roboflow(
    name: str, location: str, instance_type: str = "train"
):
    """Register a COCO JSON format file obtained from Roboflow for Detectron2
    instance detection.

    Parameters
    ----------
    name : str
        The name of the dataset obtained from Roboflow.
    location : str
        The root path of the dataset downloaded from Roboflow.
    instance_type : str, optional
        The instance type to register, defaults to 'train'.
        It is assumed the COCO JSON and related images are contained in
        the sub-directory `instance_type` of `location`.

    Returns
    -------
    str
        The name of the registered instance - which can be used by Detectron2.
    """
    # Roboflow always saves its COCO annotation file in the dir containing the
    # images with this filename.
    roboflow_coco_json_filename = "_annotations.coco.json"

    instance_name = name + "_" + instance_type
    image_root = os.path.join(location, instance_type)
    coco_json_file = os.path.join(location, instance_type, roboflow_coco_json_filename)
    register_coco_instances(instance_name, {}, coco_json_file, image_root)
    # Load the json to populate the `thing_classes` list in the Metdata
    _ = load_coco_json(coco_json_file, image_root, instance_name)
    return instance_name


def setup_detectron_config(
    model_name: str,
    train_dataset: str,
    test_dataset: str,
    output_dir: str,
    batch_size: int = 2,
    max_iter: int = 3000,
):
    """Set up a Detectron2 configuration.

    This will:
    1. Load a default detectron2 config
    2. Update the default with the detectron2 model configuration sepcified
       in args.det2_model and set initial weights from the model
    3. Derive configuration defaults from the
       detectron2 dataset
    4. Update configuration from parameters passed to this function.

    Parameters
    ----------
    model_name : str
        The name of the model (configuration and initial weights)
        to use for training.
    train_dataset : str
        The instance name of the training dataset
        (defined using `register_coco_json_from_roboflow`)
    test_dataset : str
        The instance name of the testing dataset
        (defined using `register_coco_json_from_roboflow`)
    output_dir : str
        The path of the output directory into which the configuration yaml
        and fine-tuned weights will be saved.
    batch_size : int, optional
        Number of images per batch in training.
    max_iter : int, optional
        Number of iterations in training.

    Returns
    -------
    detectron2.config.CfgNode
        The detectron2 configuration object
    """

    # Load the default configuration
    cfg = get_cfg()
    # Merge configuration defaults from the specified model
    cfg.merge_from_file(model_zoo.get_config_file(model_name))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)
    # Set up the dataset names
    cfg.DATASETS.TRAIN = (train_dataset,)
    cfg.DATASETS.TEST = (test_dataset,)
    # Derive NUM_CLASSES from COCO JSON
    coco_meta_cat = MetadataCatalog.get(train_dataset)
    num_classes = len(coco_meta_cat.thing_classes)
    # num_clsses + 1 to inculde 'background' as a class
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes + 1
    # Set up some hardcoded defaults
    # Could expose any of these as command line options later
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    # Update configuration from function parameters.
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.OUTPUT_DIR = output_dir

    return cfg


def main(args=None):
    parser = create_parser()
    args = parser.parse_args(args)

    # Shall we use wandb?
    if args.use_wandb:
        # Accept API keys from the TTY if args.wandb_key is None
        wandb.login(key=args.wandb_key)
        # There are loads of options to wandb.init, some of which we might like to expose
        # further down the line - for now I'm just using the options from the colab notebook.
        wandb.init(
            project=args.wandb_project, entity=args.wandb_entity, sync_tensorboard=True
        )

    # Download the roboflow dataset and register it in detectron2
    dataset = get_roboflow_dataset(
        args.roboflow_api_key,
        args.workspace,
        args.project,
        version_number=args.project_version,
    )

    # Register the train and test datasets with detectron2
    train_dataset = register_coco_json_from_roboflow(
        dataset.name, dataset.location, instance_type="train"
    )
    test_dataset = register_coco_json_from_roboflow(
        dataset.name, dataset.location, instance_type="test"
    )

    # Setup Detectron2 configuration from user supplied args and derive some
    # others.
    # In future if there are extra Detectron2 config vars to change then
    # we can set up a .yaml to store the config which can be passed via a
    # command line argument.
    cfg = setup_detectron_config(
        args.det2_model,
        train_dataset,
        test_dataset,
        args.output_dir,
        batch_size=args.batch_size,
        max_iter=args.max_iter,
    )

    # Fine-tune the model and put output weights in cfg.OUTPUT_DIR
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.MODEL.DEVICE = args.device
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Shutdown wandb run if we were using it in this process.
    if wandb.run is not None:
        wandb.finish()

    # Evaluate the model if requested
    if args.evaluate_model:
        val_dataset = register_coco_json_from_roboflow(
            dataset.name, dataset.location, instance_type="valid"
        )
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        predictor = DefaultPredictor(cfg)
        evaluator = COCOEvaluator(val_dataset, output_dir=cfg.OUTPUT_DIR)
        val_loader = build_detection_test_loader(cfg, val_dataset)
        inference_on_dataset(predictor.model, val_loader, evaluator)

    # Save the config YAML to output directoryß
    with open(os.path.join(cfg.OUTPUT_DIR, f"{dataset.name}_cfg.yaml"), "w") as f:
        f.write(cfg.dump())


if __name__ == "__main__":
    main()
