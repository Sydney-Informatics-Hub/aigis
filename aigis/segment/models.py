# -*- coding: utf-8 -*-
import os
import wget

def download_detectron2_model_weights(model_type):
    """
    Downloads the model weights and configuration file for the specified model type.

    Args:
        model_type (str or tuple): The type of model to download. If 'trees' or 'buildings', the corresponding
            model weights and configuration file will be downloaded. If a tuple is provided, it should contain
            the URLs for the model weights and configuration file.

    Returns:
        tuple: A tuple containing the filenames of the downloaded model weights and configuration file.

    Raises:
        ValueError: If an invalid model_type is provided.

    """
    if model_type == "trees":
        model_weights_url = "https://huggingface.co/spaces/SIH/aerial-segmentation-model-selection/resolve/main/tree_model_weights/treev3model_0012499.pth"
        config_url = "https://huggingface.co/spaces/SIH/aerial-segmentation-model-selection/resolve/main/tree_model_weights/treev3_tms_sixmaps_cfg.yaml"
    elif model_type == "buildings":
        model_weights_url = "https://huggingface.co/spaces/SIH/building-segmentation/resolve/main/model_weights/model_final.pth"
        config_url = "https://huggingface.co/spaces/SIH/building-segmentation/resolve/main/model_weights/buildings_poc_cfg.yml"
    else:
        model_weights_url, config_url = model_type

    # Download model weights
    model_weights_filename = os.path.basename(model_weights_url)
    wget.download(model_weights_url, model_weights_filename)

    # Download config file
    config_filename = os.path.basename(config_url)
    wget.download(config_url, config_filename)

    return model_weights_filename, config_filename

def download_vit_model_weights(model_type, config_url=None, model_url=None, preprocessor_url=None, training_args_url=None):
    """
    Downloads the files from the specified URLs using wget.

    Args:
        model_type (str): The type of model to download. If 'lczs', the corresponding
            files will be downloaded. If any other value is provided, the user can
            specify the URLs for the config, model, preprocessor, and training_args.

        config_url (str): The URL for the config file.

        model_url (str): The URL for the model file.

        preprocessor_url (str): The URL for the preprocessor file.

        training_args_url (str): The URL for the training_args file.

    Returns:
        list: A list of filenames of the downloaded files.

    """
    if model_type == "lczs":
        config_url = "https://huggingface.co/spaces/SIH/lcz-classification/resolve/main/ViT_LCZs_v3/config.json"
        model_url = "https://huggingface.co/spaces/SIH/lcz-classification/resolve/main/ViT_LCZs_v3/model.safetensors"
        preprocessor_url = "https://huggingface.co/spaces/SIH/lcz-classification/resolve/main/ViT_LCZs_v3/preprocessor_config.json"
        training_args_url = "https://huggingface.co/spaces/SIH/lcz-classification/resolve/main/ViT_LCZs_v3/training_args.bin"
    
    # Download config file
    config_filename = os.path.basename(config_url)
    wget.download(config_url, config_filename)
    # Download model file
    model_filename = os.path.basename(model_url)
    wget.download(model_url, model_filename)
    # Download preprocessor file
    preprocessor_filename = os.path.basename(preprocessor_url)
    wget.download(preprocessor_url, preprocessor_filename)
    # Download training args file
    training_args_filename = os.path.basename(training_args_url)
    wget.download(training_args_url, training_args_filename)
    
    return [config_filename, model_filename, preprocessor_filename, training_args_filename]