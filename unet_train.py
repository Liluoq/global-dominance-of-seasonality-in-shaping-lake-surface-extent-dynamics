"""Train a U-Net model for lake area detection.

This script trains a U-Net model using configuration parameters specified in a config file.
The model is trained on Google Earth Engine (GEE) data to detect lake/water areas.

The script checks if the specified basin (HYBAS_ID) has already been trained before proceeding.
After training, it updates the training record to mark the basin as trained.

The configuration file should specify:
    HYBAS_ID (int): Basin ID to train model for
    TRAINING_PATTERN (str): Pattern to identify training data files
    EVAL_PATTERN (str): Pattern to identify evaluation data files  
    LOCAL_SAMPLE_FOLDER (str): Folder containing training/eval samples
    INPUT_BANDS (list): List of input bands to use
    INPUT_BAND_SCALING_FACTORS (list): Scaling factors for input bands
    RESPONSE (list): Target variables to predict
    KERNEL_SIZE (int): Size of convolution kernels
    EPOCHS (int): Number of training epochs
    BATCH_SIZE (int): Training batch size
    SAVE_MODELS_AND_FIGURES_FOLDER (str): Output folder for models/figures
    KEEP_RATIO (float): Ratio of data to keep for training

Usage:
    python unet_train.py --config path/to/config.py

Args:
    --config: Path to configuration file containing model parameters
"""

import unetgee as ug
import argparse
import sys
from update_training_record import update_training_record, get_training_status
from batch_processing.load_config_module import load_config_module

parser = argparse.ArgumentParser(description='Train U-Net model on GEE')
parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')

args = parser.parse_args()
config_path = args.config

config = load_config_module(config_path)

hybas_id = config.HYBAS_ID
training_pattern = config.TRAINING_PATTERN
eval_pattern = config.EVAL_PATTERN
local_sample_folder = config.LOCAL_SAMPLE_FOLDER
input_bands = config.INPUT_BANDS
input_band_scaling_factors = config.INPUT_BAND_SCALING_FACTORS
response = config.RESPONSE
kernel_size = config.KERNEL_SIZE
epochs = config.EPOCHS
batch_size = config.BATCH_SIZE
save_models_and_figures_folder = config.SAVE_MODELS_AND_FIGURES_FOLDER
keep_ratio = config.KEEP_RATIO

if __name__ == '__main__':
    if get_training_status(hybas_id) == 'Y':
        print(f'Hybas {hybas_id} has been trained.')
        sys.exit(0)
        
    ug.unet_train(
        training_pattern=training_pattern,
        eval_pattern=eval_pattern,
        local_sample_folder=local_sample_folder,
        input_bands=input_bands,
        input_band_scaling_factors=input_band_scaling_factors,
        response=response,
        kernel_size=kernel_size,
        epochs=epochs,
        batch_size=batch_size,
        save_models_and_figures_folder=save_models_and_figures_folder,
        keep_ratio=keep_ratio
    )

    update_training_record(model_name=hybas_id, trained_status='Y')