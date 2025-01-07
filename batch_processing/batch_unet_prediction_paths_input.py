import argparse
import sys
import os
script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)
from my_unet_gdal import unet_prediction
from datetime import datetime
from dateutil import relativedelta
import update_training_record
import tensorflow as tf
import glob
import shutil
from my_unet_definition.evaluation_metrics import IoU_coef, IoU_loss

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Using U-Net to predict lake water extent')
    parser.add_argument('--wkdir', type=str, default=None, help='Working directory')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the trained model')
    parser.add_argument('--unpredicted_tfrecord_paths', nargs='+', help='List of paths to unpredicted TFRecord files')
    parser.add_argument('--predicted_tfrecord_paths', nargs='+', help='List of paths to save predicted TFRecord files')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity level')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_processes', type=int, default=1, help='Number of processes to use')
    parser.add_argument('--max_queue_size_per_process', type=int, default=4, help='Maximum queue size per process')
    parser.add_argument('--async_save', action='store_true', help='Save the output asynchronously')
    
    args = parser.parse_args()
    wkdir = args.wkdir
    model_path = args.model_path
    unpredicted_tfrecord_paths = args.unpredicted_tfrecord_paths
    predicted_tfrecord_paths = args.predicted_tfrecord_paths
    verbose = args.verbose
    batch_size = args.batch_size
    num_processes = args.num_processes
    max_queue_size_per_process = args.max_queue_size_per_process
    async_save = args.async_save
    print('*******************************************************')
    print(unpredicted_tfrecord_paths)
    if wkdir == None:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
    else:
        os.chdir(wkdir)
    
    custom_objects = {
        'IoU_loss': IoU_loss,
        'IoU_coef': IoU_coef
    }

    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    
    
    if async_save:
        futures = []
    for unpredicted_tfrecord_path, predicted_tfrecord_path in zip(unpredicted_tfrecord_paths, predicted_tfrecord_paths):
        if async_save:
            future = unet_prediction.predict_and_save_to_tfrecord_async(
                model=model,
                tfrecord_path=unpredicted_tfrecord_path,
                input_band_names=['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2', 'GSW_Occurrence', 'GSW_Recurrence'],
                input_band_scaling_factors=[0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.01, 0.01],
                input_dtype=tf.int16,
                batch_size=batch_size,
                output_band_name='Predicted_water',
                output_file=predicted_tfrecord_path,
                output_dtype=tf.int8,
                verbose=verbose,
                compression_type='GZIP',
                num_processes=num_processes,
                max_queue_size_per_process=max_queue_size_per_process
            )
            futures.append(future)
        else:
            unet_prediction.predict_and_save_to_tfrecord(
                model=model,
                tfrecord_path=unpredicted_tfrecord_path,
                input_band_names=['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2', 'GSW_Occurrence', 'GSW_Recurrence'],
                input_band_scaling_factors=[0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.01, 0.01],
                input_dtype=tf.int16,
                batch_size=batch_size,
                output_band_name='Predicted_water',
                output_file=predicted_tfrecord_path,
                output_dtype=tf.int8,
                verbose=verbose,
                compression_type='GZIP',
                num_processes=num_processes,
                max_queue_size_per_process=max_queue_size_per_process
            )
    if async_save:
        for future in futures:
            future.result()