import argparse
import sys
import os
script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)
from my_unet_gdal import unet_prediction
from datetime import datetime
import concurrent.futures
from dateutil import relativedelta
import update_training_record
import tensorflow as tf
import glob
import shutil
import subprocess
from my_unet_definition.evaluation_metrics import IoU_coef, IoU_loss

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

def divide_list_into_sublists(lst, num_subs):
    """
    Divide a list into a specified number of sub-lists.

    Args:
        lst: The list to be divided.
        num_subs: The number of sub-lists to divide into.

    Returns:
        A list of sub-lists, where each sub-list is approximately of equal size.
    """
    # Calculate the size of each chunk
    chunk_size = len(lst) // num_subs
    remainder = len(lst) % num_subs

    # Create the sub-lists
    sublists = []
    start = 0
    for i in range(num_subs):
        # Adjust the end index to account for the remainder
        end = start + chunk_size + (1 if i < remainder else 0)
        # Slice the list and add to the sublists
        sublists.append(lst[start:end])
        start = end

    return sublists

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Using U-Net to predict lake water extent')
    parser.add_argument('--wkdir', type=str, default=None, help='Working directory')
    parser.add_argument('--basin_id', type=int, default=None, help='Basin ID')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the trained model')
    parser.add_argument('--start_date', type=str, default=None, help='Start date')
    parser.add_argument('--end_date', type=str, default=None, help='End date')
    parser.add_argument('--date_fmt', type=str, default='%Y-%m-%d', help='Date format')
    parser.add_argument('--unpredicted_tfrecord_folder', type=str, help='Folder containing unpredicted TFRecord files')
    parser.add_argument('--predicted_output_folder', type=str, help='Folder to save predicted TFRecord files')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity level')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_processes', type=int, default=8, help='Number of processes to use')
    parser.add_argument('--max_queue_size_per_process', type=int, default=4, help='Maximum queue size per process')
    parser.add_argument('--add_finish_tag', action='store_true', help='Add a finish tag to the output folder')
    parser.add_argument('--async_save', action='store_true', help='Save the output asynchronously')
    parser.add_argument('--multi_tf_sessions', action='store_true', help='Use multiple TensorFlow sessions')
    parser.add_argument('--num_tf_sessions', type=int, default=1, help='Number of TensorFlow sessions')
    
    args = parser.parse_args()
    wkdir = args.wkdir
    basin_id = args.basin_id
    model_path = args.model_path
    start_date = args.start_date
    end_date = args.end_date
    date_fmt = args.date_fmt
    unpredicted_tfrecord_folder = args.unpredicted_tfrecord_folder
    predicted_output_folder = args.predicted_output_folder
    verbose = args.verbose
    batch_size = args.batch_size
    num_processes = args.num_processes
    max_queue_size_per_process = args.max_queue_size_per_process
    add_finish_tag = args.add_finish_tag
    async_save = args.async_save
    multi_tf_sessions = args.multi_tf_sessions
    num_tf_sessions = args.num_tf_sessions
    
    # Set the working directory to the directory of the script
    if wkdir == None:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
    else:
        os.chdir(wkdir)

    custom_objects = {
        'IoU_loss': IoU_loss,
        'IoU_coef': IoU_coef
    }

    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    start_date = datetime.strptime(start_date, date_fmt)
    end_date = datetime.strptime(end_date, date_fmt)
    current_date = start_date
    
    while current_date < end_date:
        current_date_str = current_date.strftime(date_fmt)
        next_month_str = (current_date + relativedelta.relativedelta(months=1)).strftime(date_fmt)
        current_unpredicted_tfrecord_folder = f'{unpredicted_tfrecord_folder}/{current_date_str}_{next_month_str}'
        tfrecord_paths = glob.glob(f'{current_unpredicted_tfrecord_folder}/*.tfrecord*')
        tfrecord_basenames = [os.path.splitext(os.path.splitext(os.path.basename(path))[0])[0] for path in tfrecord_paths]
        print('Predicting for the following basenames:')
        print(tfrecord_basenames)
        output_folder = f'{predicted_output_folder}/{current_date_str}_{next_month_str}'
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder)
        predicted_tfrecord_paths = [os.path.join(output_folder, f'{basename}_predicted.tfrecord') for basename in tfrecord_basenames]
        if multi_tf_sessions:
            unpredicted_tfrecord_paths_sublists = divide_list_into_sublists(tfrecord_paths, num_tf_sessions)
            predicted_tfrecord_paths_sublists = divide_list_into_sublists(predicted_tfrecord_paths, num_tf_sessions)
            num_processes_per_session = num_processes // num_tf_sessions
            if num_processes_per_session == 0:
                num_processes_per_session = 1
            tf_session_futures = []
            sub_commands = [[
                'python', '-u', os.path.join(script_dir, 'batch_unet_prediction_paths_input.py'),
                '--wkdir', wkdir,
                '--model_path', model_path,
                '--unpredicted_tfrecord_paths', *unpredicted_tfrecord_paths_sublists[i],
                '--predicted_tfrecord_paths', *predicted_tfrecord_paths_sublists[i],
                '--verbose', str(verbose),
                '--batch_size', str(batch_size),
                '--num_processes', str(num_processes_per_session),
                '--max_queue_size_per_process', str(max_queue_size_per_process),
                '--async_save' if async_save else '',
            ] for i in range(num_tf_sessions)]
            for i in range(len(sub_commands)):
                sub_commands[i] = [arg for arg in sub_commands[i] if arg]
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for sub_command in sub_commands:
                    tf_session_futures.append(executor.submit(subprocess.run, sub_command, check=True))
                for future in tf_session_futures:
                    future.result()
        else:
            if async_save:
                futures = []
            for tfrecord_path, predicted_tfrecord_path in zip(tfrecord_paths, predicted_tfrecord_paths):
                if async_save:
                    future = unet_prediction.predict_and_save_to_tfrecord_async(
                        model=model,
                        tfrecord_path=tfrecord_path,
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
                        tfrecord_path=tfrecord_path,
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
        if add_finish_tag:
            finish_tag_path = os.path.join(output_folder, 'finished.txt')
            with open(finish_tag_path, 'w') as f:
                f.write('Finished')
        current_date = current_date + relativedelta.relativedelta(months=1)