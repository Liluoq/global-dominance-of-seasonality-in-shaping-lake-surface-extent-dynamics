import argparse
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)
from my_unet_gdal import reconstruct_tile_from_prediction
import glob
from datetime import datetime
from dateutil import relativedelta
import tensorflow as tf
import shutil
import re

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Reconstruct TIF files from predicted TFRecord files')
    parser.add_argument('--wkdir', type=str, default=None, help='Working directory')
    parser.add_argument('--basin_id', type=int, default=None, help='Basin ID')
    parser.add_argument('--start_date', type=str, default=None, help='Start date')
    parser.add_argument('--end_date', type=str, default=None, help='End date')
    parser.add_argument('--date_fmt', type=str, default='%Y-%m-%d', help='Date format')
    parser.add_argument('--reconstructed_output_folder', type=str, default=None, help='Folder to save reconstructed TIF files')
    parser.add_argument('--window_position_folder', type=str, default=None, help='Folder containing window positions')
    parser.add_argument('--predicted_tfrecord_folder', type=str, default=None, help='Folder containing predicted TFRecord files')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity level')
    parser.add_argument('--num_processes', type=int, default=8, help='Number of processes to use')
    parser.add_argument('--clear_input', action='store_true', help='Clear the input folder')
    parser.add_argument('--add_finish_tag', action='store_true', help='Add a finish tag to the output folder')
    
    args = parser.parse_args()
    wkdir = args.wkdir
    basin_id = args.basin_id
    start_date = args.start_date
    end_date = args.end_date
    date_fmt = args.date_fmt
    reconstructed_output_folder = args.reconstructed_output_folder
    window_position_folder = args.window_position_folder
    predicted_tfrecord_folder = args.predicted_tfrecord_folder
    verbose = args.verbose
    num_processes = args.num_processes
    clear_input = args.clear_input
    add_finish_tag = args.add_finish_tag
    
    # Set the working directory to the directory of the script
    if wkdir == None:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
    else:
        os.chdir(wkdir)
        
    start_date = datetime.strptime(start_date, date_fmt)
    end_date = datetime.strptime(end_date, date_fmt)
    current_date = start_date
    
    while current_date < end_date:
        current_date_str = current_date.strftime(date_fmt)
        next_month_str = (current_date + relativedelta.relativedelta(months=1)).strftime(date_fmt)
        current_output_folder = f'{reconstructed_output_folder}/{current_date_str}_{next_month_str}'
        if os.path.exists(current_output_folder):
            print(f'{current_output_folder} already exists. Removing it.')
            shutil.rmtree(current_output_folder)
        os.makedirs(current_output_folder)
        current_window_position_folder = f'{window_position_folder}/{current_date_str}_{next_month_str}'
        predicted_tfrecord_paths = glob.glob(f'{predicted_tfrecord_folder}/{current_date_str}_{next_month_str}/*_predicted.tfrecord*')
        predicted_tfrecord_basename_noexts = [os.path.basename(predicted_tfrecord_path).split('.')[0] for predicted_tfrecord_path in predicted_tfrecord_paths]
        output_tif_paths = [os.path.join(current_output_folder, f'{predicted_tfrecord_basename_noext}_reconstructed.tif') for predicted_tfrecord_basename_noext in predicted_tfrecord_basename_noexts]
        window_position_paths = [os.path.join(current_window_position_folder, f'{re.sub("_predicted", "", predicted_tfrecord_basename_noext)}_window_positions.json') for predicted_tfrecord_basename_noext in predicted_tfrecord_basename_noexts]
        proj_metadata_paths = [os.path.join(current_window_position_folder, f'{re.sub("_predicted", "", predicted_tfrecord_basename_noext)}_proj_metadata.json') for predicted_tfrecord_basename_noext in predicted_tfrecord_basename_noexts]
        reconstruct_tile_from_prediction.reconstruct_tif_from_predicted_tfrecord_parallel(
            predicted_tfrecord_paths=predicted_tfrecord_paths,
            output_tif_paths=output_tif_paths,
            predicted_window_size=128,
            predicted_dtype=tf.int8,
            predicted_band_name='Predicted_water',
            window_position_paths=window_position_paths,
            tgt_projection_metadata_paths=proj_metadata_paths,
            verbose=verbose,
            compression_type='GZIP',
            num_processes=num_processes
        )
        if clear_input:
            if os.path.exists(current_window_position_folder):
                if verbose != 0:
                    print(f'Removing {current_window_position_folder}')
                shutil.rmtree(current_window_position_folder)
            if os.path.exists(f'{predicted_tfrecord_folder}/{current_date_str}_{next_month_str}'):
                if verbose != 0:
                    print(f'Removing {predicted_tfrecord_folder}/{current_date_str}_{next_month_str}')
                shutil.rmtree(f'{predicted_tfrecord_folder}/{current_date_str}_{next_month_str}')
        if add_finish_tag:
            with open(os.path.join(current_output_folder, 'finished.txt'), 'w') as f:
                f.write('Finished')
        current_date += relativedelta.relativedelta(months=1)