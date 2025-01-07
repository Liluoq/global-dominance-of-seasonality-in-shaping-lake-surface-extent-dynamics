import subprocess
import os
import argparse
from datetime import datetime
from dateutil import relativedelta
from load_config_module import load_config_module
import time
from concurrent.futures import ProcessPoolExecutor, as_completed, FIRST_COMPLETED

if __name__ == "__main__":
    
    
    
    parser = argparse.ArgumentParser(description='Batch processing for global lake area prediction')
    parser.add_argument('--mode', type=str, default='full', help='Mode of the batch processing')
    parser.add_argument('--wait', type=int, default=30, help='Wait time in seconds')
    parser.add_argument('--all_start_date', type=str, default='2001-01-01', help='Start date of the batch processing')
    parser.add_argument('--all_end_date', type=str, default='2024-01-01', help='End date of the batch processing')
    parser.add_argument('--wkdir', type=str, default='/WORK/Data/global_lake_area', help='Working directory')
    parser.add_argument('--basin_id', type=int, default=8020000010, help='Basin ID')
    parser.add_argument('--num_processes', type=int, default=8, help='Number of processes to use')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    
    args = parser.parse_args()
    mode = args.mode
    wait = args.wait
    all_start_date = args.all_start_date
    all_end_date = args.all_end_date
    wkdir = args.wkdir
    basin_id = str(args.basin_id)
    num_processes = str(args.num_processes)
    config_path = args.config
    config = load_config_module(config_path)
    
    if mode not in ['full', 'generate_tfrecord', 'unet_prediction', 'prediction_reconstruction', 'mosaic', 'clear_finish_tag']:
        raise ValueError(f'Unknown mode: {mode}')
    
    all_start_date = datetime.strptime(all_start_date, config.DATE_FORMAT)
    all_end_date = datetime.strptime(all_end_date, config.DATE_FORMAT)
    
    code_dir = config.CODE_DIR
    os.chdir(code_dir)
    batch_tfrecord_generate_path = config.BATCH_TFRECORD_GENERATION_SCRIPT
    batch_unet_prediction_path = config.BATCH_UNET_PREDICTION_SCRIPT
    batch_prediction_reconstruction_path = config.BATCH_PREDICTION_RECONSTRUCTION_SCRIPT
    batch_mosaic_path = config.BATCH_MOSAIC_SCRIPT
    
    gsw_occurrence_and_recurrence_path = config.GSW_OCCURRENCE_AND_RECURRENCE_PATH
    boundary_path = config.BOUNDARY_SHP_PATH
    unpredicted_tfrecord_output_folder =  config.UNPREDICTED_TFRECORD_OUTPUT_FOLDER
    input_modis_500m_folder = config.INPUT_MODIS_500M_FOLDER
    input_modis_500m_name_pattern = config.INPUT_MODIS_500M_NAME_PATTERN
    tile_size_x = config.TILE_SIZE_X
    tile_size_y = config.TILE_SIZE_Y
    verbose = config.VERBOSE
    num_processes = num_processes
    resample_alg = config.RESAMPLE_ALG
    
    model_selection = config.MODEL_SELECTION
    formatted_model_selection = config.FORMATTED_MODEL_SELECTION
    model_path = config.MODEL_PATH
    predicted_output_folder = config.PREDICTED_OUTPUT_FOLDER
    prediction_batch_size = config.PREDICTION_BATCH_SIZE
    prediction_max_queue_size_per_process = config.PREDICTION_MAX_QUEUE_SIZE_PER_PROCESS
    prediction_async_save = config.PREDICTION_ASYNC_SAVE
    prediction_multi_tf_sessions = config.PREDICTION_MULTI_TF_SESSIONS
    prediction_num_tf_sessions = config.PREDICTION_NUM_TF_SESSIONS
    
    reconstructed_output_folder = config.RECONSTUCTED_OUTPUT_FOLDER
    
    mosaic_output_folder = config.MOSAIC_OUTPUT_FOLDER
    
    if mode == 'clear_finish_tag':
        current_date = all_start_date
        while current_date < all_end_date:
            #clear the finished tag before starting the batch processing
            start_date = current_date.strftime(config.DATE_FORMAT)
            end_date = (current_date + relativedelta.relativedelta(months=1)).strftime(config.DATE_FORMAT)
            #clear the finished tag for unpredicted tfrecord folder
            current_unpredicted_tfrecord_folder = f'{unpredicted_tfrecord_output_folder}/{start_date}_{end_date}'
            if os.path.exists(os.path.join(wkdir, current_unpredicted_tfrecord_folder, 'finished.txt')):
                os.remove(os.path.join(wkdir, current_unpredicted_tfrecord_folder, 'finished.txt'))
            #clear the finished tag for predicted output folder
            current_predicted_tfrecord_folder = f'{predicted_output_folder}/{start_date}_{end_date}'
            if os.path.exists(os.path.join(wkdir, current_predicted_tfrecord_folder, 'finished.txt')):
                os.remove(os.path.join(wkdir, current_predicted_tfrecord_folder, 'finished.txt'))
            #clear the finished tag for reconstructed output folder
            current_reconstructed_output_folder = f'{reconstructed_output_folder}/{start_date}_{end_date}'
            if os.path.exists(os.path.join(wkdir, current_reconstructed_output_folder, 'finished.txt')):
                os.remove(os.path.join(wkdir, current_reconstructed_output_folder, 'finished.txt'))
            current_date += relativedelta.relativedelta(months=1)
    
    elif mode == 'mosaic':
        num_processes = int(num_processes)
        def mosaic_job(command_batch_mosaic, wkdir, reconstructed_output_folder, start_date, end_date, wait):
            current_reconstructed_output_folder = f'{reconstructed_output_folder}/{start_date}_{end_date}'
            finished_file_path = os.path.join(wkdir, current_reconstructed_output_folder, 'finished.txt')
            
            # Wait for the finished tag
            while not os.path.exists(finished_file_path):
                if wait > 0:
                    print(f'Waiting {wait} seconds for the reconstruction {current_reconstructed_output_folder} to finish.')
                    time.sleep(wait)
                else:
                    raise Exception(f'Reconstructed output folder {current_reconstructed_output_folder} does not contain the finished tag.')

            # Execute the mosaic command
            subprocess.run(command_batch_mosaic, check=True)
        
        current_date = all_start_date
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = []
            while current_date < all_end_date:
                start_date = current_date.strftime(config.DATE_FORMAT)
                end_date = (current_date + relativedelta.relativedelta(months=1)).strftime(config.DATE_FORMAT)
                
                command_batch_mosaic = [
                    'python', '-u', batch_mosaic_path,
                    '--wkdir', wkdir,
                    '--basin_id', basin_id,
                    '--start_date', start_date,
                    '--end_date', end_date,
                    '--date_fmt', config.DATE_FORMAT,
                    '--input_folder', reconstructed_output_folder,
                    '--input_file_name_base', 'reconstructed',
                    '--output_folder', mosaic_output_folder,
                    '--verbose', verbose
                ]
                
                # Submit job if there are available cores
                if len(futures) < num_processes:
                    future = executor.submit(mosaic_job, command_batch_mosaic, wkdir, reconstructed_output_folder,
                                            start_date, end_date, wait)
                    futures.append(future)
                    current_date += relativedelta.relativedelta(months=1)
                else:
                    # Properly handle completed futures
                    print('Waiting for a job to complete...')
                    completed_future = next(as_completed(futures))
                    futures.remove(completed_future)
            
            # Wait for all jobs to complete
            for future in as_completed(futures):
                future.result()  # Handle results or exceptions if needed
    
    else:
        current_date = all_start_date
        while current_date < all_end_date:
            start_date = current_date.strftime(config.DATE_FORMAT)
            end_date = (current_date + relativedelta.relativedelta(months=1)).strftime(config.DATE_FORMAT)
            
            command_batch_tfrecord_generate = [
                'python', '-u', batch_tfrecord_generate_path,
                '--wkdir', wkdir,
                '--basin_id', basin_id,
                '--start_date', start_date,
                '--end_date', end_date,
                '--gsw_occurrence_and_recurrence_path', gsw_occurrence_and_recurrence_path,
                '--boundary_path', boundary_path,
                '--output_folder', unpredicted_tfrecord_output_folder,
                '--date_fmt', config.DATE_FORMAT,
                '--input_folder', input_modis_500m_folder,
                '--input_name_pattern', input_modis_500m_name_pattern,
                '--tile_size_x', tile_size_x,
                '--tile_size_y', tile_size_y,
                '--verbose', verbose,
                '--num_processes', num_processes,
                '--add_finished_tag',
                '--resample_alg', resample_alg
            ]
            
            command_batch_unet_prediction = [
                'python', '-u', batch_unet_prediction_path,
                '--wkdir', wkdir,
                '--basin_id', basin_id,
                '--model_path', model_path,
                '--start_date', start_date,
                '--end_date', end_date,
                '--date_fmt', config.DATE_FORMAT,
                '--unpredicted_tfrecord_folder', unpredicted_tfrecord_output_folder,
                '--predicted_output_folder', predicted_output_folder,
                '--verbose', verbose,
                '--batch_size', prediction_batch_size,
                '--num_processes', num_processes,
                '--add_finish_tag',
                '--max_queue_size_per_process', prediction_max_queue_size_per_process,
                '--async_save' if prediction_async_save else '',
                '--multi_tf_sessions' if prediction_multi_tf_sessions else '',
                '--num_tf_sessions', prediction_num_tf_sessions
            ]
            command_batch_unet_prediction = [arg for arg in command_batch_unet_prediction if arg]
            
            command_batch_prediction_reconstruction = [
                'python', '-u', batch_prediction_reconstruction_path,
                '--wkdir', wkdir,
                '--basin_id', basin_id,
                '--start_date', start_date,
                '--end_date', end_date,
                '--date_fmt', config.DATE_FORMAT,
                '--reconstructed_output_folder', reconstructed_output_folder,
                '--window_position_folder', unpredicted_tfrecord_output_folder,
                '--predicted_tfrecord_folder', predicted_output_folder,
                '--verbose', verbose,
                '--num_processes', num_processes,
                '--clear_input',
                '--add_finish_tag'
            ]
            
            
            if mode == 'full':
                subprocess.run(command_batch_tfrecord_generate, check=True)
                
                subprocess.run(command_batch_unet_prediction, check=True)
                
                subprocess.run(command_batch_prediction_reconstruction, check=True)
            
            elif mode == 'generate_tfrecord':
                subprocess.run(command_batch_tfrecord_generate, check=True)
            
            elif mode == 'unet_prediction':
                current_unpredicted_tfrecord_folder = f'{unpredicted_tfrecord_output_folder}/{start_date}_{end_date}'
                while not os.path.exists(os.path.join(wkdir, current_unpredicted_tfrecord_folder, 'finished.txt')):
                    if wait > 0:
                        print(f'Waiting {wait} seconds for the TFRecord generation {current_unpredicted_tfrecord_folder} to finish.')
                        time.sleep(wait)
                    else:
                        raise Exception(f'Unpredicted TFRecord folder {current_unpredicted_tfrecord_folder} does not contain the finished tag.')
                subprocess.run(command_batch_unet_prediction, check=True)
            
            elif mode == 'prediction_reconstruction':
                current_predicted_tfrecord_folder = f'{predicted_output_folder}/{start_date}_{end_date}'
                while not os.path.exists(os.path.join(wkdir, current_predicted_tfrecord_folder, 'finished.txt')):
                    if wait > 0:
                        print(f'Waiting {wait} seconds for the TFRecord generation {current_predicted_tfrecord_folder} to finish.')
                        time.sleep(wait)
                    else:
                        raise Exception(f'Predicted TFRecord folder {current_predicted_tfrecord_folder} does not contain the finished tag.')
                subprocess.run(command_batch_prediction_reconstruction, check=True)
            
            current_date += relativedelta.relativedelta(months=1)