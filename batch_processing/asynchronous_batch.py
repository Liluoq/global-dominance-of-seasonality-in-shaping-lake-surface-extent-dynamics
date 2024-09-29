import subprocess
import os
import concurrent.futures
from datetime import datetime
from dateutil.relativedelta import relativedelta
from load_config_module import load_config_module
from ..update_training_record import get_batch_processed_status, update_training_record
import sys
import signal
import argparse

class ManagedSubprocess:
    def __init__(self):
        self.processes = []

    def run(self, command, log_path):
        """Run a command as a subprocess and log its output. Allows for external termination."""
        with open(log_path, "w") as log_file:
            process = subprocess.Popen(command, stdout=log_file, stderr=log_file, start_new_session=True)
            self.processes.append(process)
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, command)

    def terminate_all(self):
        for process in self.processes:
            if process.poll() is None:  # Process is still running
                # Send SIGTERM to all processes in the process group
                os.killpg(process.pid, signal.SIGTERM)
        self.processes.clear()

if __name__ == '__main__':
    
    # Define a signal handler
    def signal_handler(sig, frame):
        print('You pressed Ctrl+C!')
        managed_subprocess.terminate_all()
        sys.exit(0)
    # Register the signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    parser = argparse.ArgumentParser(description='Asynchronous batch processing')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    
    args = parser.parse_args()
    config_path = args.config
    config = load_config_module(config_path)
    
    all_start_date = config.START_DATE
    all_end_date = config.END_DATE
    wait = config.WAIT
    
    code_dir = config.CODE_DIR
    os.chdir(code_dir)
    batch_full_path = config.BATCH_FULL_SCRIPT
    
    wkdir = config.WKDIR
    basin_id = config.BASIN_ID
    
    command_clear_finish_tag = [
        'python', '-u', batch_full_path,
        '--mode', 'clear_finish_tag',
        '--all_start_date', all_start_date,
        '--all_end_date', all_end_date,
        '--wkdir', wkdir,
        '--basin_id', basin_id,
        '--num_processes', config.NUM_PROCESSES_CLEAR_FINISH_TAG,
        '--config', config_path
    ]
    
    command_tfrecord_mode = [
        'python', '-u', batch_full_path,
        '--mode', 'generate_tfrecord',
        '--all_start_date', all_start_date,
        '--all_end_date', all_end_date,
        '--wkdir', wkdir,
        '--basin_id', basin_id,
        '--num_processes', config.NUM_PROCESSES_TFRECORD_GENERATION,
        '--config', config_path
    ]
    
    command_unet_prediction_mode = [
        'python', '-u', batch_full_path,
        '--mode', 'unet_prediction',
        '--all_start_date', all_start_date,
        '--all_end_date', all_end_date,
        '--wait', wait,
        '--wkdir', wkdir,
        '--basin_id', basin_id,
        '--num_processes', config.NUM_PROCESSES_UNET_PREDICTION,
        '--config', config_path
    ]
    
    command_predicted_reconstruction_mode = [
        'python', '-u', batch_full_path,
        '--mode', 'prediction_reconstruction',
        '--all_start_date', all_start_date,
        '--all_end_date', all_end_date,
        '--wait', wait,
        '--wkdir', wkdir,
        '--basin_id', basin_id,
        '--num_processes', config.NUM_PROCESSES_PREDICTION_RECONSTRUCTION,
        '--config', config_path
    ]
    
    command_mosaic_mode = [
        'python', '-u', batch_full_path,
        '--mode', 'mosaic',
        '--all_start_date', all_start_date,
        '--all_end_date', all_end_date,
        '--wait', wait,
        '--wkdir', wkdir,
        '--basin_id', basin_id,
        '--num_processes', config.NUM_PROCESSES_MOSAIC,
        '--config', config_path
    ]
    
    managed_subprocess = ManagedSubprocess()
    
    log_folder = os.path.join(wkdir, 'logs', basin_id)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    generate_tfrecord_log_path = os.path.join(log_folder, 'generate_tfrecord.log')
    unet_prediction_log_path = os.path.join(log_folder, 'unet_prediction.log')
    predicted_reconstruction_log_path = os.path.join(log_folder, 'predicted_reconstruction.log')
    mosaic_log_path = os.path.join(log_folder, 'mosaic.log')
    print('Clearing finish tag...')
    subprocess.run(command_clear_finish_tag, check=True)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        print('Submitting jobs, running...')
        futures = [
            executor.submit(managed_subprocess.run, command_tfrecord_mode, generate_tfrecord_log_path),
            executor.submit(managed_subprocess.run, command_unet_prediction_mode, unet_prediction_log_path),
            executor.submit(managed_subprocess.run, command_predicted_reconstruction_mode, predicted_reconstruction_log_path),
            executor.submit(managed_subprocess.run, command_mosaic_mode, mosaic_log_path)
        ]
        
        
        try:
            for future in concurrent.futures.as_completed(futures):
                future.result()  # This will re-raise exceptions from the subprocesses
        except Exception as e:
            print("An error occurred, terminating all tasks...")
            managed_subprocess.terminate_all()
            sys.exit(1)  # Exit due to the failure

    training_record_path = config.TRAINING_RECORD_PATH
    update_training_record(model_name=int(basin_id), batch_processed='Y', file_path=training_record_path)
    
    print("All tasks completed successfully.")