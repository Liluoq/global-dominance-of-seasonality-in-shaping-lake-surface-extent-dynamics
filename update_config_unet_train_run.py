import csv
import re
import os
import subprocess
from update_training_record import get_keep_ratio

def update_config_and_run(hybas_id, keep_ratio, config_folder, config_filename, train_script_folder, train_script_filename):
    """
    Update the UNET_TRAIN_CONFIG.py file with the given HYBAS_ID and KEEP_RATIO, then run the main script.
    """
    config_path = os.path.join(config_folder, config_filename)
    train_script_path = os.path.join(train_script_folder, train_script_filename)
    
    with open(config_path, 'r') as file:
        config_content = file.read()

    # Replace HYBAS_ID and KEEP_RATIO values
    config_content = re.sub(r'HYBAS_ID = \d+', f'HYBAS_ID = {hybas_id}', config_content)
    config_content = re.sub(r'KEEP_RATIO = \d+(\.\d+)?', f'KEEP_RATIO = {keep_ratio}', config_content)

    # Write the updated content back to CONFIG.py
    with open(config_path, 'w') as file:
        file.write(config_content)

    # Run your main script here. Replace 'main_script.py' with the actual name of your script
    subprocess.run(['python', '-u', train_script_path, '--config', config_path])

if __name__ == '__main__':
    # Path to your CSV file
    training_records_folder = '/WORK/Codes/global_lake_area'
    training_records_filename = 'training_records.csv'
    training_records_path = os.path.join(training_records_folder, training_records_filename)

    config_folder = '/WORK/Codes/global_lake_area'
    config_filename = 'UNET_TRAIN_CONFIG.py'
    train_script_folder = '/WORK/Codes/global_lake_area'
    train_script_filename = 'unet_train.py'
    
    hybas_id_list = [2020057170]
    
    for basin_id in hybas_id_list:
        keep_ratio = get_keep_ratio(basin_id, training_records_path)
        update_config_and_run(basin_id, keep_ratio, config_folder, config_filename, train_script_folder, train_script_filename)