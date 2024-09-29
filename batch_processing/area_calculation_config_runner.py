from load_config_module import load_config_module
import subprocess
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the area calculation script with the given configuration')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    
    args = parser.parse_args()
    config_path = args.config
    config = load_config_module(config_path)
    
    wkdir = config.WKDIR
    basin_id = config.BASIN_ID
    start_date = config.START_DATE
    end_date = config.END_DATE
    date_fmt = config.DATE_FMT
    input_tif_folder = config.INPUT_TIF_FOLDER
    input_tif_file_name_pattern = config.INPUT_TIF_FILE_NAME_PATTERN
    output_csv_folder = config.OUTPUT_CSV_FOLDER
    output_csv_file_name_pattern = config.OUTPUT_CSV_FILE_NAME_PATTERN
    output_concatenated_csv_path = config.OUTPUT_CONCATENATED_CSV_PATH
    input_lake_shp_folder = config.INPUT_LAKE_SHP_FOLDER
    input_lake_shp_file_name = config.INPUT_LAKE_SHP_FILE_NAME
    raster_clip_temporary_folder = config.RASTER_CLIP_TEMPORARY_FOLDER
    temporary_vector_folder = config.TEMPORARY_VECTOR_FOLDER
    force_raster_binary_eq_value = config.FORCE_RASTER_BINARY_EQ_VALUE
    lake_id_field = config.LAKE_ID_FIELD
    grid_size = config.GRID_SIZE
    method = config.METHOD
    outside_value = config.OUTSIDE_VALUE
    num_processes = config.NUM_PROCESSES
    check_overlap = config.CHECK_OVERLAP
    filter_once = config.FILTER_ONCE
    verbose = config.VERBOSE
    save_lake_rasterization = config.SAVE_LAKE_RASTERIZATION
    
    batch_area_script_path = os.path.join(config.CODE_DIR, config.BATCH_AREA_SCRIPT)
    
    area_calculation_command = [
        'python', '-u', batch_area_script_path,
        '--wkdir', wkdir,
        '--basin_id', basin_id,
        '--start_date', start_date,
        '--end_date', end_date,
        '--date_fmt', date_fmt,
        '--input_tif_folder', input_tif_folder,
        '--input_tif_file_name_pattern', input_tif_file_name_pattern,
        '--output_csv_folder', output_csv_folder,
        '--output_csv_file_name_pattern', output_csv_file_name_pattern,
        '--output_concatenated_csv_path', output_concatenated_csv_path,
        '--input_lake_shp_folder', input_lake_shp_folder,
        '--input_lake_shp_file_name', input_lake_shp_file_name,
        '--raster_clip_temporary_folder', raster_clip_temporary_folder,
        '--temporary_vector_folder', temporary_vector_folder,
        '--lake_id_field', lake_id_field,
        '--grid_size', str(grid_size),
        '--method', method,
        '--outside_value', str(outside_value),
        '--num_processes', str(num_processes),
        '--check_overlap' if check_overlap else '',
        '--filter_once' if filter_once else '',
        '--verbose', str(verbose),
        '--save_lake_rasterization' if save_lake_rasterization else ''
    ]
    
    if force_raster_binary_eq_value is not None:
        area_calculation_command.extend(['--force_raster_binary_eq_value', str(force_raster_binary_eq_value)])
    
    area_calculation_command = [arg for arg in area_calculation_command if arg]
    
    subprocess.run(area_calculation_command)