import argparse
import os
import re
import pandas as pd
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)
from my_unet_gdal import area_calculation
from datetime import datetime
from dateutil import relativedelta

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Calculate area of predicted TIF files')
    parser.add_argument('--wkdir', type=str, default=None, help='Working directory')
    parser.add_argument('--basin_id', type=int, default=None, help='Basin ID')
    parser.add_argument('--start_date', type=str, default=None, help='Start date')
    parser.add_argument('--end_date', type=str, default=None, help='End date')
    parser.add_argument('--date_fmt', type=str, default='%Y-%m-%d', help='Date format')
    parser.add_argument('--input_tif_folder', type=str, default=None, help='Folder containing water classification TIF files for input')
    parser.add_argument('--input_tif_file_name_pattern', type=str, default=None, help='Pattern to format the input TIF files')
    parser.add_argument('--output_csv_folder', type=str, default=None, help='Folder to save the output CSV files')
    parser.add_argument('--output_csv_file_name_pattern', type=str, default=None, help='Pattern to format the output CSV files')
    parser.add_argument('--output_concatenated_csv_path', type=str, default=None, help='Path to save the concatenated CSV file')
    parser.add_argument('--input_lake_shp_folder', type=str, default=None, help='Folder containing lake shapefiles for input')
    parser.add_argument('--input_lake_shp_file_name', type=str, default=None, help='Lake shapefile for input')
    parser.add_argument('--raster_clip_temporary_folder', type=str, default=None, help='Temporary folder to save clipped rasters')
    parser.add_argument('--temporary_vector_folder', type=str, default=None, help='Temporary folder to save vector files')
    parser.add_argument('--force_raster_binary_eq_value', type=int, default=None, help='Force the raster to have binary values equal to this value')
    parser.add_argument('--lake_id_field', type=str, default=None, help='Field to rasterize the lake shapefile to calculate areas for each lake separately')
    parser.add_argument('--grid_size', type=int, default=None, help='Grid size for rasterization in the unit of the raster\'s srs')
    parser.add_argument('--method', type=str, help='Method to use for rasterization')
    parser.add_argument('--outside_value', type=int, default=-1, help='Values to be treated as outside of the lake boundary vector')
    parser.add_argument('--num_processes', type=int, default=None, help='Number of processes to use for parallel processing')
    parser.add_argument('--check_overlap', action='store_true', help='Check if the raster and vector overlap')
    parser.add_argument('--filter_once', action='store_true', help='Filter the lake vector once for all the rasters')
    parser.add_argument('--verbose', type=int, default=0, help='Verbosity level')
    parser.add_argument('--save_lake_rasterization', action='store_true', help='Save the rasterized lake shapefile')
    
    args = parser.parse_args()
    wkdir = args.wkdir
    basin_id = args.basin_id
    start_date = args.start_date
    end_date = args.end_date
    date_fmt = args.date_fmt
    input_tif_folder = args.input_tif_folder
    input_tif_file_name_pattern = args.input_tif_file_name_pattern
    output_csv_folder = args.output_csv_folder
    output_csv_file_name_pattern = args.output_csv_file_name_pattern
    output_concatenated_csv_path = args.output_concatenated_csv_path
    input_lake_shp_folder = args.input_lake_shp_folder
    input_lake_shp_file_name = args.input_lake_shp_file_name
    raster_clip_temporary_folder = args.raster_clip_temporary_folder
    temporary_vector_folder = args.temporary_vector_folder
    force_raster_binary_eq_value = args.force_raster_binary_eq_value
    lake_id_field = args.lake_id_field
    grid_size = args.grid_size
    method = args.method
    outside_value = args.outside_value
    num_processes = args.num_processes
    check_overlap = args.check_overlap
    filter_once = args.filter_once
    verbose = args.verbose
    save_lake_rasterization = args.save_lake_rasterization
    
    # Set the working directory to the directory of the script
    if wkdir == None:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
    else:
        os.chdir(wkdir)
    
    input_lake_shp_path = os.path.join(input_lake_shp_folder, input_lake_shp_file_name)
    
    start_date = datetime.strptime(start_date, date_fmt)
    end_date = datetime.strptime(end_date, date_fmt)
    input_water_classification_raster_paths = []
    raster_clip_temporary_folders = []
    output_csv_paths = []
    current_date = start_date
    while current_date < end_date:
        current_date_str = current_date.strftime(date_fmt)
        next_month_str = (current_date + relativedelta.relativedelta(months=1)).strftime(date_fmt)
        current_input_tif_name = input_tif_file_name_pattern.format(basin_id=basin_id, start_date=current_date_str, end_date=next_month_str)
        current_input_tif_path = os.path.join(input_tif_folder, current_input_tif_name)
        current_output_csv_name = output_csv_file_name_pattern.format(basin_id=basin_id, start_date=current_date_str, end_date=next_month_str)
        if not os.path.exists(output_csv_folder):
            os.makedirs(output_csv_folder)
        current_output_csv_path = os.path.join(output_csv_folder, current_output_csv_name)
        # if need reproject lake shapefile, add '_reprojected' to the lake shp name
        lake_shp_name_if_need_reprojection = input_lake_shp_file_name.replace('.shp', '_reprojected.shp')
        lake_shp_path_if_need_reprojection = os.path.join(input_lake_shp_folder, lake_shp_name_if_need_reprojection)
        raster_clip_temporary_folders.append(os.path.join(raster_clip_temporary_folder, f'{current_date_str}_{next_month_str}'))
        output_csv_paths.append(current_output_csv_path)
        reprojected = area_calculation.check_and_reproject_vector_srs_to_raster(
            raster_path=current_input_tif_path,
            vector_path=input_lake_shp_path,
            output_path=lake_shp_path_if_need_reprojection,
            verbose=verbose
        )
        if reprojected:
            input_lake_shp_path = lake_shp_path_if_need_reprojection
        
        input_water_classification_raster_paths.append(current_input_tif_path)
        current_date = current_date + relativedelta.relativedelta(months=1)
    
    if verbose != 0:
        print(f'Input water classification raster paths: {input_water_classification_raster_paths}')
    
    area_calculation.calculate_lake_area_grid_parallel(
        water_classification_raster_ds_paths=input_water_classification_raster_paths,
        lake_shape_path=input_lake_shp_path,
        raster_clip_temporary_folders=raster_clip_temporary_folders,
        temporary_vector_folder=temporary_vector_folder,
        output_csv_paths=output_csv_paths,
        lake_id_field=lake_id_field,
        grid_size=grid_size,
        method=method,
        outside_value=outside_value,
        force_raster_binary_eq_value=force_raster_binary_eq_value,
        num_processes=num_processes,
        check_overlap=check_overlap,
        filter_once=filter_once,
        verbose=verbose,
        save_lake_rasterization=save_lake_rasterization
    )
    
    all_data = pd.DataFrame()
    # Loop through each file in the directory
    for filepath in output_csv_paths:
        filename = os.path.basename(filepath)
        if filename.endswith('_area.csv'):
            # Extract the basin_id and start_date using regex
            match = re.match(r'(\d+)_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})_.*_area\.csv', filename)
            if match:
                basin_id, start_date, end_date = match.groups()
                
                # Read the CSV file
                df = pd.read_csv(filepath)
                
                # Add a column for the start_date with the area as the value
                df['Start_Date'] = start_date  # Add the start_date as a new column for pivoting
                
                # Append this data to the all_data DataFrame
                all_data = pd.concat([all_data, df], ignore_index=True)
    # Pivot the DataFrame to get the desired format
    pivot_df = all_data.pivot_table(index='Hylak_id', columns='Start_Date', values='area', aggfunc='sum')

    # Ensure Lake_ID remains as a regular column
    pivot_df.reset_index(inplace=True)  # This now keeps Lake_ID as a column after resetting the index

    output_concatenated_csv_folder = os.path.dirname(output_concatenated_csv_path)
    if not os.path.exists(output_concatenated_csv_folder):
        os.makedirs(output_concatenated_csv_folder)
    # Save the final DataFrame to a new CSV file
    pivot_df.to_csv(output_concatenated_csv_path, index=False)