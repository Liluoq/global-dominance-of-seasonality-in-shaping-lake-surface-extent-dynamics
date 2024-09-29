import argparse
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)
from my_unet_gdal import generate_tfrecord_from_tile
from my_unet_gdal import reproject_to_target_tile
from datetime import datetime
from dateutil import relativedelta
from osgeo import gdal
import glob
import shutil

print("Hello, world!")
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate TFRecord files from Geotiff images')
    parser.add_argument('--wkdir', type=str, default=None, help='Working directory')
    parser.add_argument('--basin_id', type=int, default=None, help='Basin ID')
    parser.add_argument('--start_date', type=str, default=None, help='Start date')
    parser.add_argument('--end_date', type=str, default=None, help='End date')
    parser.add_argument('--gsw_occurrence_and_recurrence_path', type=str, default=None, help='Path to the GSW occurrence and recurrence raster')
    parser.add_argument('--boundary_path', type=str, default=None, help='Path to the boundary shapefile')
    parser.add_argument('--output_folder', type=str, default=None, help='Output folder')
    parser.add_argument('--date_fmt', type=str, default='%Y-%m-%d', help='Date format')
    parser.add_argument('--input_folder', type=str, default=None, help='Input folder')
    parser.add_argument('--input_name_pattern', type=str, default=None, help='Input name pattern with start and end date to be filled')
    parser.add_argument('--tile_size_x', type=int, default=512, help='Tile size in the x direction')
    parser.add_argument('--tile_size_y', type=int, default=512, help='Tile size in the y direction')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity level')
    parser.add_argument('--num_processes', type=int, default=8, help='Number of processes to use')
    parser.add_argument('--resample_alg', type=str, default='bilinear', help='Resampling algorithm')
    parser.add_argument('--add_finished_tag', action='store_true', help='Add a finished tag to the output folder')
    
    args = parser.parse_args()
    wkdir = args.wkdir
    basin_id = args.basin_id
    start_date = args.start_date
    end_date = args.end_date
    gsw_occurrence_and_recurrence_path = args.gsw_occurrence_and_recurrence_path
    boundary_path = args.boundary_path
    output_folder = args.output_folder
    date_fmt = args.date_fmt
    input_folder = args.input_folder
    input_name_pattern = args.input_name_pattern
    tile_size_x = args.tile_size_x
    tile_size_y = args.tile_size_y
    verbose = args.verbose
    num_processes = args.num_processes
    resample_alg = args.resample_alg
    if resample_alg == 'bilinear':
        resample_alg = gdal.GRA_Bilinear
    elif resample_alg == 'nearest':
        resample_alg = gdal.GRA_NearestNeighbour
    else:
        raise ValueError(f'Unknown resampling algorithm: {resample_alg}')
    add_finished_tag = args.add_finished_tag
    
    # Set the working directory to the directory of the script'
    if wkdir == None:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
    else:
        os.chdir(wkdir)
        
    target_srs_path = gsw_occurrence_and_recurrence_path
    tgt_srs_wkt, x_res, y_res = reproject_to_target_tile.get_target_srs_resolution(target_srs_path, return_wkt=True)

    start_date = datetime.strptime(start_date, date_fmt)
    end_date = datetime.strptime(end_date, date_fmt)
    current_date = start_date
        
    while current_date < end_date:
        current_date_str = current_date.strftime(date_fmt)
        next_month_str = (current_date + relativedelta.relativedelta(months=1)).strftime(date_fmt)
        print(f'Generating tfrecords from {current_date_str} to {next_month_str} for basin {basin_id}')
        # Check if the current working directory is different from the desired one
        if os.getcwd() != wkdir:
            raise Exception("Current working directory is different from the desired directory. Aborting.")
        #in case there are multiple images for the same date, resulted from the tiling of GEE
        current_input_name_pattern = input_name_pattern.format(current_date_str=current_date_str, next_month_str=next_month_str)
        input_tifs = glob.glob(f'{input_folder}/{current_input_name_pattern}')
        print(input_tifs)
        current_output_folder = os.path.join(output_folder, f'{current_date_str}_{next_month_str}')
        if os.path.exists(current_output_folder):
            print(f'Output folder {current_output_folder} already exists. Deleting and re-creating it.')
            shutil.rmtree(current_output_folder)
        os.makedirs(current_output_folder)
        
        for input_tif in input_tifs:
            
            temporary_tile_folder = f'./temporary_files/{basin_id}/{current_date_str}_{next_month_str}'
            if os.path.exists(temporary_tile_folder):
                print(f'Temporary folder {temporary_tile_folder} already exists. Deleting and re-creating it.')
                shutil.rmtree(temporary_tile_folder)
            os.makedirs(temporary_tile_folder)
            
            generate_tfrecord_from_tile.reproject_and_convert_to_tfrecord_by_tile(
                input_tif_path=input_tif,
                temporary_tile_folder=temporary_tile_folder,
                tile_size_x=tile_size_x,
                tile_size_y=tile_size_y,
                tile_width_buffer=16,
                tile_height_buffer=16,
                output_folder=current_output_folder,
                band_names=['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2', 'GSW_Occurrence', 'GSW_Recurrence'],
                tgt_srs_wkt=tgt_srs_wkt,
                x_res=x_res,
                y_res=y_res,
                resample_alg=resample_alg,
                output_dtype='int16',                                                                                                                                                                                                              
                to_combine_raster=gsw_occurrence_and_recurrence_path,
                window_size=128,
                window_overlap=0,
                verbose=verbose,
                num_processes=num_processes,
                remove_files=True,
                compression_type='GZIP',
                boundary_path=boundary_path
            )
        
        if add_finished_tag:
            with open(os.path.join(current_output_folder, 'finished.txt'), 'w') as f:
                f.write('Finished')
                
        current_date = current_date + relativedelta.relativedelta(months=1)
    