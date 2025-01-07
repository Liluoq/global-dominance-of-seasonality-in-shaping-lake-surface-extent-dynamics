import argparse
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)
from osgeo import gdal
from my_unet_gdal.reproject_to_target_tile import mosaic_tiles
from datetime import datetime
from dateutil import relativedelta

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mosaic tiles')
    parser.add_argument('--wkdir', type=str, default=None, help='Working directory')
    parser.add_argument('--basin_id', type=int, default=None, help='Basin ID')
    parser.add_argument('--start_date', type=str, default=None, help='Start date')
    parser.add_argument('--end_date', type=str, default=None, help='End date')
    parser.add_argument('--date_fmt', type=str, default='%Y-%m-%d', help='Date format')
    parser.add_argument('--input_folder', type=str, default=None, help='Input folder')
    parser.add_argument('--input_file_name_base', type=str, default=None, help='Input file name base')
    parser.add_argument('--output_folder', type=str, default=None, help='Output folder')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity level')
    
    args = parser.parse_args()
    wkdir = args.wkdir
    basin_id = args.basin_id
    start_date = args.start_date
    end_date = args.end_date
    date_fmt = args.date_fmt
    input_folder = args.input_folder
    input_file_name_base = args.input_file_name_base
    output_folder = args.output_folder
    verbose = args.verbose

    start_date = datetime.strptime(start_date, date_fmt)
    end_date = datetime.strptime(end_date, date_fmt)
    current_date = start_date
    # Set the working directory to the directory of the script
    if wkdir == None:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
    else:
        os.chdir(wkdir)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    while current_date < end_date:
        current_date_str = current_date.strftime(date_fmt)
        next_month_str = (current_date + relativedelta.relativedelta(months=1)).strftime(date_fmt)
        current_input_folder = os.path.join(input_folder, f'{current_date_str}_{next_month_str}')
        current_output_tif = os.path.join(output_folder, f'{basin_id}_{current_date_str}_{next_month_str}_water_mosaic.tif')
        
        mosaic_tiles(
            input_folder=current_input_folder,
            output_tif=current_output_tif,
            input_tile_file_basename=input_file_name_base,
            verbose=verbose,
            output_dtype=gdal.GDT_Int16,
            remove_tile=False
        )
        
        current_date += relativedelta.relativedelta(months=1)
        

    
