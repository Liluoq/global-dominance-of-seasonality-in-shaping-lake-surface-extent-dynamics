import sys
sys.path.append('../')
from my_unet_gdal.reproject_to_target_tile import mosaic_tiles
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os
from osgeo import gdal
from concurrent.futures import ProcessPoolExecutor

def process_date_range(basin_id, start_date, input_gsw_folder_pattern, gsw_image_file_name_pattern, output_file_path_pattern):
    current_input_gsw_folder = input_gsw_folder_pattern.format(basin_id=basin_id)
    current_date_str = start_date.strftime('%Y-%m-%d')
    next_month_str = (start_date + relativedelta(months=1)).strftime('%Y-%m-%d')
    current_gsw_image_file_name_pattern = gsw_image_file_name_pattern.format(basin_id=basin_id, start_date=current_date_str, end_date=next_month_str)
    current_output_file_path = output_file_path_pattern.format(basin_id=basin_id, start_date=current_date_str, end_date=next_month_str)
    current_output_folder = os.path.dirname(current_output_file_path)
    if not os.path.exists(current_output_folder):
        os.makedirs(current_output_folder, exist_ok=True)
    mosaic_tiles(
        input_folder=current_input_gsw_folder,
        output_tif=current_output_file_path,
        input_tile_file_basename=current_gsw_image_file_name_pattern,
        remove_tile=False,
        verbose=1,
        output_dtype=gdal.GDT_Int16,
        temporary_vrt_prefix=current_date_str
    )

if __name__ == '__main__':
    processed_basin_id_list = [
        1020000010, 8020000010, 4020000010, 5020000010, 2020000010, 9020000010, 
        7020000010, 6020000010, 3020000010, 1020011530, 8020008900, 4020006940, 
        5020015660, 2020003440, 7020014250, 6020006540, 3020003790, 1020018110, 
        8020010700, 4020015090, 5020037270, 2020018240, 7020021430, 6020008320, 
        3020005240, 1020021940, 8020020760, 4020024190, 5020049720, 2020024230, 
        7020024600, 6020014330, 3020008670, 1020027430, 8020022890, 4020034510, 
        2020033490, 7020038340, 6020017370, 1020034170, 8020032840, # lack 3020009320
        2020041390, 7020046750, 6020021870, 1020035180, 8020044560, 5020082270, 
        2020057170, 7020047840, 6020029280, 2020065840, 7020065090, 2020071190, 
        4020050210, 3020024310, 4020050220, 1020040190, 4020050290, 4020050470
    ]
    basin_id_list = [
        1020000010, 8020000010, 4020000010, 5020000010, 2020000010, 9020000010, 
        7020000010, 6020000010, 3020000010, 1020011530, 8020008900, 4020006940, 
        5020015660, 2020003440, 7020014250, 6020006540, 3020003790, 1020018110, 
        8020010700, 4020015090, 5020037270, 2020018240, 7020021430, 6020008320, 
        3020005240, 1020021940, 8020020760, 4020024190, 5020049720, 2020024230, 
        7020024600, 6020014330, 3020008670, 1020027430, 8020022890, 4020034510, 
        2020033490, 7020038340, 6020017370, 1020034170, 8020032840, 3020009320,
        2020041390, 7020046750, 6020021870, 1020035180, 8020044560, 5020082270, 
        2020057170, 7020047840, 6020029280, 2020065840, 7020065090, 2020071190, 
        4020050210, 3020024310, 4020050220, 1020040190, 4020050290, 4020050470
    ]
    input_gsw_folder_pattern = '/WORK/Data/global_lake_area/gsw_images/gsw_images_laea_{basin_id}'
    gsw_image_file_name_pattern = '{basin_id}_gsw_30m_{start_date}_{end_date}'
    output_file_path_pattern = '/WORK/Data/global_lake_area/gsw_images/mosaic/{basin_id}/{basin_id}_gsw_30m_{start_date}_{end_date}.tif'
    
    for basin_id in basin_id_list:
        if basin_id in processed_basin_id_list:
            continue
        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = []
            start_date = '2001-01-01'
            end_date = '2022-01-01'
            date_fmt = '%Y-%m-%d'
            start_date = datetime.strptime(start_date, date_fmt)
            end_date = datetime.strptime(end_date, date_fmt)
            current_date = start_date
            while current_date < end_date:
                next_month_date = current_date + relativedelta(months=1)
                futures.append(
                    executor.submit(
                        process_date_range,
                        basin_id,
                        current_date,
                        input_gsw_folder_pattern,
                        gsw_image_file_name_pattern,
                        output_file_path_pattern
                    )
                )
                current_date = next_month_date
        
            for future in futures:
                future.result()  # This will re-raise any exceptions caught during processing