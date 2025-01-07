import os
import geopandas as gpd
from datetime import datetime
from dateutil import relativedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

import sample_generation


def process_single_month(args):
    """Process a single month data for a given basin.
    
    Args:
        args (tuple): Contains:
            hybas_id (int): Basin ID
            current_date (datetime): Start date for the month
            lake_shp_gdf (GeoDataFrame): Lake boundaries geodataframe
            patterns (dict): Dictionary of file path patterns for input/output files
            date_fmt (str): Date format string for file paths
            
    Returns:
        tuple: Contains:
            hybas_id (int): Input basin ID
            current_date (datetime): Input start date
            n_samples (int): Number of samples generated
            
    Raises:
        Exception: If there is an error processing the data
    """
    try:
        hybas_id, current_date, lake_shp_gdf, patterns, date_fmt = args
        current_end_date = current_date + relativedelta.relativedelta(months=1)
        
        my_water_raster_path = patterns['my_water'].format(
            basin_id=hybas_id, 
            start_date=current_date.strftime(date_fmt), 
            end_date=current_end_date.strftime(date_fmt)
        )
        gsw_raster_path = patterns['gsw'].format(
            basin_id=hybas_id, 
            start_date=current_date.strftime(date_fmt), 
            end_date=current_end_date.strftime(date_fmt)
        )
        save_path = patterns['save'].format(
            basin_id=hybas_id, 
            start_date=current_date.strftime(date_fmt), 
            end_date=current_end_date.strftime(date_fmt)
        )
        
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        current_samples = sample_generation.generate_samples_single_gdf(
            sample_boundary_gdf=lake_shp_gdf,
            raster_paths=[my_water_raster_path, gsw_raster_path],
            sample_col_names=['my_water', 'gsw'],
            max_points_per_cell=20000,
            sample_grid_size=10,
            save_path=save_path,
            verbose=False
        )
        
        return hybas_id, current_date, len(current_samples)
        
    except Exception as e:
        print(f"Error processing hybas_id={hybas_id}, date={current_date.strftime(date_fmt)}: {str(e)}")
        raise

if __name__ == '__main__':
    use_parallel = True
    hybas_id_list = [
        1020000010, 1020011530, 1020018110, 1020021940, 1020027430, 1020034170, 1020035180, 1020040190,
        2020000010, 2020003440, 2020018240, 2020024230, 2020033490, 2020041390, 2020057170, 2020065840, 2020071190,
        3020000010, 3020003790, 3020005240, 3020008670, 3020009320, 3020024310,
        4020000010, 4020006940, 4020015090, 4020024190, 4020034510, 4020050210, 4020050220, 4020050290, 4020050470,
        5020000010, 5020015660, 5020037270, 5020049720, 5020082270, 
        6020000010, 6020006540, 6020008320, 6020014330, 6020017370, 6020021870, 6020029280,
        7020000010, 7020014250, 7020021430, 7020024600, 7020038340, 7020046750, 7020047840, 7020065090,
        8020000010, 8020008900, 8020010700, 8020020760, 8020022890, 8020032840, 8020044560,
        9020000010
    ]
    
    processed_basin_id_list = [
        1020000010, 1020011530, 1020018110, 1020021940, 1020027430, 1020034170, 1020035180, 1020040190,
        2020000010, 2020003440, 2020018240, 2020024230, 2020033490, 2020041390, 2020057170, 2020065840, 2020071190,
        3020000010, 3020003790, 3020005240, 3020008670, 3020009320, 3020024310,
        4020000010, 4020006940, 4020015090, 4020024190, 4020034510, 4020050210, 4020050220, 4020050290, 4020050470,
        5020000010, 5020015660, 5020037270, 5020049720, 5020082270, 
        6020000010, 6020006540, 6020008320, 6020014330, 6020017370, 6020021870, 6020029280,
        7020000010, 7020014250, 7020021430, 7020024600, 
        8020008900, 8020010700, 8020022890, 8020032840,
    ]
    start_date = '2001-01-01'
    end_date = '2022-01-01'
    date_fmt = '%Y-%m-%d'
    start_date = datetime.strptime(start_date, date_fmt)
    end_date = datetime.strptime(end_date, date_fmt)
    
    lake_shp_path_pattern = '/WORK/Data/global_lake_area/lake_shps/HydroLAKES_updated_using_GLAKES/per_basin_no_contained_buffered/hylak_buffered_updated_no_contained_{basin_id}_reprojected.shp'

    my_water_raster_path_pattern = '/WORK/Data/global_lake_area/mosaic_tifs/{basin_id}/{basin_id}_{start_date}_{end_date}_water_mosaic.tif'
    gsw_raster_path_pattern = '/WORK/Data/global_lake_area/gsw_images/mosaic/{basin_id}/{basin_id}_gsw_30m_{start_date}_{end_date}.tif'
    
    save_path_pattern = '/WORK/Data/global_lake_area/revision/accuracy_samples/{basin_id}/{basin_id}_{start_date}_{end_date}_samples.pkl'
    
    # Create a list of all tasks
    patterns = {
        'my_water': my_water_raster_path_pattern,
        'gsw': gsw_raster_path_pattern,
        'save': save_path_pattern
    }
    
    for hybas_id in hybas_id_list:
        if hybas_id in processed_basin_id_list:
            continue
        lake_shp_path = lake_shp_path_pattern.format(basin_id=hybas_id)
        lake_shp_gdf = gpd.read_file(lake_shp_path)
        all_tasks = []
        current_date = start_date
        while current_date < end_date:
            all_tasks.append((hybas_id, current_date, lake_shp_gdf, patterns, date_fmt))
            current_date = current_date + relativedelta.relativedelta(months=1)

        if use_parallel:
            # Process tasks in parallel with progress bar
            with ProcessPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(process_single_month, task) for task in all_tasks]
                
                with tqdm(total=len(futures), desc="Processing samples") as pbar:
                    for future in as_completed(futures):
                        hybas_id, date, num_samples = future.result()  # This will raise any exceptions from the task
                        pbar.set_description(f"Processed {hybas_id} {date.strftime(date_fmt)}: {num_samples} samples")
                        pbar.update(1)
        else:
            for task in all_tasks:
                print(f"Processing {task}")
                process_single_month(task)
        