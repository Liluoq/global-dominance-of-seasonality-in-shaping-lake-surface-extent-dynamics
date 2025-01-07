import numpy as np
import pandas as pd
import os
import geopandas as gpd
from area_postprocessing import cloud_cover_ratio_calculation
from datetime import datetime
from dateutil.relativedelta import relativedelta

def main():
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
    for basin_id in hybas_id_list:
        print(f'Processing basin {basin_id}')
        cloud_cover_csv_folder = '/WORK/Data/global_lake_area/area_csvs/missing_data_area_concatenated'
        cloud_cover_csv_filename = f'{basin_id}_missing_data_area_concatenated.csv'
        cloud_cover_csv_path = os.path.join(cloud_cover_csv_folder, cloud_cover_csv_filename)
        cloud_cover_df = pd.read_csv(cloud_cover_csv_path)
        
        start_date = '2001-01-01'
        end_date = '2024-01-01'
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        current_date = start_date
        area_columns = []
        while current_date < end_date:
            area_columns.append(current_date.strftime('%Y-%m-01'))
            current_date = current_date + relativedelta(months=1)
        
        lake_shp_folder = '/WORK/Data/global_lake_area/lake_shps/HydroLAKES_updated_using_GLAKES/per_basin_no_contained_buffered'
        lake_shp_filename = f'hylak_buffered_updated_no_contained_{basin_id}_reprojected.shp'
        lake_shp_path = os.path.join(lake_shp_folder, lake_shp_filename)
        lake_shp_gdf = gpd.read_file(lake_shp_path)
        
        cloud_cover_ratio_df_save_folder = '/WORK/Data/global_lake_area/area_csvs/cloud_cover_ratio'
        if not os.path.exists(cloud_cover_ratio_df_save_folder):
            os.makedirs(cloud_cover_ratio_df_save_folder)
        cloud_cover_ratio_df_save_filename = f'{basin_id}_cloud_cover_ratio.csv'
        cloud_cover_ratio_df_save_path = os.path.join(cloud_cover_ratio_df_save_folder, cloud_cover_ratio_df_save_filename)
        
        cloud_cover_ratio_df = cloud_cover_ratio_calculation(
            cloud_cover_area_df=cloud_cover_df,
            lake_shp_gdf=lake_shp_gdf,
            cloud_cover_area_columns=area_columns,
            cloud_cover_lake_id_column_name='Hylak_id',
            lake_shp_lake_id_column_name='Hylak_id',
            lake_shp_lake_geom_column_name='geometry'
        )
        
        cloud_cover_ratio_df.to_csv(cloud_cover_ratio_df_save_path, index=False)
        
if __name__ == '__main__':
    main()