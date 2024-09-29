import geopandas as gpd
import pandas as pd
import os

if __name__ == '__main__':
    
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
    grid_pkl_gdf_paths = [
        f'/WORK/Data/global_lake_area/area_csvs/grids/pkl/grid_{basin_id}.pkl' for basin_id in hybas_id_list
    ]
    concatenated_grid_gdf_save_paths = ['/WORK/Data/global_lake_area/area_csvs/grids/pkl/grid_current.pkl',
                                        '/WORK/Data/global_lake_area/area_csvs/grids/csv/grid_current.csv']
    
    grid_gdfs = [pd.read_pickle(grid_pkl_gdf_path) for grid_pkl_gdf_path in grid_pkl_gdf_paths]
    for grid_gdf in grid_gdfs:
        if not isinstance(grid_gdf, gpd.GeoDataFrame):
            raise ValueError('The loaded grid is not a GeoDataFrame.')
    
    concatenated_grid_gdf = pd.concat(grid_gdfs, ignore_index=True, axis=0)
    
    for concatenated_grid_gdf_save_path in concatenated_grid_gdf_save_paths:
        concatenated_grid_gdf_save_folder = os.path.dirname(concatenated_grid_gdf_save_path)
        if not os.path.exists(concatenated_grid_gdf_save_folder):
            os.makedirs(concatenated_grid_gdf_save_folder)
        if not os.path.exists(concatenated_grid_gdf_save_folder):
            os.makedirs(concatenated_grid_gdf_save_folder)
        if concatenated_grid_gdf_save_path.endswith('.pkl'):
            concatenated_grid_gdf.to_pickle(concatenated_grid_gdf_save_path)
        elif concatenated_grid_gdf_save_path.endswith('.csv'):
            concatenated_grid_gdf.drop('geometry', axis=1, inplace=False).to_csv(concatenated_grid_gdf_save_path, index=False)
        else:
            raise ValueError('The file extension is not recognized.')