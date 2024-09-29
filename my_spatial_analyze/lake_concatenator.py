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
    CONCATENATE_TYPE = 'areas'
    DROP_CASPIAN = True
    
    if CONCATENATE_TYPE == 'levels':
        lake_pkl_gdf_paths = [
            f'/WORK/Data/global_lake_area/area_csvs/levels/level_{basin_id}.pkl' for basin_id in hybas_id_list
        ]
        concatenated_lake_gdf_save_paths = ['/WORK/Data/global_lake_area/area_csvs/levels/levels_all.pkl',
                                            '/WORK/Data/global_lake_area/area_csvs/levels/levels_all.csv']
    elif CONCATENATE_TYPE == 'areas':
        lake_pkl_gdf_paths = [
            f'/WORK/Data/global_lake_area/area_csvs/lake_wise_masked_and_analyzed_areas_with_geometries/pkl/lake_wise_masked_and_analyzed_areas_with_geometries_{basin_id}.pkl' for basin_id in hybas_id_list
        ]
        concatenated_lake_gdf_save_paths = ['/WORK/Data/global_lake_area/area_csvs/lakes/pkl/lakes_all.pkl',
                                            '/WORK/Data/global_lake_area/area_csvs/lakes/csv/lakes_all.csv']
    elif CONCATENATE_TYPE == 'GEDI_corrected_levels':
        lake_pkl_gdf_paths = [
            f'/WORK/Data/global_lake_area/area_csvs/levels_corrected_using_GEDI/level_{basin_id}_corrected.pkl' for basin_id in hybas_id_list
        ]
        concatenated_lake_gdf_save_paths = ['/WORK/Data/global_lake_area/area_csvs/levels_corrected_using_GEDI/level_all_corrected.pkl',
                                            '/WORK/Data/global_lake_area/area_csvs/levels_corrected_using_GEDI/level_all_corrected.csv']
    elif CONCATENATE_TYPE == 'ICESat2_corrected_levels':
        lake_pkl_gdf_paths = [
            f'/WORK/Data/global_lake_area/area_csvs/levels_corrected_using_ICESat2/level_{basin_id}_corrected.pkl' for basin_id in hybas_id_list
        ]
        concatenated_lake_gdf_save_paths = ['/WORK/Data/global_lake_area/area_csvs/levels_corrected_using_ICESat2/level_all_corrected.pkl',
                                            '/WORK/Data/global_lake_area/area_csvs/levels_corrected_using_ICESat2/level_all_corrected.csv']
    
    lake_gdfs = [pd.read_pickle(lake_pkl_gdf_path) for lake_pkl_gdf_path in lake_pkl_gdf_paths]
    for lake_gdf in lake_gdfs:
        if not isinstance(lake_gdf, gpd.GeoDataFrame):
            raise ValueError('The loaded lake is not a GeoDataFrame.')
    
    concatenated_lake_gdf = pd.concat(lake_gdfs, ignore_index=True, axis=0)
    print(len(concatenated_lake_gdf))
    if DROP_CASPIAN:
        concatenated_lake_gdf.drop(concatenated_lake_gdf[concatenated_lake_gdf['Hylak_id'] == 1].index, inplace=True)
    for concatenated_lake_gdf_save_path in concatenated_lake_gdf_save_paths:
        concatenated_lake_gdf_save_folder = os.path.dirname(concatenated_lake_gdf_save_path)
        if not os.path.exists(concatenated_lake_gdf_save_folder):
            os.makedirs(concatenated_lake_gdf_save_folder)
        if not os.path.exists(concatenated_lake_gdf_save_folder):
            os.makedirs(concatenated_lake_gdf_save_folder)
        if concatenated_lake_gdf_save_path.endswith('.pkl'):
            concatenated_lake_gdf.to_pickle(concatenated_lake_gdf_save_path)
        elif concatenated_lake_gdf_save_path.endswith('.csv'):
            concatenated_lake_gdf.drop('geometry', axis=1, inplace=False).to_csv(concatenated_lake_gdf_save_path, index=False)
        else:
            raise ValueError('The file extension is not recognized.')