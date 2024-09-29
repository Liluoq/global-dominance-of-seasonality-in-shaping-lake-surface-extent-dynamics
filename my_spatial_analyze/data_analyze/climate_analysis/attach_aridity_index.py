import pandas as pd
import geopandas as gpd
import numpy as np
import dask.dataframe as dd
import os

def attach_aridity_index_to_lakes(
    lake_df,
    lake_atlas_gdf,
    lake_id_column_name='Hylak_id',
    aridity_index_column_name='ari_ix_uav',
    aridity_index_scale_factor=0.01,
    humid_arid_threshold=0.65,
    climate_zone_column_name='climate_zone',
    save_paths=None,
    verbose=2
):
    lake_atlas_gdf = lake_atlas_gdf[[lake_id_column_name, aridity_index_column_name]]
    lake_atlas_gdf[aridity_index_column_name] = lake_atlas_gdf[aridity_index_column_name] * aridity_index_scale_factor
    lake_df = lake_df.merge(lake_atlas_gdf, on=lake_id_column_name, how='left')
    lake_df[climate_zone_column_name] = np.where(lake_df[aridity_index_column_name] > humid_arid_threshold, 'humid', 'arid')
    for save_path in save_paths:
        save_folder = os.path.dirname(save_path)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        if verbose != 0:
            print(f'Saving the lake data with aridity index to {save_path}.')
        if save_path.endswith('.pkl'):
            lake_df.to_pickle(save_path)
        elif save_path.endswith('.csv'):
            lake_df.drop('geometry', axis=1, inplace=False).to_csv(save_path, index=False)
        else:
            raise ValueError('Unsupported file format for saving.')
        
if __name__ == '__main__':
    lake_pkl_path = '/WORK/Data/global_lake_area/area_csvs/lakes/pkl/lakes_all.pkl'
    lake_atlas_gdb_path = '/WORK/Data/global_lake_area/hydroATLAS/pkl/lake_atlas.pkl'
    aridity_index_column_name = 'ari_ix_uav'
    save_paths = ['/WORK/Data/global_lake_area/area_csvs/lakes/pkl/lakes_all_with_aridity_index.pkl',
                  '/WORK/Data/global_lake_area/area_csvs/lakes/csv/lakes_all_with_aridity_index.csv']
    print(f'Loading the lake atlas data from {lake_atlas_gdb_path}.')
    lake_atlas_gdf = pd.read_pickle(lake_atlas_gdb_path)
    print(f'Loading the lake data from {lake_pkl_path}.')
    lake_df = pd.read_pickle(lake_pkl_path)
    attach_aridity_index_to_lakes(
        lake_df=lake_df,
        lake_atlas_gdf=lake_atlas_gdf,
        aridity_index_column_name=aridity_index_column_name,
        save_paths=save_paths
    )