import sys
sys.path.append('/WORK/Codes/global_lake_area')
from batch_processing.load_config_module import load_config_module
from area_to_level import convert_area_df_to_level_df
import pandas as pd
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config', type=str, required=True)
    
    args = parser.parse_args()
    config_path = args.config
    config = load_config_module(config_path)
    
    gdf_paths = config.GDF_PATHS
    area_columns = config.AREA_COLUMNS
    globathy_nc_path = config.GLOBATHY_NC_PATH
    id_column_name = config.ID_COLUMN_NAME
    columns_to_drop = config.COLUMNS_TO_DROP
    save_paths = config.SAVE_PATHS
    geometry_columns = config.GEOMETRY_COLUMNS
    
    for gdf_path, save_path_sublist in zip(gdf_paths, save_paths):
        if not gdf_path.endswith('.pkl'):
            raise ValueError('The input file must be a pickle file. (Because LLQ makes the pkl version have full geometries)')
        print(f'Converting {gdf_path}')
        gdf_to_convert = pd.read_pickle(gdf_path)
        
        converted_level_df = convert_area_df_to_level_df(
            area_df=gdf_to_convert,
            area_columns=area_columns,
            globathy_nc_path=globathy_nc_path,
            id_column_name=id_column_name,
            columns_to_drop=columns_to_drop
        )
        for save_path in save_path_sublist:
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                
            if save_path.endswith('.pkl'):
                converted_level_df.to_pickle(save_path)
            elif save_path.endswith('.csv'):
                converted_level_df.drop(columns=geometry_columns).to_csv(save_path, index=False)