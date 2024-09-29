from area_postprocessing import mask_area_when_frozen, fix_unmasked_lakes, mask_area_when_cloud_contaminated
from attach_geometry_and_generate_grid import time_series_analysis_on_df, attach_geometry_to_concatenated_areas, generate_grid_from_geometry_added_concatenated_areas
import sys
sys.path.append('/WORK/Codes/global_lake_area')
from batch_processing.load_config_module import load_config_module
import pandas as pd
import geopandas as gpd
import argparse
import os
import netCDF4 as nc

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    
    args = parser.parse_args()
    config_path = args.config
    
    config = load_config_module(config_path)
    
    basin_id = config.BASIN_ID
    raw_concatenated_csv_path = config.RAW_CONCATENATED_CSV_PATH
    cloud_cover_ratio_csv_path = config.CLOUD_COVER_RATIO_CSV_PATH
    area_columns = config.AREA_COLUMNS
    lake_id_column = config.LAKE_ID_COLUMN
    lakeensemblr_nc_path = config.LAKEENSEMBLR_NC_PATH
    ice_ratio_threshold = config.ICE_RATIO_THRESHOLD
    types_of_analysis = config.TYPES_OF_ANALYSIS
    analyzed_column_names = config.ANALYZED_COLUMN_NAMES
    unit_scale = config.UNIT_SCALE
    period = config.PERIOD
    parallel_for_time_series_analysis = config.PARALLEL_FOR_TIME_SERIES_ANALYSIS
    attach_geometry = config.ATTACH_GEOMETRY
    if attach_geometry:
        lake_shp_path = config.LAKE_SHP_PATH
        to_calculate_area_lake_shp_path = config.TO_CALCULATE_AREA_LAKE_SHP_PATH
        lake_id_column_in_shp = config.LAKE_ID_COLUMN_IN_SHP
        common_lake_id_name = config.COMMON_LAKE_ID_NAME
        grid_size = config.GRID_SIZE
    fix_unmasked_lakes_flag = config.FIX_UNMASKED_LAKES_FLAG
    save_paths = config.SAVE_PATHS
    verbose = config.VERBOSE
    mask_column_prefix = config.MASK_COLUMN_PREFIX
    
    mk_test_sinificance_level = config.MK_TEST_SIGNIFICANCE_LEVEL
    
    generate_grid_flag = config.GENERATE_GRID_FLAG
    if generate_grid_flag:
        additional_agg_dict_when_generating_grid = config.ADDITIONAL_AGG_DICT_WHEN_GENERATING_GRID
        grid_save_paths = config.GRID_SAVE_PATHS
    
    calculation_mask_columns = [f'{mask_column_prefix}_{area_column}' for area_column in area_columns]
    
    raw_concatenated_df = pd.read_csv(raw_concatenated_csv_path)
    
    cloud_cover_ratio_df = pd.read_csv(cloud_cover_ratio_csv_path)
    
    cloud_masked_concatenated_df = mask_area_when_cloud_contaminated(
        area_concatenated_df=raw_concatenated_df,
        cloud_cover_ratio_df=cloud_cover_ratio_df,
        area_df_lake_id_column_name=lake_id_column,
        cloud_cover_ratio_df_lake_id_column_name=lake_id_column,
        area_columns=area_columns, 
        cloud_cover_ratio_columns=area_columns,
        cloud_cover_threshold=0.05,
        already_frozen_masked=False,
        outlier_detection=True,
        lof_n_neighbors=12
    )
    
    num_lakes = raw_concatenated_df[lake_id_column].nunique()
    
    if attach_geometry:
        if verbose != 0:
            print(f'Attaching geometry to the analyzed data for basin {basin_id}, with {num_lakes} lakes.')
        geometry_attached_cloud_masked_concatenated_df = attach_geometry_to_concatenated_areas(
            concatenated_area_csv=cloud_masked_concatenated_df,
            lake_shp_path=lake_shp_path,
            lake_id_field_in_csv=lake_id_column,
            lake_id_field_in_shp=lake_id_column_in_shp,
            common_id_name=common_lake_id_name,
            cast_to_centroid=True,
            output_crs=None, # By default, it will be the same as the shapefile
            verbose=verbose
        )
        to_calculate_area_lake_gdf = gpd.read_file(to_calculate_area_lake_shp_path)
        geometry_attached_cloud_masked_concatenated_df['Bd_area'] = to_calculate_area_lake_gdf['geometry'].area
    
    if verbose != 0:
        print(f'Masking area when frozen for basin {basin_id}, with {num_lakes} lakes.')
    
    with nc.Dataset(lakeensemblr_nc_path) as lakeensemblr_nc:
        if verbose != 0:
            print(lakeensemblr_nc)
        ice_ratio_lake_id_array = lakeensemblr_nc.variables[lake_id_column][:]
        ice_ratio_array = lakeensemblr_nc.variables['ice_coverage_ratio'][:]
        
    geometry_attached_cloud_frozen_masked_concatenated_df = mask_area_when_frozen(
        area_concatenated_df=geometry_attached_cloud_masked_concatenated_df,
        area_columns=area_columns,
        lake_id_column=lake_id_column,
        ice_ratio_lake_id_array=ice_ratio_lake_id_array.copy(),
        ice_ratio_array=ice_ratio_array.copy(),
        ice_ratio_threshold=ice_ratio_threshold,
        mask_column_prefix=mask_column_prefix,
        use_parallel=True,
        outlier_detection=True,
        lof_n_neighbors=12,
        lake_area_column_name='Lake_area'
    )
        
    if fix_unmasked_lakes_flag:
        if not attach_geometry:
            raise ValueError('Cannot fix unmasked lakes without attaching geometry.')
        if verbose != 0:
            print(f'Fixing unmasked lakes for basin {basin_id}, with {num_lakes} lakes.')
        
        geometry_attached_cloud_frozen_masked_concatenated_df = fix_unmasked_lakes(
            area_partially_masked_gdf=geometry_attached_cloud_frozen_masked_concatenated_df,
            area_columns=area_columns,
            lake_id_column=lake_id_column,
            lake_geom_column='centroid',
            area_mask_flag_column='Area_masked',
            ice_ratio_lake_id_array=ice_ratio_lake_id_array.copy(),
            ice_ratio_array=ice_ratio_array.copy(),
            ice_ratio_threshold=ice_ratio_threshold,
            use_parallel=True,
            outlier_detection=True,
            lof_n_neighbors=12,
            lake_area_column_name='Lake_area'
        )
    
    for type_of_analysis, analyzed_column_name in zip(types_of_analysis, analyzed_column_names):
        if verbose != 0:
            print(f'Analyzing {type_of_analysis} for basin {basin_id}, with {num_lakes} lakes.')
        geometry_attached_analyzed_frozen_cloud_masked_concatenated_df = time_series_analysis_on_df(
            df=geometry_attached_cloud_frozen_masked_concatenated_df,
            time_series_columns=area_columns,
            type_of_analysis=type_of_analysis,
            output_column_name=analyzed_column_name,
            unit_scale=unit_scale,
            period=period,
            calculation_mask_columns=calculation_mask_columns,
            parallel=parallel_for_time_series_analysis
        )
    
    if save_paths is not None:
        for save_path in save_paths:
            save_folder = os.path.dirname(save_path)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            if verbose != 0:
                print(f'Saving the masked and analyzed data to {save_path}.')
            if attach_geometry:
                if save_path.endswith('.pkl'):
                    geometry_attached_analyzed_frozen_cloud_masked_concatenated_df.to_pickle(save_path)
                elif save_path.endswith('.csv'): # 'geometry' used in the following line is the polygon of the lake boundary
                    geometry_attached_analyzed_frozen_cloud_masked_concatenated_df.drop('geometry', axis=1, inplace=False).to_csv(save_path, index=False)
                    
    if generate_grid_flag:
        if verbose != 0:
            print(f'Generating grid for basin {basin_id}, with {num_lakes} lakes.')
        grid_gdf = generate_grid_from_geometry_added_concatenated_areas(
            geometry_added_concatenated_areas_gdf=geometry_attached_analyzed_frozen_cloud_masked_concatenated_df,
            grid_size=grid_size,
            area_columns=area_columns,
            additional_agg_dict=additional_agg_dict_when_generating_grid,
            geometry_to_use_column='geometry', # using the lake boundary polygon to ensure grid covers large lakes
            verbose=verbose
        )
        grids_columns = grid_gdf.columns
        for col in grids_columns:
            if 'significant' in col:
                grid_gdf[f'{col}_significant_ratio'] = grid_gdf[col]/grid_gdf['lake_count']
        
        for grid_save_path in grid_save_paths:
            grid_save_folder = os.path.dirname(grid_save_path)
            if not os.path.exists(grid_save_folder):
                os.makedirs(grid_save_folder)
            if verbose != 0:
                print(f'Saving the grid to {grid_save_path}.')
            if grid_save_path.endswith('.pkl'):
                grid_gdf.to_pickle(grid_save_path)
            elif grid_save_path.endswith('.csv'):
                grid_gdf.drop('geometry', axis=1, inplace=False).to_csv(grid_save_path, index=False)