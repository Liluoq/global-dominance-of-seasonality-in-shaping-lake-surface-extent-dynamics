import pandas as pd
import geopandas as gpd
import numpy as np
import os
from attach_geometry_and_generate_grid import time_series_analysis_on_df, generate_grid_from_geometry_added_concatenated_areas
import sys
sys.path.append('/WORK/Codes/global_lake_area')
from batch_processing.load_config_module import load_config_module


def sum_positive(values):
    return values[values > 0].sum()

def sum_negative(values):
    return values[values < 0].sum()

def mean_positive(values):
    return values[values > 0].mean()

def mean_negative(values):
    return values[values < 0].mean()

def calculate_statistics_along_direction(
    gdf, 
    column, 
    axis='longitude', 
    step=1.0, 
    stat_type='mean'
):
    if axis not in ['longitude', 'latitude']:
        raise ValueError("Axis must be either 'longitude' or 'latitude'")
    
    valid_stats = ['mean', 'median', 'sum', 'count', 'sum_positive', 'sum_negative', 'mean_positive', 'mean_negative']
    if stat_type not in valid_stats:
        raise ValueError(f"stat_type must be one of {valid_stats}")
    
    # Extract centroid coordinates
    gdf['centroid'] = gdf.geometry.centroid
    if axis == 'longitude':
        gdf['coord'] = gdf.centroid.x
    else:
        gdf['coord'] = gdf.centroid.y
    
    # Create bins
    min_coord = gdf['coord'].min()
    max_coord = gdf['coord'].max()
    bins = np.arange(min_coord, max_coord + step, step)
    gdf['bin'] = pd.cut(gdf['coord'], bins=bins, include_lowest=True, precision=6)
    
    # Define custom aggregation functions
    agg_funcs = {
        'mean': 'mean',
        'median': 'median',
        'sum': 'sum',
        'count': 'count',
        'sum_positive': sum_positive,
        'sum_negative': sum_negative,
        'mean_positive': mean_positive,
        'mean_negative': mean_negative
    }
    
    # Calculate the specified statistic
    stats = gdf.groupby('bin')[column].agg(agg_funcs[stat_type]).reset_index()
    stats.columns = ['bin', stat_type]  # Set the column name to stat_type
    stats['bin_center'] = stats['bin'].apply(lambda x: x.mid)
    
    return stats

def grid_time_series_analysis(
    grid_gdf,
    time_series_columns,
    type_of_analysis,
    output_column_name,
    unit_scale=1.0,
    mk_test_significance_level=0.05
):
    analyzed_grid_gdf = time_series_analysis_on_df(
        df=grid_gdf,
        time_series_columns=time_series_columns,
        type_of_analysis=type_of_analysis,
        output_column_name=output_column_name,
        unit_scale=unit_scale,
        period=13,
        reshape_period=12,
        calculation_mask_columns=None,
        mk_test_significance_level=mk_test_significance_level,
        parallel=False
    )
    return analyzed_grid_gdf

if __name__ == '__main__':
    TEST = False
    GENERATE_FULL_GRID_AREA = True
    TIME_SERIES_ANALYZE = False
    if TEST:
        # Sample GeoDataFrame
        data = {
            'geometry': [
                'POLYGON ((-1 -1, -1 0, 0 0, 0 -1, -1 -1))',
                'POLYGON ((0 -1, 0 0, 1 0, 1 -1, 0 -1))',
                'POLYGON ((-1 0, -1 1, 0 1, 0 0, -1 0))',
                'POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))'
            ],
            'value': [10, 20, 15, 25]
        }
        gdf = gpd.GeoDataFrame(data, geometry=gpd.GeoSeries.from_wkt(data['geometry']))
        gdf.set_crs(epsg=4326, inplace=True)
        print(gdf.geometry.centroid)
        # Calculate statistics along latitude with a step of 1.5 degrees
        stats_df = calculate_statistics_along_direction(gdf, 'value', axis='latitude', step=0.5)
        print(stats_df)
        print(stats_df['bin'].iloc[0].left, stats_df['bin'].iloc[0].right)
    
    if GENERATE_FULL_GRID_AREA:
        analyzed_column_agg_type = [
            'sum', 'sum', 'sum',
            'median', 'median', 'median',
            ('sum', 'pos_count', 'neg_count'), 'median', ('sum', 'pos_count', 'neg_count'),
            'median',
            ('sum', 'pos_count', 'neg_count'), ('sum', 'pos_count', 'neg_count'), 'median',
            'median',
            'sum', 'sum', 'median',
            'sum', 'sum', 'median',
            'median',
            'sum', 'sum', 'sum',
            'sum', 'sum', 'sum',
            'sum',
            'median',
            'sum',
            'median',
            'median'
        ]
        aggregate_column_names = [
            'mean_area', 'mean_seasonal_amplitude', 'mean_annual_mean_first_difference',
            'mean_seasonal_amplitude_percentage', 'mean_annual_mean_first_difference_percentage', 'mean_apportion_entropy_percentage', 
            'linear_trend_of_standard_deviation_per_period', 'linear_trend_of_standard_deviation_percentage_per_period', 'linear_trend_of_stl_trend_per_period',
            'linear_trend_of_stl_trend_percentage_per_period',
            'linear_trend_of_annual_mean', 'linear_trend_of_annual_mean_first_difference', 'linear_trend_of_annual_mean_first_difference_percentage',
            'linear_trend_of_apportion_entropy_percentage',
            'rao_mk_test_on_stl_trend_slope', 'rao_mk_test_on_seasonal_amplitude_slope', 'rao_mk_test_on_seasonal_amplitude_percentage_slope',
            'rao_mk_test_on_annual_mean_slope', 'rao_mk_test_on_annual_mean_first_difference_slope', 'rao_mk_test_on_annual_mean_first_difference_percentage_slope',
            'rao_mk_test_on_apportion_entropy_percentage_slope',
            'rao_mk_test_on_stl_trend_significant', 'rao_mk_test_on_seasonal_amplitude_significant', 'rao_mk_test_on_seasonal_amplitude_percentage_significant',
            'rao_mk_test_on_annual_mean_significant', 'rao_mk_test_on_annual_mean_first_difference_significant', 'rao_mk_test_on_annual_mean_first_difference_percentage_significant',
            'rao_mk_test_on_apportion_entropy_percentage_significant',
            'seasonality_dominance_percentage',
            'annual_means_std',
            'percentage_of_extreme_low_water_compared_with_long_term_changes',
            'percentage_of_extreme_low_water_compared_with_long_term_trends_plus_average_seasonlity'
        ]
        additional_agg_dict_for_generating_grid = {
            col: agg_type for col, agg_type in zip(aggregate_column_names, analyzed_column_agg_type)
        }
        
        bootstrap_ks_test_column_pairs = [
            ['mean_annual_std_percentage_of_first_period', 'mean_annual_std_percentage_of_second_period'],
            ['mean_annual_mean_first_difference_percentage_of_first_period', 'mean_annual_mean_first_difference_percentage_of_second_period'],
            ['mean_annual_mean_percentage_of_first_period', 'mean_annual_mean_percentage_of_second_period']
        ]
        bootstrap_ks_test_output_column_names = [
            'annual_std_percentage_change_mw_test_p',
            'annual_mean_first_difference_percentage_change_mw_test_p',
            'annual_mean_percentage_change_mw_test_p'
        ]
        
        config = load_config_module('LAKE_WISE_AREA_POSTPROCESSING_CONFIG.py')
        area_columns = config.AREA_COLUMNS
        lake_columns_oi = [config.LAKE_ID_COLUMN] + area_columns + ['centroid', 'geometry', 'Lake_type'] + aggregate_column_names + [item for sublist in bootstrap_ks_test_column_pairs for item in sublist]
        lake_full_pkl_path = '/WORK/Data/global_lake_area/area_csvs/lakes/pkl/lakes_all_with_all_additional_attributes.pkl'
        lake_full_gdf = pd.read_pickle(lake_full_pkl_path)[lake_columns_oi]
        
        
        selection_type = 'all'
        
        if selection_type == 'all':
            grid_gdf_save_paths = [
                '/WORK/Data/global_lake_area/area_csvs/grids/pkl/grid_all_medium.pkl',
                '/WORK/Data/global_lake_area/area_csvs/grids/csv/grid_all_medium.csv'
            ]
        elif selection_type == 'natural':
            lake_full_gdf.drop(lake_full_gdf[lake_full_gdf['Lake_type'] != 1].index, inplace=True)
            grid_gdf_save_paths = [
                '/WORK/Data/global_lake_area/area_csvs/grids/pkl/grid_all_natural.pkl',
                '/WORK/Data/global_lake_area/area_csvs/grids/csv/grid_all_natural.csv'
            ]
        elif selection_type == 'artificial':
            lake_full_gdf.drop(lake_full_gdf[lake_full_gdf['Lake_type'] == 1].index, inplace=True)
            grid_gdf_save_paths = [
                '/WORK/Data/global_lake_area/area_csvs/grids/pkl/grid_all_artificial.pkl',
                '/WORK/Data/global_lake_area/area_csvs/grids/csv/grid_all_artificial.csv'
            ]
        print('Generating grid from the concatenated lake GeoDataFrame...')

        grid_gdf = generate_grid_from_geometry_added_concatenated_areas(
            geometry_added_concatenated_areas_gdf=lake_full_gdf,
            grid_size=0.5,
            area_columns=area_columns,
            grid_extent=[-180, -90, 180, 90],
            geometry_to_use_column='geometry',
            additional_agg_dict=additional_agg_dict_for_generating_grid,
            bootstrap_ks_test_column_pairs=None,
            bootstrap_ks_test_output_columns=None
        )
        grid_gdf = grid_gdf[grid_gdf['lake_count'] != 0]
        grid_gdf['annual_std_increase_percentage'] = grid_gdf['linear_trend_of_standard_deviation_per_period_pos_count'] / (grid_gdf['linear_trend_of_standard_deviation_per_period_pos_count'] + grid_gdf['linear_trend_of_standard_deviation_per_period_neg_count']) * 100
        grid_gdf['annual_mean_increase_percentage'] = grid_gdf['linear_trend_of_annual_mean_pos_count'] / (grid_gdf['linear_trend_of_annual_mean_pos_count'] + grid_gdf['linear_trend_of_annual_mean_neg_count']) * 100
        grid_gdf['annual_mean_first_difference_increase_percentage'] = grid_gdf['linear_trend_of_annual_mean_first_difference_pos_count'] / (grid_gdf['linear_trend_of_annual_mean_first_difference_pos_count'] + grid_gdf['linear_trend_of_annual_mean_first_difference_neg_count']) * 100
        for save_path in grid_gdf_save_paths:
            save_folder = os.path.dirname(save_path)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            if save_path.endswith('.pkl'):
                grid_gdf.to_pickle(save_path)
            elif save_path.endswith('.csv'):
                grid_gdf.drop('geometry', axis=1, inplace=False).to_csv(save_path, index=False)
            else:
                raise ValueError('The file extension is not recognized.')
    
    if TIME_SERIES_ANALYZE:
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
        config = load_config_module('LAKE_WISE_AREA_POSTPROCESSING_CONFIG.py')
        print(len(hybas_id_list))
        for basin_id in hybas_id_list:
            print(f'Processing basin {basin_id}...')
            grid_pkl_gdf_path = f'/WORK/Data/global_lake_area/area_csvs/grids/pkl/grid_{basin_id}.pkl'
            grid_gdf_save_paths = [
                f'/WORK/Data/global_lake_area/area_csvs/grids/pkl/grid_{basin_id}.pkl',
                f'/WORK/Data/global_lake_area/area_csvs/grids/csv/grid_{basin_id}.csv'
            ]
            grid_gdf = pd.read_pickle(grid_pkl_gdf_path)
            time_series_columns = config.AREA_COLUMNS
            types_of_analysis = [
                'mean', 
                'mean_annual_standard_deviation', 
                'mean_annual_mean_first_difference',
                'mean_annual_standard_deviation_percentage', 
                'mean_annual_mean_first_difference_percentage',
                'linear_trend_of_stl_trend_per_period',
                'linear_trend_of_standard_deviation_per_period',
                'linear_trend_of_standard_deviation_percentage_per_period',
                'linear_trend_of_annual_mean',
                'linear_trend_of_annual_mean_first_difference',
                'linear_trend_of_annual_mean_first_difference_percentage',
                'rao_mk_test_on_stl_linear_trend',
                'rao_mk_test_on_standard_deviation',
                'rao_mk_test_on_standard_deviation_percentage',
                'rao_mk_test_on_annual_mean',
                'rao_mk_test_on_annual_mean_first_difference',
                'rao_mk_test_on_annual_mean_first_difference_percentage',
            ]
            output_column_names = [
                'grid_mean_area',
                'grid_mean_seasonal_amplitude',
                'grid_mean_annual_mean_first_difference',
                'grid_mean_seasonal_amplitude_percentage',
                'grid_mean_annual_mean_first_difference_percentage',
                'grid_linear_trend_of_stl_trend_per_period',
                'grid_linear_trend_of_standard_deviation_per_period',
                'grid_linear_trend_of_standard_deviation_percentage_per_period',
                'grid_linear_trend_of_annual_mean',
                'grid_linear_trend_of_annual_mean_first_difference',
                'grid_linear_trend_of_annual_mean_first_difference_percentage',
                'grid_rao_mk_test_on_stl_linear_trend',
                'grid_rao_mk_test_on_standard_deviation',
                'grid_rao_mk_test_on_standard_deviation_percentage',
                'grid_rao_mk_test_on_annual_mean',
                'grid_rao_mk_test_on_annual_mean_first_difference',
                'grid_rao_mk_test_on_annual_mean_first_difference_percentage'
            ]
            for type_of_analysis, output_column_name in zip(types_of_analysis, output_column_names):
                grid_gdf = grid_time_series_analysis(
                    grid_gdf=grid_gdf,
                    time_series_columns=time_series_columns,
                    type_of_analysis=type_of_analysis,
                    output_column_name=output_column_name,
                    unit_scale=1e-6,
                    mk_test_significance_level=0.05
                )
            
            for save_path in grid_gdf_save_paths:
                save_folder = os.path.dirname(save_path)
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                if save_path.endswith('.pkl'):
                    grid_gdf.to_pickle(save_path)
                elif save_path.endswith('.csv'):
                    grid_gdf.drop('geometry', axis=1, inplace=False).to_csv(save_path, index=False)
                else:
                    raise ValueError('The file extension is not recognized.')