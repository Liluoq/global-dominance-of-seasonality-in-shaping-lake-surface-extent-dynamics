# Config for lake-wise area postprocessing

from datetime import datetime
from dateutil import relativedelta

BASIN_ID = 9020000010
RAW_CONCATENATED_CSV_PATH = f'/WORK/Data/global_lake_area/area_csvs/concatenated/{BASIN_ID}_concatenated.csv'
CLOUD_COVER_RATIO_CSV_PATH = f'/WORK/Data/global_lake_area/area_csvs/cloud_cover_ratio/{BASIN_ID}_cloud_cover_ratio.csv'

area_start_date = '2001-01-01'
area_end_date = '2024-01-01'
date_fmt = '%Y-%m-%d'
area_start_date = datetime.strptime(area_start_date, date_fmt)
area_end_date = datetime.strptime(area_end_date, date_fmt)
AREA_COLUMNS = []
current_date = area_start_date
while current_date < area_end_date:
    AREA_COLUMNS.append(current_date.strftime(date_fmt))
    current_date = current_date + relativedelta.relativedelta(months=1)
    
LAKE_ID_COLUMN = 'Hylak_id'
LAKEENSEMBLR_NC_PATH = '/WORK/Data/global_lake_area/lake_1d_model_results/lakes_ice_coverage_median.nc'
ICE_RATIO_THRESHOLD = 0.1
TYPES_OF_ANALYSIS = [
    'mean', 'mean_annual_standard_deviation', 'mean_annual_mean_first_difference',
    'mean_annual_standard_deviation_percentage', 'mean_annual_mean_first_difference_percentage', 'mean_apportion_entropy_percentage',
    'linear_trend_of_standard_deviation_per_period', 'linear_trend_of_standard_deviation_percentage_per_period', 'linear_trend_of_stl_trend_per_period',
    'linear_trend_of_stl_trend_percentage_per_period',
    'linear_trend_of_annual_mean', 'linear_trend_of_annual_mean_first_difference', 'linear_trend_of_annual_mean_first_difference_percentage',
    'linear_trend_of_apportion_entropy_percentage',
    'rao_mk_test_on_stl_linear_trend', 'rao_mk_test_on_standard_deviation', 'rao_mk_test_on_standard_deviation_percentage',
    'rao_mk_test_on_annual_mean', 'rao_mk_test_on_annual_mean_first_difference', 'rao_mk_test_on_annual_mean_first_difference_percentage',
    'rao_mk_test_on_apportion_entropy_percentage',
    'difference_in_annual_mean_std_of_two_periods', 'difference_in_annual_mean_apportion_entropy_of_two_periods',
    'difference_in_annual_std_of_two_periods',
    'relative_difference_in_annual_std_of_two_periods',
    'difference_in_annual_mean_first_difference_std_of_two_periods',
    'relative_difference_in_annual_mean_first_difference_std_of_two_periods',
    'difference_in_monthly_first_differece_abs_annual_mean_of_two_periods',
    'mean_annual_std_percentage_of_first_period',
    'mean_annual_std_percentage_of_second_period',
    'mean_annual_mean_first_difference_percentage_of_first_period',
    'mean_annual_mean_first_difference_percentage_of_second_period',
    'mean_annual_mean_percentage_of_first_period',
    'mean_annual_mean_percentage_of_second_period'
]
ANALYZED_COLUMN_NAMES = [
    'mean_area', 'mean_seasonal_amplitude', 'mean_annual_mean_first_difference',
    'mean_seasonal_amplitude_percentage', 'mean_annual_mean_first_difference_percentage', 'mean_apportion_entropy_percentage',
    'linear_trend_of_standard_deviation_per_period', 'linear_trend_of_standard_deviation_percentage_per_period', 'linear_trend_of_stl_trend_per_period',
    'linear_trend_of_stl_trend_percentage_per_period',
    'linear_trend_of_annual_mean', 'linear_trend_of_annual_mean_first_difference', 'linear_trend_of_annual_mean_first_difference_percentage',
    'linear_trend_of_apportion_entropy_percentage',
    'rao_mk_test_on_stl_trend', 'rao_mk_test_on_seasonal_amplitude', 'rao_mk_test_on_seasonal_amplitude_percentage',
    'rao_mk_test_on_annual_mean', 'rao_mk_test_on_annual_mean_first_difference', 'rao_mk_test_on_annual_mean_first_difference_percentage',
    'rao_mk_test_on_apportion_entropy_percentage',
    'difference_in_annual_mean_std_of_two_periods', 'difference_in_annual_mean_apportion_entropy_of_two_periods',
    'difference_in_annual_std_of_two_periods',
    'relative_difference_in_annual_std_of_two_periods',
    'difference_in_annual_mean_first_difference_std_of_two_periods',
    'relative_difference_in_annual_mean_first_difference_std_of_two_periods',
    'difference_in_monthly_first_differece_abs_annual_mean_of_two_periods',
    'mean_annual_std_percentage_of_first_period',
    'mean_annual_std_percentage_of_second_period',
    'mean_annual_mean_first_difference_percentage_of_first_period',
    'mean_annual_mean_first_difference_percentage_of_second_period',
    'mean_annual_mean_percentage_of_first_period',
    'mean_annual_mean_percentage_of_second_period'
]
UNIT_SCALE = 1e-6
PERIOD = 13
PARALLEL_FOR_TIME_SERIES_ANALYSIS = True
MK_TEST_SIGNIFICANCE_LEVEL = 0.05
ATTACH_GEOMETRY = True
if ATTACH_GEOMETRY:
    LAKE_SHP_PATH = f'/WORK/Data/global_lake_area/lake_shps/HydroLAKES_updated_using_GLAKES/per_basin_no_contained/hylak_unbuffered_updated_no_contained_{BASIN_ID}.shp'
    TO_CALCULATE_AREA_LAKE_SHP_PATH = f'/WORK/Data/global_lake_area/lake_shps/HydroLAKES_updated_using_GLAKES/per_basin_no_contained_buffered/hylak_buffered_updated_no_contained_{BASIN_ID}_reprojected.shp'
    LAKE_ID_COLUMN_IN_SHP = 'Hylak_id'
    COMMON_LAKE_ID_NAME = 'Hylak_id'
    GRID_SIZE = 1.0
    SAVE_PATHS = [f'/WORK/Data/global_lake_area/area_csvs/lake_wise_masked_and_analyzed_areas_with_geometries/csv/lake_wise_masked_and_analyzed_areas_with_geometries_{BASIN_ID}.csv',
                    f'/WORK/Data/global_lake_area/area_csvs/lake_wise_masked_and_analyzed_areas_with_geometries/pkl/lake_wise_masked_and_analyzed_areas_with_geometries_{BASIN_ID}.pkl']
else:
    SAVE_PATHS = [f'/WORK/Data/global_lake_area/area_csvs/lake_wise_masked_and_analyzed_areas/lake_wise_masked_and_analyzed_areas_{BASIN_ID}.csv']
FIX_UNMASKED_LAKES_FLAG = True
MASK_COLUMN_PREFIX = 'frozen'
VERBOSE = 2

GENERATE_GRID_FLAG = False
if GENERATE_GRID_FLAG:
    analyzed_column_agg_type = [
        'sum', 'sum', 'sum',
        'mean', 'mean', 'mean',
        'sum', 'mean', 'sum',
        'sum', 'sum', 'mean',
        'mean',
        'sum', 'sum', 'mean',
        'sum', 'sum', 'mean',
        'mean',
        'sum', 'sum', 'sum',
        'sum', 'sum', 'sum',
        'sum',
        'sum', 'mean'
    ]
    aggregate_column_names = [
        'mean_area', 'mean_seasonal_amplitude', 'mean_annual_mean_first_difference',
        'mean_seasonal_amplitude_percentage', 'mean_annual_mean_first_difference_percentage', 'mean_apportion_entropy_percentage', 
        'linear_trend_of_standard_deviation_per_period', 'linear_trend_of_standard_deviation_percentage_per_period', 'linear_trend_of_stl_trend_per_period',
        'linear_trend_of_annual_mean', 'linear_trend_of_annual_mean_first_difference', 'linear_trend_of_annual_mean_first_difference_percentage',
        'linear_trend_of_apportion_entropy_percentage',
        'rao_mk_test_on_stl_trend_slope', 'rao_mk_test_on_seasonal_amplitude_slope', 'rao_mk_test_on_seasonal_amplitude_percentage_slope',
        'rao_mk_test_on_annual_mean_slope', 'rao_mk_test_on_annual_mean_first_difference_slope', 'rao_mk_test_on_annual_mean_first_difference_percentage_slope',
        'rao_mk_test_on_apportion_entropy_percentage_slope',
        'rao_mk_test_on_stl_trend_significant', 'rao_mk_test_on_seasonal_amplitude_significant', 'rao_mk_test_on_seasonal_amplitude_percentage_significant',
        'rao_mk_test_on_annual_mean_significant', 'rao_mk_test_on_annual_mean_first_difference_significant', 'rao_mk_test_on_annual_mean_first_difference_percentage_significant',
        'rao_mk_test_on_apportion_entropy_percentage_significant',
        'difference_in_annual_mean_std_of_two_periods', 'difference_in_annual_mean_apportion_entropy_of_two_periods'
    ]
    ADDITIONAL_AGG_DICT_WHEN_GENERATING_GRID = {
        col: agg_type for col, agg_type in zip(aggregate_column_names, analyzed_column_agg_type)
    }
    GRID_SIGNIFICANT_THRESHOLD = 0.5
    GRID_SAVE_PATHS = [f'/WORK/Data/global_lake_area/area_csvs/grids/pkl/grid_{BASIN_ID}.pkl',
                       f'/WORK/Data/global_lake_area/area_csvs/grids/csv/grid_{BASIN_ID}.csv']