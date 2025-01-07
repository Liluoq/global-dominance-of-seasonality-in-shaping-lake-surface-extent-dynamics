from datetime import datetime
from dateutil import relativedelta

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

GDF_PATHS = [f'/WORK/Data/global_lake_area/area_csvs/lake_wise_masked_and_analyzed_areas_with_geometries/pkl/lake_wise_masked_and_analyzed_areas_with_geometries_{basin_id}.pkl' for basin_id in hybas_id_list]

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

GLOBATHY_NC_PATH = '/WORK/Data/global_lake_area/nc_data/globathy/GLOBathy_hAV_relationships.nc'

ID_COLUMN_NAME = 'Hylak_id'
COLUMNS_TO_DROP = [
    'mean_area', 'mean_seasonal_amplitude', 'mean_annual_mean_first_difference',
    'mean_seasonal_amplitude_percentage', 'mean_annual_mean_first_difference_percentage', 'mean_apportion_entropy_percentage',
    'linear_trend_of_standard_deviation_per_period', 'linear_trend_of_standard_deviation_percentage_per_period', 'linear_trend_of_stl_trend_per_period',
    'linear_trend_of_annual_mean', 'linear_trend_of_annual_mean_first_difference', 'linear_trend_of_annual_mean_first_difference_percentage',
    'linear_trend_of_apportion_entropy_percentage',
    'rao_mk_test_on_stl_trend_p', 'rao_mk_test_on_seasonal_amplitude_p', 'rao_mk_test_on_seasonal_amplitude_percentage_p',
    'rao_mk_test_on_annual_mean_p', 'rao_mk_test_on_annual_mean_first_difference_p', 'rao_mk_test_on_annual_mean_first_difference_percentage_p',
    'rao_mk_test_on_apportion_entropy_percentage_p',
    'rao_mk_test_on_stl_trend_slope', 'rao_mk_test_on_seasonal_amplitude_slope', 'rao_mk_test_on_seasonal_amplitude_percentage_slope',
    'rao_mk_test_on_annual_mean_slope', 'rao_mk_test_on_annual_mean_first_difference_slope', 'rao_mk_test_on_annual_mean_first_difference_percentage_slope',
    'rao_mk_test_on_apportion_entropy_percentage_slope',
    'rao_mk_test_on_stl_trend_significant', 'rao_mk_test_on_seasonal_amplitude_significant', 'rao_mk_test_on_seasonal_amplitude_percentage_significant',
    'rao_mk_test_on_annual_mean_significant', 'rao_mk_test_on_annual_mean_first_difference_significant', 'rao_mk_test_on_annual_mean_first_difference_percentage_significant',
    'rao_mk_test_on_apportion_entropy_percentage_significant',
]

TYPES_OF_ANALYSIS = [
    'mean', 'mean_annual_standard_deviation', 'mean_annual_mean_first_difference',
    'mean_annual_standard_deviation_percentage', 'mean_annual_mean_first_difference_percentage', 'mean_apportion_entropy_percentage',
    'linear_trend_of_standard_deviation_per_period', 'linear_trend_of_standard_deviation_percentage_per_period', 'linear_trend_of_stl_trend_per_period',
    'linear_trend_of_annual_mean', 'linear_trend_of_annual_mean_first_difference', 'linear_trend_of_annual_mean_first_difference_percentage',
    'linear_trend_of_apportion_entropy_percentage',
    'rao_mk_test_on_stl_linear_trend', 'rao_mk_test_on_standard_deviation', 'rao_mk_test_on_standard_deviation_percentage',
    'rao_mk_test_on_annual_mean', 'rao_mk_test_on_annual_mean_first_difference', 'rao_mk_test_on_annual_mean_first_difference_percentage',
    'rao_mk_test_on_apportion_entropy_percentage'
]
ANALYZED_COLUMN_NAMES = [
    'mean_area', 'mean_seasonal_amplitude', 'mean_annual_mean_first_difference',
    'mean_seasonal_amplitude_percentage', 'mean_annual_mean_first_difference_percentage', 'mean_apportion_entropy_percentage',
    'linear_trend_of_standard_deviation_per_period', 'linear_trend_of_standard_deviation_percentage_per_period', 'linear_trend_of_stl_trend_per_period',
    'linear_trend_of_annual_mean', 'linear_trend_of_annual_mean_first_difference', 'linear_trend_of_annual_mean_first_difference_percentage',
    'linear_trend_of_apportion_entropy_percentage',
    'rao_mk_test_on_stl_trend', 'rao_mk_test_on_seasonal_amplitude', 'rao_mk_test_on_seasonal_amplitude_percentage',
    'rao_mk_test_on_annual_mean', 'rao_mk_test_on_annual_mean_first_difference', 'rao_mk_test_on_annual_mean_first_difference_percentage',
    'rao_mk_test_on_apportion_entropy_percentage'
]

SAVE_PATHS = [
    [f'/WORK/Data/global_lake_area/area_csvs/levels/level_{basin_id}.csv', f'/WORK/Data/global_lake_area/area_csvs/levels/level_{basin_id}.pkl'] for basin_id in hybas_id_list
]

GEOMETRY_COLUMNS = ['geometry']