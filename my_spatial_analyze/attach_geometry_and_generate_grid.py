import geopandas as gpd
import numpy as np
import pandas as pd
import os
from shapely.geometry import Polygon
import math
from scipy.stats import entropy
from statsmodels.tsa.seasonal import STL
from pymannkendall import hamed_rao_modification_test
from pyhomogeneity import pettitt_test
from pandarallel import pandarallel
from hurst import compute_Hc
# ignore mk test warnings of dividing by zero or invalid sqrt
import warnings
from scipy.stats import ks_2samp
from scipy.stats import mannwhitneyu
warnings.filterwarnings('ignore', category=RuntimeWarning, module='pymannkendall')


print('Initializing pandarallel...')
pandarallel.initialize(progress_bar=False, nb_workers=12, use_memory_fs = False)

def bootstrap_ks_test_two_sided(sample1, sample2, num_bootstrap=1000):
    # Calculate the observed K-S statistic
    observed_stat, _ = ks_2samp(sample1, sample2)
    
    # Combine both samples
    combined_sample = np.concatenate([sample1, sample2])
    
    # Initialize an array to store bootstrap statistics
    bootstrap_stats = np.zeros(num_bootstrap)
    
    # Perform bootstrap resampling
    for i in range(num_bootstrap):
        # Resample with replacement from the combined sample
        resampled1 = np.random.choice(combined_sample, size=len(sample1), replace=True)
        resampled2 = np.random.choice(combined_sample, size=len(sample2), replace=True)
        
        # Calculate the K-S statistic for the resampled data
        bootstrap_stat, _ = ks_2samp(resampled1, resampled2)
        bootstrap_stats[i] = bootstrap_stat
    
    # Calculate the p-value
    p_value = np.mean(bootstrap_stats >= observed_stat)
    
    return observed_stat, p_value

def bootstrap_ks_test_one_sided(sample1, sample2, num_bootstrap=1000, alternative='greater'):
    if sample1.size == 0 or sample2.size == 0:
        return np.nan, np.nan
    observed_stat, _ = ks_2samp(sample1, sample2, alternative=alternative)
    combined_sample = np.concatenate([sample1, sample2])
    bootstrap_stats = np.zeros(num_bootstrap)
    for i in range(num_bootstrap):
        resampled1 = np.random.choice(combined_sample, size=len(sample1), replace=True)
        resampled2 = np.random.choice(combined_sample, size=len(sample2), replace=True)
        bootstrap_stat, _ = ks_2samp(resampled1, resampled2, alternative=alternative)
        bootstrap_stats[i] = bootstrap_stat
    p_value = np.mean(bootstrap_stats >= observed_stat) if alternative == 'greater' else np.mean(bootstrap_stats <= observed_stat)
    return observed_stat, p_value

def bootstrap_mannwhitneyu_test(data1, data2, n_bootstrap=1000, alternative='two-sided'):
    if data1.size == 0 or data2.size == 0:
        return np.nan, np.nan
    # Compute the observed Mann-Whitney U statistic
    observed_stat, _ = mannwhitneyu(data1, data2, alternative=alternative)
    
    # Combine the datasets
    combined = np.concatenate([data1, data2])
    
    # Perform bootstrapping
    bootstrap_stats = []
    np.random.seed(42)  # For reproducibility
    for _ in range(n_bootstrap):
        np.random.shuffle(combined)
        new_data1 = combined[:len(data1)]
        new_data2 = combined[len(data1):]
        stat, _ = mannwhitneyu(new_data1, new_data2, alternative=alternative)
        bootstrap_stats.append(stat)
    
    # Calculate the p-value for the one-sided test
    bootstrap_stats = np.array(bootstrap_stats)
    
    if alternative == 'greater':
        p_value = np.mean(bootstrap_stats >= observed_stat)
    elif alternative == 'less':
        p_value = np.mean(bootstrap_stats <= observed_stat)
    else:  # 'two-sided'
        p_value = np.mean(np.abs(bootstrap_stats) >= np.abs(observed_stat))
    
    return observed_stat, p_value

def attach_geometry_to_concatenated_areas(
    concatenated_area_csv,
    lake_shp_path: str,
    lake_id_field_in_csv: str,
    lake_id_field_in_shp: str,
    common_id_name: str,
    cast_to_centroid: bool = False,
    output_pkl_path: str = None,
    output_crs=None,
    verbose=0 
):
    if verbose != 0:
        print('Reading lake shapefile and concatenated area CSV...')
    lake_shp_gdf = gpd.read_file(lake_shp_path)
    
    if isinstance(concatenated_area_csv, str):
        concatenated_area_df = pd.read_csv(concatenated_area_csv)
    elif isinstance(concatenated_area_csv, pd.DataFrame):
        concatenated_area_df = concatenated_area_csv
    else:
        raise ValueError('concatenated_area_csv must be either a path to a CSV file or a DataFrame')
    
    if verbose == 2:
        print('Lake shapefile:')
        print(lake_shp_gdf.head())
        print('Concatenated area CSV:')
        print(concatenated_area_df.head())
    if lake_id_field_in_shp != common_id_name:
        lake_shp_gdf[common_id_name] = lake_shp_gdf[lake_id_field_in_shp]
        lake_shp_gdf.drop(columns=[lake_id_field_in_shp], inplace=True)
    if lake_id_field_in_csv != common_id_name:
        concatenated_area_df[common_id_name] = concatenated_area_df[lake_id_field_in_csv]
        concatenated_area_df.drop(columns=[lake_id_field_in_csv], inplace=True)
    
    if verbose != 0:
        print('Filtering lake shapefile...')
    filtered_lake_shp_gdf = lake_shp_gdf[lake_shp_gdf[common_id_name].isin(concatenated_area_df[common_id_name])]
    if verbose == 2:
        print('Original size of lake shapefile:', len(lake_shp_gdf))
        print('Filtered size of lake shapefile:', len(filtered_lake_shp_gdf))
    if output_crs is not None:
        if verbose != 0:
            print(f'Reprojecting lake shapefile to {output_crs}...')
        filtered_lake_shp_gdf = filtered_lake_shp_gdf.to_crs(output_crs)
    if verbose != 0:
        print('Merging lake shapefile with concatenated area CSV...')
    concatenated_area_df = concatenated_area_df.merge(filtered_lake_shp_gdf, on=common_id_name, how='inner')
    if verbose == 2:
        print(f'filtered_lake_shp_gdf CRS: {filtered_lake_shp_gdf.crs}')
    concatenated_area_gdf = gpd.GeoDataFrame(concatenated_area_df, geometry='geometry', crs=filtered_lake_shp_gdf.crs)
    
    if cast_to_centroid:
        if verbose != 0:
            print('Casting to centroid...')
        concatenated_area_gdf['centroid'] = concatenated_area_gdf['geometry'].centroid
        concatenated_area_gdf.set_geometry('centroid', inplace=True)
        
    if output_pkl_path is not None:
        output_pkl_folder = os.path.dirname(output_pkl_path)
        if not os.path.exists(output_pkl_folder):
            os.makedirs(output_pkl_folder)
        if verbose != 0:
            print(f'Saving to {output_pkl_path}...')
            print("Here's the first 5 rows of the GeoDataFrame:")
            print(concatenated_area_gdf.head())
        concatenated_area_gdf.to_pickle(output_pkl_path)
    
    return concatenated_area_gdf

def generate_grid_from_geometry_added_concatenated_areas(
    geometry_added_concatenated_areas_gdf,
    grid_size: float,
    area_columns: list,
    op_area_column='sum',
    output_pkl_path=None,
    grid_extent=None,  # [minx, miny, maxx, maxy],
    additional_agg_dict=None,
    geometry_to_use_column=None,
    verbose=0,
    bootstrap_ks_test_column_pairs=None,
    bootstrap_ks_test_output_columns=None
):
    if grid_extent:
        if type(grid_extent) != list:
            raise ValueError('grid_extent must be a list like [minx, miny, maxx, maxy]')
    
    if isinstance(geometry_added_concatenated_areas_gdf, str):
        if verbose != 0:
            print(f'Reading geometry-added concatenated areas from {geometry_added_concatenated_areas_gdf}...')
        geometry_added_concatenated_areas_gdf = pd.read_pickle(geometry_added_concatenated_areas_gdf)
    else:
        if not isinstance(geometry_added_concatenated_areas_gdf, gpd.GeoDataFrame):
            raise ValueError('geometry_added_concatenated_areas_gdf must be either a path to a GeoDataFrame or a GeoDataFrame')

    if geometry_to_use_column is None:
        geometry_to_use_column = geometry_added_concatenated_areas_gdf.geometry.name
    else:
        geometry_added_concatenated_areas_gdf.set_geometry(geometry_to_use_column, inplace=True)
    if verbose == 2:
        print(f'Using geometry {geometry_to_use_column}')
    if verbose == 2:
        print('Geometry-added concatenated areas:')
        print(geometry_added_concatenated_areas_gdf.head())
        print(f'using geometry {geometry_added_concatenated_areas_gdf.geometry.name}')
    if not isinstance(geometry_added_concatenated_areas_gdf, gpd.GeoDataFrame):
        raise ValueError('geometry_added_concatenated_areas_path must point to a GeoDataFrame')
    
    if grid_extent is None:
        grid_extent = geometry_added_concatenated_areas_gdf.total_bounds
    
    minx, miny, maxx, maxy = grid_extent
    
    if verbose != 0:
        print('Generating grid cells...')
        print(f'Extent: {minx, miny, maxx, maxy}')
    
    # Generate the grid cells and assign a unique grid_id
    cols = math.ceil((maxx - minx) / grid_size)
    rows = math.ceil((maxy - miny) / grid_size)
    horizontal_lines = [minx + i * grid_size for i in range(cols + 1)]
    vertical_lines = [miny + j * grid_size for j in range(rows + 1)]

    polygons = []
    grid_ids = []
    grid_id = 0  # Initialize grid_id
    for i in range(len(horizontal_lines)-1):
        for j in range(len(vertical_lines)-1):
            current_polygon = Polygon([
                (horizontal_lines[i], vertical_lines[j]),
                (horizontal_lines[i + 1], vertical_lines[j]),
                (horizontal_lines[i + 1], vertical_lines[j + 1]),
                (horizontal_lines[i], vertical_lines[j + 1]),
                (horizontal_lines[i], vertical_lines[j])
            ])
            if verbose == 2:
                #test if the polygon is valid
                if not current_polygon.is_valid:
                    print(f'Polygon {grid_id} is not valid')
                    print(current_polygon)
            polygons.append(current_polygon)
            grid_ids.append(grid_id)
            grid_id += 1

    # Create a new GeoDataFrame for the grid with grid_id
    grid_gdf = gpd.GeoDataFrame({'grid_id': grid_ids, 'geometry': polygons}, crs=geometry_added_concatenated_areas_gdf.crs)

    # Perform spatial join
    if verbose != 0:
        print('Performing spatial join...')
    joined_gdf = gpd.sjoin(grid_gdf, geometry_added_concatenated_areas_gdf, how='inner', predicate='intersects')
    # Aggregate data
    aggregation_dict = {col: op_area_column for col in area_columns}
    aggregation_dict['index_right'] = 'count'  # Count the number of lakes
    if additional_agg_dict is not None:
        def mode(series):
            return series.mode()[0] if not series.mode().empty else np.nan
        for k, v in additional_agg_dict.items():
            if v == 'mode':
                additional_agg_dict[k] = mode
            elif isinstance(v, tuple):
                rep_v = tuple([type_of_agg if type_of_agg != 'mode' else mode for type_of_agg in v])
                additional_agg_dict[k] = rep_v
    
        def pos_count(series):
            return np.sum(series > 0)
        for k, v in additional_agg_dict.items():
            if v == 'pos_count':
                additional_agg_dict[k] = pos_count
            elif isinstance(v, tuple):
                rep_v = tuple([type_of_agg if type_of_agg != 'pos_count' else pos_count for type_of_agg in v])
                additional_agg_dict[k] = rep_v
        
        def neg_count(series):
            return np.sum(series < 0)
        for k, v in additional_agg_dict.items():
            if v == 'neg_count':
                additional_agg_dict[k] = neg_count
            elif isinstance(v, tuple):
                rep_v = tuple([type_of_agg if type_of_agg != 'neg_count' else neg_count for type_of_agg in v])
                additional_agg_dict[k] = rep_v
                
        aggregation_dict.update(additional_agg_dict)
    aggregated_data = joined_gdf.groupby('grid_id').agg(aggregation_dict)
    aggregated_data.columns = ['_'.join(col).strip() for col in aggregated_data.columns.values]
    aggregated_data.rename(columns={'index_right_count': 'lake_count'}, inplace=True)
    aggregated_data.rename(columns={f'{col}_sum': col for col in area_columns}, inplace=True)
    # Join the aggregated data back to the grid dataframe to maintain geometry
    final_gdf = grid_gdf.join(aggregated_data, on='grid_id')
    # Fill NaN values with 0 for grids without any lakes
    final_gdf[area_columns + ['lake_count']] = final_gdf[area_columns + ['lake_count']].fillna(0)

    if bootstrap_ks_test_column_pairs is not None:
        if bootstrap_ks_test_output_columns is None:
            raise ValueError('bootstrap_ks_test_output_columns must be provided')
        
        for bootstrap_ks_test_column_pair, bootstrap_ks_test_output_column in zip(bootstrap_ks_test_column_pairs, bootstrap_ks_test_output_columns):
            def apply_mw_test_and_get_p(x, alternative):
                print(f'Applying KS test for grid {x["grid_id"].iloc[0]} with alternative {alternative}')
                sample1 = x[bootstrap_ks_test_column_pair[0]].to_numpy().flatten()
                sample2 = x[bootstrap_ks_test_column_pair[1]].to_numpy().flatten()
                assert len(sample1) == len(sample2), 'The two samples must have the same length'
                nan_mask = np.logical_or(np.isnan(sample1), np.isnan(sample2))
                sample1 = sample1[~nan_mask]
                sample2 = sample2[~nan_mask]
                assert len(sample1) == len(sample2), 'The two samples must have the same length after removing NaN values'
                _, p_value = bootstrap_mannwhitneyu_test(sample1, sample2, alternative=alternative)
                return p_value
            def apply_mw_test_and_get_p_greater(x):
                return apply_mw_test_and_get_p(x, 'greater')
            def apply_mw_test_and_get_p_less(x):
                return apply_mw_test_and_get_p(x, 'less')
            current_ks_result = joined_gdf.groupby('grid_id').parallel_apply(apply_mw_test_and_get_p_greater).reset_index()
            current_ks_result.columns = ['grid_id', f'{bootstrap_ks_test_output_column}_greater']
            final_gdf = final_gdf.merge(current_ks_result, on='grid_id', how='left')
            current_ks_result = joined_gdf.groupby('grid_id').parallel_apply(apply_mw_test_and_get_p_less).reset_index()
            current_ks_result.columns = ['grid_id', f'{bootstrap_ks_test_output_column}_less']
            final_gdf = final_gdf.merge(current_ks_result, on='grid_id', how='left')
                
    corrected_final_gdf = gpd.GeoDataFrame({'geometry': gpd.GeoSeries([final_gdf.iloc[i]['geometry'] for i in range(len(final_gdf))])}, geometry='geometry', crs=final_gdf.crs)
    corrected_final_gdf = corrected_final_gdf.join(final_gdf.drop('geometry', axis=1))
    if verbose == 2:
        print('Final GeoDataFrame:')
        print(corrected_final_gdf.head())
    
    if output_pkl_path is not None:
        # Save to GeoPackage
        if verbose != 0:
            print(f'Saving to {output_pkl_path}...')
        output_pkl_folder = os.path.dirname(output_pkl_path)
        if not os.path.exists(output_pkl_folder):
            os.makedirs(output_pkl_folder)
        corrected_final_gdf.to_pickle(output_pkl_path)

    return corrected_final_gdf

def time_series_analysis_on_df(
    df: pd.DataFrame,
    time_series_columns: list,
    type_of_analysis: str,
    output_column_name: str,
    unit_scale: float = 1,
    period=None,
    reshape_period=12,
    calculation_mask_columns=None,
    parallel=True,
    mk_test_significance_level=0.05
):
    if calculation_mask_columns is not None:
        assert len(time_series_columns) == len(calculation_mask_columns), 'time_series_columns and calculation_mask_columns must have the same length'
    
    allowed_analysis_types = [
        'mean', 
        'mean_annual_standard_deviation', 
        'mean_annual_mean_first_difference',
        'mean_annual_standard_deviation_percentage', 
        'mean_annual_mean_first_difference_percentage',
        'mean_apportion_entropy_percentage',
        'linear_trend_per_period', 
        'linear_trend_of_standard_deviation_per_period', 
        'linear_trend_of_standard_deviation_percentage_per_period',
        'linear_trend_of_stl_trend_per_period',
        'linear_trend_of_stl_trend_percentage_per_period',
        'linear_trend_of_annual_mean',
        'linear_trend_of_annual_mean_percentage',
        'linear_trend_of_annual_mean_first_difference',
        'linear_trend_of_annual_mean_first_difference_percentage',
        'linear_trend_of_apportion_entropy_percentage',
        'rao_mk_test_on_stl_linear_trend',
        'rao_mk_test_on_standard_deviation',
        'rao_mk_test_on_standard_deviation_percentage',
        'rao_mk_test_on_annual_mean',
        'rao_mk_test_on_annual_mean_first_difference',
        'rao_mk_test_on_annual_mean_first_difference_percentage',
        'rao_mk_test_on_apportion_entropy_percentage',
        'pettitt_test_on_stl_linear_trend',
        'pettitt_test_on_standard_deviation',
        'pettitt_test_on_annual_mean',
        'pettitt_test_on_annual_mean_first_difference',
        'difference_in_annual_mean_std_of_two_periods',
        'difference_in_annual_mean_apportion_entropy_of_two_periods',
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
    
    if type_of_analysis not in allowed_analysis_types:
        print(f'You are using {type_of_analysis} analysis')
        raise ValueError('type_of_analysis must be one of the following:', allowed_analysis_types)
    if type_of_analysis == 'mean':
        def calculate_mean(row):
            return np.mean([row[col]*unit_scale for col in time_series_columns])
        analyze_func = calculate_mean
    
    elif type_of_analysis == 'mean_annual_standard_deviation':
        def calculate_mean_annual_standard_deviation(row):
            areas = np.array([row[col]*unit_scale for col in time_series_columns])
            #reshape areas to (period, numperiods) to calculate the standard deviation of each period
            if calculation_mask_columns is not None:
                masks = np.array([row[col] for col in calculation_mask_columns])
                areas[masks] = np.nan
            areas = areas.reshape(-1, reshape_period)
            annual_standard_deviations = np.nanstd(areas, axis=1)
            return np.nanmean(annual_standard_deviations)
        analyze_func = calculate_mean_annual_standard_deviation
        
    elif type_of_analysis == 'mean_annual_standard_deviation_percentage':
        def calculate_mean_annual_standard_deviation_percentage(row):
            areas = np.array([row[col]*unit_scale for col in time_series_columns])
            #reshape areas to (period, numperiods) to calculate the standard deviation of each period
            if calculation_mask_columns is not None:
                masks = np.array([row[col] for col in calculation_mask_columns])
                areas[masks] = np.nan
            areas = areas.reshape(-1, reshape_period)
            annual_standard_deviations = np.nanstd(areas, axis=1)
            return np.nanmean(annual_standard_deviations)/np.nanmean(areas)
        analyze_func = calculate_mean_annual_standard_deviation_percentage
        
    elif type_of_analysis == 'mean_annual_mean_first_difference':
        def calculate_mean_annual_mean_first_difference(row):
            areas = np.array([row[col]*unit_scale for col in time_series_columns])
            if calculation_mask_columns is not None:
                masks = np.array([row[col] for col in calculation_mask_columns])
                areas[masks] = np.nan
            #reshape areas to (period, numperiods) to calculate the standard deviation of each period
            areas = areas.reshape(-1, reshape_period)
            annual_means = np.nanmean(areas, axis=1)
            annual_means_diff = np.abs(np.diff(annual_means))
            return np.nanmean(annual_means_diff)
        analyze_func = calculate_mean_annual_mean_first_difference
    
    elif type_of_analysis == 'mean_annual_mean_first_difference_percentage':
        def calculate_mean_annual_mean_first_difference_percentage(row):
            areas = np.array([row[col]*unit_scale for col in time_series_columns])
            if calculation_mask_columns is not None:
                masks = np.array([row[col] for col in calculation_mask_columns])
                areas[masks] = np.nan
            #reshape areas to (period, numperiods) to calculate the standard deviation of each period
            areas = areas.reshape(-1, reshape_period)
            annual_means = np.nanmean(areas, axis=1)
            annual_means_diff = np.abs(np.diff(annual_means))
            return np.nanmean(annual_means_diff)/np.nanmean(annual_means[:len(annual_means_diff)])
        analyze_func = calculate_mean_annual_mean_first_difference_percentage
    
    elif type_of_analysis == 'mean_apportion_entropy_percentage':
        def calculate_mean_apportion_entropy_percentage(row):
            areas = np.array([row[col]*unit_scale for col in time_series_columns])
            if calculation_mask_columns is not None:
                masks = np.array([row[col] for col in calculation_mask_columns])
                areas[masks] = np.nan
            #reshape areas to (period, numperiods) to calculate the standard deviation of each period
            areas = areas.reshape(-1, reshape_period)
            annual_sums = np.nansum(areas, axis=1)
            annual_apportion_entropy_percentage = np.zeros(annual_sums.shape)
            for i in range(annual_sums.shape[0]):
                non_nan_area_time_series = areas[i][~np.isnan(areas[i])]
                if len(non_nan_area_time_series) == 0 or annual_sums[i] == 0:
                    continue
                area_pk = non_nan_area_time_series/annual_sums[i]
                annual_apportion_entropy_percentage[i] = entropy(area_pk)
                current_largest_entropy = np.log(len(non_nan_area_time_series))
                annual_apportion_entropy_percentage[i] /= current_largest_entropy
            return np.nanmean(annual_apportion_entropy_percentage)

        analyze_func = calculate_mean_apportion_entropy_percentage
    
    elif type_of_analysis == 'linear_trend_per_period':
        if period is None:
            raise ValueError('period must be provided for linear_trend_per_period analysis')
        def calculate_linear_trend(row):
            return np.polyfit(range(len(time_series_columns)), [row[col]*unit_scale for col in time_series_columns], 1)[0]*period
        analyze_func = calculate_linear_trend
    
    elif type_of_analysis == 'linear_trend_of_standard_deviation_per_period':
        if period is None:
            raise ValueError('period must be provided for linear_trend_of_period_standard_deviation analysis')
        def calculate_linear_trend_of_period_standard_deviation(row):
            areas = np.array([row[col]*unit_scale for col in time_series_columns])
            if calculation_mask_columns is not None:
                masks = np.array([row[col] for col in calculation_mask_columns])
                areas[masks] = np.nan
            #reshape areas to (period, numperiods) to calculate the standard deviation of each period
            areas = areas.reshape(-1, reshape_period)
            standard_deviations = np.nanstd(areas, axis=1)
            return np.polyfit(range(len(standard_deviations)), standard_deviations, 1)[0]
        analyze_func = calculate_linear_trend_of_period_standard_deviation
    
    elif type_of_analysis == 'linear_trend_of_standard_deviation_percentage_per_period':
        if period is None:
            raise ValueError('period must be provided for linear_trend_of_period_standard_deviation analysis')
        def calculate_linear_trend_of_period_standard_deviation_percentage(row):
            areas = np.array([row[col]*unit_scale for col in time_series_columns])
            if calculation_mask_columns is not None:
                masks = np.array([row[col] for col in calculation_mask_columns])
                areas[masks] = np.nan
            #reshape areas to (period, numperiods) to calculate the standard deviation of each period
            areas = areas.reshape(-1, reshape_period)
            standard_deviations = np.nanstd(areas, axis=1)
            mean_std = np.nanmean(standard_deviations)
            return np.polyfit(range(len(standard_deviations)), standard_deviations, 1)[0] / mean_std * 100
        analyze_func = calculate_linear_trend_of_period_standard_deviation_percentage
    
    elif type_of_analysis == 'linear_trend_of_stl_seasonal_max_minus_min_per_period':
        if period is None:
            raise ValueError('period must be provided for linear_trend_of_stl_seasonal_max_minus_min_per_period analysis')
        def calculate_linear_trend_of_stl_seasonal_max_minus_min_per_period(row):
            areas = np.array([row[col]*unit_scale for col in time_series_columns])
            #do stl decomposition on areas and get the seasonal terms
            stl_result = STL(areas, period=period, robust=True).fit()
            seasonal_terms = stl_result.seasonal
            #reshape areas to (period, numperiods) to calculate the standard deviation of each period
            seasonal_terms = seasonal_terms.reshape(-1, reshape_period)
            maxs = np.max(seasonal_terms, axis=1)
            mins = np.min(seasonal_terms, axis=1)
            return np.polyfit(range(len(maxs)), maxs - mins, 1)[0]
        analyze_func = calculate_linear_trend_of_stl_seasonal_max_minus_min_per_period
    
    elif type_of_analysis == 'linear_trend_of_stl_trend_per_period':
        if period is None:
            raise ValueError('period must be provided for linear_trend_of_stl_trend_per_period analysis')
        def calculate_linear_trend_of_stl_trend(row):
            areas = np.array([row[col]*unit_scale for col in time_series_columns])
            #do stl decomposition on areas and get the seasonal terms
            stl_result = STL(areas, period=period, robust=True).fit()
            trend_terms = stl_result.trend
            return np.polyfit(range(len(trend_terms)), trend_terms, 1)[0]*period
        analyze_func = calculate_linear_trend_of_stl_trend
        
    elif type_of_analysis == 'linear_trend_of_stl_trend_percentage_per_period':
        if period is None:
            raise ValueError('period must be provided for linear_trend_of_stl_trend_per_period analysis')
        def calculate_linear_trend_of_stl_trend_percentage_per_period(row):
            areas = np.array([row[col]*unit_scale for col in time_series_columns])
            #do stl decomposition on areas and get the seasonal terms
            stl_result = STL(areas, period=period, robust=True).fit()
            trend_terms = stl_result.trend
            return np.polyfit(range(len(trend_terms)), trend_terms, 1)[0]/np.nanmean(trend_terms)*100
        analyze_func = calculate_linear_trend_of_stl_trend_percentage_per_period
        
    elif type_of_analysis == 'linear_trend_of_annual_mean':
        if period is None:
            raise ValueError('period must be provided for linear_trend_of_annual_mean analysis')
        def calculate_linear_trend_of_annual_mean(row):
            areas = np.array([row[col]*unit_scale for col in time_series_columns])
            if calculation_mask_columns is not None:
                masks = np.array([row[col] for col in calculation_mask_columns])
                areas[masks] = np.nan
            #reshape areas to (period, numperiods) to calculate the standard deviation of each period
            areas = areas.reshape(-1, reshape_period)
            annual_means = np.nanmean(areas, axis=1)
            return np.polyfit(range(len(annual_means)), annual_means, 1)[0]
        analyze_func = calculate_linear_trend_of_annual_mean
        
    elif type_of_analysis == 'linear_trend_of_annual_mean_percentage':
        if period is None:
            raise ValueError('period must be provided for linear_trend_of_annual_mean analysis')
        def calculate_linear_trend_of_annual_mean_percentage(row):
            areas = np.array([row[col]*unit_scale for col in time_series_columns])
            if calculation_mask_columns is not None:
                masks = np.array([row[col] for col in calculation_mask_columns])
                areas[masks] = np.nan
            #reshape areas to (period, numperiods) to calculate the standard deviation of each period
            areas = areas.reshape(-1, reshape_period)
            annual_means = np.nanmean(areas, axis=1)
            return np.polyfit(range(len(annual_means)), annual_means, 1)[0]/np.nanmean(annual_means)*100
        analyze_func = calculate_linear_trend_of_annual_mean_percentage
    
    elif type_of_analysis == 'linear_trend_of_annual_mean_first_difference':
        if period is None:
            raise ValueError('period must be provided for linear_trend_of_annual_mean_first_difference analysis')
        def calculate_linear_trend_of_annual_mean_first_difference(row):
            areas = np.array([row[col]*unit_scale for col in time_series_columns])
            if calculation_mask_columns is not None:
                masks = np.array([row[col] for col in calculation_mask_columns])
                areas[masks] = np.nan
            #reshape areas to (period, numperiods) to calculate the standard deviation of each period
            areas = areas.reshape(-1, reshape_period)
            annual_means = np.nanmean(areas, axis=1)
            annual_means_diff = np.abs(np.diff(annual_means))
            return np.polyfit(range(len(annual_means_diff)), annual_means_diff, 1)[0]
        analyze_func = calculate_linear_trend_of_annual_mean_first_difference
        
    elif type_of_analysis == 'linear_trend_of_annual_mean_first_difference_percentage':
        if period is None:
            raise ValueError('period must be provided for linear_trend_of_annual_mean_first_difference analysis')
        def calculate_linear_trend_of_annual_mean_first_difference_percentage(row):
            areas = np.array([row[col]*unit_scale for col in time_series_columns])
            if calculation_mask_columns is not None:
                masks = np.array([row[col] for col in calculation_mask_columns])
                areas[masks] = np.nan
            #reshape areas to (period, numperiods) to calculate the standard deviation of each period
            areas = areas.reshape(-1, reshape_period)
            annual_means = np.nanmean(areas, axis=1)
            annual_means_diff = np.abs(np.diff(annual_means))
            mean_annual_means_diff = np.nanmean(annual_means_diff)
            return np.polyfit(range(len(annual_means_diff)), annual_means_diff, 1)[0] / mean_annual_means_diff * 100
        analyze_func = calculate_linear_trend_of_annual_mean_first_difference_percentage
        
    elif type_of_analysis == 'linear_trend_of_apportion_entropy_percentage':
        if period is None:
            raise ValueError('period must be provided for linear_trend_of_apportion_entropy_percentage analysis')
        def calculate_linear_trend_of_apportion_entropy_percentage(row):
            areas = np.array([row[col]*unit_scale for col in time_series_columns])
            if calculation_mask_columns is not None:
                masks = np.array([row[col] for col in calculation_mask_columns])
                areas[masks] = np.nan
            #reshape areas to (period, numperiods) to calculate the standard deviation of each period
            areas = areas.reshape(-1, reshape_period)
            annual_sums = np.nansum(areas, axis=1)
            annual_apportion_entropy_percentage = np.zeros(annual_sums.shape)
            for i in range(annual_sums.shape[0]):
                non_nan_area_time_series = areas[i][~np.isnan(areas[i])]
                if len(non_nan_area_time_series) == 0 or annual_sums[i] == 0:
                    continue
                area_pk = non_nan_area_time_series/annual_sums[i]
                annual_apportion_entropy_percentage[i] = entropy(area_pk)
                current_largest_entropy = np.log(len(non_nan_area_time_series))
                annual_apportion_entropy_percentage[i] /= current_largest_entropy
            return np.polyfit(range(len(annual_apportion_entropy_percentage)), annual_apportion_entropy_percentage, 1)[0]
        analyze_func = calculate_linear_trend_of_apportion_entropy_percentage
        
    elif type_of_analysis == 'rao_mk_test_on_stl_linear_trend':
        if period is None:
            raise ValueError('period must be provided for rao_mk_test_on_stl_linear_trend analysis')
        def calculate_rao_mk_test_on_stl_linear_trend(row):
            areas = np.array([row[col]*unit_scale for col in time_series_columns])
            #do stl decomposition on areas and get the seasonal terms
            stl_result = STL(areas, period=period, robust=True).fit()
            trend_terms = stl_result.trend
            try:
                test_results = hamed_rao_modification_test(trend_terms)
                return {'p': test_results.p, 'trend': test_results.trend, 'slope': test_results.slope}
            except ZeroDivisionError:
                return {'p': np.nan, 'trend': np.nan, 'slope': np.nan}
        analyze_func = calculate_rao_mk_test_on_stl_linear_trend
        
    elif type_of_analysis == 'rao_mk_test_on_standard_deviation':
        if period is None:
            raise ValueError('period must be provided for rao_mk_test_on_standard_deviation analysis')
        def calculate_rao_mk_test_on_standard_deviation(row):
            areas = np.array([row[col]*unit_scale for col in time_series_columns])
            #reshape areas to (period, numperiods) to calculate the standard deviation of each period
            if calculation_mask_columns is not None:
                masks = np.array([row[col] for col in calculation_mask_columns])
                areas[masks] = np.nan
            areas = areas.reshape(-1, reshape_period)
            standard_deviations = np.nanstd(areas, axis=1)
            try:
                test_results = hamed_rao_modification_test(standard_deviations, alpha=0.1)
                return {'p': test_results.p, 'trend': test_results.trend, 'slope': test_results.slope}
            except ZeroDivisionError:
                return {'p': np.nan, 'trend': np.nan, 'slope': np.nan}
        analyze_func = calculate_rao_mk_test_on_standard_deviation
        
    elif type_of_analysis == 'rao_mk_test_on_standard_deviation_percentage':
        if period is None:
            raise ValueError('period must be provided for rao_mk_test_on_standard_deviation analysis')
        def calculate_rao_mk_test_on_standard_deviation_percentage(row):
            areas = np.array([row[col]*unit_scale for col in time_series_columns])
            #reshape areas to (period, numperiods) to calculate the standard deviation of each period
            if calculation_mask_columns is not None:
                masks = np.array([row[col] for col in calculation_mask_columns])
                areas[masks] = np.nan
            areas = areas.reshape(-1, reshape_period)
            standard_deviations = np.nanstd(areas, axis=1)
            mean_area = np.nanmean(areas)
            try:
                test_results = hamed_rao_modification_test(standard_deviations/mean_area, alpha=0.1)
                return {'p': test_results.p, 'trend': test_results.trend, 'slope': test_results.slope}
            except ZeroDivisionError:
                return {'p': np.nan, 'trend': np.nan, 'slope': np.nan}
        analyze_func = calculate_rao_mk_test_on_standard_deviation_percentage
        
    elif type_of_analysis == 'rao_mk_test_on_annual_mean':
        if period is None:
            raise ValueError('period must be provided for rao_mk_test_on_annual_mean analysis')
        def calculate_rao_mk_test_on_annual_mean(row):
            areas = np.array([row[col]*unit_scale for col in time_series_columns])
            if calculation_mask_columns is not None:
                masks = np.array([row[col] for col in calculation_mask_columns])
                areas[masks] = np.nan
            #reshape areas to (period, numperiods) to calculate the standard deviation of each period
            areas = areas.reshape(-1, reshape_period)
            annual_means = np.nanmean(areas, axis=1)
            try:
                test_result = hamed_rao_modification_test(annual_means, alpha=0.1)
                return {'p': test_result.p, 'trend': test_result.trend, 'slope': test_result.slope}
            except ZeroDivisionError:
                return {'p': np.nan, 'trend': np.nan, 'slope': np.nan}
        analyze_func = calculate_rao_mk_test_on_annual_mean
        
    elif type_of_analysis == 'rao_mk_test_on_annual_mean_first_difference':
        if period is None:
            raise ValueError('period must be provided for rao_mk_test_on_annual_mean_first_difference analysis')
        def calculate_rao_mk_test_on_annual_mean_first_difference(row):
            areas = np.array([row[col]*unit_scale for col in time_series_columns])
            if calculation_mask_columns is not None:
                masks = np.array([row[col] for col in calculation_mask_columns])
                areas[masks] = np.nan
            #reshape areas to (period, numperiods) to calculate the standard deviation of each period
            areas = areas.reshape(-1, reshape_period)
            annual_means = np.nanmean(areas, axis=1)
            annual_means_diff = np.abs(np.diff(annual_means))
            try:
                test_result = hamed_rao_modification_test(annual_means_diff, alpha=0.1)
                return {'p': test_result.p, 'trend': test_result.trend, 'slope': test_result.slope}
            except ZeroDivisionError:
                return {'p': np.nan, 'trend': np.nan, 'slope': np.nan}
        analyze_func = calculate_rao_mk_test_on_annual_mean_first_difference
        
    elif type_of_analysis == 'rao_mk_test_on_annual_mean_first_difference_percentage':
        if period is None:
            raise ValueError('period must be provided for rao_mk_test_on_annual_mean_first_difference analysis')
        def calculate_rao_mk_test_on_annual_mean_first_difference_percentage(row):
            areas = np.array([row[col]*unit_scale for col in time_series_columns])
            if calculation_mask_columns is not None:
                masks = np.array([row[col] for col in calculation_mask_columns])
                areas[masks] = np.nan
            #reshape areas to (period, numperiods) to calculate the standard deviation of each period
            areas = areas.reshape(-1, reshape_period)
            annual_means = np.nanmean(areas, axis=1)
            annual_means_diff = np.abs(np.diff(annual_means))
            mean_area = np.nanmean(areas)
            try:
                test_result = hamed_rao_modification_test(annual_means_diff/mean_area, alpha=0.1)
                return {'p': test_result.p, 'trend': test_result.trend, 'slope': test_result.slope}
            except ZeroDivisionError:
                return {'p': np.nan, 'trend': np.nan, 'slope': np.nan}
        analyze_func = calculate_rao_mk_test_on_annual_mean_first_difference_percentage
        
    elif type_of_analysis == 'rao_mk_test_on_apportion_entropy_percentage':
        if period is None:
            raise ValueError('period must be provided for rao_mk_test_on_apportion_entropy_percentage analysis')
        def calculate_rao_mk_test_on_apportion_entropy_percentage(row):
            areas = np.array([row[col]*unit_scale for col in time_series_columns])
            if calculation_mask_columns is not None:
                masks = np.array([row[col] for col in calculation_mask_columns])
                areas[masks] = np.nan
            #reshape areas to (period, numperiods) to calculate the standard deviation of each period
            areas = areas.reshape(-1, reshape_period)
            annual_sums = np.nansum(areas, axis=1)
            annual_apportion_entropy_percentage = np.zeros(annual_sums.shape)
            for i in range(annual_sums.shape[0]):
                non_nan_area_time_series = areas[i][~np.isnan(areas[i])]
                if len(non_nan_area_time_series) == 0 or annual_sums[i] == 0:
                    continue
                area_pk = non_nan_area_time_series/annual_sums[i]
                annual_apportion_entropy_percentage[i] = entropy(area_pk)
                current_largest_entropy = np.log(len(non_nan_area_time_series))
                annual_apportion_entropy_percentage[i] /= current_largest_entropy
            try:
                test_result = hamed_rao_modification_test(annual_apportion_entropy_percentage, alpha=0.1)
                return {'p': test_result.p, 'trend': test_result.trend, 'slope': test_result.slope}
            except ZeroDivisionError:
                return {'p': np.nan, 'trend': np.nan, 'slope': np.nan}
        analyze_func = calculate_rao_mk_test_on_apportion_entropy_percentage
        
    elif type_of_analysis == 'pettitt_test_on_stl_linear_trend':
        if period is None:
            raise ValueError('period must be provided for pettitt_test_on_stl_linear_trend analysis')
        def calculate_pettitt_test_on_stl_linear_trend(row):
            areas = np.array([row[col]*unit_scale for col in time_series_columns])
            #do stl decomposition on areas and get the seasonal terms
            stl_result = STL(areas, period=period, robust=True).fit()
            trend_terms = stl_result.trend
            test_result = pettitt_test(trend_terms)
            return {'p': test_result.p, 'cp': test_result.cp}
        analyze_func = calculate_pettitt_test_on_stl_linear_trend
        
    elif type_of_analysis == 'pettitt_test_on_standard_deviation':
        if period is None:
            raise ValueError('period must be provided for pettitt_test_on_standard_deviation analysis')
        def calculate_pettitt_test_on_standard_deviation(row):
            areas = np.array([row[col]*unit_scale for col in time_series_columns])
            #reshape areas to (period, numperiods) to calculate the standard deviation of each period
            if calculation_mask_columns is not None:
                masks = np.array([row[col] for col in calculation_mask_columns])
                areas[masks] = np.nan
            areas = areas.reshape(-1, reshape_period)
            standard_deviations = np.nanstd(areas, axis=1)
            test_result = pettitt_test(standard_deviations)
            return {'p': test_result.p, 'cp': test_result.cp}
        analyze_func = calculate_pettitt_test_on_standard_deviation
    
    elif type_of_analysis == 'pettitt_test_on_annual_mean':
        if period is None:
            raise ValueError('period must be provided for pettitt_test_on_annual_mean analysis')
        def calculate_pettitt_test_on_annual_mean(row):
            areas = np.array([row[col]*unit_scale for col in time_series_columns])
            if calculation_mask_columns is not None:
                masks = np.array([row[col] for col in calculation_mask_columns])
                areas[masks] = np.nan
            #reshape areas to (period, numperiods) to calculate the standard deviation of each period
            areas = areas.reshape(-1, reshape_period)
            annual_means = np.nanmean(areas, axis=1)
            test_result = pettitt_test(annual_means)
            return {'p': test_result.p, 'cp': test_result.cp}
        analyze_func = calculate_pettitt_test_on_annual_mean
        
    elif type_of_analysis == 'pettitt_test_on_annual_mean_first_difference':
        if period is None:
            raise ValueError('period must be provided for pettitt_test_on_annual_mean_first_difference analysis')
        def calculate_pettitt_test_on_annual_mean_first_difference(row):
            areas = np.array([row[col]*unit_scale for col in time_series_columns])
            if calculation_mask_columns is not None:
                masks = np.array([row[col] for col in calculation_mask_columns])
                areas[masks] = np.nan
            #reshape areas to (period, numperiods) to calculate the standard deviation of each period
            areas = areas.reshape(-1, reshape_period)
            annual_means = np.nanmean(areas, axis=1)
            annual_means_diff = np.abs(np.diff(annual_means))
            test_result = pettitt_test(annual_means_diff)
            return {'p': test_result.p, 'cp': test_result.cp}
        analyze_func = calculate_pettitt_test_on_annual_mean_first_difference
        
    elif type_of_analysis == 'linear_trend_of_annual_mean':
        if period is None:
            raise ValueError('period must be provided for linear_trend_of_annual_mean analysis')
        def calculate_linear_trend_of_annual_mean(row):
            areas = np.array([row[col]*unit_scale for col in time_series_columns])
            if calculation_mask_columns is not None:
                masks = np.array([row[col] for col in calculation_mask_columns])
                areas[masks] = np.nan
            #reshape areas to (period, numperiods) to calculate the standard deviation of each period
            areas = areas.reshape(-1, reshape_period)
            annual_means = np.nanmean(areas, axis=1)
            return np.polyfit(range(len(annual_means)), annual_means, 1)[0]
        analyze_func = calculate_linear_trend_of_annual_mean
       
    elif type_of_analysis == 'linear_trend_of_annual_mean_first_difference':
        if period is None:
            raise ValueError('period must be provided for linear_trend_of_annual_mean_first_difference analysis')
        def calculate_linear_trend_of_annual_mean_first_difference(row):
            areas = np.array([row[col]*unit_scale for col in time_series_columns])
            if calculation_mask_columns is not None:
                masks = np.array([row[col] for col in calculation_mask_columns])
                areas[masks] = np.nan
            #reshape areas to (period, numperiods) to calculate the standard deviation of each period
            areas = areas.reshape(-1, reshape_period)
            annual_means = np.nanmean(areas, axis=1)
            annual_means_diff = np.abs(np.diff(annual_means))
            return np.polyfit(range(len(annual_means_diff)), annual_means_diff, 1)[0]
        analyze_func = calculate_linear_trend_of_annual_mean_first_difference
        
    elif type_of_analysis == 'difference_in_annual_mean_std_of_two_periods':
        if period is None:
            raise ValueError('period must be provided for difference_in_annual_mean_std_of_two_periods analysis')
        def calculate_difference_in_annual_mean_std_of_two_periods(row):
            areas = np.array([row[col]*unit_scale for col in time_series_columns])
            if calculation_mask_columns is not None:
                masks = np.array([row[col] for col in calculation_mask_columns])
                areas[masks] = np.nan
            #reshape areas to (period, numperiods) to calculate the standard deviation of each period
            areas = areas.reshape(-1, reshape_period)
            annual_means = np.nanmean(areas, axis=1)
            std_of_first_period = np.nanstd(annual_means[:11])
            std_of_second_period = np.nanstd(annual_means[12:])
            if np.isnan(std_of_first_period) or np.isnan(std_of_second_period):
                return np.nan
            else:
                return std_of_second_period - std_of_first_period
        analyze_func = calculate_difference_in_annual_mean_std_of_two_periods
    
    elif type_of_analysis == 'difference_in_annual_mean_apportion_entropy_of_two_periods':
        if period is None:
            raise ValueError('period must be provided for difference_in_annual_mean_apportion_entropy_of_two_periods analysis')
        def calculate_difference_in_annual_mean_apportion_entropy_of_two_periods(row):
            areas = np.array([row[col]*unit_scale for col in time_series_columns])
            if calculation_mask_columns is not None:
                masks = np.array([row[col] for col in calculation_mask_columns])
                areas[masks] = np.nan
            #reshape areas to (period, numperiods) to calculate the standard deviation of each period
            areas = areas.reshape(-1, reshape_period)
            annual_means = np.nanmean(areas, axis=1)
            first_period_annual_sum = np.nansum(areas[:11])
            second_period_annual_sum = np.nansum(areas[12:])
            first_period_non_nan_area_time_series = areas[:11][~np.isnan(areas[:11])]
            second_period_non_nan_area_time_series = areas[12:][~np.isnan(areas[12:])]
            if len(first_period_non_nan_area_time_series) == 0 or len(second_period_non_nan_area_time_series) == 0:
                return np.nan
            first_period_area_pk = first_period_non_nan_area_time_series/first_period_annual_sum
            second_period_area_pk = second_period_non_nan_area_time_series/second_period_annual_sum
            first_period_largest_entropy = np.log(len(first_period_non_nan_area_time_series))
            second_period_largest_entropy = np.log(len(second_period_non_nan_area_time_series))
            first_period_entropy_percentage = entropy(first_period_area_pk)/first_period_largest_entropy
            second_period_entropy_percentage = entropy(second_period_area_pk)/second_period_largest_entropy
            if np.isnan(first_period_entropy_percentage) or np.isnan(second_period_entropy_percentage):
                return np.nan
            else:
                return second_period_entropy_percentage - first_period_entropy_percentage
        analyze_func = calculate_difference_in_annual_mean_apportion_entropy_of_two_periods
    
    elif type_of_analysis == 'difference_in_annual_std_of_two_periods':
        if period is None:
            raise ValueError('period must be provided for difference_in_annual_std_of_two_periods analysis')
        def calculate_difference_in_annual_std_of_two_periods(row):
            areas = np.array([row[col]*unit_scale for col in time_series_columns])
            if calculation_mask_columns is not None:
                masks = np.array([row[col] for col in calculation_mask_columns])
                areas[masks] = np.nan
            #reshape areas to (period, numperiods) to calculate the standard deviation of each period
            areas = areas.reshape(-1, reshape_period)
            annual_std = np.nanstd(areas, axis=1)
            first_period_annual_std = annual_std[:11]
            sceond_period_annual_std = annual_std[12:]
            first_period_mean_std = np.nanmean(first_period_annual_std)
            second_period_mean_std = np.nanmean(sceond_period_annual_std)
            if np.isnan(first_period_mean_std) or np.isnan(second_period_mean_std):
                return np.nan
            else:
                return second_period_mean_std - first_period_mean_std
        analyze_func = calculate_difference_in_annual_std_of_two_periods
    
    elif type_of_analysis == 'relative_difference_in_annual_std_of_two_periods':
        if period is None:
            raise ValueError('period must be provided for relative_difference_in_annual_std_of_two_periods analysis')
        def calculate_relative_difference_in_annual_std_of_two_periods(row):
            areas = np.array([row[col]*unit_scale for col in time_series_columns])
            if calculation_mask_columns is not None:
                masks = np.array([row[col] for col in calculation_mask_columns])
                areas[masks] = np.nan
            #reshape areas to (period, numperiods) to calculate the standard deviation of each period
            areas = areas.reshape(-1, reshape_period)
            annual_std = np.nanstd(areas, axis=1)
            first_period_annual_std = annual_std[:11]
            sceond_period_annual_std = annual_std[12:]
            first_period_mean_std = np.nanmean(first_period_annual_std)
            second_period_mean_std = np.nanmean(sceond_period_annual_std)
            if np.isnan(first_period_mean_std) or np.isnan(second_period_mean_std):
                return np.nan
            if first_period_mean_std <= 1e-5 or second_period_mean_std <= 1e-5:
                return np.nan
            else:
                return 2*(second_period_mean_std - first_period_mean_std)/(first_period_mean_std+second_period_mean_std) * 100
        analyze_func = calculate_relative_difference_in_annual_std_of_two_periods
    
    elif type_of_analysis == 'difference_in_annual_mean_first_difference_std_of_two_periods':
        if period is None:
            raise ValueError('period must be provided for difference_in_annual_mean_first_difference_of_two_periods analysis')
        def calculate_difference_in_annual_mean_first_difference_std_of_two_periods(row):
            areas = np.array([row[col]*unit_scale for col in time_series_columns])
            if calculation_mask_columns is not None:
                masks = np.array([row[col] for col in calculation_mask_columns])
                areas[masks] = np.nan
            #reshape areas to (period, numperiods) to calculate the standard deviation of each period
            areas = areas.reshape(-1, reshape_period)
            annual_means = np.nanmean(areas, axis=1)
            first_period_annual_means_diff = np.diff(annual_means[:11])
            second_period_annual_means_diff = np.diff(annual_means[12:])
            first_period_diff_std = np.nanstd(first_period_annual_means_diff)
            second_period_diff_std = np.nanstd(second_period_annual_means_diff)
            if np.isnan(first_period_diff_std) or np.isnan(second_period_diff_std):
                return np.nan
            else:
                return second_period_diff_std - first_period_diff_std
        analyze_func = calculate_difference_in_annual_mean_first_difference_std_of_two_periods
    
    elif type_of_analysis == 'relative_difference_in_annual_mean_first_difference_std_of_two_periods':
        if period is None:
            raise ValueError('period must be provided for relative_difference_in_annual_mean_first_difference_std_of_two_periods analysis')
        def calculate_relative_difference_in_annual_mean_first_difference_std_of_two_periods(row):
            areas = np.array([row[col]*unit_scale for col in time_series_columns])
            if calculation_mask_columns is not None:
                masks = np.array([row[col] for col in calculation_mask_columns])
                areas[masks] = np.nan
            #reshape areas to (period, numperiods) to calculate the standard deviation of each period
            areas = areas.reshape(-1, reshape_period)
            annual_means = np.nanmean(areas, axis=1)
            first_period_annual_means_diff = np.diff(annual_means[:11])
            second_period_annual_means_diff = np.diff(annual_means[12:])
            first_period_diff_std = np.nanstd(first_period_annual_means_diff)
            second_period_diff_std = np.nanstd(second_period_annual_means_diff)
            if np.isnan(first_period_diff_std) or np.isnan(second_period_diff_std):
                return np.nan
            if first_period_diff_std <= 1e-5 or second_period_diff_std <= 1e-5:
                return np.nan
            else:
                return 2*(second_period_diff_std - first_period_diff_std)/(first_period_diff_std + second_period_diff_std) * 100
        analyze_func = calculate_relative_difference_in_annual_mean_first_difference_std_of_two_periods
    
    elif type_of_analysis == 'difference_in_monthly_first_differece_abs_annual_mean_of_two_periods':
        if period is None:
            raise ValueError('period must be provided for difference_in_monthly_first_differece_annual_mean_of_two_periods analysis')
        def calculate_difference_in_monthly_first_differece_abs_annual_mean_of_two_periods(row):
            areas = np.array([row[col]*unit_scale for col in time_series_columns])
            if calculation_mask_columns is not None:
                masks = np.array([row[col] for col in calculation_mask_columns])
                areas[masks] = np.nan
            #reshape areas to (period, numperiods) to calculate the standard deviation of each period
            areas = areas.reshape(-1, reshape_period)
            monthly_diffs = np.diff(areas)
            monthly_diffs_abs = np.abs(monthly_diffs)
            monthly_diffs_abs_annual_mean = np.nanmean(monthly_diffs_abs, axis=1)
            monthly_diffs_abs_annual_mean_first_period = np.nanmean(monthly_diffs_abs_annual_mean[:11])
            monthly_diffs_abs_annual_mean_second_period = np.nanmean(monthly_diffs_abs_annual_mean[12:])
            
            if np.isnan(monthly_diffs_abs_annual_mean_first_period) or np.isnan(monthly_diffs_abs_annual_mean_second_period):
                return np.nan
            else:
                return monthly_diffs_abs_annual_mean_second_period - monthly_diffs_abs_annual_mean_first_period
            
        analyze_func = calculate_difference_in_monthly_first_differece_abs_annual_mean_of_two_periods
    
    elif type_of_analysis == 'mean_annual_std_percentage_of_first_period':
        def calculate_mean_annual_std_percentage_of_first_period(row):
            areas = np.array([row[col]*unit_scale for col in time_series_columns])
            if calculation_mask_columns is not None:
                masks = np.array([row[col] for col in calculation_mask_columns])
                areas[masks] = np.nan
            #reshape areas to (period, numperiods) to calculate the standard deviation of each period
            areas = areas.reshape(-1, reshape_period)
            annual_std = np.nanstd(areas, axis=1)
            annual_std_first_period = np.nanmean(annual_std[:11])
            annual_std_second_period = np.nanmean(annual_std[12:])
            if np.isnan(annual_std_first_period) or np.isnan(annual_std_second_period):
                return np.nan
            return annual_std_first_period / (annual_std_first_period + annual_std_second_period) * 100
        analyze_func = calculate_mean_annual_std_percentage_of_first_period
    
    elif type_of_analysis == 'mean_annual_std_percentage_of_second_period':
        def calculate_mean_annual_std_percentage_of_second_period(row):
            areas = np.array([row[col]*unit_scale for col in time_series_columns])
            if calculation_mask_columns is not None:
                masks = np.array([row[col] for col in calculation_mask_columns])
                areas[masks] = np.nan
            #reshape areas to (period, numperiods) to calculate the standard deviation of each period
            areas = areas.reshape(-1, reshape_period)
            annual_std = np.nanstd(areas, axis=1)
            annual_std_first_period = np.nanmean(annual_std[:11])
            annual_std_second_period = np.nanmean(annual_std[12:])
            if np.isnan(annual_std_first_period) or np.isnan(annual_std_second_period):
                return np.nan
            return annual_std_second_period / (annual_std_first_period + annual_std_second_period) * 100
        analyze_func = calculate_mean_annual_std_percentage_of_second_period
    
    elif type_of_analysis == 'mean_annual_mean_first_difference_percentage_of_first_period':
        def calculate_mean_annual_mean_first_difference_percentage_of_first_period(row):
            areas = np.array([row[col]*unit_scale for col in time_series_columns])
            if calculation_mask_columns is not None:
                masks = np.array([row[col] for col in calculation_mask_columns])
                areas[masks] = np.nan
            #reshape areas to (period, numperiods) to calculate the standard deviation of each period
            areas = areas.reshape(-1, reshape_period)
            annual_means = np.nanmean(areas, axis=1)
            annual_means_first_diff_first_period = np.abs(np.diff(annual_means[:11]))
            annual_means_first_diff_second_period = np.abs(np.diff(annual_means[12:]))
            annual_means_first_diff_first_period_mean = np.nanmean(annual_means_first_diff_first_period)
            annual_means_first_diff_second_period_mean = np.nanmean(annual_means_first_diff_second_period)
            if np.isnan(annual_means_first_diff_first_period_mean) or np.isnan(annual_means_first_diff_second_period_mean):
                return np.nan
            return annual_means_first_diff_first_period_mean / (annual_means_first_diff_first_period_mean + annual_means_first_diff_second_period_mean) * 100
        analyze_func = calculate_mean_annual_mean_first_difference_percentage_of_first_period
        
    elif type_of_analysis == 'mean_annual_mean_first_difference_percentage_of_second_period':
        def calculate_mean_annual_mean_first_difference_percentage_of_second_period(row):
            areas = np.array([row[col]*unit_scale for col in time_series_columns])
            if calculation_mask_columns is not None:
                masks = np.array([row[col] for col in calculation_mask_columns])
                areas[masks] = np.nan
            #reshape areas to (period, numperiods) to calculate the standard deviation of each period
            areas = areas.reshape(-1, reshape_period)
            annual_means = np.nanmean(areas, axis=1)
            annual_means_first_diff_first_period = np.abs(np.diff(annual_means[:11]))
            annual_means_first_diff_second_period = np.abs(np.diff(annual_means[12:]))
            annual_means_first_diff_first_period_mean = np.nanmean(annual_means_first_diff_first_period)
            annual_means_first_diff_second_period_mean = np.nanmean(annual_means_first_diff_second_period)
            if np.isnan(annual_means_first_diff_first_period_mean) or np.isnan(annual_means_first_diff_second_period_mean):
                return np.nan
            return annual_means_first_diff_second_period_mean / (annual_means_first_diff_first_period_mean + annual_means_first_diff_second_period_mean) * 100
        analyze_func = calculate_mean_annual_mean_first_difference_percentage_of_second_period
    
    elif type_of_analysis == 'mean_annual_mean_percentage_of_first_period':
        def calculate_mean_annual_mean_percentage_of_first_period(row):
            areas = np.array([row[col]*unit_scale for col in time_series_columns])
            if calculation_mask_columns is not None:
                masks = np.array([row[col] for col in calculation_mask_columns])
                areas[masks] = np.nan
            #reshape areas to (period, numperiods) to calculate the standard deviation of each period
            areas = areas.reshape(-1, reshape_period)
            annual_means = np.nanmean(areas, axis=1)
            annual_means_first_period_mean = np.nanmean(annual_means[:11])
            annual_means_second_period_mean = np.nanmean(annual_means[12:])
            if np.isnan(annual_means_first_period_mean) or np.isnan(annual_means_second_period_mean):
                return np.nan
            return annual_means_first_period_mean / (annual_means_first_period_mean + annual_means_second_period_mean) * 100
        analyze_func = calculate_mean_annual_mean_percentage_of_first_period
    
    elif type_of_analysis == 'mean_annual_mean_percentage_of_second_period':
        def calculate_mean_annual_mean_percentage_of_second_period(row):
            areas = np.array([row[col]*unit_scale for col in time_series_columns])
            if calculation_mask_columns is not None:
                masks = np.array([row[col] for col in calculation_mask_columns])
                areas[masks] = np.nan
            #reshape areas to (period, numperiods) to calculate the standard deviation of each period
            areas = areas.reshape(-1, reshape_period)
            annual_means = np.nanmean(areas, axis=1)
            annual_means_first_period_mean = np.nanmean(annual_means[:11])
            annual_means_second_period_mean = np.nanmean(annual_means[12:])
            if np.isnan(annual_means_first_period_mean) or np.isnan(annual_means_second_period_mean):
                return np.nan
            return annual_means_second_period_mean / (annual_means_first_period_mean + annual_means_second_period_mean) * 100
        analyze_func = calculate_mean_annual_mean_percentage_of_second_period
    
    if 'rao_mk_test' in type_of_analysis or 'pettitt_test' in type_of_analysis:
        if parallel:
            result = df.parallel_apply(analyze_func, axis=1)
        else:
            result = df.apply(analyze_func, axis=1)
        result = result.to_dict()
        result_df = pd.DataFrame(result).T
        result_df.columns = [f'{output_column_name}_{col}' for col in result_df.columns]
        if 'mk_test' in type_of_analysis:
            significance_level = mk_test_significance_level
        result_df[f'{output_column_name}_significant'] = result_df[f'{output_column_name}_p'] < significance_level
        for col in result_df.columns:
            df[col] = result_df[col]
    
    else:
        if parallel:
            df[output_column_name] = df.parallel_apply(analyze_func, axis=1)
        else:
            df[output_column_name] = df.apply(analyze_func, axis=1)
    
    return df