import pandas as pd
import numpy as np
import geopandas as gpd 
import os
import sys
from pandarallel import pandarallel
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.tsa.seasonal import STL

def low_water_extreme_flag_generation(
    lake_lse_df,
    area_columns,
    area_mask_columns,
    low_water_extreme_flag_column_names,
    extreme_threshold=0.1,
    parallel=False,
    parallel_num_workers=6,
    detrend=True
):
    """Generate flags indicating low water extreme events for each lake.
    
    This function analyzes lake surface area time series data to identify low water extreme events
    based on a specified threshold percentile. The data can optionally be detrended before analysis.
    
    Args:
        lake_lse_df (pd.DataFrame): DataFrame containing lake surface area time series data
        area_columns (list): Column names containing surface area measurements
        area_mask_columns (list): Column names containing mask flags for invalid measurements
        low_water_extreme_flag_column_names (list): Names for output columns containing extreme flags
        extreme_threshold (float, optional): Percentile threshold for extreme events (0-1). Defaults to 0.1.
        parallel (bool, optional): Whether to use parallel processing. Defaults to False.
        parallel_num_workers (int, optional): Number of parallel workers if parallel=True. Defaults to 6.
        detrend (bool, optional): Whether to remove annual mean trend before analysis. Defaults to True.
        
    Returns:
        pd.DataFrame: DataFrame containing boolean flags indicating low water extreme events
    """

    def generate_low_water_extreme_flag(row):
        areas = np.array([row[col] for col in area_columns])
        masks = np.array([row[col] for col in area_mask_columns])
        areas[masks] = np.nan
        areas = areas.reshape(-1, 12)
        area_annual_means = np.nanmean(areas, axis=1)
        if detrend:
            areas = areas - area_annual_means[:, np.newaxis]
        else:
            pass
        n_cols = areas.shape[1]
        low_water_extreme_flags = np.zeros(areas.shape, dtype=bool)
        for i in range(n_cols):
            current_col = areas[:, i]
            n_valid_values = np.sum(~np.isnan(current_col))
            if n_valid_values < 1/extreme_threshold:
                pass
            else:
                extreme_threshold_value = np.nanpercentile(current_col, extreme_threshold*100)
                low_water_extreme_flags[:, i] = current_col < extreme_threshold_value
        low_water_extreme_flags = low_water_extreme_flags.flatten()
        return pd.Series(low_water_extreme_flags)
    if parallel:
        pandarallel.initialize(progress_bar=False, nb_workers=parallel_num_workers, use_memory_fs = False)
        low_water_extreme_flags_df = lake_lse_df.parallel_apply(generate_low_water_extreme_flag, axis=1)
    else:
        low_water_extreme_flags_df = lake_lse_df.apply(generate_low_water_extreme_flag, axis=1)
    low_water_extreme_flags_df.columns = low_water_extreme_flag_column_names
    return low_water_extreme_flags_df
        
def high_water_extreme_flag_generation(
    lake_lse_df,
    area_columns,
    area_mask_columns,
    high_water_extreme_flag_column_names,
    extreme_threshold=0.9,
    parallel=False,
    parallel_num_workers=6,
    detrend=True
):
    """Generate boolean flags indicating high water extreme events for each lake.
    
    This function analyzes lake surface area time series data to identify high water extreme events
    based on a percentile threshold. It can optionally remove annual mean trends and use parallel processing.
    
    Args:
        lake_lse_df (pd.DataFrame): DataFrame containing lake surface area time series data
        area_columns (list): Names of columns containing area measurements
        area_mask_columns (list): Names of columns containing boolean masks for invalid data
        high_water_extreme_flag_column_names (list): Names for output columns containing extreme flags
        extreme_threshold (float, optional): Percentile threshold for extreme events (0-1). Defaults to 0.9.
        parallel (bool, optional): Whether to use parallel processing. Defaults to False.
        parallel_num_workers (int, optional): Number of parallel workers if parallel=True. Defaults to 6.
        detrend (bool, optional): Whether to remove annual mean trend before analysis. Defaults to True.
        
    Returns:
        pd.DataFrame: DataFrame containing boolean flags indicating high water extreme events
    """

    def generate_high_water_extreme_flag(row):
        areas = np.array([row[col] for col in area_columns])
        masks = np.array([row[col] for col in area_mask_columns])
        areas[masks] = np.nan
        areas = areas.reshape(-1, 12)
        area_annual_means = np.nanmean(areas, axis=1)
        if detrend:
            areas = areas - area_annual_means[:, np.newaxis]
        else:
            pass
        n_cols = areas.shape[1]
        high_water_extreme_flags = np.zeros(areas.shape, dtype=bool)
        for i in range(n_cols):
            current_col = areas[:, i]
            n_valid_values = np.sum(~np.isnan(current_col))
            if n_valid_values < 1/(1-extreme_threshold):
                pass
            else:
                extreme_threshold_value = np.nanpercentile(current_col, extreme_threshold*100)
                high_water_extreme_flags[:, i] = current_col > extreme_threshold_value
        high_water_extreme_flags = high_water_extreme_flags.flatten()
        return pd.Series(high_water_extreme_flags)
    if parallel:
        pandarallel.initialize(progress_bar=False, nb_workers=parallel_num_workers, use_memory_fs = False)
        high_water_extreme_flags_df = lake_lse_df.parallel_apply(generate_high_water_extreme_flag, axis=1)
    else:
        high_water_extreme_flags_df = lake_lse_df.apply(generate_high_water_extreme_flag, axis=1)
    high_water_extreme_flags_df.columns = high_water_extreme_flag_column_names
    return high_water_extreme_flags_df
        
def increases_in_low_water_extremes_and_pval_of_two_periods(
    lake_lse_df_with_low_water_extreme_flags,
    first_period_low_water_extreme_flag_columns,
    first_period_area_mask_columns,
    second_period_low_water_extreme_flag_columns,
    second_period_area_mask_columns,
    output_column_prefix,
    significant_level=0.05,
    parallel=False,
    parallel_num_workers=6,
    reshape_period=12
):
    """Calculate increases in low water extreme events between two time periods.

    Args:
        lake_lse_df_with_low_water_extreme_flags (pd.DataFrame): DataFrame containing low water extreme flags
        first_period_low_water_extreme_flag_columns (list): Column names for first period extreme flags
        first_period_area_mask_columns (list): Column names for first period area masks
        second_period_low_water_extreme_flag_columns (list): Column names for second period extreme flags
        second_period_area_mask_columns (list): Column names for second period area masks
        output_column_prefix (str): Prefix for output column names
        significant_level (float, optional): P-value threshold for significance. Defaults to 0.05.
        parallel (bool, optional): Whether to use parallel processing. Defaults to False.
        parallel_num_workers (int, optional): Number of parallel workers if parallel=True. Defaults to 6.
        reshape_period (int, optional): Number of observations per year for reshaping. Defaults to 12.

    Returns:
        pd.DataFrame: DataFrame containing:
            - Frequency change in extreme events between periods
            - Count change in extreme events between periods  
            - P-value from proportion test comparing periods
    """

    def calculate_increases_in_low_water_extremes_and_pval(row):
        first_period_low_water_extreme_flags = row[first_period_low_water_extreme_flag_columns].to_numpy().flatten()
        first_period_area_masks = row[first_period_area_mask_columns].to_numpy().flatten().astype(bool)
        second_period_low_water_extreme_flags = row[second_period_low_water_extreme_flag_columns].to_numpy().flatten()
        second_period_area_masks = row[second_period_area_mask_columns].to_numpy().flatten().astype(bool)
        first_period_low_water_extreme_flags = first_period_low_water_extreme_flags[~first_period_area_masks]
        second_period_low_water_extreme_flags = second_period_low_water_extreme_flags[~second_period_area_masks]
        n_first_period_extreme = np.sum(first_period_low_water_extreme_flags)
        n_second_period_extreme = np.sum(second_period_low_water_extreme_flags)
        nobs_first_period = len(first_period_low_water_extreme_flags)
        nobs_second_period = len(second_period_low_water_extreme_flags)
        if nobs_first_period == 0 or nobs_second_period == 0:
            return pd.Series([np.nan, np.nan])
        stat, pval = proportions_ztest([n_first_period_extreme, n_second_period_extreme], [nobs_first_period, nobs_second_period])
        increase_in_extremes_frequency = n_second_period_extreme / nobs_second_period - n_first_period_extreme / nobs_first_period
        increase_in_extremes_count = n_second_period_extreme - n_first_period_extreme
        return pd.Series([increase_in_extremes_frequency, increase_in_extremes_count, pval])
    if parallel:
        pandarallel.initialize(progress_bar=False, nb_workers=parallel_num_workers, use_memory_fs = False)
        increases_in_extremes_df = lake_lse_df_with_low_water_extreme_flags.parallel_apply(calculate_increases_in_low_water_extremes_and_pval, axis=1)
    else:
        increases_in_extremes_df = lake_lse_df_with_low_water_extreme_flags.apply(calculate_increases_in_low_water_extremes_and_pval, axis=1)
    increases_in_extremes_df.columns = [f'{output_column_prefix}_frequency', f'{output_column_prefix}_count', f'{output_column_prefix}_pval']
    return increases_in_extremes_df

def increases_in_high_water_extremes_and_pval_of_two_periods(
    lake_lse_df_with_high_water_extreme_flags,
    first_period_high_water_extreme_flag_columns,
    first_period_area_mask_columns,
    second_period_high_water_extreme_flag_columns,
    second_period_area_mask_columns,
    output_column_prefix,
    significant_level=0.05,
    parallel=False,
    parallel_num_workers=6,
    reshape_period=12
):
    """Calculate increases in high water extreme events between two time periods.

    Args:
        lake_lse_df_with_high_water_extreme_flags (pd.DataFrame): DataFrame containing high water extreme flags
        first_period_high_water_extreme_flag_columns (list): Column names for first period extreme flags
        first_period_area_mask_columns (list): Column names for first period area masks
        second_period_high_water_extreme_flag_columns (list): Column names for second period extreme flags
        second_period_area_mask_columns (list): Column names for second period area masks
        output_column_prefix (str): Prefix for output column names
        significant_level (float, optional): P-value threshold for significance. Defaults to 0.05.
        parallel (bool, optional): Whether to use parallel processing. Defaults to False.
        parallel_num_workers (int, optional): Number of parallel workers if parallel=True. Defaults to 6.
        reshape_period (int, optional): Number of observations per year for reshaping. Defaults to 12.

    Returns:
        pd.DataFrame: DataFrame containing:
            - Frequency change in extreme events between periods
            - Count change in extreme events between periods
            - P-value from proportion test comparing periods
    """

    def calculate_increases_in_high_water_extremes_and_pval(row):
        first_period_high_water_extreme_flags = row[first_period_high_water_extreme_flag_columns].to_numpy().flatten()
        first_period_area_masks = row[first_period_area_mask_columns].to_numpy().flatten().astype(bool)
        second_period_high_water_extreme_flags = row[second_period_high_water_extreme_flag_columns].to_numpy().flatten()
        second_period_area_masks = row[second_period_area_mask_columns].to_numpy().flatten().astype(bool)
        first_period_high_water_extreme_flags = first_period_high_water_extreme_flags[~first_period_area_masks]
        second_period_high_water_extreme_flags = second_period_high_water_extreme_flags[~second_period_area_masks]
        n_first_period_extreme = np.sum(first_period_high_water_extreme_flags)
        n_second_period_extreme = np.sum(second_period_high_water_extreme_flags)
        nobs_first_period = len(first_period_high_water_extreme_flags)
        nobs_second_period = len(second_period_high_water_extreme_flags)
        if nobs_first_period == 0 or nobs_second_period == 0:
            return pd.Series([np.nan, np.nan])
        stat, pval = proportions_ztest([n_first_period_extreme, n_second_period_extreme], [nobs_first_period, nobs_second_period])
        increase_in_extremes_frequency = n_second_period_extreme / nobs_second_period - n_first_period_extreme / nobs_first_period
        increase_in_extremes_count = n_second_period_extreme - n_first_period_extreme
        return pd.Series([increase_in_extremes_frequency, increase_in_extremes_count, pval])
    if parallel:
        pandarallel.initialize(progress_bar=False, nb_workers=parallel_num_workers, use_memory_fs = False)
        increases_in_extremes_df = lake_lse_df_with_high_water_extreme_flags.parallel_apply(calculate_increases_in_high_water_extremes_and_pval, axis=1)
    else:
        increases_in_extremes_df = lake_lse_df_with_high_water_extreme_flags.apply(calculate_increases_in_high_water_extremes_and_pval, axis=1)
    increases_in_extremes_df.columns = [f'{output_column_prefix}_frequency', f'{output_column_prefix}_count', f'{output_column_prefix}_pval']
    return increases_in_extremes_df

def mean_deviations_of_low_water_extremes_from_annual_means_period(
    lake_lse_df_with_low_water_extreme_flags,
    period_low_water_extreme_flag_columns,
    period_area_columns,
    period_area_mask_columns,
    output_column_name,
    reshape_period=12,
    parallel=False,
    parallel_num_workers=6,
    unit_scale=1e-6
):
    """Deprecated.
    """

    def calculate_mean_deviations_of_low_water_extremes_from_annual_means(row):
        period_low_water_extreme_flags = row[period_low_water_extreme_flag_columns].to_numpy().flatten().astype(bool)
        period_areas = row[period_area_columns].to_numpy().flatten().astype(float)*unit_scale
        period_area_masks = row[period_area_mask_columns].to_numpy().flatten().astype(bool)
        assert period_areas.shape == period_low_water_extreme_flags.shape, 'Areas and low water extreme flags shape does not match'
        assert period_areas.shape == period_area_masks.shape, 'Areas and area masks shape does not match'
        if np.sum(~period_area_masks) == 0:
            return pd.Series([np.nan])
        if np.sum(period_low_water_extreme_flags) == 0:
            return pd.Series([np.nan])
        period_areas = np.where(period_area_masks, np.nan, period_areas)
        period_areas = period_areas.reshape(-1, reshape_period)
        period_low_water_extreme_flags = period_low_water_extreme_flags.reshape(-1, reshape_period)
        
        period_annual_means = np.nanmean(period_areas, axis=1)
        period_annual_means_array = np.repeat(period_annual_means, reshape_period).reshape(-1, reshape_period)
        assert period_annual_means_array.shape == period_areas.shape, 'Annual means array shape does not match with period areas shape'
        deviations_from_annual_means = period_areas - period_annual_means_array
        deviations_from_annual_means_of_low_water_extremes = deviations_from_annual_means[period_low_water_extreme_flags].flatten()
        mean_deviations_from_annual_means_of_low_water_extremes = np.nanmean(deviations_from_annual_means_of_low_water_extremes)
        if np.isnan(mean_deviations_from_annual_means_of_low_water_extremes):
            print(f'areas = {period_areas}')
            print(f'low_water_extreme_flags = {period_low_water_extreme_flags}')
            print(f'area_masks = {period_area_masks}')
        return pd.Series([mean_deviations_from_annual_means_of_low_water_extremes])
    if parallel:
        pandarallel.initialize(progress_bar=False, nb_workers=parallel_num_workers, use_memory_fs = False)
        mean_deviations_df = lake_lse_df_with_low_water_extreme_flags.parallel_apply(calculate_mean_deviations_of_low_water_extremes_from_annual_means, axis=1)
    else:
        mean_deviations_df = lake_lse_df_with_low_water_extreme_flags.apply(calculate_mean_deviations_of_low_water_extremes_from_annual_means, axis=1)
        
    mean_deviations_df.columns = [output_column_name]
    return mean_deviations_df
    
def max_deviations_of_low_water_extremes_from_annual_means_period(
    lake_lse_df_with_low_water_extreme_flags,
    period_low_water_extreme_flag_columns,
    period_area_columns,
    period_area_mask_columns,
    output_column_name,
    reshape_period=12,
    parallel=False,
    parallel_num_workers=6,
    unit_scale=1e-6
):
    """Deprecated.
    """

    def calculate_max_deviations_of_low_water_extremes_from_annual_means(row):
        period_low_water_extreme_flags = row[period_low_water_extreme_flag_columns].to_numpy().flatten().astype(bool)
        period_areas = row[period_area_columns].to_numpy().flatten().astype(float)*unit_scale
        period_area_masks = row[period_area_mask_columns].to_numpy().flatten().astype(bool)
        assert period_areas.shape == period_low_water_extreme_flags.shape, 'Areas and low water extreme flags shape does not match'
        assert period_areas.shape == period_area_masks.shape, 'Areas and area masks shape does not match'
        if np.sum(~period_area_masks) == 0:
            return pd.Series([np.nan])
        if np.sum(period_low_water_extreme_flags) == 0:
            return pd.Series([np.nan])
        period_areas = np.where(period_area_masks, np.nan, period_areas)
        period_areas = period_areas.reshape(-1, reshape_period)
        period_low_water_extreme_flags = period_low_water_extreme_flags.reshape(-1, reshape_period)
        
        period_annual_means = np.nanmean(period_areas, axis=1)
        period_annual_means_array = np.repeat(period_annual_means, reshape_period).reshape(-1, reshape_period)
        assert period_annual_means_array.shape == period_areas.shape, 'Annual means array shape does not match with period areas shape'
        deviations_from_annual_means = period_areas - period_annual_means_array
        deviations_from_annual_means_of_low_water_extremes = deviations_from_annual_means[period_low_water_extreme_flags].flatten()
        max_deviations_from_annual_means_of_low_water_extremes = np.nanmax(deviations_from_annual_means_of_low_water_extremes)
        if np.isnan(max_deviations_from_annual_means_of_low_water_extremes):
            print(f'areas = {period_areas}')
            print(f'low_water_extreme_flags = {period_low_water_extreme_flags}')
            print(f'area_masks = {period_area_masks}')
        return pd.Series([max_deviations_from_annual_means_of_low_water_extremes])
    if parallel:
        pandarallel.initialize(progress_bar=False, nb_workers=parallel_num_workers, use_memory_fs = False)
        max_deviations_df = lake_lse_df_with_low_water_extreme_flags.parallel_apply(calculate_max_deviations_of_low_water_extremes_from_annual_means, axis=1)
    else:
        max_deviations_df = lake_lse_df_with_low_water_extreme_flags.apply(calculate_max_deviations_of_low_water_extremes_from_annual_means, axis=1)
    
    max_deviations_df.columns = [output_column_name]
    return max_deviations_df
    
def min_deviations_of_low_water_extremes_from_annual_means_period(
    lake_lse_df_with_low_water_extreme_flags,
    period_low_water_extreme_flag_columns,
    period_area_columns,
    period_area_mask_columns,
    output_column_name,
    reshape_period=12,
    parallel=False,
    parallel_num_workers=6,
    unit_scale=1e-6
):
    """Calculate minimum (negative, for absolute will be maximum) deviations of low water extremes from annual means for each lake.

    Args:
        lake_lse_df_with_low_water_extreme_flags (pandas.DataFrame): DataFrame containing lake data with low water extreme flags
        period_low_water_extreme_flag_columns (list): Column names containing low water extreme flags for a given period
        period_area_columns (list): Column names containing area values for a given period
        period_area_mask_columns (list): Column names containing area mask values for a given period
        output_column_name (str): Name of output column for minimum deviations
        reshape_period (int, optional): Number of months per period for reshaping. Defaults to 12.
        parallel (bool, optional): Whether to use parallel processing. Defaults to False.
        parallel_num_workers (int, optional): Number of workers for parallel processing. Defaults to 6.
        unit_scale (float, optional): Scale factor to apply to area values. Defaults to 1e-6.

    Returns:
        pandas.DataFrame: DataFrame containing minimum deviations from annual means for low water extremes
    """

    def calculate_min_deviations_of_low_water_extremes_from_annual_means(row):
        period_low_water_extreme_flags = row[period_low_water_extreme_flag_columns].to_numpy().flatten().astype(bool)
        period_areas = row[period_area_columns].to_numpy().flatten().astype(float)*unit_scale
        period_area_masks = row[period_area_mask_columns].to_numpy().flatten().astype(bool)
        assert period_areas.shape == period_low_water_extreme_flags.shape, 'Areas and low water extreme flags shape does not match'
        assert period_areas.shape == period_area_masks.shape, 'Areas and area masks shape does not match'
        if np.sum(~period_area_masks) == 0:
            return pd.Series([np.nan])
        if np.sum(period_low_water_extreme_flags) == 0:
            return pd.Series([np.nan])
        period_areas = np.where(period_area_masks, np.nan, period_areas)
        period_areas = period_areas.reshape(-1, reshape_period)
        period_low_water_extreme_flags = period_low_water_extreme_flags.reshape(-1, reshape_period)
        
        period_annual_means = np.nanmean(period_areas, axis=1)
        period_annual_means_array = np.repeat(period_annual_means, reshape_period).reshape(-1, reshape_period)
        assert period_annual_means_array.shape == period_areas.shape, 'Annual means array shape does not match with period areas shape'
        deviations_from_annual_means = period_areas - period_annual_means_array
        deviations_from_annual_means_of_low_water_extremes = deviations_from_annual_means[period_low_water_extreme_flags].flatten()
        min_deviations_from_annual_means_of_low_water_extremes = np.nanmin(deviations_from_annual_means_of_low_water_extremes)
        if np.isnan(min_deviations_from_annual_means_of_low_water_extremes):
            print(f'areas = {period_areas}')
            print(f'low_water_extreme_flags = {period_low_water_extreme_flags}')
            print(f'area_masks = {period_area_masks}')
        return pd.Series([min_deviations_from_annual_means_of_low_water_extremes])
    if parallel:
        pandarallel.initialize(progress_bar=False, nb_workers=parallel_num_workers, use_memory_fs = False)
        min_deviations_df = lake_lse_df_with_low_water_extreme_flags.parallel_apply(calculate_min_deviations_of_low_water_extremes_from_annual_means, axis=1)
    else:
        min_deviations_df = lake_lse_df_with_low_water_extreme_flags.apply(calculate_min_deviations_of_low_water_extremes_from_annual_means, axis=1)
    
    min_deviations_df.columns = [output_column_name]
    return min_deviations_df
    
def max_deviations_of_high_water_extremes_from_annual_means_period(
    lake_lse_df_with_high_water_extreme_flags,
    period_high_water_extreme_flag_columns,
    period_area_columns,
    period_area_mask_columns,
    output_column_name,
    reshape_period=12,
    parallel=False,
    parallel_num_workers=6,
    unit_scale=1e-6
):
    """Calculate maximum deviations of high water extremes from annual means for each lake.
    
    Parameters
    ----------
    lake_lse_df_with_high_water_extreme_flags : pandas.DataFrame
        DataFrame containing lake area data and high water extreme flags
    period_high_water_extreme_flag_columns : list
        Column names containing high water extreme flags
    period_area_columns : list 
        Column names containing area values
    period_area_mask_columns : list
        Column names containing area mask values
    output_column_name : str
        Name of output column for maximum deviations
    reshape_period : int, optional
        Number of months to reshape data into, defaults to 12
    parallel : bool, optional
        Whether to use parallel processing, defaults to False
    parallel_num_workers : int, optional
        Number of workers for parallel processing, defaults to 6
    unit_scale : float, optional
        Factor to scale area values by, defaults to 1e-6
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing maximum deviations from annual means for high water extremes
        
    Notes
    -----
    - Areas are reshaped into periods of reshape_period months
    - NaN values are returned for lakes with no valid data or no high water extremes
    - Area values are scaled by unit_scale factor before calculations
    """

    def calculate_max_deviations_of_high_water_extremes_from_annual_means(row):
        period_high_water_extreme_flags = row[period_high_water_extreme_flag_columns].to_numpy().flatten().astype(bool)
        period_areas = row[period_area_columns].to_numpy().flatten().astype(float)*unit_scale
        period_area_masks = row[period_area_mask_columns].to_numpy().flatten().astype(bool)
        assert period_areas.shape == period_high_water_extreme_flags.shape, 'Areas and high water extreme flags shape does not match'
        assert period_areas.shape == period_area_masks.shape, 'Areas and area masks shape does not match'
        if np.sum(~period_area_masks) == 0:
            return pd.Series([np.nan])
        if np.sum(period_high_water_extreme_flags) == 0:
            return pd.Series([np.nan])
        period_areas = np.where(period_area_masks, np.nan, period_areas)
        period_areas = period_areas.reshape(-1, reshape_period)
        period_high_water_extreme_flags = period_high_water_extreme_flags.reshape(-1, reshape_period)
        
        period_annual_means = np.nanmean(period_areas, axis=1)
        period_annual_means_array = np.repeat(period_annual_means, reshape_period).reshape(-1, reshape_period)
        assert period_annual_means_array.shape == period_areas.shape, 'Annual means array shape does not match with period areas shape'
        deviations_from_annual_means = period_areas - period_annual_means_array
        deviations_from_annual_means_of_high_water_extremes = deviations_from_annual_means[period_high_water_extreme_flags].flatten()
        max_deviations_from_annual_means_of_high_water_extremes = np.nanmax(deviations_from_annual_means_of_high_water_extremes)
        if np.isnan(max_deviations_from_annual_means_of_high_water_extremes):
            print(f'areas = {period_areas}')
            print(f'high_water_extreme_flags = {period_high_water_extreme_flags}')
            print(f'area_masks = {period_area_masks}')
        return pd.Series([max_deviations_from_annual_means_of_high_water_extremes])
    if parallel:
        pandarallel.initialize(progress_bar=False, nb_workers=parallel_num_workers, use_memory_fs = False)
        max_deviations_df = lake_lse_df_with_high_water_extreme_flags.parallel_apply(calculate_max_deviations_of_high_water_extremes_from_annual_means, axis=1)
    else:
        max_deviations_df = lake_lse_df_with_high_water_extreme_flags.apply(calculate_max_deviations_of_high_water_extremes_from_annual_means, axis=1)
    
    max_deviations_df.columns = [output_column_name]
    return max_deviations_df
    
def calculate_stl_trend_term_anomaly_stds(
    lake_lse_df,
    area_columns,
    output_column_name,
    period_for_stl=13,
    parallel=False,
    parallel_num_workers=6,
    unit_scale=1e-6
):
    def calculate_stl_trend_term__std(row):
        areas = row[area_columns].to_numpy().flatten().astype(float)*unit_scale
        stl_result = STL(areas, period=period_for_stl, robust=True).fit()
        trend_terms = stl_result.trend 
        trend_terms_std = np.nanstd(trend_terms)
        return pd.Series([trend_terms_std])
        
    if parallel:
        pandarallel.initialize(progress_bar=False, nb_workers=parallel_num_workers, use_memory_fs = False)
        trend_terms_stds_df = lake_lse_df.parallel_apply(calculate_stl_trend_term__std, axis=1)
    else:
        trend_terms_stds_df = lake_lse_df.apply(calculate_stl_trend_term__std, axis=1)
    trend_terms_stds_df.columns = [output_column_name]
    return trend_terms_stds_df

def calculate_annual_means_stds(
    lake_lse_df,
    area_columns,
    area_mask_columns,
    output_column_name,
    reshape_period=12,
    parallel=False,
    parallel_num_workers=6,
    unit_scale=1e-6
):
    """Calculate standard deviation of annual means for each lake.

    Args:
        lake_lse_df (pandas.DataFrame): DataFrame containing lake data
        area_columns (list): Column names containing area values
        area_mask_columns (list): Column names containing area mask values
        output_column_name (str): Name of output column for standard deviation
        reshape_period (int, optional): Number of months per period for reshaping. Defaults to 12.
        parallel (bool, optional): Whether to use parallel processing. Defaults to False.
        parallel_num_workers (int, optional): Number of workers for parallel processing. Defaults to 6.
        unit_scale (float, optional): Scale factor to apply to area values. Defaults to 1e-6.

    Returns:
        pandas.DataFrame: DataFrame containing standard deviation of annual means
    """
    def calculate_annual_means_std(row):
        areas = row[area_columns].to_numpy().flatten().astype(float)*unit_scale
        area_masks = row[area_mask_columns].to_numpy().flatten().astype(bool)
        areas = np.where(area_masks, np.nan, areas)
        areas = areas.reshape(-1, reshape_period)
        annual_means = np.nanmean(areas, axis=1)
        annual_means_std = np.nanstd(annual_means)
        return pd.Series([annual_means_std])
    if parallel:
        pandarallel.initialize(progress_bar=False, nb_workers=parallel_num_workers, use_memory_fs = False)
        annual_means_stds_df = lake_lse_df.parallel_apply(calculate_annual_means_std, axis=1)
    else:
        annual_means_stds_df = lake_lse_df.apply(calculate_annual_means_std, axis=1)
    annual_means_stds_df.columns = [output_column_name]
    return annual_means_stds_df