import netCDF4 as nc
import numpy as np
from datetime import datetime
from dateutil import relativedelta
import os
import geopandas as gpd
import pandas as pd
from joblib import Parallel, delayed
from time import sleep
import pdb
from pyproj import CRS
from sklearn.neighbors import LocalOutlierFactor

def get_ice_height_from_lakeensemblr_nc(
    nc_file_path:str
):
    with nc.Dataset(nc_file_path) as data:
        ice_height = data.variables['ice_height'][:, 0:3, 0, 0, 0] # should be 11688 x 3 and a numpy masked array
        if not isinstance(ice_height, np.ma.MaskedArray):
            raise ValueError(f'ice_height is not a masked array, instead it is {type(ice_height)}')
    return ice_height

def generate_monthly_ice_mask_for_area(
    nc_file_path:str,
    model_start_date:str,
    model_end_date:str,
    area_start_date:str,
    area_end_date:str,
    date_fmt:str='%Y-%m-%d',
    model_selection_strategy:str='min',
    missing_value=1e20
):
    if model_selection_strategy not in ['min', 'median', 'max']:
        raise ValueError(f'Invalid model_selection_strategy: {model_selection_strategy}, must be one of ["min", "median", "max"]')
    if model_selection_strategy == 'min':
        model_selection_func = np.min
    elif model_selection_strategy == 'median':
        model_selection_func = np.median
    elif model_selection_strategy == 'max':
        model_selection_func = np.max
    
    model_start_date = datetime.strptime(model_start_date, date_fmt)
    model_end_date = datetime.strptime(model_end_date, date_fmt)
    area_start_date = datetime.strptime(area_start_date, date_fmt)
    area_end_date = datetime.strptime(area_end_date, date_fmt)
    
    start_day_index = (area_start_date - model_start_date).days
    end_day_index = (area_end_date - model_start_date).days if area_end_date <= model_end_date else (model_end_date - model_start_date).days
    
    ice_height_oi = get_ice_height_from_lakeensemblr_nc(nc_file_path)
    
    current_date = area_start_date
    monthly_days_with_ice_ratio = []
    while current_date < area_end_date and current_date < model_end_date:
        current_month_start_index = (current_date - model_start_date).days
        current_month_end_index = (current_date + relativedelta.relativedelta(months=1) - model_start_date).days
        
        current_month_ice_height = ice_height_oi[current_month_start_index:current_month_end_index, :]
        current_month_ice_height_per_day = model_selection_func(current_month_ice_height, axis=1)
        
        current_month_days_with_ice_array = current_month_ice_height_per_day > 0
        current_month_days_with_ice = np.sum(current_month_days_with_ice_array)
        current_month_total_valid_days = current_month_days_with_ice_array.count()
        
        current_month_days_with_ice_ratio = current_month_days_with_ice / current_month_total_valid_days
        monthly_days_with_ice_ratio.append(current_month_days_with_ice_ratio)
        
        current_date = current_date + relativedelta.relativedelta(months=1)
        
    if area_end_date > model_end_date:
        five_years_before_model_end_date = model_end_date - relativedelta.relativedelta(years=5)
        monthly_days_with_ice_ratio_five_years_before_index = (five_years_before_model_end_date.year - area_start_date.year) * 12 + five_years_before_model_end_date.month - area_start_date.month
        monthly_days_with_ice_ratio_recent_five_years = monthly_days_with_ice_ratio[monthly_days_with_ice_ratio_five_years_before_index:]
        assert len(monthly_days_with_ice_ratio_recent_five_years) == 60, f'Length of monthly_days_with_ice_ratio_recent_five_years is not 60, instead it is {len(monthly_days_with_ice_ratio_recent_five_years)}'
        monthly_days_with_ice_ratio_recent_five_years = np.array(monthly_days_with_ice_ratio_recent_five_years).reshape(-1, 12)
        monthly_days_with_ice_ratio_recent_five_years_avg = np.mean(monthly_days_with_ice_ratio_recent_five_years, axis=0).tolist()
        
        years_to_fill = area_end_date.year - model_end_date.year
        for i in range(years_to_fill):
            monthly_days_with_ice_ratio = monthly_days_with_ice_ratio + monthly_days_with_ice_ratio_recent_five_years_avg
            
    assert len(monthly_days_with_ice_ratio) == ((area_end_date.year - area_start_date.year) * 12 + area_end_date.month - area_start_date.month), f'Length of monthly_days_with_ice_ratio is not {((area_end_date.year - area_start_date.year) * 12 + area_end_date.month - area_start_date.month)}, instead it is {len(monthly_days_with_ice_ratio)}'
    
    return monthly_days_with_ice_ratio

def get_ice_existence_from_mylake_txt(
    txt_file_path:str
):
    mylake_df = pd.read_csv(txt_file_path)
    columns = mylake_df.columns.to_list()
    columns.remove('zz')
    surface_temp = mylake_df.iloc[0][columns].to_numpy()
    
    ice_existence = surface_temp == 0
    return ice_existence

def generate_monthly_ice_mask_for_area_from_mylake_txt(
    txt_file_path:str,
    model_start_date:str,
    model_end_date:str,
    area_start_date:str,
    area_end_date:str,
    date_fmt:str='%Y-%m-%d'
):
    model_start_date = datetime.strptime(model_start_date, date_fmt)
    model_end_date = datetime.strptime(model_end_date, date_fmt)
    area_start_date = datetime.strptime(area_start_date, date_fmt)
    area_end_date = datetime.strptime(area_end_date, date_fmt)
    
    start_day_index = (area_start_date - model_start_date).days
    end_day_index = (area_end_date - model_start_date).days if area_end_date <= model_end_date else (model_end_date - model_start_date).days
    
    ice_existence_array = get_ice_existence_from_mylake_txt(txt_file_path)
    
    current_date = area_start_date
    monthly_days_with_ice_ratio = []
    while current_date < area_end_date and current_date < model_end_date:
        current_month_start_index = (current_date - model_start_date).days
        current_month_end_index = (current_date + relativedelta.relativedelta(months=1) - model_start_date).days
        
        current_month_days_with_ice_array = ice_existence_array[current_month_start_index:current_month_end_index]
        
        current_month_days_with_ice = np.sum(current_month_days_with_ice_array)
        current_month_total_valid_days = current_month_days_with_ice_array.count()
        
        current_month_days_with_ice_ratio = current_month_days_with_ice / current_month_total_valid_days
        monthly_days_with_ice_ratio.append(current_month_days_with_ice_ratio)
        
        current_date = current_date + relativedelta.relativedelta(months=1)
        
    if area_end_date > model_end_date:
        five_years_before_model_end_date = model_end_date - relativedelta.relativedelta(years=5)
        monthly_days_with_ice_ratio_five_years_before_index = (five_years_before_model_end_date.year - area_start_date.year) * 12 + five_years_before_model_end_date.month - area_start_date.month
        monthly_days_with_ice_ratio_recent_five_years = monthly_days_with_ice_ratio[monthly_days_with_ice_ratio_five_years_before_index:]
        assert len(monthly_days_with_ice_ratio_recent_five_years) == 60, f'Length of monthly_days_with_ice_ratio_recent_five_years is not 60, instead it is {len(monthly_days_with_ice_ratio_recent_five_years)}'
        monthly_days_with_ice_ratio_recent_five_years = np.array(monthly_days_with_ice_ratio_recent_five_years).reshape(-1, 12)
        monthly_days_with_ice_ratio_recent_five_years_avg = np.mean(monthly_days_with_ice_ratio_recent_five_years, axis=0).tolist()
        
        years_to_fill = area_end_date.year - model_end_date.year
        for i in range(years_to_fill):
            monthly_days_with_ice_ratio = monthly_days_with_ice_ratio + monthly_days_with_ice_ratio_recent_five_years_avg
            
    assert len(monthly_days_with_ice_ratio) == ((area_end_date.year - area_start_date.year) * 12 + area_end_date.month - area_start_date.month), f'Length of monthly_days_with_ice_ratio is not {((area_end_date.year - area_start_date.year) * 12 + area_end_date.month - area_start_date.month)}, instead it is {len(monthly_days_with_ice_ratio)}'
    
    return monthly_days_with_ice_ratio
    
def mask_area_when_frozen(
    area_concatenated_df,
    area_columns,
    lake_id_column,
    ice_ratio_lake_id_array: np.ndarray,
    ice_ratio_array: np.ndarray,
    ice_ratio_threshold=0.3,
    mask_column_prefix=None,
    use_parallel=True,
    n_jobs=-1,
    outlier_detection=False,
    lof_n_neighbors=12,
    lake_area_column_name=None
):
    if outlier_detection and lake_area_column_name is None:
        raise ValueError('lake_area_column_name must be provided when outlier_detection is True.')
    def fill_nan_values(lake_areas):
        # Create a copy of the array to avoid changing the original data
        filled_areas = lake_areas.copy().astype(float)
        # Forward fill: fill NaN with the latest previous non-NaN value
        valid_idx = np.where(~np.isnan(filled_areas))[0]  # indices of non-NaN values
        if valid_idx.size == 0:
            return filled_areas # all nans
        for i in range(1, len(filled_areas)):
            if np.isnan(filled_areas[i]) and i > valid_idx[0]:  # if NaN and not at the start
                filled_areas[i] = filled_areas[i - 1]
        
        # Backward fill: fill NaN at the start of the array
        for i in range(valid_idx[0]):
            filled_areas[i] = filled_areas[valid_idx[0]]

        return filled_areas
    
    area_concatenated_df['Area_masked'] = False
        
    if mask_column_prefix is not None:
        mask_flag_columns = {f'{mask_column_prefix}_{col}': pd.Series([False] * len(area_concatenated_df), index=area_concatenated_df.index) for col in area_columns}
        area_concatenated_df = pd.concat([area_concatenated_df, pd.DataFrame(mask_flag_columns)], axis=1)
    lake_ids = area_concatenated_df[lake_id_column].unique()
    assert isinstance(lake_ids, np.ndarray), f'lake_ids is not a numpy array, instead it is {type(lake_ids)}'
    mask = np.isin(ice_ratio_lake_id_array, lake_ids)
    ice_ratio_lake_id_array = ice_ratio_lake_id_array[mask]
    ice_ratio_array = ice_ratio_array[mask]
    
    area_mask_columns = [f'{mask_column_prefix}_{col}' for col in area_columns] if mask_column_prefix is not None else None
    def process_row(index, row, ice_ratio_array, ice_ratio_threshold, mask_column_prefix, area_columns):
        if ice_ratio_array is not None:
            row['Area_masked'] = True
            if outlier_detection and lake_area_column_name is not None:
                
                area_array = row[area_columns].to_numpy().flatten().astype(float)
                # if all 0 in area_array, then return
                if np.all(area_array == 0):
                    return index, row
                num_years = len(area_array) // 12
                if len(area_array) % 12 != 0:
                    raise ValueError('Length of area array is not a multiple of 12.')
                ice_anomaly = ice_ratio_array > ice_ratio_threshold
                non_frozen_annual_median = np.nanmedian(np.where(ice_anomaly, np.nan, area_array).reshape(-1, 12), axis=1)
                non_frozen_annual_median = np.where(np.isnan(non_frozen_annual_median), np.nanmedian(area_array.reshape(-1, 12), axis=1), non_frozen_annual_median)
                non_frozen_annual_median = np.repeat(non_frozen_annual_median, 12)
                non_frozen_annual_std = np.nanstd(np.where(ice_anomaly, np.nan, area_array).reshape(-1, 12), axis=1)
                non_frozen_annual_std = np.where(np.isnan(non_frozen_annual_std), np.nanstd(area_array.reshape(-1, 12), axis=1), non_frozen_annual_std)
                non_frozen_annual_std = np.repeat(non_frozen_annual_std, 12)
                monthly_ice_anomaly = ice_anomaly.reshape(-1, 12).copy()
                for row_idx in range(monthly_ice_anomaly.shape[0]):
                    current_year_monthly_ice_anomaly = monthly_ice_anomaly[row_idx]
                    current_year_monthly_ice_anomaly_false_indices = np.where(~current_year_monthly_ice_anomaly)[0]
                    if current_year_monthly_ice_anomaly_false_indices.size != 0:
                        last_false_index = current_year_monthly_ice_anomaly_false_indices[-1]
                        current_year_monthly_ice_anomaly[last_false_index] = True
                        first_false_index = current_year_monthly_ice_anomaly_false_indices[0]
                        current_year_monthly_ice_anomaly[first_false_index] = True
                        if current_year_monthly_ice_anomaly_false_indices.size > 2:
                            second_last_false_index = current_year_monthly_ice_anomaly_false_indices[-2]
                            current_year_monthly_ice_anomaly[second_last_false_index] = True
                            second_first_false_index = current_year_monthly_ice_anomaly_false_indices[1]
                            current_year_monthly_ice_anomaly[second_first_false_index] = True
                monthly_ice_anomaly = monthly_ice_anomaly.flatten()
                filtered_area_array = np.where(ice_anomaly, non_frozen_annual_median, area_array)
                #possible_frozen_area_anomaly = np.logical_or(filtered_area_array > non_frozen_annual_median + 3.5*non_frozen_annual_std, filtered_area_array < non_frozen_annual_median - 3.5*non_frozen_annual_std)
                #possible_frozen_area_anomaly = possible_frozen_area_anomaly & monthly_ice_anomaly
                #filtered_area_array = np.where(possible_frozen_area_anomaly, non_frozen_annual_median, filtered_area_array)
                time_steps = np.arange(len(filtered_area_array))
                if np.max(filtered_area_array) == np.min(filtered_area_array):
                    return index, row
                X = np.concatenate([time_steps.reshape(-1, 1), filtered_area_array.reshape(-1, 1)], axis=1)
                X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
                X[:, 0] = X[:, 0] * 5
                lof_scores = LocalOutlierFactor(n_neighbors=lof_n_neighbors).fit_predict(X)
                lof_mask = lof_scores == -1
                lof_mask = lof_mask.flatten()
                lof_mask = monthly_ice_anomaly & lof_mask
                
                area_and_lof_mask = ice_anomaly | lof_mask# | possible_frozen_area_anomaly
                masked_area_array = np.where(area_and_lof_mask, np.nan, area_array)
                
                if mask_column_prefix is not None:
                    row[area_mask_columns] = area_and_lof_mask
                
                filled_area_array = fill_nan_values(masked_area_array)
                row[area_columns] = filled_area_array
            
            else:    
                for i, area_column in enumerate(area_columns):
                    if ice_ratio_array[i] > ice_ratio_threshold:
                        row[area_column] = np.nan
                        if mask_column_prefix is not None:
                            row[f'{mask_column_prefix}_{area_column}'] = True
                        
                # fill the nan values
                current_lake_areas = row[area_columns].to_numpy()
                current_filled_lake_areas = fill_nan_values(current_lake_areas)
                for i, area_column in enumerate(area_columns):
                    if np.isnan(row[area_column]):
                        row[area_column] = current_filled_lake_areas[i]
        return index, row
    
    if use_parallel:
        print('Using parallel processing to mask areas...')
        
        index_and_row_and_ice_ratios = [(index, row, ice_ratio_array[np.where(ice_ratio_lake_id_array == row[lake_id_column])[0]][0] if np.where(ice_ratio_lake_id_array == row[lake_id_column])[0].size != 0 else None) 
                          for index, row in area_concatenated_df.iterrows()]

        results = Parallel(n_jobs=n_jobs)(delayed(process_row)(index, row, ice_ratio_array, ice_ratio_threshold, mask_column_prefix, area_columns)
                                      for index, row, ice_ratio_array in index_and_row_and_ice_ratios)
        print('parallel processing done.')
        print('Updating the DataFrame...')
        # Collect updated rows
        updated_rows = {index: row for index, row in results}
        updated_df = pd.DataFrame.from_dict(updated_rows, orient='index')
        
        # Bulk update the original DataFrame
        area_concatenated_df.update(updated_df)
        
    else:
        for index, row in area_concatenated_df.iterrows():
            current_lake_id = row[lake_id_column]
            current_lake_id_index = np.where(ice_ratio_lake_id_array == current_lake_id)[0]
            if current_lake_id_index.size != 0:
                current_ice_ratio = ice_ratio_array[current_lake_id_index][0]
                process_row(index, row, current_ice_ratio, ice_ratio_threshold, mask_column_prefix, area_columns)
    return area_concatenated_df

def fix_unmasked_lakes(
    area_partially_masked_gdf: gpd.GeoDataFrame,
    area_columns,
    lake_id_column,
    lake_geom_column,
    area_mask_flag_column,
    ice_ratio_lake_id_array: np.ndarray,
    ice_ratio_array: np.ndarray,
    ice_ratio_threshold=0.3,
    mask_column_prefix=None,
    use_parallel=True,
    outlier_detection=False,
    lof_n_neighbors=12,
    lake_area_column_name=None
):
    if outlier_detection and lake_area_column_name is None:
        raise ValueError('lake_area_column_name must be provided when outlier_detection is True.')
    
    print('Fixing unmasked lakes...')
    masked_gdf = area_partially_masked_gdf[area_partially_masked_gdf[area_mask_flag_column]]
    unmasked_gdf = area_partially_masked_gdf[~area_partially_masked_gdf[area_mask_flag_column]]
    sindex = masked_gdf.sindex
    if unmasked_gdf.empty:
        print('No unmasked lakes to fix.')
        return area_partially_masked_gdf
    def get_nearest_lake_id(row):
        # Get the nearest geometry's index
        nearest_idx = sindex.nearest(row[lake_geom_column])[1][0]
        # Get the lake id of the nearest geometry
        nearest_lake_id = masked_gdf.iloc[nearest_idx][lake_id_column]
        return nearest_lake_id
    unmasked_gdf['Nearest_masked_lake_id'] = unmasked_gdf.apply(get_nearest_lake_id, axis=1)
    
    fixed_unmasked_gdf = mask_area_when_frozen(
        area_concatenated_df=unmasked_gdf,
        area_columns=area_columns,
        lake_id_column='Nearest_masked_lake_id',
        ice_ratio_lake_id_array=ice_ratio_lake_id_array.copy(),
        ice_ratio_array=ice_ratio_array.copy(),
        ice_ratio_threshold=ice_ratio_threshold,
        mask_column_prefix=mask_column_prefix,
        use_parallel=use_parallel,
        outlier_detection=outlier_detection,
        lof_n_neighbors=lof_n_neighbors,
        lake_area_column_name=lake_area_column_name
    )
    print('HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    fixed_unmasked_gdf.drop(columns=['Nearest_masked_lake_id'], inplace=True)
    
    area_partially_masked_gdf.update(fixed_unmasked_gdf)
    
    return area_partially_masked_gdf

def cloud_cover_ratio_calculation(
    cloud_cover_area_df,
    lake_shp_gdf,
    cloud_cover_area_columns,
    cloud_cover_lake_id_column_name,
    lake_shp_lake_id_column_name,
    lake_shp_lake_geom_column_name
):
    print(lake_shp_gdf.crs)
    for index, row in cloud_cover_area_df.iterrows():
        current_lake_id = row[cloud_cover_lake_id_column_name]
        current_lake_geom = lake_shp_gdf[lake_shp_gdf[lake_shp_lake_id_column_name] == current_lake_id][lake_shp_lake_geom_column_name].iloc[0]
        current_lake_geom_area = current_lake_geom.area
        current_cloud_cover_area_array = row[cloud_cover_area_columns].to_numpy()
        current_cloud_cover_ratio_array = current_cloud_cover_area_array / current_lake_geom_area
        cloud_cover_area_df.loc[index, cloud_cover_area_columns] = current_cloud_cover_ratio_array
        
    return cloud_cover_area_df

def mask_area_when_cloud_contaminated(
    area_concatenated_df,
    cloud_cover_ratio_df,
    area_df_lake_id_column_name,
    cloud_cover_ratio_df_lake_id_column_name,
    area_columns,
    cloud_cover_ratio_columns,
    cloud_cover_threshold=0.025,
    already_frozen_masked=False,
    outlier_detection=False,
    lof_n_neighbors=12
):
    if already_frozen_masked:
        raise ValueError('This function is not designed to mask already frozen lakes.')
    
    def fill_nan_values(lake_areas):
        # If all nan, then return all zero
        if np.all(np.isnan(lake_areas)):
            return np.zeros_like(lake_areas)
        
        # Create a copy of the array to avoid changing the original data
        filled_areas = lake_areas.copy().astype(float)
        
        # Create a pandas Series from the array
        series = pd.Series(filled_areas)
        
        # Perform local linear interpolation
        series_interpolated = series.interpolate(method='linear', limit_direction='both')
        
        # Convert the Series back to a numpy array
        filled_areas = series_interpolated.to_numpy()
        
        return filled_areas
    
    for index, row in area_concatenated_df.iterrows():
        current_lake_id = int(row[area_df_lake_id_column_name])
        current_cloud_cover_ratio_row = cloud_cover_ratio_df[cloud_cover_ratio_df[cloud_cover_ratio_df_lake_id_column_name] == current_lake_id]
        current_area_array = row[area_columns].to_numpy().flatten()
        if np.all(current_area_array == 0) or np.max(current_area_array) == np.min(current_area_array):
            continue
        time_steps = np.arange(len(current_area_array))
        X = np.concatenate([time_steps.reshape(-1, 1), current_area_array.reshape(-1, 1)], axis=1)
        X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X[:, 0] = X[:, 0] * 5
        lof_scores = LocalOutlierFactor(n_neighbors=lof_n_neighbors).fit_predict(X)
        lof_anomaly = lof_scores == -1
        lof_anomaly = lof_anomaly.flatten()
        current_cloud_cover_ratio_array = current_cloud_cover_ratio_row[cloud_cover_ratio_columns].to_numpy().flatten()
        if len(current_area_array) != len(current_cloud_cover_ratio_array):
            print(f'Error for lake {current_lake_id}: Length of area array is {len(current_area_array)}, length of cloud cover ratio array is {len(current_cloud_cover_ratio_array)}')
            continue
        current_area_array = np.where(current_cloud_cover_ratio_array > cloud_cover_threshold, np.nan, current_area_array)
        lof_and_cloud_mask = np.logical_and(lof_anomaly, current_cloud_cover_ratio_array > 0.001)
        current_area_array = np.where(lof_and_cloud_mask, np.nan, current_area_array)
        current_area_array = fill_nan_values(current_area_array)
        area_concatenated_df.loc[index, area_columns] = current_area_array
        
    return area_concatenated_df