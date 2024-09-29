import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandarallel import pandarallel

print('Initializing pandarallel...')
pandarallel.initialize(progress_bar=False, nb_workers=6, use_memory_fs = False)


def calculating_annual_stds(
    lake_lse_df,
    area_columns,
    output_column_names,
    unit_scale: float = 1,
    reshape_period=12,
    calculation_mask_columns=None,
):
    def calculate_annual_std(row):
        areas = np.array([row[col]*unit_scale for col in area_columns])
        if calculation_mask_columns is not None:
            masks = np.array([row[col] for col in calculation_mask_columns])
            areas[masks] = np.nan
        #reshape areas to (period, numperiods) to calculate the standard deviation of each period
        areas = areas.reshape(-1, reshape_period)
        annual_stds = np.nanstd(areas, axis=1)
        return pd.Series(annual_stds)
    annual_stds_df = lake_lse_df.parallel_apply(calculate_annual_std, axis=1)
    annual_stds_df.columns = output_column_names
    return annual_stds_df

def calculating_annual_mean_first_differences(
    lake_lse_df,
    area_columns,
    output_column_names,
    unit_scale: float = 1,
    reshape_period=12,
    calculation_mask_columns=None,
):
    def calculate_annual_mean_first_difference(row):
        areas = np.array([row[col]*unit_scale for col in area_columns])
        if calculation_mask_columns is not None:
            masks = np.array([row[col] for col in calculation_mask_columns])
            areas[masks] = np.nan
        #reshape areas to (period, numperiods) to calculate the standard deviation of each period
        areas = areas.reshape(-1, reshape_period)
        annual_means = np.nanmean(areas, axis=1)
        annual_mean_first_differences = np.abs(np.diff(annual_means))
        return pd.Series(annual_mean_first_differences)
    annual_mean_first_diff_df = lake_lse_df.parallel_apply(calculate_annual_mean_first_difference, axis=1)
    annual_mean_first_diff_df.columns = output_column_names
    return annual_mean_first_diff_df