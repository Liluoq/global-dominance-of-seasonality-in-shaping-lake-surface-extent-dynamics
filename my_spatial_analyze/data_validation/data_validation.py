import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
from dateutil import relativedelta
import sys
sys.path.append('/WORK/Codes/global_lake_area/my_spatial_analyze')
from area_to_volume import area_to_volume_hydrolakes
from sklearn.metrics import r2_score
from scipy.interpolate import interpn
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import datashader as ds
from datashader.mpl_ext import dsshow
from scipy.stats import gaussian_kde
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import linregress, pearsonr, spearmanr
import os
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

def plot_using_datashader(ax, x, y, add_colorbar=True):

    df = pd.DataFrame(dict(x=x, y=y))
    dsartist = dsshow(
        df,
        ds.Point("x", "y"),
        ds.count(),
        norm="linear",
        aspect="equal",
        ax=ax
    )
    if add_colorbar:
        fig = ax.figure
        # Inset color bar
        axins = inset_axes(ax,
                        width="5%",  # width = 5% of parent_bbox width
                        height="30%",  # height : 30%
                        loc='lower right',
                        bbox_to_anchor=(0.1, 0.1, 0.8, 0.8),
                        bbox_transform=ax.transAxes,
                        borderpad=0,
                        )
        fig.colorbar(dsartist, cax=axins, label='Density')

def plot_using_scipy_gaussian_kde(ax, x, y, add_colorbar=True):
    # Calculate the point density
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    fig = ax.figure
    scatter = ax.scatter(x, y, c=z, s=5)
    if add_colorbar:
        # Inset color bar
        axins = inset_axes(ax,
                        width="5%",  # width = 5% of parent_bbox width
                        height="30%",  # height : 30%
                        loc='lower right',
                        bbox_to_anchor=(0.05, 0.1, 0.7, 0.8),
                        bbox_transform=ax.transAxes,
                        borderpad=0,
                        )
        fig.colorbar(scatter, cax=axins, label='Density')
    return ax

def compare_with_glolakes(
    glolakes_csv_path,
    my_lakes_area_pkl_path,
    my_lakes_area_unit_scale=1e-6,
    glolakes_id_column_name='Hylak_id',
    my_lakes_id_column_name='Hylak_id',
    my_lakes_start_date='2001-01-01',
    my_lakes_end_date='2024-01-01',
    glolakes_start_date='1984-01-01',
    glolakes_end_date='2021-01-01',
    date_fmt='%Y-%m-%d',
    ax=None,
    lake_size_bins=None,
    lake_size_labels=None,
    plot_mean=False,
    save_path=None
):
    if not my_lakes_area_pkl_path.endswith('.pkl'):
        raise ValueError('The file extension of the my_lakes_area_pkl_path is not recognized. The file extension should be .pkl.')
    glolakes_df = pd.read_csv(glolakes_csv_path)
    my_lakes_gdf = pd.read_pickle(my_lakes_area_pkl_path)
    my_lakes_gdf.drop(my_lakes_gdf[my_lakes_gdf['Slope_100'] == -1].index, inplace=True)
    
    my_lakes_start_date = datetime.strptime(my_lakes_start_date, '%Y-%m-%d')
    my_lakes_end_date = datetime.strptime(my_lakes_end_date, '%Y-%m-%d')
    glolakes_start_date = datetime.strptime(glolakes_start_date, '%Y-%m-%d')
    glolakes_end_date = datetime.strptime(glolakes_end_date, '%Y-%m-%d')
    
    compare_start_date = max(my_lakes_start_date, glolakes_start_date)
    compare_end_date = min(my_lakes_end_date, glolakes_end_date)
    
    selected_area_columns = []
    current_date = compare_start_date
    while current_date < compare_end_date:
        selected_area_columns.append(current_date.strftime(date_fmt))
        current_date = current_date + relativedelta.relativedelta(months=1)
    my_lakes_frozen_columns = [f'frozen_{col}' for col in selected_area_columns]
    glolakes_selected_columns = [glolakes_id_column_name] + selected_area_columns
    my_lakes_selected_columns = [my_lakes_id_column_name, 'Slope_100', 'Lake_area'] + selected_area_columns + my_lakes_frozen_columns
    
    glolakes_selected_df = glolakes_df[glolakes_selected_columns]
    my_lakes_selected_gdf = my_lakes_gdf[my_lakes_selected_columns]
    
    glolakes_selected_df.columns = [glolakes_id_column_name] + [f'glolakes_{col}' for col in selected_area_columns]
    my_lakes_selected_gdf.columns = [my_lakes_id_column_name, 'Slope_100', 'Lake_area'] + [f'my_lakes_{col}' for col in selected_area_columns] + my_lakes_frozen_columns
    
    merged_df = pd.merge(glolakes_selected_df, my_lakes_selected_gdf, left_on=glolakes_id_column_name, right_on=my_lakes_id_column_name, how='inner')
    
    glolakes_renamed_selected_volume_columns = [f'glolakes_{col}' for col in selected_area_columns]
    my_lakes_renamed_selected_volume_columns = [f'my_lakes_{col}' for col in selected_area_columns]
    for my_lakes_col in my_lakes_renamed_selected_volume_columns:
        merged_df[my_lakes_col] = merged_df.apply(lambda row: area_to_volume_hydrolakes(row[my_lakes_col]*my_lakes_area_unit_scale, row['Slope_100']), axis=1)
    if lake_size_bins is None or lake_size_labels is None:
        if plot_mean:
            glolakes_volumes = group_df[glolakes_renamed_selected_volume_columns].to_numpy()
            my_lakes_volumes = group_df[my_lakes_renamed_selected_volume_columns].to_numpy()
            my_lakes_frozen_mask = merged_df[my_lakes_frozen_columns].to_numpy().astype(bool)
            glolakes_volumes = np.where(glolakes_volumes == 0, np.nan, glolakes_volumes)
            my_lakes_volumes = np.where(my_lakes_volumes == 0, np.nan, my_lakes_volumes)
            my_lakes_volumes = np.where(my_lakes_frozen_mask == True, 0, my_lakes_volumes)
            glolakes_volumes = np.nanmean(glolakes_volumes, axis=1)
            my_lakes_volumes = np.nanmean(my_lakes_volumes, axis=1)
            glolakes_volumes = np.where(np.isnan(glolakes_volumes), 0, glolakes_volumes)
            my_lakes_volumes = np.where(np.isnan(my_lakes_volumes), 0, my_lakes_volumes)
            assert len(glolakes_volumes) == len(my_lakes_volumes), 'The number of lakes in the two datasets are not the same.'
        else:
            glolakes_volumes = group_df[glolakes_renamed_selected_volume_columns].to_numpy().flatten()
            my_lakes_volumes = group_df[my_lakes_renamed_selected_volume_columns].to_numpy().flatten()
            my_lakes_frozen_mask = merged_df[my_lakes_frozen_columns].to_numpy().flatten().astype(bool)
            my_lakes_volumes = np.where(my_lakes_frozen_mask == True, 0, my_lakes_volumes)
        
        # Replace zeros with NaNs and remove NaNs
        glolakes_volumes[glolakes_volumes == 0] = np.nan
        my_lakes_volumes[my_lakes_volumes == 0] = np.nan
        
        # Create a mask for NaNs
        nan_mask = np.isnan(glolakes_volumes) | np.isnan(my_lakes_volumes)
        # Remove NaNs
        glolakes_volumes = glolakes_volumes[~nan_mask]
        my_lakes_volumes = my_lakes_volumes[~nan_mask]
        assert len(glolakes_volumes) == len(my_lakes_volumes), 'The number of lakes in the two datasets are not the same.'
        
        current_ax = axs[i]
        
        # Number of scatter points (N)
        N = len(glolakes_volumes)
        # Calculate R-square
        if len(glolakes_volumes) > 1:  # Ensure there are enough points to calculate R-square
            r2 = r2_score(my_lakes_volumes, glolakes_volumes)
        else:
            r2 = float('nan')  # Not enough data points to calculate R-square
        if N < 100000:
            plot_using_scipy_gaussian_kde(current_ax, glolakes_volumes, my_lakes_volumes)
        else:
            plot_using_datashader(current_ax, glolakes_volumes, my_lakes_volumes)
                    
        # Annotate the plot with R-square and N without a box
        current_ax.text(0.05, 0.95, f'$R^2$ = {r2:.2f}\nN = {N}', transform=current_ax.transAxes, 
                        verticalalignment='top', fontsize=12)
        
        current_ax.set_title(f'Lake size group: All')
        current_ax.set_xlabel('Glolakes Volumes')
        current_ax.set_ylabel('My Lakes Volumes')
        
        current_ax.set_aspect('equal', adjustable='box')
        #set both axis to have the same limits
        current_ax.set_xlim(min(glolakes_volumes.min(), my_lakes_volumes.min()), max(glolakes_volumes.max(), my_lakes_volumes.max()))
        current_ax.set_ylim(min(glolakes_volumes.min(), my_lakes_volumes.min()), max(glolakes_volumes.max(), my_lakes_volumes.max()))
        #set both axis to be log scale
        current_ax.set_xscale('log')
        current_ax.set_yscale('log')
        
        return ax
    
    elif lake_size_bins is not None and lake_size_labels is not None:
        print(f'Plotting on multiple axes, the passed ax will be ignored.')
        merged_df.loc[:, 'lake_size_group'] = pd.cut(merged_df['Lake_area'], bins=lake_size_bins, labels=lake_size_labels)
        
        lake_size_groups = merged_df['lake_size_group'].unique()
        # Remove groups with no rows
        lake_size_groups = [group for group in lake_size_groups if len(merged_df[merged_df['lake_size_group'] == group]) > 0]
        num_groups = len(lake_size_groups)

        plot_per_row = 3
        num_rows = num_groups // plot_per_row
        if num_groups % plot_per_row != 0:
            num_rows += 1
        num_cols = plot_per_row
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 5*num_rows))
        axs = axs.flatten()
        
        for i, group in enumerate(lake_size_groups):
            group_df = merged_df[merged_df['lake_size_group'] == group]
            if plot_mean:
                glolakes_volumes = group_df[glolakes_renamed_selected_volume_columns].to_numpy()
                my_lakes_volumes = group_df[my_lakes_renamed_selected_volume_columns].to_numpy()
                my_lakes_frozen_mask = group_df[my_lakes_frozen_columns].to_numpy().astype(bool)
                glolakes_volumes = np.where(glolakes_volumes == 0, np.nan, glolakes_volumes)
                my_lakes_volumes = np.where(my_lakes_volumes == 0, np.nan, my_lakes_volumes)
                my_lakes_volumes = np.where(my_lakes_frozen_mask == True, 0, my_lakes_volumes)
                glolakes_volumes = np.nanmean(glolakes_volumes, axis=1)
                my_lakes_volumes = np.nanmean(my_lakes_volumes, axis=1)
                glolakes_volumes = np.where(np.isnan(glolakes_volumes), 0, glolakes_volumes)
                my_lakes_volumes = np.where(np.isnan(my_lakes_volumes), 0, my_lakes_volumes)
                assert len(glolakes_volumes) == len(my_lakes_volumes), 'The number of lakes in the two datasets are not the same.'
            else:
                glolakes_volumes = group_df[glolakes_renamed_selected_volume_columns].to_numpy().flatten()
                my_lakes_volumes = group_df[my_lakes_renamed_selected_volume_columns].to_numpy().flatten()
                my_lakes_frozen_mask = group_df[my_lakes_frozen_columns].to_numpy().flatten().astype(bool)
                my_lakes_volumes = np.where(my_lakes_frozen_mask == True, 0, my_lakes_volumes)
            # Replace zeros with NaNs
            glolakes_volumes[glolakes_volumes == 0] = np.nan
            my_lakes_volumes[my_lakes_volumes == 0] = np.nan

            # Create a mask for NaNs
            nan_mask = np.isnan(glolakes_volumes) | np.isnan(my_lakes_volumes)

            # Remove NaNs
            glolakes_volumes = glolakes_volumes[~nan_mask]
            my_lakes_volumes = my_lakes_volumes[~nan_mask]
            current_ax = axs[i]
            
            # Number of scatter points (N)
            N = len(glolakes_volumes)
            # Calculate R-square
            if len(glolakes_volumes) > 1:  # Ensure there are enough points to calculate R-square
                r2 = r2_score(my_lakes_volumes, glolakes_volumes)
            else:
                r2 = float('nan')  # Not enough data points to calculate R-square
            if N < 100000:
                plot_using_scipy_gaussian_kde(current_ax, glolakes_volumes, my_lakes_volumes, add_colorbar=True)
            else:
                plot_using_datashader(current_ax, glolakes_volumes, my_lakes_volumes, add_colorbar=True)
                        
            # Annotate the plot with R-square and N without a box
            current_ax.text(0.05, 0.95, f'$R^2$ = {r2:.2f}\nN = {N}', transform=current_ax.transAxes, 
                            verticalalignment='top', fontsize=12)
            
            current_ax.set_title(f'Lake size group: {group}')
            current_ax.set_xlabel('Mean volume (Glolakes)')
            current_ax.set_ylabel('Mean volume (Our results)')
            
            #set both axis to be log scale
            current_ax.set_xscale('log')
            current_ax.set_yscale('log')
            current_ax.set_aspect('equal', adjustable='box')
            #set both axis to have the same limits
            min_limit = min(glolakes_volumes.min(), my_lakes_volumes.min())
            max_limit = max(glolakes_volumes.max(), my_lakes_volumes.max())
            current_ax.set_xlim(min_limit, max_limit)
            current_ax.set_ylim(min_limit, max_limit)
            
            # Add a 45 degree line
            current_ax.plot([min_limit, max_limit], [min_limit, max_limit], 'k--', zorder=2.1)
    
    if save_path:
        if save_path.endswith('.png'):
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
def smape(A, F):
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE) between two arrays,
    ignoring NaNs in the calculation.
    
    Parameters:
    A (numpy array): Actual values.
    F (numpy array): Forecasted values.
    
    Returns:
    float: SMAPE value.
    """
    # Ignore NaNs in calculation
    valid = ~np.isnan(A) & ~np.isnan(F)
    A, F = A[valid], F[valid]
    assert len(A) == len(F), 'The lengths of the two arrays are not the same.'
    if len(A) == 0 or len(F) == 0:
        return np.nan
    return 100 * np.mean(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))
        
def cal_nrmse(actual, predicted, normalization='range'):
    """
    Calculate Normalized Root Mean Squared Error (NRMSE).
    
    Parameters:
    actual (numpy array): Actual values.
    predicted (numpy array): Predicted values.
    normalization (str): Normalization method ('range' or 'mean').
    
    Returns:
    float: NRMSE value (%).
    """
    valid = ~np.isnan(actual) & ~np.isnan(predicted)
    actual, predicted = actual[valid], predicted[valid]
    
    if len(actual) == 0 or len(predicted) == 0:
        return np.nan
    
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    
    if normalization == 'range':
        nrmse_value = rmse / (np.max(actual) - np.min(actual))
    elif normalization == 'mean':
        nrmse_value = rmse / np.mean(actual)
    else:
        raise ValueError("Normalization method must be 'range' or 'mean'")
    
    return nrmse_value*100
        
def compare_with_glolakes_lake_wise(
    glolakes_csv_path,
    my_lakes_area_pkl_path,
    my_lakes_area_unit_scale=1e-6,
    glolakes_id_column_name='Hylak_id',
    my_lakes_id_column_name='Hylak_id',
    my_lakes_start_date='2001-01-01',
    my_lakes_end_date='2024-01-01',
    glolakes_start_date='1984-01-01',
    glolakes_end_date='2021-01-01',
    date_fmt='%Y-%m-%d',
    min_valid_points=12,
    ax_list=None,
    save_path=None
):
    if not my_lakes_area_pkl_path.endswith('.pkl'):
        raise ValueError('The file extension of the my_lakes_area_pkl_path is not recognized. The file extension should be .pkl.')
    glolakes_df = pd.read_csv(glolakes_csv_path)
    my_lakes_gdf = pd.read_pickle(my_lakes_area_pkl_path)
    my_lakes_gdf.drop(my_lakes_gdf[my_lakes_gdf['Slope_100'] == -1].index, inplace=True)
    
    my_lakes_start_date = datetime.strptime(my_lakes_start_date, '%Y-%m-%d')
    my_lakes_end_date = datetime.strptime(my_lakes_end_date, '%Y-%m-%d')
    glolakes_start_date = datetime.strptime(glolakes_start_date, '%Y-%m-%d')
    glolakes_end_date = datetime.strptime(glolakes_end_date, '%Y-%m-%d')
    
    compare_start_date = max(my_lakes_start_date, glolakes_start_date)
    compare_end_date = min(my_lakes_end_date, glolakes_end_date)
    
    selected_area_columns = []
    current_date = compare_start_date
    while current_date < compare_end_date:
        selected_area_columns.append(current_date.strftime(date_fmt))
        current_date = current_date + relativedelta.relativedelta(months=1)
    glolakes_selected_columns = [glolakes_id_column_name] + selected_area_columns
    my_lakes_frozen_columns = [f'frozen_{col}' for col in selected_area_columns]
    my_lakes_selected_columns = [my_lakes_id_column_name, 'Slope_100', 'Lake_area'] + selected_area_columns + my_lakes_frozen_columns
    
    glolakes_selected_df = glolakes_df[glolakes_selected_columns]
    my_lakes_selected_gdf = my_lakes_gdf[my_lakes_selected_columns]
    
    glolakes_selected_df.columns = [glolakes_id_column_name] + [f'glolakes_{col}' for col in selected_area_columns]
    my_lakes_selected_gdf.columns = [my_lakes_id_column_name, 'Slope_100', 'Lake_area'] + [f'my_lakes_{col}' for col in selected_area_columns] + my_lakes_frozen_columns
    
    merged_df = pd.merge(glolakes_selected_df, my_lakes_selected_gdf, left_on=glolakes_id_column_name, right_on=my_lakes_id_column_name, how='inner')
    
    glolakes_renamed_selected_volume_columns = [f'glolakes_{col}' for col in selected_area_columns]
    my_lakes_renamed_selected_volume_columns = [f'my_lakes_{col}' for col in selected_area_columns]
    for my_lakes_col in my_lakes_renamed_selected_volume_columns:
        merged_df[my_lakes_col] = merged_df.apply(lambda row: area_to_volume_hydrolakes(row[my_lakes_col]*my_lakes_area_unit_scale, row['Slope_100']), axis=1)
        
    glolakes_volumes = merged_df[glolakes_renamed_selected_volume_columns].to_numpy()
    my_lakes_volumes = merged_df[my_lakes_renamed_selected_volume_columns].to_numpy()
    my_lakes_frozen_mask = merged_df[my_lakes_frozen_columns].to_numpy().astype(bool)
    
    glolakes_volumes[glolakes_volumes == 0] = np.nan
    my_lakes_volumes[my_lakes_volumes == 0] = np.nan
    
    nan_mask = np.isnan(glolakes_volumes) | np.isnan(my_lakes_volumes) | my_lakes_frozen_mask
    glolakes_volumes[nan_mask] = np.nan
    my_lakes_volumes[nan_mask] = np.nan
    
    nrmses = []
    smapes = []
    coef_corrs = []
    
    assert glolakes_volumes.shape == my_lakes_volumes.shape, 'The shapes of the two arrays are not the same.'
    num_lakes = glolakes_volumes.shape[0]
    
    for i in range(num_lakes):
        valid_mask = ~np.isnan(glolakes_volumes[i]) & ~np.isnan(my_lakes_volumes[i])
        valid_count = np.sum(valid_mask)
        if valid_count < min_valid_points:
            nrmse = np.nan
            smape_val = np.nan
            corr_coef = np.nan
        else:
            nrmse = cal_nrmse(glolakes_volumes[i][valid_mask], my_lakes_volumes[i][valid_mask])
            smape_val = smape(glolakes_volumes[i][valid_mask], my_lakes_volumes[i][valid_mask])
            corr_coef = np.corrcoef(glolakes_volumes[i][valid_mask], my_lakes_volumes[i][valid_mask])[0, 1]
        nrmses.append(nrmse)
        smapes.append(smape_val)
        coef_corrs.append(corr_coef)
    
    nrmse = np.array(nrmses)
    smape_val = np.array(smapes)
    corr_coef = np.array(coef_corrs)
    #remove na
    nrmse = nrmse[~np.isnan(nrmse)]
    nrmse = nrmse[nrmse < 200]
    smape_val = smape_val[~np.isnan(smape_val)]
    corr_coef = corr_coef[~np.isnan(corr_coef)]
    
    if ax_list is None:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        ax_list = axs.flatten()
    else:
        assert len(ax_list) == 3, 'The length of the passed ax_list is not 2.'
    
    for ax, metrix in zip(ax_list, [nrmse, smape_val, corr_coef]):
        ax.hist(metrix, bins=50, color='skyblue')
        if metrix is nrmse:
            xlabel = 'NRMSE (%)'
        elif metrix is smape_val:
            xlabel = 'SMAPE (%)'
        elif metrix is corr_coef:
            xlabel = 'Correlation coefficient'
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Frequency')
        #calculate median and plot vertical line
        median_val = np.median(metrix)
        ax.axvline(median_val, color='k', linestyle='dashed', linewidth=1)
        ax.text(0.6, 0.9, f'Median: {median_val:.2f}', transform=ax.transAxes, verticalalignment='top', fontsize=12)
        #ax.set_title('Histogram of ' + ('Correlation coefficient' if metrix is corr_coef else 'SMAPE (%)'))
    
    if save_path:
        if save_path.endswith('.png'):
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(save_path)
    else:
        plt.show()
        
def compare_with_zhao_2018(
    zhao_2018_csv_path,
    my_lakes_area_pkl_path,
    my_lakes_area_unit_scale=1e-6,
    my_lakes_id_column_name='Grand_id',
    zhao_2018_id_column_name='Grand_id',
    my_lakes_start_date='2001-01-01',
    my_lakes_end_date='2024-01-01',
    zhao_2018_start_date='1984-03-01',
    zhao_2018_end_date='2019-01-01',
    date_fmt='%Y-%m-%d',
    ax=None,
    lake_size_bins=None,
    lake_size_labels=None,
    plot_mean=False,
    save_path=None
):
    if not my_lakes_area_pkl_path.endswith('.pkl'):
        raise ValueError('The file extension of the my_lakes_area_pkl_path is not recognized. The file extension should be .pkl.')
    zhao_2018_df = pd.read_csv(zhao_2018_csv_path)
    my_lakes_gdf = pd.read_pickle(my_lakes_area_pkl_path)
    
    my_lakes_start_date = datetime.strptime(my_lakes_start_date, '%Y-%m-%d')
    my_lakes_end_date = datetime.strptime(my_lakes_end_date, '%Y-%m-%d')
    zhao_2018_start_date = datetime.strptime(zhao_2018_start_date, '%Y-%m-%d')
    zhao_2018_end_date = datetime.strptime(zhao_2018_end_date, '%Y-%m-%d')
    
    compare_start_date = max(my_lakes_start_date, zhao_2018_start_date)
    compare_end_date = min(my_lakes_end_date, zhao_2018_end_date)
    
    selected_area_columns = []
    current_date = compare_start_date
    while current_date < compare_end_date:
        selected_area_columns.append(current_date.strftime(date_fmt))
        current_date = current_date + relativedelta.relativedelta(months=1)
    my_lakes_frozen_columns = [f'frozen_{col}' for col in selected_area_columns]
    zhao_2018_selected_columns = [zhao_2018_id_column_name] + selected_area_columns
    my_lakes_selected_columns = [my_lakes_id_column_name, 'Lake_area'] + selected_area_columns + my_lakes_frozen_columns
    
    zhao_2018_selected_df = zhao_2018_df[zhao_2018_selected_columns]
    my_lakes_selected_gdf = my_lakes_gdf[my_lakes_selected_columns]
    
    zhao_2018_selected_df.columns = [zhao_2018_id_column_name] + [f'zhao_2018_{col}' for col in selected_area_columns]
    my_lakes_selected_gdf.columns = [my_lakes_id_column_name, 'Lake_area'] + [f'my_lakes_{col}' for col in selected_area_columns] + my_lakes_frozen_columns
    
    merged_df = pd.merge(zhao_2018_selected_df, my_lakes_selected_gdf, left_on=zhao_2018_id_column_name, right_on=my_lakes_id_column_name, how='inner')
    
    zhao_2018_renamed_selected_area_columns = [f'zhao_2018_{col}' for col in selected_area_columns]
    my_lakes_renamed_selected_area_columns = [f'my_lakes_{col}' for col in selected_area_columns]
    
    for my_lakes_col in my_lakes_renamed_selected_area_columns:
        merged_df[my_lakes_col] = merged_df.apply(lambda row: row[my_lakes_col]*my_lakes_area_unit_scale, axis=1)
    
    if lake_size_bins is None or lake_size_labels is None:
        if plot_mean:
            zhao_2018_areas = group_df[zhao_2018_renamed_selected_area_columns].to_numpy()
            my_lakes_areas = group_df[my_lakes_renamed_selected_area_columns].to_numpy()
            my_lakes_frozen_mask = merged_df[my_lakes_frozen_columns].to_numpy().astype(bool)
            zhao_2018_areas = np.where(zhao_2018_areas == 0, np.nan, zhao_2018_areas)
            my_lakes_areas = np.where(my_lakes_areas == 0, np.nan, my_lakes_areas)
            my_lakes_areas = np.where(my_lakes_areas == 0, np.nan, my_lakes_areas)
            my_lakes_areas = np.where(my_lakes_frozen_mask == True, 0, my_lakes_areas)
            zhao_2018_areas = np.nanmean(zhao_2018_areas, axis=1)
            my_lakes_areas = np.nanmean(my_lakes_areas, axis=1)
            zhao_2018_areas = np.where(np.isnan(zhao_2018_areas), 0, zhao_2018_areas)
            my_lakes_areas = np.where(np.isnan(my_lakes_areas), 0, my_lakes_areas)
            assert len(zhao_2018_areas) == len(my_lakes_areas), 'The number of lakes in the two datasets are not the same.'
        else:
            zhao_2018_areas = group_df[zhao_2018_renamed_selected_area_columns].to_numpy().flatten()
            my_lakes_areas = group_df[my_lakes_renamed_selected_area_columns].to_numpy().flatten()
            my_lakes_frozen_mask = merged_df[my_lakes_frozen_columns].to_numpy().flatten().astype(bool)
            my_lakes_areas = np.where(my_lakes_frozen_mask == True, 0, my_lakes_areas)
        
        # Replace zeros with NaNs
        zhao_2018_areas[zhao_2018_areas == 0] = np.nan
        my_lakes_areas[my_lakes_areas == 0] = np.nan

        # Create a mask for NaNs
        nan_mask = np.isnan(zhao_2018_areas) | np.isnan(my_lakes_areas)

        # Remove NaNs
        zhao_2018_areas = zhao_2018_areas[~nan_mask]
        my_lakes_areas = my_lakes_areas[~nan_mask]
        assert len(zhao_2018_areas) == len(my_lakes_areas), 'The number of lakes in the two datasets are not the same.'
        
        current_ax = axs[i]
        
        # Number of scatter points (N)
        N = len(zhao_2018_areas)
        # Calculate R-square
        if len(zhao_2018_areas) > 1:  # Ensure there are enough points to calculate R-square
            r2 = r2_score(my_lakes_areas, zhao_2018_areas)
        else:
            r2 = float('nan')  # Not enough data points to calculate R-square
        if N < 100000:
            plot_using_scipy_gaussian_kde(current_ax, zhao_2018_areas, my_lakes_areas)
        else:
            plot_using_datashader(current_ax, zhao_2018_areas, my_lakes_areas)
                    
        # Annotate the plot with R-square and N without a box
        current_ax.text(0.05, 0.95, f'$R^2$ = {r2:.2f}\nN = {N}', transform=current_ax.transAxes, 
                        verticalalignment='top', fontsize=12)
        
        current_ax.set_title(f'Lake size group: All')
        current_ax.set_xlabel('zhao_2018 areas')
        current_ax.set_ylabel('My Lakes areas')
        
        current_ax.set_aspect('equal', adjustable='box')
        #set both axis to have the same limits
        current_ax.set_xlim(min(zhao_2018_areas.min(), my_lakes_areas.min()), max(zhao_2018_areas.max(), my_lakes_areas.max()))
        current_ax.set_ylim(min(zhao_2018_areas.min(), my_lakes_areas.min()), max(zhao_2018_areas.max(), my_lakes_areas.max()))
        #set both axis to be log scale
        current_ax.set_xscale('log')
        current_ax.set_yscale('log')
        
        return ax
    
    elif lake_size_bins is not None and lake_size_labels is not None:
        print(f'Plotting on multiple axes, the passed ax will be ignored.')
        merged_df.loc[:, 'lake_size_group'] = pd.cut(merged_df['Lake_area'], bins=lake_size_bins, labels=lake_size_labels)
        
        lake_size_groups = lake_size_labels
        # Remove groups with no rows
        lake_size_groups = [group for group in lake_size_groups if len(merged_df[merged_df['lake_size_group'] == group]) > 0]
        num_groups = len(lake_size_groups)
        # sort groups by size
        
        plot_per_row = 3
        num_rows = num_groups // plot_per_row
        if num_groups % plot_per_row != 0:
            num_rows += 1
        num_cols = plot_per_row
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 5*num_rows))
        axs = axs.flatten()
        
        for i, group in enumerate(lake_size_groups):
            group_df = merged_df[merged_df['lake_size_group'] == group]
            if plot_mean:
                zhao_2018_areas = group_df[zhao_2018_renamed_selected_area_columns].to_numpy()
                my_lakes_areas = group_df[my_lakes_renamed_selected_area_columns].to_numpy()
                my_lakes_frozen_mask = group_df[my_lakes_frozen_columns].to_numpy().astype(bool)
                zhao_2018_areas = np.where(zhao_2018_areas == 0, np.nan, zhao_2018_areas)
                my_lakes_areas = np.where(my_lakes_areas == 0, np.nan, my_lakes_areas)
                my_lakes_areas = np.where(my_lakes_areas == 0, np.nan, my_lakes_areas)
                my_lakes_areas = np.where(my_lakes_frozen_mask == True, 0, my_lakes_areas)
                zhao_2018_areas = np.nanmean(zhao_2018_areas, axis=1)
                my_lakes_areas = np.nanmean(my_lakes_areas, axis=1)
                zhao_2018_areas = np.where(np.isnan(zhao_2018_areas), 0, zhao_2018_areas)
                my_lakes_areas = np.where(np.isnan(my_lakes_areas), 0, my_lakes_areas)
                assert len(zhao_2018_areas) == len(my_lakes_areas), 'The number of lakes in the two datasets are not the same.'
            else:
                zhao_2018_areas = group_df[zhao_2018_renamed_selected_area_columns].to_numpy().flatten()
                my_lakes_areas = group_df[my_lakes_renamed_selected_area_columns].to_numpy().flatten()
                my_lakes_frozen_mask = group_df[my_lakes_frozen_columns].to_numpy().flatten().astype(bool)
                my_lakes_areas = np.where(my_lakes_frozen_mask == True, 0, my_lakes_areas)
            # Replace zeros with NaNs
            zhao_2018_areas[zhao_2018_areas == 0] = np.nan
            my_lakes_areas[my_lakes_areas == 0] = np.nan

            # Create a mask for NaNs
            nan_mask = np.isnan(zhao_2018_areas) | np.isnan(my_lakes_areas)

            # Remove NaNs
            zhao_2018_areas = zhao_2018_areas[~nan_mask]
            my_lakes_areas = my_lakes_areas[~nan_mask]
            assert len(zhao_2018_areas) == len(my_lakes_areas), 'The number of lakes in the two datasets are not the same.'
            current_ax = axs[i]
            
            # Number of scatter points (N)
            N = len(zhao_2018_areas)
            # Calculate R-square
            if len(zhao_2018_areas) > 1:  # Ensure there are enough points to calculate R-square
                r2 = r2_score(my_lakes_areas, zhao_2018_areas)
            else:
                r2 = float('nan')  # Not enough data points to calculate R-square
            if N < 100000:
                plot_using_scipy_gaussian_kde(current_ax, zhao_2018_areas, my_lakes_areas, add_colorbar=True)
            else:
                plot_using_datashader(current_ax, zhao_2018_areas, my_lakes_areas, add_colorbar=True)
                        
            # Annotate the plot with R-square and N without a box
            current_ax.text(0.05, 0.95, f'$R^2$ = {r2:.2f}\nN = {N}', transform=current_ax.transAxes, 
                            verticalalignment='top', fontsize=12)
            
            current_ax.set_title(f'Lake size group: {group}')
            current_ax.set_xlabel('Mean area (zhao_2018)')
            current_ax.set_ylabel('Mean area (Our results)')
            
            #set both axis to be log scale
            current_ax.set_xscale('log')
            current_ax.set_yscale('log')
            current_ax.set_aspect('equal', adjustable='box')
            #set both axis to have the same limits
            min_limit = min(zhao_2018_areas.min(), my_lakes_areas.min())
            max_limit = max(zhao_2018_areas.max(), my_lakes_areas.max())
            current_ax.set_xlim(min_limit, max_limit)
            current_ax.set_ylim(min_limit, max_limit)
            
            # Add a 45 degree line
            current_ax.plot([min_limit, max_limit], [min_limit, max_limit], 'k--', zorder=2.1)
    
    if save_path:
        if save_path.endswith('.png'):
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
        
def compare_with_zhao_2018_lake_wise(
    zhao_2018_csv_path,
    my_lakes_area_pkl_path,
    my_lakes_area_unit_scale=1e-6,
    my_lakes_id_column_name='Grand_id',
    zhao_2018_id_column_name='Grand_id',
    my_lakes_start_date='2001-01-01',
    my_lakes_end_date='2024-01-01',
    zhao_2018_start_date='1984-03-01',
    zhao_2018_end_date='2019-01-01',
    date_fmt='%Y-%m-%d',
    min_valid_points=12,
    ax_list=None,
    save_path=None
):
    if not my_lakes_area_pkl_path.endswith('.pkl'):
        raise ValueError('The file extension of the my_lakes_area_pkl_path is not recognized. The file extension should be .pkl.')
    zhao_2018_df = pd.read_csv(zhao_2018_csv_path)
    my_lakes_gdf = pd.read_pickle(my_lakes_area_pkl_path)
    
    my_lakes_start_date = datetime.strptime(my_lakes_start_date, '%Y-%m-%d')
    my_lakes_end_date = datetime.strptime(my_lakes_end_date, '%Y-%m-%d')
    zhao_2018_start_date = datetime.strptime(zhao_2018_start_date, '%Y-%m-%d')
    zhao_2018_end_date = datetime.strptime(zhao_2018_end_date, '%Y-%m-%d')
    
    compare_start_date = max(my_lakes_start_date, zhao_2018_start_date)
    compare_end_date = min(my_lakes_end_date, zhao_2018_end_date)
    
    selected_area_columns = []
    current_date = compare_start_date
    while current_date < compare_end_date:
        selected_area_columns.append(current_date.strftime(date_fmt))
        current_date = current_date + relativedelta.relativedelta(months=1)
    zhao_2018_selected_columns = [zhao_2018_id_column_name] + selected_area_columns
    my_lakes_frozen_columns = [f'frozen_{col}' for col in selected_area_columns]
    my_lakes_selected_columns = [my_lakes_id_column_name, 'Lake_area'] + selected_area_columns + my_lakes_frozen_columns
    
    zhao_2018_selected_df = zhao_2018_df[zhao_2018_selected_columns]
    my_lakes_selected_gdf = my_lakes_gdf[my_lakes_selected_columns]
    
    zhao_2018_selected_df.columns = [zhao_2018_id_column_name] + [f'zhao_2018_{col}' for col in selected_area_columns]
    my_lakes_selected_gdf.columns = [my_lakes_id_column_name, 'Lake_area'] + [f'my_lakes_{col}' for col in selected_area_columns] + my_lakes_frozen_columns
    
    merged_df = pd.merge(zhao_2018_selected_df, my_lakes_selected_gdf, left_on=zhao_2018_id_column_name, right_on=my_lakes_id_column_name, how='inner')
    
    zhao_2018_renamed_selected_area_columns = [f'zhao_2018_{col}' for col in selected_area_columns]
    my_lakes_renamed_selected_area_columns = [f'my_lakes_{col}' for col in selected_area_columns]
    
    for my_lakes_col in my_lakes_renamed_selected_area_columns:
        merged_df[my_lakes_col] = merged_df.apply(lambda row: row[my_lakes_col]*my_lakes_area_unit_scale, axis=1)
        
    zhao_2018_areas = merged_df[zhao_2018_renamed_selected_area_columns].to_numpy()
    my_lakes_areas = merged_df[my_lakes_renamed_selected_area_columns].to_numpy()
    my_lakes_frozen_mask = merged_df[my_lakes_frozen_columns].to_numpy().astype(bool)
    zhao_2018_areas[zhao_2018_areas == 0] = np.nan
    my_lakes_areas[my_lakes_areas == 0] = np.nan
    
    nan_mask = np.isnan(zhao_2018_areas) | np.isnan(my_lakes_areas) | my_lakes_frozen_mask
    zhao_2018_areas[nan_mask] = np.nan
    my_lakes_areas[nan_mask] = np.nan
    
    nrmses = []
    smapes = []
    corr_coefs = []
    
    assert zhao_2018_areas.shape == my_lakes_areas.shape, 'The shapes of the two arrays are not the same.'
    num_lakes = zhao_2018_areas.shape[0]
    for i in range(num_lakes):
        valid_mask = ~np.isnan(zhao_2018_areas[i]) & ~np.isnan(my_lakes_areas[i])
        valid_count = np.sum(valid_mask)
        if valid_count < min_valid_points:
            nrmse = np.nan
            smape_val = np.nan
            corr_coef = np.nan
        else:
            nrmse = cal_nrmse(zhao_2018_areas[i][valid_mask], my_lakes_areas[i][valid_mask])
            smape_val = smape(zhao_2018_areas[i][valid_mask], my_lakes_areas[i][valid_mask])
            corr_coef = np.corrcoef(zhao_2018_areas[i][valid_mask], my_lakes_areas[i][valid_mask])[0, 1]
        nrmses.append(nrmse)
        smapes.append(smape_val)
        corr_coefs.append(corr_coef)
    
    nrmse = np.array(nrmses)
    smape_val = np.array(smapes)
    corr_coef = np.array(corr_coefs)
    #remove na
    nrmse = nrmse[~np.isnan(nrmse)]
    nrmse = nrmse[nrmse < 200]
    smape_val = smape_val[~np.isnan(smape_val)]
    corr_coef = corr_coef[~np.isnan(corr_coef)]
    
    if ax_list is None:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        ax_list = axs.flatten()
    else:
        assert len(ax_list) == 3, 'The length of the passed ax_list is not 2.'
    
    for ax, metrix in zip(ax_list, [nrmse, smape_val, corr_coef]):
        ax.hist(metrix, bins=50, color='skyblue')
        if metrix is nrmse:
            xlabel = 'NRMSE (%)'
        elif metrix is smape_val:
            xlabel = 'SMAPE (%)'
        elif metrix is corr_coef:
            xlabel = 'Correlation coefficient'
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Frequency')
        #calculate median and plot vertical line
        median_val = np.median(metrix)
        ax.axvline(median_val, color='k', linestyle='dashed', linewidth=1)
        ax.text(0.6, 0.9, f'Median: {median_val:.2f}', transform=ax.transAxes, verticalalignment='top', fontsize=12)
        #ax.set_title('Histogram of ' + ('Correlation coefficient' if metrix is corr_coef else 'SMAPE (%)'))
    
    if save_path:
        if save_path.endswith('.png'):
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(save_path)
    else:
        plt.show()
        
def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
        
def compare_with_xingdongLi_2019_lake_wise(
    lake_names:list,
    xingdongLi_2019_csv_path_pattern,
    my_lakes_level_path_pattern,
    anomaly_type='start',
    save_path=None,
    return_cc=True,
    my_lakes_using_area=False
):
    #set global font size to 12
    plt.rcParams.update({'font.size': 16})
    accepted_anomaly_types = ['start', 'mean']
    if not anomaly_type in accepted_anomaly_types:
        raise ValueError(f'The passed anomaly_type is not in the accepted anomaly types: {accepted_anomaly_types}')
    
    num_lakes = len(lake_names)
    num_cols = 8
    num_rows = num_lakes // num_cols
    if num_lakes % num_cols != 0:
        num_rows += 1
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 5*num_rows), layout='tight')
    ax_list = axs.flatten()
    all_cc = []
    for lake_name, ax in zip(lake_names, ax_list):
        xingdongLi_2019_csv_path = xingdongLi_2019_csv_path_pattern.format(lake_name)
        my_lakes_level_path = my_lakes_level_path_pattern.format(lake_name)
        
        xingdongLi_2019_df = pd.read_csv(xingdongLi_2019_csv_path)
        my_lakes_level_df = pd.read_csv(my_lakes_level_path)
        
        selected_dates = xingdongLi_2019_df.columns.tolist()
        my_lakes_frozen_columns = [f'frozen_{col}' for col in selected_dates]
        
        xingdongli_levels = xingdongLi_2019_df[selected_dates].to_numpy().flatten()
        my_lakes_levels = my_lakes_level_df[selected_dates].to_numpy().flatten()
        frozen_mask = my_lakes_level_df[my_lakes_frozen_columns].to_numpy().flatten().astype(bool)
        
        xingdongli_levels = xingdongli_levels[~frozen_mask]
        my_lakes_levels = my_lakes_levels[~frozen_mask]
        if my_lakes_using_area:
            my_lakes_levels = my_lakes_levels * 1e-6
        
        if len(xingdongli_levels) == 0 or len(my_lakes_levels) == 0:
            print(f'No valid data points for {lake_name}. Skipping...')
            continue
        
        if anomaly_type == 'start':
            xingdongli_levels = xingdongli_levels - xingdongli_levels[0]
            my_lakes_levels = my_lakes_levels - my_lakes_levels[0]
        elif anomaly_type == 'mean':
            xingdongli_levels = xingdongli_levels - np.nanmean(xingdongli_levels)
            my_lakes_levels = my_lakes_levels - np.nanmean(my_lakes_levels)
        
        if not len(xingdongli_levels) == len(my_lakes_levels):
            print(f'Lengths of the two arrays are not the same for {lake_name}. Skipping...')
            continue
        num_points = len(xingdongli_levels)
        
        ax.scatter(xingdongli_levels, my_lakes_levels, color='blue', s=20, alpha=0.6)
    
        # Plot the 1:1 line
        min_val = min(min(xingdongli_levels), min(my_lakes_levels))
        max_val = max(max(xingdongli_levels), max(my_lakes_levels))
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        if not my_lakes_using_area:
            
            ax.plot([min_val, max_val], [min_val, max_val], 'k-', linewidth=2)
        
        # Calculate and plot the regression line
        slope, intercept, r_value, p_value, std_err = linregress(xingdongli_levels, my_lakes_levels)
        ax.plot([min_val, max_val], [slope * min_val + intercept, slope * max_val + intercept], 'r--', linewidth=2)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        # Calculate CC and RMSE
        #cc = np.corrcoef(xingdongli_levels, my_lakes_levels)[0, 1]
        #all_cc.append(cc)
        #calculate spearman cc
        cc = spearmanr(xingdongli_levels, my_lakes_levels)[0]
        all_cc.append(cc)
        if not my_lakes_using_area:
            rmse = calculate_rmse(my_lakes_levels, xingdongli_levels)
        
            # Add text box with CC and RMS
            textstr = f'N={num_points}\nCC={cc:.3f}\nRMSE={rmse:.2f}m'
        else:
            textstr = f'N={num_points}\nCC={cc:.3f}'
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, verticalalignment='top')
        
        # Set labels, title, and limits
        ax.set_title(lake_name)
        ax.set_xlabel(r'Li $\mathit{et\,al.}$ (2019)')
        ax.set_ylabel('Lake area (Our results)')
        if not my_lakes_using_area:
            ax.set_xlim([min_val, max_val])
            ax.set_ylim([min_val, max_val])
        ax.grid(True)
    
    if save_path:
        if save_path.endswith('.png'):
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        elif save_path.endswith('.pdf'):
            plt.savefig(save_path, bbox_inches='tight', format='pdf')
        else:
            plt.savefig(save_path)
    else:
        plt.show()
    
    if return_cc:
        return all_cc
        
def compare_with_GREALM_lake_wise(
    input_data_folder,
    GREALM_filename_identifier,
    my_lakes_filename_pattern,
    num_to_plot,
    anomaly_type='start',
    n_cols=8,
    save_path=None
):
    accepted_anomaly_types = ['start', 'mean']
    if not anomaly_type in accepted_anomaly_types:
        raise ValueError(f'The passed anomaly_type is not in the accepted anomaly types: {accepted_anomaly_types}')
    
    files_in_folder = os.listdir(input_data_folder)
    GREALM_filenames = [file for file in files_in_folder if GREALM_filename_identifier in file]
    lake_ids = [file.split('_')[0] for file in GREALM_filenames]
    my_lakes_filenames = [my_lakes_filename_pattern.format(lake_id) for lake_id in lake_ids]
    
    num_lakes = len(lake_ids)
    selected_filename_indices = np.random.choice(num_lakes, num_to_plot, replace=False)
    
    selected_GREALM_filenames = [GREALM_filenames[i] for i in selected_filename_indices]
    selected_my_lakes_filenames = [my_lakes_filenames[i] for i in selected_filename_indices]
    selected_lake_ids = [lake_ids[i] for i in selected_filename_indices]
    
    n_rows = num_to_plot // n_cols
    if num_to_plot % n_cols != 0:
        n_rows += 1
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    ax_list = axs.flatten()
    
    for lake_id, GREALM_filename, my_lakes_filename, ax in zip(selected_lake_ids, selected_GREALM_filenames, selected_my_lakes_filenames, ax_list):
        GREALM_df = pd.read_csv(os.path.join(input_data_folder, GREALM_filename))
        my_lakes_df = pd.read_csv(os.path.join(input_data_folder, my_lakes_filename))
        
        selected_dates = GREALM_df.columns.tolist()
        my_lakes_frozen_columns = [f'frozen_{col}' for col in selected_dates]
        
        GREALM_levels = GREALM_df[selected_dates].to_numpy().flatten()
        my_lakes_levels = my_lakes_df[selected_dates].to_numpy().flatten()
        frozen_mask = my_lakes_df[my_lakes_frozen_columns].to_numpy().flatten().astype(bool)
        
        GREALM_levels = GREALM_levels[~frozen_mask]
        my_lakes_levels = my_lakes_levels[~frozen_mask]
        
        if len(GREALM_levels) == 0 or len(my_lakes_levels) == 0:
            print(f'No valid data points for {GREALM_filename}. Skipping...')
            continue
        
        if anomaly_type == 'start':
            GREALM_levels = GREALM_levels - GREALM_levels[0]
            my_lakes_levels = my_lakes_levels - my_lakes_levels[0]
        elif anomaly_type == 'mean':
            GREALM_levels = GREALM_levels - np.nanmean(GREALM_levels)
            my_lakes_levels = my_lakes_levels - np.nanmean(my_lakes_levels)
        
        if not len(GREALM_levels) == len(my_lakes_levels):
            print(f'Lengths of the two arrays are not the same for {GREALM_filename}. Skipping...')
            continue
        num_points = len(GREALM_levels)
        
        # Remove 0.1 quantile and 0.9 quantile
        #GREALM_quantile_mask = (GREALM_levels > np.quantile(GREALM_levels, 0.1)) & (GREALM_levels < np.quantile(GREALM_levels, 0.9))
        #my_lakes_quantile_mask = (my_lakes_levels > np.quantile(my_lakes_levels, 0.1)) & (my_lakes_levels < np.quantile(my_lakes_levels, 0.9))
        #valid_mask = GREALM_quantile_mask & my_lakes_quantile_mask
        #GREALM_levels = GREALM_levels[valid_mask]
        #my_lakes_levels = my_lakes_levels[valid_mask]
        
        ax.scatter(GREALM_levels, my_lakes_levels, color='blue', s=24, alpha=0.6)
    
        # Plot the 1:1 line
        min_val = min(min(GREALM_levels), min(my_lakes_levels))
        max_val = max(max(GREALM_levels), max(my_lakes_levels))
        ax.plot([min_val, max_val], [min_val, max_val], 'k-', linewidth=2)
        
        # Calculate and plot the regression line
        slope, intercept, r_value, p_value, std_err = linregress(GREALM_levels, my_lakes_levels)
        ax.plot([min_val, max_val], [slope * min_val + intercept, slope * max_val + intercept], 'r--', linewidth=2)
        
        # Calculate CC and RMSE
        cc = spearmanr(GREALM_levels, my_lakes_levels)[0]
        rmse = calculate_rmse(my_lakes_levels, GREALM_levels)
        
        # Add text box with CC and RMSE
        textstr = f'N={num_points}\nCC={cc:.3f}\nRMSE={rmse:.2f}m'
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top')
        
        # Set labels, title, and limits
        ax.set_title(lake_id)
        ax.set_xlabel('Reference Levels')
        ax.set_ylabel('My Lake Levels')
        ax.set_xlim([min_val, max_val])
        ax.set_ylim([min_val, max_val])
        ax.grid(True)
    
    if save_path:
        if save_path.endswith('.png'):
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(save_path)
    else:
        plt.show()
        
def compare_corrected_water_levels_with_ICESat2(
    corrected_water_levels_csv_path,
    ICESat2_water_levels_monthly_median_csv_path,
    ICESat2_water_levels_monthly_min_csv_path,
    ICESat2_water_levels_monthly_max_csv_path,
    ICESat2_water_levels_monthly_count_csv_path,
    corrected_water_level_column_names,
    ICESat2_water_level_column_names,
    comparison_metrics_save_path=None,
    corrected_water_levels_lake_id_column_name='Hylak_id',
    ICESat2_water_levels_lake_id_column_name='id',
    correction_status_column_name='water_level_corrected',
    minimum_valid_points_for_comparison=5,
    corrected_water_level_avg_area_property_column_name='Lake_area',
    allow_half_outside=False
):
    corrected_water_levels_df = pd.read_csv(corrected_water_levels_csv_path)
    ICESat2_water_levels_median_df = pd.read_csv(ICESat2_water_levels_monthly_median_csv_path)
    ICESat2_water_levels_min_df = pd.read_csv(ICESat2_water_levels_monthly_min_csv_path)
    ICESat2_water_levels_max_df = pd.read_csv(ICESat2_water_levels_monthly_max_csv_path)
    ICESat2_water_levels_count_df = pd.read_csv(ICESat2_water_levels_monthly_count_csv_path)
    
    common_water_level_column_names = [col for col in corrected_water_level_column_names if col in ICESat2_water_level_column_names]
    mask_flag_column_names = [f'frozen_{col}' for col in common_water_level_column_names]
    
    comparison_metrics = {}
    comparison_metrics['Hylak_id'] = []
    comparison_metrics['N_points'] = []
    comparison_metrics['R_squared'] = []
    comparison_metrics['RMSE'] = []
    comparison_metrics['NRMSE'] = []
    comparison_metrics['pearsonr'] = []
    comparison_metrics['spearmanr'] = []
    comparison_metrics['Within_percentage'] = []
    comparison_metrics['Weighted_within_percentage'] = []
    comparison_metrics['Avg_area'] = []
    comparison_metrics['Lake_type'] = []
    for index, row in corrected_water_levels_df.iterrows():
        current_lake_id = row[corrected_water_levels_lake_id_column_name]
        if not row[correction_status_column_name]:
            #print(f'Lake {current_lake_id} has not been corrected. Skipping...')
            continue
        ICESat2_water_levels_median_row = ICESat2_water_levels_median_df[ICESat2_water_levels_median_df[ICESat2_water_levels_lake_id_column_name] == current_lake_id]
        ICESat2_water_levels_min_row = ICESat2_water_levels_min_df[ICESat2_water_levels_min_df[ICESat2_water_levels_lake_id_column_name] == current_lake_id]
        ICESat2_water_levels_max_row = ICESat2_water_levels_max_df[ICESat2_water_levels_max_df[ICESat2_water_levels_lake_id_column_name] == current_lake_id]
        ICESat2_water_levels_count_row = ICESat2_water_levels_count_df[ICESat2_water_levels_count_df[ICESat2_water_levels_lake_id_column_name] == current_lake_id]
        # assert the length of the rows are the same
        if len(ICESat2_water_levels_median_row) == 0 or len(ICESat2_water_levels_min_row) == 0 or len(ICESat2_water_levels_max_row) == 0 or len(ICESat2_water_levels_count_row) == 0:
            continue
        assert len(ICESat2_water_levels_median_row) == len(ICESat2_water_levels_min_row) == len(ICESat2_water_levels_max_row) == len(ICESat2_water_levels_count_row), 'The number of rows are not the same.'

        corrected_water_levels_array = row[common_water_level_column_names].to_numpy().astype(float)
        ICESat2_water_levels_median_array = ICESat2_water_levels_median_row[common_water_level_column_names].to_numpy().flatten().astype(float)
        ICESat2_water_levels_min_array = ICESat2_water_levels_min_row[common_water_level_column_names].to_numpy().flatten().astype(float)
        ICESat2_water_levels_max_array = ICESat2_water_levels_max_row[common_water_level_column_names].to_numpy().flatten().astype(float)
        ICESat2_water_levels_count_array = ICESat2_water_levels_count_row[common_water_level_column_names].to_numpy().flatten().astype(float)
        frozen_mask = row[mask_flag_column_names].to_numpy().flatten().astype(bool)
        
        nan_mask = np.isnan(corrected_water_levels_array) | np.isnan(ICESat2_water_levels_median_array) | frozen_mask
        
        corrected_water_levels_array = corrected_water_levels_array[~nan_mask]
        ICESat2_water_levels_median_array = ICESat2_water_levels_median_array[~nan_mask]
        ICESat2_water_levels_min_array = ICESat2_water_levels_min_array[~nan_mask]
        ICESat2_water_levels_max_array = ICESat2_water_levels_max_array[~nan_mask]
        ICESat2_water_levels_count_array = ICESat2_water_levels_count_array[~nan_mask]
        
        inf_mask = np.isinf(corrected_water_levels_array) | np.isinf(ICESat2_water_levels_median_array)
        corrected_water_levels_array = corrected_water_levels_array[~inf_mask]
        ICESat2_water_levels_median_array = ICESat2_water_levels_median_array[~inf_mask]
        ICESat2_water_levels_min_array = ICESat2_water_levels_min_array[~inf_mask]
        ICESat2_water_levels_max_array = ICESat2_water_levels_max_array[~inf_mask]
        ICESat2_water_levels_count_array = ICESat2_water_levels_count_array[~inf_mask]
        assert len(corrected_water_levels_array) == len(ICESat2_water_levels_median_array), 'The number of data points are not the same.'
        assert len(corrected_water_levels_array) == len(ICESat2_water_levels_min_array) == len(ICESat2_water_levels_max_array) == len(ICESat2_water_levels_count_array), 'The number of data points are not the same.'
        
        N = len(corrected_water_levels_array)
        if N < minimum_valid_points_for_comparison:
            #print(f'Lake {current_lake_id} does not have enough valid data points for comparison. Skipping...')
            continue
        
        index_with_largest_count = np.argmax(ICESat2_water_levels_count_array)
        # calculate anomaly
        corrected_water_levels_array = corrected_water_levels_array - corrected_water_levels_array[index_with_largest_count]
        ICESat2_water_levels_median_array = ICESat2_water_levels_median_array - ICESat2_water_levels_median_array[index_with_largest_count]
        
        ICESat2_water_levels_min_first = ICESat2_water_levels_min_array[index_with_largest_count]
        assert not isinstance(ICESat2_water_levels_min_first, np.ndarray), 'The first element of ICESat2_water_levels_min_array is an array.'
        ICESat2_water_levels_max_first = ICESat2_water_levels_max_array[index_with_largest_count]
        assert not isinstance(ICESat2_water_levels_max_first, np.ndarray), 'The first element of ICESat2_water_levels_max_array is an array.'
        ICESat2_water_levels_min_array = ICESat2_water_levels_min_array - ICESat2_water_levels_max_first
        ICESat2_water_levels_max_array = ICESat2_water_levels_max_array - ICESat2_water_levels_min_first
        
        ICESat2_water_levels_count_weight_array = ICESat2_water_levels_count_array / np.sum(ICESat2_water_levels_count_array)
        
        if allow_half_outside:
        
            corrected_water_levels_within_ICESat2_range_flag_array = (corrected_water_levels_array >= ICESat2_water_levels_min_array) & (corrected_water_levels_array <= ICESat2_water_levels_max_array)
            corrected_water_levels_without_but_not_too_much_flag_array = (corrected_water_levels_array < ICESat2_water_levels_min_array) & (corrected_water_levels_array >= ICESat2_water_levels_min_array - (ICESat2_water_levels_max_array - ICESat2_water_levels_min_array) / 2) | (corrected_water_levels_array > ICESat2_water_levels_max_array) & (corrected_water_levels_array <= ICESat2_water_levels_max_array + (ICESat2_water_levels_max_array - ICESat2_water_levels_min_array) / 2)
            corrected_water_levels_within_ICESat2_range_percentage = np.sum(corrected_water_levels_within_ICESat2_range_flag_array) / N * 100 + np.sum(corrected_water_levels_without_but_not_too_much_flag_array) / N * 50
            
            corrected_water_levels_within_ICESat2_range_weighted_percentage = np.sum(corrected_water_levels_within_ICESat2_range_flag_array * ICESat2_water_levels_count_weight_array) * 100 + np.sum(corrected_water_levels_without_but_not_too_much_flag_array * ICESat2_water_levels_count_weight_array) * 50
        
        else:
            corrected_water_levels_within_ICESat2_range_flag_array = (corrected_water_levels_array >= ICESat2_water_levels_min_array) & (corrected_water_levels_array <= ICESat2_water_levels_max_array)
            corrected_water_levels_within_ICESat2_range_percentage = np.sum(corrected_water_levels_within_ICESat2_range_flag_array) / N * 100
            
            corrected_water_levels_within_ICESat2_range_weighted_percentage = np.sum(corrected_water_levels_within_ICESat2_range_flag_array * ICESat2_water_levels_count_weight_array) * 100
        
        R_squared = r2_score(ICESat2_water_levels_median_array, corrected_water_levels_array)
        RMSE = np.sqrt(mean_squared_error(ICESat2_water_levels_median_array, corrected_water_levels_array))
        NRMSE = cal_nrmse(ICESat2_water_levels_median_array, corrected_water_levels_array)
        pearsonr_val = pearsonr(ICESat2_water_levels_median_array, corrected_water_levels_array)[0]
        spearmanr_val = spearmanr(ICESat2_water_levels_median_array, corrected_water_levels_array)[0]
        
        comparison_metrics['Hylak_id'].append(current_lake_id)
        comparison_metrics['N_points'].append(N)
        comparison_metrics['R_squared'].append(R_squared)
        comparison_metrics['RMSE'].append(RMSE)
        comparison_metrics['NRMSE'].append(NRMSE)
        comparison_metrics['pearsonr'].append(pearsonr_val)
        comparison_metrics['spearmanr'].append(spearmanr_val)
        comparison_metrics['Within_percentage'].append(corrected_water_levels_within_ICESat2_range_percentage)
        comparison_metrics['Weighted_within_percentage'].append(corrected_water_levels_within_ICESat2_range_weighted_percentage)
        comparison_metrics['Avg_area'].append(row[corrected_water_level_avg_area_property_column_name])
        comparison_metrics['Lake_type'].append(row['Lake_type'])
        
    comparison_metrics_df = pd.DataFrame(comparison_metrics)
    if comparison_metrics_save_path is not None:
        comparison_metrics_save_folder = os.path.dirname(comparison_metrics_save_path)
        if not os.path.exists(comparison_metrics_save_folder):
            os.makedirs(comparison_metrics_save_folder)
        comparison_metrics_df.to_csv(comparison_metrics_save_path, index=False)
    
    return comparison_metrics_df

def plot_comparison_metrics_of_ICESat2(
    metrics_csv_path,
    lake_size_bins=None,
    lake_size_labels=None,
    save_path=None
):
    comparison_metrics_df = pd.read_csv(metrics_csv_path)
    import seaborn as sns
    if lake_size_bins is None or lake_size_labels is None:
        fig, axs = plt.subplots(3, 2, figsize=(15, 10))
        ax_list = axs.flatten()
        
        for i, metric in enumerate(['RMSE', 'NRMSE', 'pearsonr', 'spearmanr', 'Within_percentage', 'Weighted_within_percentage']):
            current_ax = ax_list[i]
            sns.histplot(comparison_metrics_df[metric], bins=50, color='skyblue', ax=current_ax)
            current_ax.set_xlabel(metric)
            current_ax.set_ylabel('Frequency')
            current_ax.axvline(comparison_metrics_df[metric].median(), color='k', linestyle='dashed', linewidth=1)
            current_ax.text(0.6, 0.9, f'Median: {comparison_metrics_df[metric].median():.2f}', transform=current_ax.transAxes, verticalalignment='top', fontsize=12)
                
    elif lake_size_bins is not None and lake_size_labels is not None:
        # now plot using boxplot rather than histogram, grouped by lake size in the order of lake_size_labels
        # first add a column to the dataframe for lake size group
        comparison_metrics_df.loc[:, 'lake_size_group'] = pd.cut(comparison_metrics_df['Avg_area'], bins=lake_size_bins, labels=lake_size_labels)
        
        fig, axs = plt.subplots(3, 2, figsize=(15, 10))
        ax_list = axs.flatten()
        
        for i, metric in enumerate(['RMSE', 'NRMSE', 'pearsonr', 'spearmanr', 'Within_percentage', 'Weighted_within_percentage']):
            current_ax = ax_list[i]
            sns.boxplot(x='lake_size_group', y=metric, data=comparison_metrics_df, ax=current_ax, order=lake_size_labels)
            current_ax.set_xlabel('Lake size group')
            current_ax.set_ylabel(metric)

            # if metric is RMSE or NRMSE, set ylim to be 0 to 10 and 0 to 100 respectively
            if metric in ['RMSE']:
                current_ax.set_ylim(0, 10)
            elif metric in ['NRMSE']:
                current_ax.set_ylim(0, 100)
            elif metric in ['pearsonr', 'spearmanr']:
                current_ax.set_ylim(-1, 1)
            
    if save_path:
        if save_path.endswith('.png'):
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(save_path)