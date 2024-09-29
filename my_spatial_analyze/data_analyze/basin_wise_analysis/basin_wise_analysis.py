import geopandas as gpd
import pandas as pd 
import numpy as np
import os 
import sys
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
sys.path.append('/WORK/Codes/global_lake_area/')
from my_spatial_analyze.visualization import plot_grid

def merge_hydrobasins_of_multiple_regions(
    hydrobasins_shp_path_pattern,
    region_abbreviation_list,
    output_merged_pkl_path
):
    """
    Merge the HydroBASINS shapefiles of multiple regions into a single GeoDataFrame and save it as a pickle file.
    """
    hydrobasins_gdf_list = []
    for region_abbreviation in region_abbreviation_list:
        hydrobasins_shp_path = hydrobasins_shp_path_pattern.format(region_abbreviation)
        hydrobasins_gdf = gpd.read_file(hydrobasins_shp_path)
        hydrobasins_gdf_list.append(hydrobasins_gdf)
    
    merged_hydrobasins_gdf = pd.concat(hydrobasins_gdf_list)
    output_folder = os.path.dirname(output_merged_pkl_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    merged_hydrobasins_gdf.to_pickle(output_merged_pkl_path)

def convert_hydrobasins_to_pkl(
    hydrobasins_shp_path,
    output_pkl_path
):
    """
    Convert the HydroBASINS shapefile to a pandas DataFrame and save it as a pickle file.
    """
    hydrobasins_gdf = gpd.read_file(hydrobasins_shp_path)
    output_folder = os.path.dirname(output_pkl_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    hydrobasins_gdf.to_pickle(output_pkl_path)
    
def add_basin_wise_statistics_to_hydrobasins(
    hydrobasins_gdf,
    lake_lse_gdf,
    column_name_to_calculate,
    statistics_type='median',
    basin_id_column_name='HYBAS_ID'
):
    """
    Add basin-wise statistics to the HydroBASINS GeoDataFrame.
    
    Args:
        hydrobasins_gdf (geopandas.GeoDataFrame): A GeoDataFrame representing the HydroBASINS dataset.
        lake_lse_gdf (geopandas.GeoDataFrame): A GeoDataFrame representing the lake LSE dataset.
        column_name_to_calculate (str): The name of the column in lake_lse_gdf to calculate the statistics for.
        statistics_type (str): The type of statistics to calculate. Can be 'median' or 'mean'.
    
    Returns:
        geopandas.GeoDataFrame: The HydroBASINS GeoDataFrame with the added column.
    """
    if statistics_type not in ['median', 'mean', 'sum', 'mode', 'positive_percentage']:
        raise ValueError("statistics_type must be one of 'median', 'mean', 'sum', 'mode', or 'positive_percentage'")
    
    # Check if the HydroBASINS and lake LSE GeoDataFrames have the same CRS
    if hydrobasins_gdf.crs != lake_lse_gdf.crs:
        raise ValueError("The CRS of hydrobasins_gdf and lake_lse_gdf must be the same")
    
    # Perform a spatial join between the HydroBASINS and lake LSE GeoDataFrames
    hydrobasins_with_lake_lse = gpd.sjoin(hydrobasins_gdf, lake_lse_gdf, how='inner', op='intersects')
    
    # Group the data by the basin ID and calculate the statistics
    if statistics_type == 'median':
        basin_wise_statistics = hydrobasins_with_lake_lse.groupby(basin_id_column_name)[column_name_to_calculate].median()
        basin_wise_statistics.name = f'{column_name_to_calculate}_median'
    elif statistics_type == 'mean':
        basin_wise_statistics = hydrobasins_with_lake_lse.groupby(basin_id_column_name)[column_name_to_calculate].mean()
        basin_wise_statistics.name = f'{column_name_to_calculate}_mean'
    elif statistics_type == 'sum':
        basin_wise_statistics = hydrobasins_with_lake_lse.groupby(basin_id_column_name)[column_name_to_calculate].sum()
        basin_wise_statistics.name = f'{column_name_to_calculate}_sum'
    elif statistics_type == 'mode':
        def mode(x):
            # remove rows with nans first
            x = x.dropna()
            # if empty return np.nan
            if x.empty:
                return np.nan
            return x.value_counts().sort_values(ascending=False).index[0]
        basin_wise_statistics = hydrobasins_with_lake_lse.groupby(basin_id_column_name).apply(lambda x: mode(x[column_name_to_calculate]))
        basin_wise_statistics.name = f'{column_name_to_calculate}_mode'
    elif statistics_type == 'positive_percentage':
        def positive_percentage(x):
            # remove rows with nans first
            x = x.dropna()
            if x.empty:
                return np.nan
            positive_count = (x > 0).sum()
            negtive_count = (x < 0).sum()
            positive_percentage = positive_count / (positive_count + negtive_count) * 100
            return positive_percentage
        basin_wise_statistics = hydrobasins_with_lake_lse.groupby(basin_id_column_name).apply(lambda x: positive_percentage(x[column_name_to_calculate]))
        basin_wise_statistics.name = f'{column_name_to_calculate}_positive_percentage'
    print(basin_wise_statistics)
    # Add the basin-wise statistics to the HydroBASINS GeoDataFrame
    hydrobasins_gdf = hydrobasins_gdf.merge(basin_wise_statistics, left_on=basin_id_column_name, right_index=True, suffixes=('', f'_{statistics_type}'))
    
    return hydrobasins_gdf

def calculate_and_add_reservoir_contribution_to_hydrobasins(
    hydrobasins_gdf,
    lake_lse_gdf,
    to_calculate_column_name,
    lake_type_column='Lake_type',
    basin_id_column_name='HYBAS_ID'
):
    # Check if the CRS of both GeoDataFrames is the same
    if hydrobasins_gdf.crs != lake_lse_gdf.crs:
        raise ValueError("The CRS of hydrobasins_gdf and lake_lse_gdf must be the same")
    
    # Perform a spatial join
    hydrobasins_with_lake_lse = gpd.sjoin(
        hydrobasins_gdf, lake_lse_gdf, how='inner', op='intersects'
    )
    
    # Fill NaN values with zeros
    hydrobasins_with_lake_lse[to_calculate_column_name] = (
        hydrobasins_with_lake_lse[to_calculate_column_name].fillna(0)
    )
    
    # Filter out reservoirs (Lake_type != 1)
    hydrobasins_with_lake_lse_reservoirs = hydrobasins_with_lake_lse[
        hydrobasins_with_lake_lse[lake_type_column] != 1
    ]
    hydrobasins_with_lake_lse_reservoirs[to_calculate_column_name] = (
        hydrobasins_with_lake_lse_reservoirs[to_calculate_column_name].fillna(0)
    )
    
    # Group by basin ID and sum
    total_sums_reservoirs = hydrobasins_with_lake_lse_reservoirs.groupby(
        basin_id_column_name
    )[to_calculate_column_name].sum()
    
    total_sums_sll = hydrobasins_with_lake_lse.groupby(
        basin_id_column_name
    )[to_calculate_column_name].sum()
    
    # Handle division by zero
    total_sums_sll = total_sums_sll.replace(0, np.nan)
    
    # Calculate reservoir contribution
    reservoir_contribution = (total_sums_reservoirs / total_sums_sll) * 100
    reservoir_contribution.replace([np.inf, -np.inf], np.nan, inplace=True)
    reservoir_contribution.name = f'{to_calculate_column_name}_reservoir_contribution'
    
    # Count the number of reservoirs per basin
    reservoir_counts = hydrobasins_with_lake_lse_reservoirs.groupby(
        basin_id_column_name
    ).size()
    reservoir_counts.name = 'reservoir_count'
    
    # Merge the reservoir contribution and counts into hydrobasins_gdf
    hydrobasins_gdf = hydrobasins_gdf.merge(
        reservoir_contribution, left_on=basin_id_column_name, right_index=True, how='left'
    )
    hydrobasins_gdf = hydrobasins_gdf.merge(
        reservoir_counts, left_on=basin_id_column_name, right_index=True, how='left'
    )
    
    # Replace NaN counts with zero and convert to integer
    hydrobasins_gdf['reservoir_count'] = hydrobasins_gdf['reservoir_count'].fillna(0).astype(int)
    
    return hydrobasins_gdf

def plot_hydrobasins_map(
    hydrobasins_gdf,
    column_to_plot,
    cmap=None,
    projection=ccrs.PlateCarree(),
    ax=None
):
    """
    Plot the HydroBASINS GeoDataFrame on a map.
    
    Args:
        hydrobasins_gdf (geopandas.GeoDataFrame): A GeoDataFrame representing the HydroBASINS dataset.
        column_to_plot (str): The name of the column in hydrobasins_gdf to plot.
        cmap (str): The name of the colormap to use.
        projection (cartopy.crs.Projection): The projection to use for the map.
        ax (matplotlib.axes.Axes): The axes to plot on. If None, a new figure and axes will be created.
    """
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': projection})
    else:
        fig = ax.figure
    
    if cmap is None:
        cmap = 'viridis'
    
    hydrobasins_gdf.plot(column=column_to_plot, cmap=cmap, ax=ax, legend=True)
    
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.COASTLINE)
    land_feature = cfeature.NaturalEarthFeature(
        'physical', 'land', '50m',
        edgecolor='face',
        facecolor='grey'
    )

    ax.add_feature(land_feature, zorder=1.5)
    
    land_feature = cfeature.NaturalEarthFeature(
        'physical', 'land', '50m',
        edgecolor='face',
        facecolor='grey'
    )

    ax.add_feature(land_feature, zorder=1.5)
    
    return fig, ax

def plot_hydrobasins_map_new(
    hydrobasins_gdf,
    ax=None,
    projection=ccrs.PlateCarree(),
    title='HydroBASINS Map',
    cmap=None,
    column_to_plot=None,
    vmin=None,
    vmax=None,
    div_colorbar=True,
    use_log_scale_color=False,
    linthresh_for_log_color=None,
    use_discrete_color=False,
    discrete_bins=None,
    gridlines=True,
    extent=None,
    add_rivers=True,
    add_ocean=False,
    set_global=True,
    fix_for_antimeridian=False,
    draw_colorbar=True,
    edgecolor='black',
    show=True
):
    plot_grid(
        grid_gdf=hydrobasins_gdf,
        ax=ax,
        projection=projection,
        title=title,
        cmap=cmap,
        color_column=column_to_plot,
        vmin=vmin,
        vmax=vmax,
        edgecolor=edgecolor,
        div_colorbar=div_colorbar,
        use_log_scale_color=use_log_scale_color,
        linthresh_for_log_color=linthresh_for_log_color,
        use_discrete_color=use_discrete_color,
        discrete_bins=discrete_bins,
        gridlines=gridlines,
        extent=extent,
        add_rivers=add_rivers,
        add_ocean=add_ocean,
        set_global=set_global,
        fix_for_antimeridian=fix_for_antimeridian,
        draw_colorbar=draw_colorbar,
        show=show
    )