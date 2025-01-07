import basin_wise_analysis
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from datetime import datetime
from dateutil.relativedelta import relativedelta

if __name__ == '__main__':
    area_time_series_included = False
    calculated_hydrobasins_pkl_path = '/WORK/Data/global_lake_area/hydrobasins/merged/hydrobasins_lev03_with_statistics.pkl'
    calculated_hydrobasins_csv_path = '/WORK/Data/global_lake_area/hydrobasins/merged/hydrobasins_lev03_with_statistics.csv'
    
    merged_hydrobasins_pkl_path = '/WORK/Data/global_lake_area/hydrobasins/merged/hydrobasins_lev03.pkl'
    lake_lse_pkl_path = '/WORK/Data/global_lake_area/area_csvs/lakes/csv/lakes_all_with_aridity_and_permafrost_type.csv'
    merged_hydrobasins_gdf = pd.read_pickle(merged_hydrobasins_pkl_path)
    print(merged_hydrobasins_gdf.columns.tolist())
    lake_lse_gdf = pd.read_csv(lake_lse_pkl_path)
    lake_lse_lat_column = 'Pour_lat'
    lake_lse_lon_column = 'Pour_long'
    lake_lse_gdf['geometry'] = [Point(xy) for xy in zip(lake_lse_gdf[lake_lse_lon_column], lake_lse_gdf[lake_lse_lat_column])]
    lake_lse_gdf = gpd.GeoDataFrame(lake_lse_gdf, crs='EPSG:4326')
    #project hydrobasins to lake_lse_gdf's crs if not the same
    if lake_lse_gdf.crs != merged_hydrobasins_gdf.crs:
        merged_hydrobasins_gdf = merged_hydrobasins_gdf.to_crs(lake_lse_gdf.crs)
    column_names_and_statistics_to_calculate = {
        'linear_trend_of_standard_deviation_per_period': 'sum',
        'linear_trend_of_standard_deviation_percentage_per_period': 'median',
        'linear_trend_of_annual_mean': 'sum',
        'linear_trend_of_stl_trend_per_period': 'sum',
        'linear_trend_of_stl_trend_percentage_per_period': 'median',
        'linear_trend_of_annual_mean_first_difference': 'sum',
        'linear_trend_of_annual_mean_first_difference_percentage': 'median',
        'ari_ix_uav': 'median',
        'EXTENT': 'mode'
    }
    
    if area_time_series_included:
        area_start_date = '2001-01-01'
        area_end_date = '2024-01-01'
        date_fmt = '%Y-%m-%d'
        area_start_date = datetime.strptime(area_start_date, date_fmt)
        area_end_date = datetime.strptime(area_end_date, date_fmt)
        current_date = area_start_date
        area_columns = []
        while current_date < area_end_date:
            area_columns.append(current_date.strftime(date_fmt))
            current_date += relativedelta(months=1)
        
        for area_column in area_columns:
            column_names_and_statistics_to_calculate[area_column] = 'sum'
    
    for column_name, statistics_type in column_names_and_statistics_to_calculate.items():
        merged_hydrobasins_gdf = basin_wise_analysis.add_basin_wise_statistics_to_hydrobasins(
            hydrobasins_gdf=merged_hydrobasins_gdf,
            lake_lse_gdf=lake_lse_gdf,
            column_name_to_calculate=column_name,
            statistics_type=statistics_type
        )
        
    # add relative contributions of reservoirs to total seasonality to compare with Cooley et al. (2021) Fig. 3
    merged_hydrobasins_gdf = basin_wise_analysis.calculate_and_add_reservoir_contribution_to_hydrobasins(
        hydrobasins_gdf=merged_hydrobasins_gdf,
        lake_lse_gdf=lake_lse_gdf,
        to_calculate_column_name='mean_seasonal_amplitude'
    )
        
    merged_hydrobasins_gdf.to_pickle(calculated_hydrobasins_pkl_path)
    # before saving to csv, drop the geometry column
    merged_hydrobasins_gdf.drop('geometry', axis=1).to_csv(calculated_hydrobasins_csv_path, index=False)