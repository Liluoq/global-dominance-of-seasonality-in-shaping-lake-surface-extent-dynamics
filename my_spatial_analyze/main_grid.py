import geopandas as gpd
import attach_geometry_and_generate_grid as ag
import visualization as vis
import numpy as np
from datetime import datetime
import os
import matplotlib.pyplot as plt
import pandas as pd
from dateutil import relativedelta
import cartopy.crs as ccrs



if __name__ == '__main__':
    ATTACH_GEOMETRY = False
    GENERATE_GRID = False
    TIME_SERIES_ANALYSIS_AT_GRID_SCALE = False
    VISUALIZATION = True
    ONLY_WITH_PANELS = False
    
    basin_id = 'all_medium'
    wkdir = '/WORK/Data/global_lake_area'
    concatenated_area_csv_path = os.path.join(wkdir, f'area_csvs/concatenated/{basin_id}_concatenated.csv')
    lake_shp_path = os.path.join(wkdir, f'lake_shps/hylak_buffered_{basin_id}.shp')
    lake_id_field_in_csv = 'Hylak_id'
    lake_id_field_in_shp = 'Hylak_id'
    common_id_name = 'Hylak_id'
    cast_to_centroid = True
    output_concatenated_pkl_path = os.path.join(wkdir, f'area_csvs/concatenated_areas_with_lakes/concatenated_areas_with_lakes_{basin_id}.pkl')
    verbose = 2
    
    if ATTACH_GEOMETRY:
        ag.attach_geometry_to_concatenated_areas(
            concatenated_area_csv=concatenated_area_csv_path,
            lake_shp_path=lake_shp_path,
            lake_id_field_in_csv=lake_id_field_in_csv,
            lake_id_field_in_shp=lake_id_field_in_shp,
            common_id_name=common_id_name,
            cast_to_centroid=cast_to_centroid,
            output_pkl_path=output_concatenated_pkl_path,
            verbose=verbose
        )
    
    start_date = '2001-01-01'
    end_date = '2024-01-01'
    date_fmt = '%Y-%m-%d'
    start_date = datetime.strptime(start_date, date_fmt)
    end_date = datetime.strptime(end_date, date_fmt)
    dates = [current_date.strftime(date_fmt) for current_date in 
             [start_date + relativedelta.relativedelta(months=i) for i in range((end_date.year - start_date.year) * 12 + end_date.month - start_date.month)]]
    output_grid_pkl_path = os.path.join(wkdir, f'area_csvs/grids/grid_{basin_id}.pkl')
    
    if GENERATE_GRID:
        ag.generate_grid_from_geometry_added_concatenated_areas(
            geometry_added_concatenated_areas_path=output_concatenated_pkl_path,
            grid_size=0.25,
            area_columns=dates,
            output_pkl_path=output_grid_pkl_path,
            grid_extent=None,
            geometry_to_use_column='geometry',
            verbose=2
        )
        
    if TIME_SERIES_ANALYSIS_AT_GRID_SCALE:
        analyzed_gdf_path = os.path.join(wkdir, f'area_csvs/analyzed_grids/analyzed_grid_{basin_id}.pkl')
    else:
        analyzed_gdf_path = os.path.join(wkdir, f'area_csvs/grids/pkl/grid_{basin_id}.pkl')
    if TIME_SERIES_ANALYSIS_AT_GRID_SCALE:
        gdf = pd.read_pickle(output_grid_pkl_path)
        if not isinstance(gdf, gpd.GeoDataFrame):
            raise ValueError('The loaded object is not a GeoDataFrame')
        analyzed_gdf = ag.time_series_analysis_on_df(
            df=gdf,
            time_series_columns=dates,
            type_of_analysis='linear_trend_per_period',
            output_column_name='linear_trend_km2_per_year',
            period=12,
            unit_scale=1e-6
        )
        
        analyzed_gdf = ag.time_series_analysis_on_df(
            df=analyzed_gdf,
            time_series_columns=dates,
            type_of_analysis='linear_trend_of_standard_deviation_per_period',
            output_column_name='linear_trend_of_period_standard_deviation_km2',
            period=12,
            unit_scale=1e-6
        )
        
        analyzed_gdf = ag.time_series_analysis_on_df(
            df=analyzed_gdf,
            time_series_columns=dates,
            type_of_analysis='linear_trend_of_stl_seasonal_max_minus_min_per_period',
            output_column_name='linear_trend_of_std_seasonal_period_max_minus_min_km2',
            period=12,
            unit_scale=1e-6
        )
        
        analyzed_gdf = ag.time_series_analysis_on_df(
            df=analyzed_gdf,
            time_series_columns=dates,
            type_of_analysis='linear_trend_of_stl_trend_per_period',
            output_column_name='linear_trend_of_stl_per_trend_km2_per_year',
            period=12,
            unit_scale=1e-6
        )
        
        analyzed_gdf = ag.time_series_analysis_on_df(
            df=analyzed_gdf,
            time_series_columns=dates,
            type_of_analysis='linear_trend_of_period_coefficient_of_variation',
            output_column_name='linear_trend_of_period_coefficient_of_variation_changes_per_year',
            period=12,
            unit_scale=1e-6
        )
        
        sorted_gdf = analyzed_gdf.sort_values('lake_count', ascending=False)
        print(sorted_gdf.linear_trend_km2_per_year.head())
        
        analyzed_gdf_folder = os.path.dirname(analyzed_gdf_path)
        if not os.path.exists(analyzed_gdf_folder):
            os.makedirs(analyzed_gdf_folder)
        analyzed_gdf.to_pickle(analyzed_gdf_path)
        
    if VISUALIZATION:
        save_type = 'pdf'
        set_global = False
        projection = ccrs.PlateCarree()
        save_path = os.path.join(wkdir, f'area_csvs/test_figs/analyzed_grid_mean_area_{basin_id}.{save_type}')
        save_folder = os.path.dirname(save_path)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        gdf = pd.read_pickle(analyzed_gdf_path)
        gdf['linear_trend_of_standard_deviation_percentage_per_period_median_decade'] = gdf['linear_trend_of_standard_deviation_percentage_per_period_median'] * 10
        gdf['linear_trend_of_annual_mean_first_difference_percentage_median_decade'] = gdf['linear_trend_of_annual_mean_first_difference_percentage_median'] * 10
        # convert to true percentage (because currently in area postprocessing, i didn't multiply 100 to columns below)
        gdf['mean_seasonal_amplitude_percentage_median'] = gdf['mean_seasonal_amplitude_percentage_median'] * 100
        print(gdf.columns.tolist())
        #gdf.to_csv(os.path.join(wkdir, f'area_csvs/test_figs/analyzed_grid_{basin_id}.csv'), index=False)
        if not ONLY_WITH_PANELS:
            ax = vis.plot_grid(
                grid_gdf=gdf.copy(),
                projection=projection,
                save_path=save_path,
                color_column='mean_area_sum',
                gridlines=True,
                add_rivers=True,
                use_log_scale_color=True,
                cmap='viridis',
                div_colorbar=False,
                set_global=set_global,
                extent=[-180, -60, 180, 90]
            )
            
            save_path = os.path.join(wkdir, f'area_csvs/test_figs/analyzed_grid_mean_annual_std_{basin_id}.{save_type}')
            ax = vis.plot_grid(
                grid_gdf=gdf.copy(),
                projection=projection,
                save_path=save_path,
                color_column='mean_seasonal_amplitude_sum',
                gridlines=True,
                add_rivers=True,
                use_log_scale_color=True,
                set_global=set_global,
                extent=[-180, -60, 180, 90],
                cmap='viridis',
                div_colorbar=False,
            )
            
            save_path = os.path.join(wkdir, f'area_csvs/test_figs/analyzed_grid_mean_annual_std_percentage_{basin_id}.{save_type}')
            ax = vis.plot_grid(
                grid_gdf=gdf.copy(),
                projection=projection,
                save_path=save_path,
                color_column='mean_seasonal_amplitude_percentage_median',
                gridlines=True,
                add_rivers=True,
                use_log_scale_color=False,
                set_global=set_global,
                extent=[-180, -60, 180, 90],
                cmap='Blues',
                div_colorbar=False,
                use_discrete_color=True,
                discrete_bins=[0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
            )
            
            save_path = os.path.join(wkdir, f'area_csvs/test_figs/analyzed_grid_mean_annual_apportion_entropy_percentage_{basin_id}.{save_type}')
            ax = vis.plot_grid(
                grid_gdf=gdf.copy(),
                projection=projection,
                save_path=save_path,
                color_column='mean_apportion_entropy_percentage_median',
                gridlines=True,
                add_rivers=True,
                use_log_scale_color=False,
                set_global=set_global,
                extent=[-180, -60, 180, 90],
                cmap='Blues',
                div_colorbar=False,
            )
            
            save_path = os.path.join(wkdir, f'area_csvs/test_figs/analyzed_grid_mean_annual_mean_first_difference_{basin_id}.{save_type}')
            ax = vis.plot_grid(
                grid_gdf=gdf.copy(),
                projection=projection,
                save_path=save_path,
                color_column='mean_annual_mean_first_difference_sum',
                gridlines=True,
                add_rivers=True,
                use_log_scale_color=True,
                set_global=set_global,
                extent=[-180, -60, 180, 90],
                cmap='viridis',
                div_colorbar=False,
            )
            
            save_path = os.path.join(wkdir, f'area_csvs/test_figs/percentage_of_seasonality_increase_lakes_{basin_id}.{save_type}')
            ax = vis.plot_grid(
                grid_gdf=gdf.copy(),
                projection=projection,
                save_path=save_path,
                color_column='annual_std_increase_percentage',
                gridlines=True,
                add_rivers=True,
                set_global=set_global,
                extent=[-180, -60, 180, 90],
                cmap='Blues',
                div_colorbar=False,
                use_discrete_color=True,
                discrete_bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            )
            
            save_path = os.path.join(wkdir, f'area_csvs/test_figs/analyzed_grid_annual_season_percentage_decade_{basin_id}.{save_type}')
            ax = vis.plot_grid(
                grid_gdf=gdf.copy(),
                projection=projection,
                save_path=save_path,
                color_column='linear_trend_of_standard_deviation_percentage_per_period_median_decade',
                gridlines=True,
                add_rivers=True,
                use_log_scale_color=False,
                set_global=set_global,
                extent=[-180, -60, 180, 90],
                use_discrete_color=True,
                discrete_bins=[-20, -17.5, -15, -12.5, -10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20]
            )
            
            save_path = os.path.join(wkdir, f'area_csvs/test_figs/analyzed_grid_annual_season_amplitude_trend_{basin_id}.{save_type}')
            ax = vis.plot_grid(
                grid_gdf=gdf.copy(),
                projection=projection,
                save_path=save_path,
                color_column='linear_trend_of_standard_deviation_per_period_sum',
                gridlines=True,
                add_rivers=True,
                use_discrete_color=True,
                discrete_bins=[-100, -10, -1, -0.1, 0, 0.1, 1, 10, 100],
                set_global=set_global,
                extent=[-180, -60, 180, 90]
            )
            
            save_path = os.path.join(wkdir, f'area_csvs/test_figs/analyzed_grid_stl_trend_{basin_id}.{save_type}')
            ax = vis.plot_grid(
                grid_gdf=gdf.copy(),
                projection=projection,
                save_path=save_path,
                color_column='linear_trend_of_stl_trend_per_period_sum',
                gridlines=True,
                add_rivers=True,
                use_log_scale_color=True,
                set_global=set_global,
                extent=[-180, -60, 180, 90]
            )
            
            save_path = os.path.join(wkdir, f'area_csvs/test_figs/analyzed_grid_apportion_entropy_percentage_trend_{basin_id}.{save_type}')
            ax = vis.plot_grid(
                grid_gdf=gdf.copy(),
                projection=projection,
                save_path=save_path,
                color_column='linear_trend_of_apportion_entropy_percentage_median',
                gridlines=True,
                add_rivers=True,
                use_log_scale_color=False,
                set_global=set_global,
                extent=[-180, -60, 180, 90],
                use_discrete_color=True,
                discrete_bins=[-0.25, -0.20, -0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15, 0.20, 0.25]
            )
            
            save_path = os.path.join(wkdir, f'area_csvs/test_figs/analyzed_grid_annual_mean_trend_{basin_id}.{save_type}')
            ax = vis.plot_grid(
                grid_gdf=gdf.copy(),
                projection=projection,
                save_path=save_path,
                color_column='linear_trend_of_annual_mean_sum',
                gridlines=True,
                add_rivers=True,
                use_log_scale_color=True,
                set_global=set_global,
                extent=[-180, -60, 180, 90]
            )
            
            save_path = os.path.join(wkdir, f'area_csvs/test_figs/analyzed_grid_annual_mean_first_difference_trend_{basin_id}.{save_type}')
            ax = vis.plot_grid(
                grid_gdf=gdf.copy(),
                projection=projection,
                save_path=save_path,
                color_column='linear_trend_of_annual_mean_first_difference_sum',
                gridlines=True,
                add_rivers=True,
                use_log_scale_color=True,
                set_global=set_global,
                extent=[-180, -60, 180, 90]
            )
            
            save_path = os.path.join(wkdir, f'area_csvs/test_figs/analyzed_grid_annual_mean_first_difference_percentage_trend_{basin_id}.{save_type}')
            ax = vis.plot_grid(
                grid_gdf=gdf.copy(),
                projection=projection,
                save_path=save_path,
                color_column='linear_trend_of_annual_mean_first_difference_percentage_median',
                gridlines=True,
                add_rivers=True,
                use_log_scale_color=False,
                set_global=set_global,
                extent=[-180, -60, 180, 90],
                use_discrete_color=True,
                discrete_bins=[-5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
            )
            
            save_path = os.path.join(wkdir, f'area_csvs/test_figs/analyzed_grid_annual_mean_first_difference_percentage_trend_decade_{basin_id}.{save_type}')
            ax = vis.plot_grid(
                grid_gdf=gdf.copy(),
                projection=projection,
                save_path=save_path,
                color_column='linear_trend_of_annual_mean_first_difference_percentage_median_decade',
                gridlines=True,
                add_rivers=True,
                use_log_scale_color=False,
                set_global=set_global,
                extent=[-180, -60, 180, 90],
                use_discrete_color=True,
                discrete_bins=[-20, -17.5, -15, -12.5, -10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20]
            )
        
        # Below starts plotting with panels
        """
        save_path = os.path.join(wkdir, f'area_csvs/test_figs/analyzed_grid_seasonal_amplitude_trend_panel_{basin_id}.png')
        main_plot_kwargs = {
            'projection': projection,
            'color_column': 'grid_linear_trend_of_standard_deviation_per_period',
            'gridlines': True,
            'add_rivers': True,
            'use_log_scale_color': True,
            'set_global': set_global,
            'significant_column': 'grid_rao_mk_test_on_standard_deviation_p',
            'significant_threshold': 0.05,
            'draw_colorbar': False,
            'title': 'Linear Trend of Annual Std.',
            'cmap': 'RdBu',
        }
        inset_axes_layout = {
            'main': {'width': '79.5%', 'height': '80%', 'loc': 'lower left', 
                     'bbox_to_anchor': (0.0, 0.0, 1.0, 1.0), 'borderpad': 0.0},
            'top': {'width': '79.5%', 'height': '15%', 'loc': 'upper left', 
                    'bbox_to_anchor': (0.0, 0.0, 1.0, 1.0), 'borderpad': 0.0},
            'right': {'width': '15.5%', 'height': '80%', 'loc': 'lower right', 
                      'bbox_to_anchor': (0.0, 0.0, 1.0, 1.0), 'borderpad': 0.0},
            'colorbar_ax': {'width': '18%', 'height': '3.5%', 'loc': 'upper right',
                            'bbox_to_anchor': (0.0, 0.0, 1.0, 1.0), 'borderpad': 0.0}
        }
        rdbu_cmap = plt.get_cmap('RdBu')
        color_15 = rdbu_cmap(0.15)
        color_85 = rdbu_cmap(0.85)
        ax = vis.plot_grid_with_panels(
            grid_gdf=gdf.copy(),
            panel_columns=[main_plot_kwargs['color_column']],
            panel_stat_lists=[['mean_positive', 'mean_negative']],
            panel_steps=1.0,
            panel_legend_lists=[['Increase', 'Decrease']],
            panel_color_lists=[[color_85, color_15]],
            main_plot_kwargs=main_plot_kwargs,
            inset_axes_layout=inset_axes_layout,
            save_path=save_path,
            additional_colorbar_kwargs={'orientation': 'horizontal', 'extend': 'both'}
        )
        
        save_path = os.path.join(wkdir, f'area_csvs/test_figs/analyzed_grid_stl_linear_trend_panel_{basin_id}.png')
        main_plot_kwargs = {
            'projection': projection,
            'color_column': 'grid_linear_trend_of_stl_trend_per_period',
            'gridlines': True,
            'add_rivers': True,
            'use_log_scale_color': True,
            'set_global': set_global,
            'significant_column': 'grid_rao_mk_test_on_stl_linear_trend_p',
            'significant_threshold': 0.05,
            'draw_colorbar': False,
            'title': 'Linear Trend of STL Trend (/yr)',
            'cmap': 'RdBu',
        }
        inset_axes_layout = {
            'main': {'width': '79.5%', 'height': '80%', 'loc': 'lower left', 
                     'bbox_to_anchor': (0.0, 0.0, 1.0, 1.0), 'borderpad': 0.0},
            'top': {'width': '79.5%', 'height': '15%', 'loc': 'upper left', 
                    'bbox_to_anchor': (0.0, 0.0, 1.0, 1.0), 'borderpad': 0.0},
            'right': {'width': '15.5%', 'height': '80%', 'loc': 'lower right', 
                      'bbox_to_anchor': (0.0, 0.0, 1.0, 1.0), 'borderpad': 0.0},
            'colorbar_ax': {'width': '18%', 'height': '3.5%', 'loc': 'upper right',
                            'bbox_to_anchor': (0.0, 0.0, 1.0, 1.0), 'borderpad': 0.0}
        }
        rdbu_cmap = plt.get_cmap('RdBu')
        color_15 = rdbu_cmap(0.15)
        color_85 = rdbu_cmap(0.85)
        ax = vis.plot_grid_with_panels(
            grid_gdf=gdf.copy(),
            panel_columns=[main_plot_kwargs['color_column']],
            panel_stat_lists=[['mean_positive', 'mean_negative']],
            panel_steps=1.0,
            panel_legend_lists=[['Increase', 'Decrease']],
            panel_color_lists=[[color_85, color_15]],
            main_plot_kwargs=main_plot_kwargs,
            inset_axes_layout=inset_axes_layout,
            save_path=save_path,
            additional_colorbar_kwargs={'orientation': 'horizontal', 'extend': 'both'}
        )
        
        save_path = os.path.join(wkdir, f'area_csvs/test_figs/analyzed_grid_annual_mean_trend_panel_{basin_id}.png')
        main_plot_kwargs = {
            'projection': projection,
            'color_column': 'grid_linear_trend_of_annual_mean',
            'gridlines': True,
            'add_rivers': True,
            'use_log_scale_color': True,
            'set_global': set_global,
            'significant_column': 'grid_rao_mk_test_on_annual_mean_p',
            'significant_threshold': 0.05,
            'draw_colorbar': False,
            'title': 'Linear Trend of Annual Mean',
            'cmap': 'RdBu',
        }
        inset_axes_layout = {
            'main': {'width': '79.5%', 'height': '80%', 'loc': 'lower left', 
                     'bbox_to_anchor': (0.0, 0.0, 1.0, 1.0), 'borderpad': 0.0},
            'top': {'width': '79.5%', 'height': '15%', 'loc': 'upper left', 
                    'bbox_to_anchor': (0.0, 0.0, 1.0, 1.0), 'borderpad': 0.0},
            'right': {'width': '15.5%', 'height': '80%', 'loc': 'lower right', 
                      'bbox_to_anchor': (0.0, 0.0, 1.0, 1.0), 'borderpad': 0.0},
            'colorbar_ax': {'width': '18%', 'height': '3.5%', 'loc': 'upper right',
                            'bbox_to_anchor': (0.0, 0.0, 1.0, 1.0), 'borderpad': 0.0}
        }
        rdbu_cmap = plt.get_cmap('RdBu')
        color_15 = rdbu_cmap(0.15)
        color_85 = rdbu_cmap(0.85)
        ax = vis.plot_grid_with_panels(
            grid_gdf=gdf.copy(),
            panel_columns=[main_plot_kwargs['color_column']],
            panel_stat_lists=[['mean_positive', 'mean_negative']],
            panel_steps=1.0,
            panel_legend_lists=[['Increase', 'Decrease']],
            panel_color_lists=[[color_85, color_15]],
            main_plot_kwargs=main_plot_kwargs,
            inset_axes_layout=inset_axes_layout,
            save_path=save_path,
            additional_colorbar_kwargs={'orientation': 'horizontal', 'extend': 'both'}
        )
        
        save_path = os.path.join(wkdir, f'area_csvs/test_figs/analyzed_grid_annual_mean_first_difference_trend_panel_{basin_id}.png')
        main_plot_kwargs = {
            'projection': projection,
            'color_column': 'grid_linear_trend_of_annual_mean_first_difference',
            'gridlines': True,
            'add_rivers': True,
            'use_log_scale_color': True,
            'set_global': set_global,
            'significant_column': 'grid_rao_mk_test_on_annual_mean_first_difference_p',
            'significant_threshold': 0.05,
            'draw_colorbar': False,
            'title': 'Linear Trend of Annual Mean First Difference',
            'cmap': 'RdBu',
        }
        inset_axes_layout = {
            'main': {'width': '79.5%', 'height': '80%', 'loc': 'lower left', 
                     'bbox_to_anchor': (0.0, 0.0, 1.0, 1.0), 'borderpad': 0.0},
            'top': {'width': '79.5%', 'height': '15%', 'loc': 'upper left', 
                    'bbox_to_anchor': (0.0, 0.0, 1.0, 1.0), 'borderpad': 0.0},
            'right': {'width': '15.5%', 'height': '80%', 'loc': 'lower right', 
                      'bbox_to_anchor': (0.0, 0.0, 1.0, 1.0), 'borderpad': 0.0},
            'colorbar_ax': {'width': '18%', 'height': '3.5%', 'loc': 'upper right',
                            'bbox_to_anchor': (0.0, 0.0, 1.0, 1.0), 'borderpad': 0.0}
        }
        rdbu_cmap = plt.get_cmap('RdBu')
        color_15 = rdbu_cmap(0.15)
        color_85 = rdbu_cmap(0.85)
        ax = vis.plot_grid_with_panels(
            grid_gdf=gdf.copy(),
            panel_columns=[main_plot_kwargs['color_column']],
            panel_stat_lists=[['mean_positive', 'mean_negative']],
            panel_steps=1.0,
            panel_legend_lists=[['Increase', 'Decrease']],
            panel_color_lists=[[color_85, color_15]],
            main_plot_kwargs=main_plot_kwargs,
            inset_axes_layout=inset_axes_layout,
            save_path=save_path,
            additional_colorbar_kwargs={'orientation': 'horizontal', 'extend': 'both'}
        )
        
        save_path = os.path.join(wkdir, f'area_csvs/test_figs/analyzed_grid_annual_mean_first_difference_percentage_change_panel_{basin_id}.png')
        main_plot_kwargs = {
            'projection': projection,
            'color_column': 'grid_linear_trend_of_annual_mean_first_difference_percentage',
            'gridlines': True,
            'add_rivers': True,
            'use_log_scale_color': False,
            'set_global': set_global,
            'significant_column': 'grid_rao_mk_test_on_annual_mean_first_difference_percentage_p',
            'significant_threshold': 0.05,
            'draw_colorbar': False,
            'title': 'Annual Mean First Difference Percentage from 2001 to 2023',
            'cmap': 'RdBu',
            'use_discrete_color':True,
            'discrete_bins':[-0.25, -0.20, -0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15, 0.20, 0.25],
        }
        inset_axes_layout = {
            'main': {'width': '79.5%', 'height': '80%', 'loc': 'lower left', 
                     'bbox_to_anchor': (0.0, 0.0, 1.0, 1.0), 'borderpad': 0.0},
            'top': {'width': '79.5%', 'height': '15%', 'loc': 'upper left', 
                    'bbox_to_anchor': (0.0, 0.0, 1.0, 1.0), 'borderpad': 0.0},
            'right': {'width': '15.5%', 'height': '80%', 'loc': 'lower right', 
                      'bbox_to_anchor': (0.0, 0.0, 1.0, 1.0), 'borderpad': 0.0},
            'colorbar_ax': {'width': '18%', 'height': '3.5%', 'loc': 'upper right',
                            'bbox_to_anchor': (0.0, 0.0, 1.0, 1.0), 'borderpad': 0.0}
        }
        rdbu_cmap = plt.get_cmap('RdBu')
        color_15 = rdbu_cmap(0.15)
        color_85 = rdbu_cmap(0.85)
        ax = vis.plot_grid_with_panels(
            grid_gdf=gdf.copy(),
            panel_columns=[main_plot_kwargs['color_column']],
            panel_stat_lists=[['mean_positive', 'mean_negative']],
            panel_steps=1.0,
            panel_legend_lists=[['Increase', 'Decrease']],
            panel_color_lists=[[color_85, color_15]],
            main_plot_kwargs=main_plot_kwargs,
            inset_axes_layout=inset_axes_layout,
            save_path=save_path,
            additional_colorbar_kwargs={'orientation': 'horizontal', 'extend': 'both'}
        )
        
        save_path = os.path.join(wkdir, f'area_csvs/test_figs/analyzed_grid_standard_deviation_percentage_change_panel_{basin_id}.png')
        main_plot_kwargs = {
            'projection': projection,
            'color_column': 'grid_linear_trend_of_standard_deviation_percentage_per_period',
            'gridlines': True,
            'add_rivers': True,
            'use_log_scale_color': False,
            'set_global': set_global,
            'significant_column': 'grid_rao_mk_test_on_standard_deviation_percentage_p',
            'significant_threshold': 0.05,
            'draw_colorbar': False,
            'title': 'Annual Std. Percentage from 2001 to 2023',
            'cmap': 'RdBu',
            'use_discrete_color':True,
            'discrete_bins':[-0.25, -0.20, -0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15, 0.20, 0.25],
        }
        inset_axes_layout = {
            'main': {'width': '79.5%', 'height': '80%', 'loc': 'lower left', 
                     'bbox_to_anchor': (0.0, 0.0, 1.0, 1.0), 'borderpad': 0.0},
            'top': {'width': '79.5%', 'height': '15%', 'loc': 'upper left', 
                    'bbox_to_anchor': (0.0, 0.0, 1.0, 1.0), 'borderpad': 0.0},
            'right': {'width': '15.5%', 'height': '80%', 'loc': 'lower right', 
                      'bbox_to_anchor': (0.0, 0.0, 1.0, 1.0), 'borderpad': 0.0},
            'colorbar_ax': {'width': '18%', 'height': '3.5%', 'loc': 'upper right',
                            'bbox_to_anchor': (0.0, 0.0, 1.0, 1.0), 'borderpad': 0.0}
        }
        rdbu_cmap = plt.get_cmap('RdBu')
        color_15 = rdbu_cmap(0.15)
        color_85 = rdbu_cmap(0.85)
        ax = vis.plot_grid_with_panels(
            grid_gdf=gdf.copy(),
            panel_columns=[main_plot_kwargs['color_column']],
            panel_stat_lists=[['mean_positive', 'mean_negative']],
            panel_steps=1.0,
            panel_legend_lists=[['Increase', 'Decrease']],
            panel_color_lists=[[color_85, color_15]],
            main_plot_kwargs=main_plot_kwargs,
            inset_axes_layout=inset_axes_layout,
            save_path=save_path,
            additional_colorbar_kwargs={'orientation': 'horizontal', 'extend': 'both'}
        )
        
        save_path = os.path.join(wkdir, f'area_csvs/test_figs/analyzed_grid_mean_area_panel_{basin_id}.png')
        main_plot_kwargs = {
            'projection': projection,
            'color_column': 'grid_mean_area',
            'gridlines': True,
            'add_rivers': True,
            'use_log_scale_color': True,
            'set_global': set_global,
            'draw_colorbar': False,
            'cmap': 'Blues',
            'div_colorbar': False,
            'title': None
        }
        inset_axes_layout = {
            'main': {'width': '79.5%', 'height': '80%', 'loc': 'lower left', 
                     'bbox_to_anchor': (0.0, 0.0, 1.0, 1.0), 'borderpad': 0.0},
            'top': {'width': '79.5%', 'height': '17.5%', 'loc': 'upper left', 
                    'bbox_to_anchor': (0.0, 0.0, 1.0, 1.0), 'borderpad': 0.0},
            'right': {'width': '18.0%', 'height': '80%', 'loc': 'lower right', 
                      'bbox_to_anchor': (0.0, 0.0, 1.0, 1.0), 'borderpad': 0.0},
        }
        ax = vis.plot_grid_with_panels(
            grid_gdf=gdf.copy(),
            panel_columns=[main_plot_kwargs['color_column']],
            panel_stat_lists=[['mean']],
            panel_steps=1.0,
            panel_legend_lists=[['Average area']],
            panel_color_lists=[['blue']],
            main_plot_kwargs=main_plot_kwargs,
            inset_axes_layout=inset_axes_layout,
            save_path=save_path
        )
        """