from attach_geometry_and_generate_grid import time_series_analysis_on_df
from statsmodels.tsa.seasonal import STL
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.dates as mdates
from datetime import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import os
import matplotlib.patches as mpatches
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


# This function should be deprecated
def lake_wise_lse_analyze_and_plot(
    lake_lse_df,
    lake_id_column_name,
    area_columns,
    period=12,
    reshape_period=12,
    do_analyze=True,
    do_plotting=True,
    unit_scale=1e-6,
    stl_trend_column_name='stl_trend',
    stl_season_column_name='stl_season_max_minus_min',
    mean_area_column_name='mean_area',
    lake_size_bins=None,
    lake_size_labels=None,
    save_path=None
):
    print('This function should be deprecated. Use lake_wise_lse_analyze_and_plot_v2 instead.')
    if do_analyze == False:
        if stl_trend_column_name not in lake_lse_df.columns:
            raise ValueError(f'{stl_trend_column_name} not in columns of lake_lse_df')
        if stl_season_column_name not in lake_lse_df.columns:
            raise ValueError(f'{stl_season_column_name} not in columns of lake_lse_df')
        if mean_area_column_name not in lake_lse_df.columns:
            raise ValueError(f'{mean_area_column_name} not in columns of lake_lse_df')
        
    else:
        print(f'Analyzing {lake_lse_df.shape[0]} lakes for {stl_trend_column_name}')
        lake_lse_df = time_series_analysis_on_df(
            df=lake_lse_df,
            time_series_columns=area_columns,
            type_of_analysis='linear_trend_of_stl_trend_per_period',
            output_column_name=stl_trend_column_name,
            period=period,
            unit_scale=1e-6,
            reshape_period=reshape_period
        )
        
        print(f'Analyzing {lake_lse_df.shape[0]} lakes for {stl_season_column_name}')
        lake_lse_df = time_series_analysis_on_df(
            df=lake_lse_df,
            time_series_columns=area_columns,
            type_of_analysis='linear_trend_of_stl_seasonal_max_minus_min_per_period',
            output_column_name=stl_season_column_name,
            period=period,
            unit_scale=1e-6,
            reshape_period=reshape_period
        )
        
        print(f'Analyzing {lake_lse_df.shape[0]} lakes for {mean_area_column_name}')
        lake_lse_df = time_series_analysis_on_df(
            df=lake_lse_df,
            time_series_columns=area_columns,
            type_of_analysis='mean',
            output_column_name=mean_area_column_name,
            period=period,
            unit_scale=1e-6,
            reshape_period=reshape_period
        )
    
    # Create a new column based on the positive/negative values of stl_trend_column_name
    lake_lse_df['trend_direction'] = np.where(lake_lse_df[stl_trend_column_name] >= 0, 'Positive', 'Negative')
    lake_lse_df['abs_trend'] = np.abs(lake_lse_df[stl_trend_column_name])
    
    lake_lse_df['seasonal_direction'] = np.where(lake_lse_df[stl_season_column_name] >= 0, 'Positive', 'Negative')
    lake_lse_df['abs_seasonal'] = np.abs(lake_lse_df[stl_season_column_name])
    
    metric_type = ['Trend'] * lake_lse_df.shape[0] + ['Seasonal'] * lake_lse_df.shape[0]
    metric_value = lake_lse_df['abs_trend'].to_list() + lake_lse_df['abs_seasonal'].to_list()
    metric_direction = lake_lse_df['trend_direction'].to_list() + lake_lse_df['seasonal_direction'].to_list()
    
    lake_lse_analyzed_for_boxplot = pd.DataFrame({
        'metric_type': metric_type,
        'metric_value': metric_value,
        'metric_direction': metric_direction
    })
    
    #Drop those that have a mean area of 0
    lake_lse_df = lake_lse_df[lake_lse_df[mean_area_column_name] > 0]
    
    if lake_size_bins is None and lake_size_labels is None:
        lake_size_bins = [0, 1, 10, 50, 100, 500, np.inf]
        lake_size_labels = ['0-1', '1-10', '10-50', '50-100', '100-500', '>500']
    
    lake_lse_df.loc[:, 'lake_size_group'] = pd.cut(lake_lse_df[mean_area_column_name], bins=lake_size_bins, labels=lake_size_labels)
    lake_size_groups = lake_lse_df['lake_size_group'].unique()
    trend_sum = []
    seasonality_sum = []

    for group in lake_size_groups:
        group_df = lake_lse_df[lake_lse_df['lake_size_group'] == group]
        trend_sum.append(group_df[stl_trend_column_name].sum())
        seasonality_sum.append(group_df[stl_season_column_name].sum())
    
    group_df_for_barplot = pd.DataFrame({
        'metric_type': ['Trend'] * len(lake_size_groups) + ['Seasonal'] * len(lake_size_groups),
        'metric_value': trend_sum + seasonality_sum,
        'lake_size_group': lake_size_groups.tolist()*2
    })
        
    start_date = area_columns[0]
    end_date = area_columns[-1]
    dates_mon = pd.date_range(start=start_date, end=end_date, freq='MS')
    dates_mon = [date.to_pydatetime() for date in dates_mon.to_list()]
    dates_yr = pd.date_range(start=start_date, end=end_date, freq='YS')
    dates_yr = [date.to_pydatetime() for date in dates_yr.to_list()]
    total_areas = lake_lse_df[area_columns].sum().to_numpy()*unit_scale
    total_areas_stl_result = STL(total_areas, period=period).fit()
    total_areas_trend = total_areas_stl_result.trend
    total_areas_seasonal = total_areas_stl_result.seasonal
    
    total_areas_seasonal_period_max_minus_min = total_areas_seasonal.reshape(-1, reshape_period).max(axis=1) - total_areas_seasonal.reshape(-1, reshape_period).min(axis=1)

    if do_plotting:
        layout_mosaic = '''
            aa
            bc
        '''
        fig, axs = plt.subplot_mosaic(layout_mosaic, figsize=(12, 8))
        for ax_kw, ax in axs.items():
            if ax_kw == 'a':
                ax.plot(dates_mon, total_areas, color='skyblue')
                ax.plot(dates_mon, total_areas_trend, color='blue')
                ax.set_xlim([min(dates_mon), max(dates_mon)])
                #d = np.zeros(len(total_areas_df.dates))
                #ax.fill_between(total_areas_df.dates, total_areas_df.total_areas, d, where=total_areas_df.total_areas >= d, interpolate=True, color='skyblue', alpha=0.5)
                # Add a vertical dashed line for each year
                years = np.unique([date.year for date in dates_mon])
                for year in years:
                    ax.axvline(pd.to_datetime(str(year)), color='lightgrey', linestyle='--', alpha=0.5)
                ax.xaxis.set_major_locator(mdates.YearLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            elif ax_kw == 'b':
                sns.boxplot(data=lake_lse_analyzed_for_boxplot, x='metric_type', y='metric_value', hue='metric_direction', ax=ax, log_scale=(False, True))
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                #sns.histplot(data=lake_lse_df, x='abs_trend', hue='trend_direction', log_scale=(True, False), ax=ax)
            elif ax_kw == 'c':
                sns.barplot(data=group_df_for_barplot, x='lake_size_group', y='metric_value', hue='metric_type', ax=ax, order=lake_size_labels, palette={'Trend': 'blue', 'Seasonal': 'skyblue'})
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            
            if save_path:
                if save_path.endswith('.png'):
                    plt.savefig(save_path, dpi=300, format='png', bbox_inches='tight')
                elif save_path.endswith('.pdf'):
                    plt.savefig(save_path, format='pdf', bbox_inches='tight')
                else:
                    raise ValueError('save_path must end with either ".png" or ".pdf"')
            else:
                plt.show()
                
def calculate_statistics(
    lake_wise_gdf,
    analyze_column_name,
    statistics_type='mk_significant_trend_percentage'
):
    allowed_statistics_tpyes = [
        'mk_significant_trend_percentage',
        'linear_trend_percentage',
    ]
    if statistics_type not in allowed_statistics_tpyes:
        raise ValueError(f'statistics_type must be one of {allowed_statistics_tpyes}')

    if statistics_type == 'mk_significant_trend_percentage':
        total_nums = lake_wise_gdf.shape[0]
        significant_increase_nums = lake_wise_gdf[analyze_column_name].apply(lambda x: x == 'increasing').sum()
        significant_decrease_nums = lake_wise_gdf[analyze_column_name].apply(lambda x: x == 'decreasing').sum()
        significant_increase_percentage = significant_increase_nums / total_nums * 100
        significant_decrease_percentage = significant_decrease_nums / total_nums * 100
        result_dict = {
            'total': total_nums,
            'significant_increase': significant_increase_nums,
            'significant_decrease': significant_decrease_nums,
            'significant_increase_percentage': significant_increase_percentage,
            'significant_decrease_percentage': significant_decrease_percentage
        }

        return result_dict
    
    elif statistics_type == 'linear_trend_percentage':
        total_nums = lake_wise_gdf.shape[0]
        increase_nums = lake_wise_gdf[analyze_column_name].apply(lambda x: x > 0).sum()
        decrease_nums = lake_wise_gdf[analyze_column_name].apply(lambda x: x < 0).sum()
        increase_percentage = increase_nums / total_nums * 100
        decrease_percentage = decrease_nums / total_nums * 100
        result_dict = {
            'total': total_nums,
            'increase': increase_nums,
            'decrease': decrease_nums,
            'increase_percentage': increase_percentage,
            'decrease_percentage': decrease_percentage
        }

        return result_dict
    
def basin_distribution_plot(
    basin_id_list,
    basin_display_name_dict,
    basin_color_dict,
    region_name_dict,
    region_color_dict,
    basin_gdf,
    basin_id_column_name,
    projection=ccrs.PlateCarree(),
    extent=None,
    ax=None,
    save_path=None
):
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': projection}, figsize=(15, 7.5))
    
    basin_gdf = basin_gdf.to_crs(projection)
    #simplify the geometries
    basin_gdf['simplified_geom'] = basin_gdf.simplify(1000)
    basin_gdf = basin_gdf.set_geometry('simplified_geom')
    for basin_id in basin_id_list:
        basin_gdf_single = basin_gdf[basin_gdf[basin_id_column_name] == basin_id]
        basin_display_name = basin_display_name_dict[basin_id]
        basin_color = basin_color_dict[basin_id]
        basin_gdf_single.plot(ax=ax, color=basin_color, edgecolor='black', linewidth=0.5, zorder=2.5, transform=projection)
        centroid = basin_gdf_single.geometry.centroid
        ax.text(centroid.x, centroid.y, basin_display_name, transform=projection, fontweight='bold', color='brown', fontsize='large', zorder=3)
        
    ax.spines['geo'].set_visible(False)    
    
    ax.add_feature(cfeature.COASTLINE, zorder=1.2)  # Coastlines
   
    legend_elements = [mpatches.Patch(facecolor=color, edgecolor='black', label=region)
                   for region, color in zip(region_name_dict.values(), region_color_dict.values())]
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0.0, 0.375), fontsize='large', title='Regions')
    if extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    else:
        ax.set_global()
    if save_path is not None:
        if save_path.endswith('.png'):
            plt.savefig(save_path, dpi=300, format='png', bbox_inches='tight')
        elif save_path.endswith('.pdf'):
            plt.savefig(save_path, format='pdf', bbox_inches='tight')
        elif save_path.endswith('.svg'):
            plt.savefig(save_path, format='svg', bbox_inches='tight')
        else:
            raise ValueError('save_path must end with either ".png" or ".pdf", or ".svg"')
    else:
        plt.show()
    return ax
    
def basin_wise_statistics_plot(
    basin_gdf,
    statistic_column_name,
    projection=ccrs.PlateCarree(),
    cmap=None,
    ax=None
):
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': projection}, figsize=(15, 7.5))
    
    basin_gdf = basin_gdf.to_crs(ccrs.PlateCarree().proj4_init)
    basin_gdf.plot(column=statistic_column_name, ax=ax, cmap=cmap, edgecolor='black', linewidth=0.5, zorder=2.5, transform=ccrs.PlateCarree())
    
    ax.spines['geo'].set_visible(False)
    ax.add_feature(cfeature.OCEAN, color='gainsboro', zorder=0.1)  # Ocean color
    ax.add_feature(cfeature.COASTLINE, zorder=1.2)  # Coastlines
    ax.set_global()
    plt.show()
    return ax
    
def basin_time_series_plot(
    basin_id_list,
    basin_display_name_dict,
    basin_display_name_color_dict,
    basin_lake_wise_lse_csv_path_pattern,
    area_columns,
    unit_scale=1e-6,
    n_cols=10,
    save_path=None
):
    n_plots = len(basin_id_list)
    n_rows = n_plots // n_cols
    if n_plots % n_cols != 0:
        n_rows += 1
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 1.5*n_rows))
    ax_list = axs.flatten()
    for ax, basin_id in zip(ax_list, basin_id_list):
        basin_lake_wise_lse_gdf = pd.read_csv(basin_lake_wise_lse_csv_path_pattern.format(basin_id))
        basin_display_name = basin_display_name_dict[basin_id]
        basin_display_name_color = basin_display_name_color_dict[basin_id]
        total_areas = basin_lake_wise_lse_gdf[area_columns].sum().to_numpy()*unit_scale
        total_areas_stl_result = STL(total_areas, period=13).fit()
        total_areas_trend = total_areas_stl_result.trend
        x_dates = pd.date_range(start=area_columns[0], end=area_columns[-1], freq='MS')
        ax.plot(x_dates, total_areas, color=basin_display_name_color)
        ax.plot(x_dates, total_areas_trend, color='black')
        ax.set_title(basin_display_name, color=basin_display_name_color)
        # display x labels per 3 years
        years = np.unique([date.year for date in x_dates])
        for year in years:
            ax.axvline(pd.to_datetime(str(year)), color='lightgrey', linestyle='--', alpha=0.5)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    
    if save_path is not None:
        save_folder = os.path.dirname(save_path)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        plt.savefig(save_path, dpi=300, format='png', bbox_inches='tight')
    else:
        plt.show()
        
def categorical_percentage_pie_plot(
    increase_percentages,
    decrease_percentages,
    significant_increase_percentages,
    significant_decrease_percentages,
    category_titles,
    axes_flatten=None,
    show_plot=True
):
    def plot_pie(ax, increase, decrease, sig_increase, sig_decrease):
        # Non-significant parts
        non_sig_increase = increase - sig_increase
        non_sig_decrease = decrease - sig_decrease
        
        # Pie sizes and colors
        sizes = [non_sig_increase, sig_increase, non_sig_decrease, sig_decrease]
        colors = ['deepskyblue', 'deepskyblue', 'salmon', 'salmon']
        hatch_patterns = ['', '////', '', '////']

        wedges, texts, autotexts = ax.pie(
            sizes, colors=colors, startangle=90, autopct='%1.0f%%', wedgeprops=dict(edgecolor='black', linewidth=1.5)
        )
        
        for wedge, hatch in zip(wedges, hatch_patterns):
            wedge.set_hatch(hatch)
        
        for text in texts + autotexts:
            text.set_fontsize(14)
            text.set_color('white')

    if axes_flatten is None:
        # Create subplots
        n_cols = 3
        n_rows = len(category_titles) // n_cols
        if len(category_titles) % n_cols != 0:
            n_rows += 1
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes_flatten = axes.flatten()

    # Plot each region's pie chart
    for i, ax in enumerate(axes_flatten):
        plot_pie(ax, increase_percentages[i], decrease_percentages[i], significant_increase_percentages[i], significant_decrease_percentages[i])
        ax.set_title(category_titles[i], fontsize=16)

    # Create custom legend
    legend_elements = [
        mpatches.Patch(facecolor='deepskyblue', edgecolor='black', label='Insignificant Increase'),
        mpatches.Patch(facecolor='deepskyblue', edgecolor='black', hatch='///', label='Significant Increase'),
        mpatches.Patch(facecolor='salmon', edgecolor='black', label='Insignificant Decrease'),
        mpatches.Patch(facecolor='salmon', edgecolor='black', hatch='///', label='Significant Decrease')
    ]

    fig.legend(handles=legend_elements, loc='center right', fontsize='large', title='Legend')

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Make room for the legend on the right
    if show_plot:
        plt.show()
    
def regional_percentage_pie_plot(
    significant_increase_percentages,
    increase_percentages,
    significant_decrease_percentages,
    decrease_percentages,
    regional_titles,
    axes_flatten=None
):
    categorical_percentage_pie_plot(
        increase_percentages=increase_percentages,
        decrease_percentages=decrease_percentages,
        significant_increase_percentages=significant_increase_percentages,
        significant_decrease_percentages=significant_decrease_percentages,
        category_titles=regional_titles,
        axes_flatten=axes_flatten
    )

def percentage_pie_plot_of_difference_sized_lakes(
    lse_df,
    significance_column_name,
    trend_column_name,
    lake_size_bins,
    lake_size_labels,
    lake_area_column_name,
    axes_flatten=None,
    show_plot=True
):
    lse_df['Lake Size'] = pd.cut(lse_df[lake_area_column_name], bins=lake_size_bins, labels=lake_size_labels)
    increase_percentages = []
    decrease_percentages = []
    significant_increase_percentages = []
    significant_decrease_percentages = []
    for size in lake_size_labels:
        subset = lse_df[lse_df['Lake Size'] == size]
        change_statistics = calculate_statistics(subset, trend_column_name, statistics_type='linear_trend_percentage')
        significant_change_statistics = calculate_statistics(subset, significance_column_name, statistics_type='mk_significant_trend_percentage')
        increase_percentages.append(change_statistics['increase_percentage'])
        decrease_percentages.append(change_statistics['decrease_percentage'])
        significant_increase_percentages.append(significant_change_statistics['significant_increase_percentage'])
        significant_decrease_percentages.append(significant_change_statistics['significant_decrease_percentage'])
    print(increase_percentages)
    print(decrease_percentages)
    print(significant_increase_percentages)
    print(significant_decrease_percentages)
    categorical_percentage_pie_plot(
        increase_percentages=increase_percentages,
        decrease_percentages=decrease_percentages,
        significant_increase_percentages=significant_increase_percentages,
        significant_decrease_percentages=significant_decrease_percentages,
        category_titles=lake_size_labels,
        axes_flatten=axes_flatten,
        show_plot=show_plot
    )
  
def percentage_per_category_bar_plot(
    significant_increase,
    non_significant_increase,
    significant_decrease,
    non_significant_decrease,
    bar_width=0.9,
    category_labels=None,
    bar_span_color_2d_list=None,
    xticklabel_color_list=None,
    ax=None
):
    # Number of categories
    categories = len(non_significant_increase)

    # Bar positions
    ind = np.arange(categories)
    if not len(category_labels) == len(ind):
        raise ValueError('The number of category labels must match the number of categories.')
    
    # Plotting
    if ax is None:
        fig, ax = plt.subplots(figsize=(18, 5))

    gap_between_bars = 1-bar_width
    if bar_span_color_2d_list is not None:
        num_sub_categories = [len(bar_span_color_2d_list[i]) for i in range(len(bar_span_color_2d_list))]
        bar_span_color_1d_list = [bar_span_color_2d_list[i][0] for i in range(len(bar_span_color_2d_list))]
        span_right_ends = np.cumsum(num_sub_categories) - 1
        span_left_ends = []
        for i in range(len(num_sub_categories)):
            if i == 0:
                span_left_ends.append(0)
            else:
                span_left_ends.append(span_right_ends[i-1] + 1)
        span_ranges = [(span_left_ends[i]-0.5*bar_width-0.5*gap_between_bars, span_right_ends[i]+0.5*bar_width+0.5*gap_between_bars) for i in range(len(span_left_ends))]
        for span_range, color in zip(span_ranges, bar_span_color_1d_list):
            ax.axvspan(span_range[0], span_range[1], color=color, alpha=1, zorder=0)
            
    # Significant increase
    ax.bar(ind, significant_increase, bar_width, color='steelblue', edgecolor=None, label='Significant Increase', hatch='///')

    # Non-Significant increase
    ax.bar(ind, non_significant_increase, bar_width, bottom=np.array(significant_increase), color='steelblue', edgecolor=None, label='Non-significant Increase')

    # Non-significant decrease
    ax.bar(ind, non_significant_decrease, bar_width, bottom=np.array(non_significant_increase) + np.array(significant_increase), color='chocolate', edgecolor=None, label='Non-significant Decrease')

    # Significant decrease
    ax.bar(ind, significant_decrease, bar_width, bottom=np.array(non_significant_increase) + np.array(significant_increase) + np.array(non_significant_decrease), color='chocolate', edgecolor=None, label='Significant Decrease', hatch='///')

    # Adding labels and title
    ax.set_ylabel('Percentage')
    ax.set_title('Stacked Bar Plot of Trends')
    ax.set_xticks(ind)
    if category_labels is None:
        category_labels = [f'Category {i+1}' for i in range(categories)]
    ax.set_xticklabels(category_labels)
    if xticklabel_color_list is not None:
        for i, color in enumerate(xticklabel_color_list):
            ax.get_xticklabels()[i].set_color(color)
    #ax.legend()
    ax.set_ylim(0, 100)
    ax.set_xlim(ind[0]-0.5*bar_width-0.5*gap_between_bars, ind[-1]+0.5*bar_width+0.5*gap_between_bars)
    # add a horizontal line at y=50
    ax.axhline(y=50, color='black', linestyle='--', linewidth=1)
    if bar_width == 1 and bar_span_color_2d_list is not None:
        for i in range(len(span_right_ends)):
            if i == len(span_right_ends) - 1:
                continue
            ax.axvline(x=span_right_ends[i]+0.5*bar_width, color='black', linestyle='--', linewidth=1)
            
        
    # Display the plot
    plt.show()
    
def bar_plot_of_all_lse_changes(
    lse_df,
    to_plot_column_names,
    to_plot_column_stats_types,
    to_plot_column_labels,
    to_plot_column_colors,
    lake_area_column_name,
    lake_size_bins=None,
    category_values=None,
    lake_size_labels=None,
    columns_in_twin=None,
    x_label='category',
    y_label='Changes',
    ax_title='Grouped Bar Plot of Lake Area Changes',
    bar_width=0.2,
    ax=None,
    xticklabel_rotation=0,
    show_plot=True
):
    if category_values is not None and lake_size_bins is not None:
        raise ValueError('Only one of category_values and lake_size_bins should be provided.')
    if category_values is not None:
        lse_df['category'] = lse_df[lake_area_column_name].map(dict(zip(category_values, lake_size_labels)))
    if lake_size_bins is not None:
        # Categorize lakes based on their area if bins are provided
        if lake_size_bins and lake_size_labels:
            lse_df['category'] = pd.cut(lse_df[lake_area_column_name], bins=lake_size_bins, labels=lake_size_labels)
        else:
            lse_df['category'] = 'All'

    # Prepare a DataFrame to hold the statistics
    stats_df = pd.DataFrame(columns=to_plot_column_labels, index=lake_size_labels if lake_size_labels else ['All'])

    # Calculate statistics for each category and type of change
    for size in stats_df.index:
        subset = lse_df[lse_df['category'] == size]
        for col, stat, label in zip(to_plot_column_names, to_plot_column_stats_types, to_plot_column_labels):
            if stat == 'mean':
                stats_df.loc[size, label] = subset[col].mean()
            elif stat == 'median':
                stats_df.loc[size, label] = subset[col].median()
            elif stat == 'std':
                stats_df.loc[size, label] = subset[col].std()
            elif stat == 'sum':
                stats_df.loc[size, label] = subset[col].sum()
            elif stat == 'positive_sum':
                stats_df.loc[size, label] = subset[subset[col] > 0][col].sum()
            elif stat == 'negative_sum':
                stats_df.loc[size, label] = subset[subset[col] < 0][col].sum()
            else:
                raise ValueError(f'Unknown statistic type: {stat}')
            # Add other statistics as needed

    # Plotting
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    n_bars = len(stats_df.columns)
    index = stats_df.index

    # Positions of the bars on the x-axis
    bar_positions = list(range(len(stats_df)))
    
    if columns_in_twin is not None:
        ax_twin = ax.twinx()
    
    # Create bars for each column
    for i, (column, color) in enumerate(zip(stats_df.columns, to_plot_column_colors)):
        bar_position = [x + i * bar_width for x in bar_positions]
        if columns_in_twin is not None and column in columns_in_twin:
            ax_twin.bar(bar_position, stats_df[column], width=bar_width, label=column, color=color)
        else:
            ax.bar(bar_position, stats_df[column], width=bar_width, label=column, color=color)

    # Set the x-axis ticks to be in the middle of the grouped bars
    grouped_bar_positions_2d_list = [
        [x + i * bar_width for x in bar_positions]
        for i in range(n_bars)
    ]
    grouped_bar_positions_np_array = np.array(grouped_bar_positions_2d_list).T
    grouped_bar_positions_central = np.mean(grouped_bar_positions_np_array, axis=1)
    ax.set_xticks(grouped_bar_positions_central)
    ax.set_xticklabels(index, rotation=xticklabel_rotation)

    # Adding labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(ax_title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()

    # Show plot
    if show_plot:
        plt.show()
    return ax


def plot_hbar_by_country(
    lakes_df,
    to_plot_column_name,
    to_plot_significance_column_name,
    n_countries_head=10,
    n_countries_tail=10,
    axes_flattened=None,
    bar_width=0.4,
    lake_count_threshold=1000
):
    increase_percentage = []
    decrease_percentage = []
    significant_increase_percentage = []
    significant_decrease_percentage = []
    countries = lakes_df['Country'].unique()
    countries_oi = []
    for country in countries:
        country_df = lakes_df[lakes_df['Country'] == country]
        total_lakes = country_df.shape[0]
        if total_lakes < lake_count_threshold:
            continue
        change_results = calculate_statistics(country_df, to_plot_column_name, statistics_type='linear_trend_percentage')
        significant_change_results = calculate_statistics(country_df, to_plot_significance_column_name, statistics_type='mk_significant_trend_percentage')
        increase_percentage.append(change_results['increase_percentage'])
        decrease_percentage.append(change_results['decrease_percentage'])
        significant_increase_percentage.append(significant_change_results['significant_increase_percentage'])
        significant_decrease_percentage.append(significant_change_results['significant_decrease_percentage'])
        countries_oi.append(country)
    #normalize the percentages to sum to 100%
    percentage_total = [increase_percentage[i] + decrease_percentage[i] for i in range(len(increase_percentage))]
    increase_percentage = [increase_percentage[i] / percentage_total[i] * 100 for i in range(len(increase_percentage))]
    decrease_percentage = [decrease_percentage[i] / percentage_total[i] * 100 for i in range(len(decrease_percentage))]
    significant_increase_percentage = [significant_increase_percentage[i] / percentage_total[i] * 100 for i in range(len(significant_increase_percentage))]
    significant_decrease_percentage = [significant_decrease_percentage[i] / percentage_total[i] * 100 for i in range(len(significant_decrease_percentage))]
    
    non_significant_increase_percentage = [increase_percentage[i] - significant_increase_percentage[i] for i in range(len(increase_percentage))]
    non_significant_decrease_percentage = [decrease_percentage[i] - significant_decrease_percentage[i] for i in range(len(decrease_percentage))]
    
    percentage_df = pd.DataFrame({
        'Country': countries_oi,
        'Increase': increase_percentage,
        'Decrease': decrease_percentage,
        'Significant Increase': significant_increase_percentage,
        'Significant Decrease': significant_decrease_percentage,
        'Non-significant Increase': non_significant_increase_percentage,
        'Non-significant Decrease': non_significant_decrease_percentage
    })
    #sort the percentage_df according to dscending order of the increase
    percentage_df = percentage_df.sort_values(by='Increase', ascending=False)
    # keep only the head and tails
    head_percentage_df = percentage_df.head(n_countries_head).sort_values(by='Increase', ascending=True)
    tail_percentage_df = percentage_df.tail(n_countries_tail)
    
    if axes_flattened is None:
        fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
        axes_flattened = axes.flatten()
    
    for current_df, ax in zip([head_percentage_df, tail_percentage_df], axes_flattened):
        bar_positions = np.arange(current_df.shape[0])
        country_labels = current_df['Country']
        non_significant_increase_percentage = current_df['Non-significant Increase']
        non_significant_decrease_percentage = current_df['Non-significant Decrease']
        significant_increase_percentage = current_df['Significant Increase']
        significant_decrease_percentage = current_df['Significant Decrease']
             
        ax.barh(bar_positions, significant_increase_percentage, bar_width, color='skyblue', edgecolor='black', label='Significant Increase', hatch='///')
        ax.barh(bar_positions, non_significant_increase_percentage, bar_width, left=significant_increase_percentage, color='skyblue', edgecolor='black', label='Non-significant Increase')
        ax.barh(bar_positions, non_significant_decrease_percentage, bar_width, left=significant_increase_percentage + non_significant_increase_percentage, color='salmon', edgecolor='black', label='Non-significant Decrease')
        ax.barh(bar_positions, significant_decrease_percentage, bar_width, left=significant_increase_percentage + non_significant_increase_percentage + non_significant_decrease_percentage, color='salmon', edgecolor='black', label='Significant Decrease', hatch='///')
        
        ax.axvline(x=50, color='black', linestyle='--', linewidth=1)
        
        ax.set_yticks(bar_positions)
        ax.set_yticklabels(country_labels)
        ax.set_xlabel('Percentage')
        ax.set_title('Percentages by Country')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.show()
        