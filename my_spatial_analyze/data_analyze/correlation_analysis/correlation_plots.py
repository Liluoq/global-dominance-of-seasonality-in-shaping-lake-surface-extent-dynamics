import numpy as np
import pandas as pd
import geopandas as gpd
import dask.dataframe as dd
import dask_geopandas as dgpd
import pdb
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
import sys
sys.path.append('/WORK/Codes/global_lake_area/my_spatial_analyze')
from lake_wise_lse_analyze import bar_plot_of_all_lse_changes, calculate_statistics, percentage_per_category_bar_plot
from scipy.stats import bootstrap
from visualization import handle_antimeridian
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

TICK_LABEL_SIZE = 6
AXIS_LABEL_SIZE = 8
TITLE_SIZE = 8
LEGEND_SIZE = 6
mpl.rcParams['xtick.labelsize'] = TICK_LABEL_SIZE
mpl.rcParams['ytick.labelsize'] = TICK_LABEL_SIZE
mpl.rcParams['axes.labelsize'] = AXIS_LABEL_SIZE
mpl.rcParams['axes.titlesize'] = TITLE_SIZE
mpl.rcParams['legend.fontsize'] = LEGEND_SIZE

def draw_pie_with_different_radius(
    datas,
    colors,
    ax=None # This ax should be a polar axis
):
    datas_sum = np.sum(datas)
    # Normalize datas to be between 0 and 1
    datas_normalized = [data / datas_sum for data in datas]


    # Create the pie chart
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    # Initialize the starting angle
    start_angle = 0

    # Draw each pie slice
    for color, data_norm in zip(colors, datas_normalized):
        # Calculate the width of the pie slice
        theta = 2 * np.pi * data_norm
        # Plot each slice
        ax.bar(x=start_angle, height=data_norm, width=theta, color=color, edgecolor='black', align='edge')
        # Update the starting angle for the next slice
        angle = start_angle + theta / 2
        # Annotate with the percentage
        data_norm_percentage = data_norm * 100
        ax.text(angle, data_norm / 2, f'{data_norm_percentage:0.0f}%', ha='center', va='center', color='white', fontsize=TICK_LABEL_SIZE, fontweight='bold')
        start_angle += theta
        print(f'theta in degree: {theta * 180 / np.pi}')
        print(f'start_angle in degree: {start_angle * 180 / np.pi}')
    # Display the plot
    ax.axis('off')
    return ax

 
def category_percentage_pie_plots(
    to_plot_df,
    continuous_column_names,
    continuous_group_bins_list,
    continuous_group_labels_list,
    continuous_group_colors_list,
    discrete_column_names,
    discrete_group_values_list,
    discrete_group_labels_list,
    discrete_group_colors_list,
    ax_list=None,
    save_path=None,
    pies_different_radius=False
):
    def plot_pie(data_counts, ax, title, colors, labels, threshold=2.0):
        total = data_counts.sum()
        
        def autopct(pct):
            return ('%1.1f%%' % pct) if pct > threshold else ''
        
        wedges, texts, autotexts = ax.pie(
            data_counts, labels=labels, autopct=autopct, startangle=140, colors=colors, textprops=dict(color="black"),
            pctdistance=0.85
        )
        ax.set_title(title)
        ax.axis('equal')
        
        for text in texts:
            text.set_fontsize(12)
            text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'), path_effects.Normal()])
        
        for i, autotext in enumerate(autotexts):
            pct = 100 * data_counts.iloc[i] / total
            if pct <= threshold:
                autotext.set_position((autotext.get_position()[0] * 1.4, autotext.get_position()[1] * 1.4))
            autotext.set_fontsize(12)
            autotext.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'), path_effects.Normal()])
    
    if ax_list is None:
        n_cols = 3
        if continuous_column_names is not None and discrete_column_names is not None:
            n_rows = int(np.ceil(len(continuous_column_names) + len(discrete_column_names))/n_cols)
        elif continuous_column_names is not None:
            n_rows = int(np.ceil(len(continuous_column_names))/n_cols)
        elif discrete_column_names is not None:
            n_rows = int(np.ceil(len(discrete_column_names))/n_cols)
        else :
            raise ValueError('continuous_column_names and discrete_column_names cannot be None at the same time')
        print(n_rows)
        if pies_different_radius:
            fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, subplot_kw={'projection': 'polar'}, figsize=(n_cols*5, n_rows*5))
        else:
            fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols*5, n_rows*5))
        ax_list = axes.flatten()
    ax_list_index = 0
    if continuous_column_names is not None:
        for i, continuous_column_name in enumerate(continuous_column_names):
            to_plot_df[f'{continuous_column_name}_category'] = pd.cut(to_plot_df[continuous_column_name], bins=continuous_group_bins_list[i], labels=continuous_group_labels_list[i])
            current_category_counts = to_plot_df[f'{continuous_column_name}_category'].value_counts().reindex(continuous_group_labels_list[i])
            print(current_category_counts)
            if pies_different_radius:
                draw_pie_with_different_radius(
                    datas=current_category_counts,
                    colors=continuous_group_colors_list[i],
                    ax=ax_list[ax_list_index]
                )
            else:
                plot_pie(current_category_counts, ax_list[ax_list_index], continuous_column_name, continuous_group_colors_list[i], continuous_group_labels_list[i])
            ax_list_index += 1
    if discrete_column_names is not None:
        for i, discrete_column_name in enumerate(discrete_column_names):
            to_plot_df[f'{discrete_column_name}_category'] = to_plot_df[discrete_column_name].map(dict(zip(discrete_group_values_list[i], discrete_group_labels_list[i])))
            current_category_counts = to_plot_df[f'{discrete_column_name}_category'].value_counts().reindex(discrete_group_labels_list[i])
            print(current_category_counts)
            if pies_different_radius:
                draw_pie_with_different_radius(
                    datas=current_category_counts,
                    colors=discrete_group_colors_list[i],
                    ax=ax_list[ax_list_index]
                )
            else:
                plot_pie(current_category_counts, ax_list[ax_list_index], discrete_column_name, discrete_group_colors_list[i], discrete_group_labels_list[i])
            ax_list_index += 1
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def bar_plot_of_different_groups(
    to_plot_df,
    to_plot_column_names,
    to_plot_column_colors,
    to_plot_column_labels,
    agg_to_bar_stat_types,
    to_group_by_column_name,
    group_bins=None,
    group_values=None,
    group_labels=None,
    bar_width=0.35,
    xticklabel_rotation=0,
    x_label='Category',
    y_label='Changes',
    ax_title='Changes of different groups',
    ax=None,
    show_plot=True
):
    ax = bar_plot_of_all_lse_changes(
        lse_df=to_plot_df,
        to_plot_column_names=to_plot_column_names,
        to_plot_column_stats_types=agg_to_bar_stat_types,
        to_plot_column_colors=to_plot_column_colors,
        to_plot_column_labels=to_plot_column_labels,
        lake_area_column_name=to_group_by_column_name,
        lake_size_bins=group_bins,
        category_values=group_values,
        lake_size_labels=group_labels,
        bar_width=bar_width,
        xticklabel_rotation=xticklabel_rotation,
        x_label=x_label,
        y_label=y_label,
        ax_title=ax_title,
        ax=ax,
        show_plot=show_plot
    )
    return ax
    
def percentage_per_category_bar_plot_emsemble(
    lake_lse_df,
    trend_column_name,
    significant_trend_column_name,
    category_column_name,
    category_bins=None,
    category_values=None,
    category_labels=None,
    bar_width=0.9,
    bar_span_color_2d_list=None,
    xticklabel_color_list=None,
    ax=None
):
    if category_bins is None and category_values is None:
        raise ValueError('Either category_bins or category_values should be provided.')
    if category_bins is not None and category_values is not None:
        raise ValueError('Only one of category_bins and category_values should be provided.')
    if category_bins is not None:
        lake_lse_df['category'] = pd.cut(lake_lse_df[category_column_name], bins=category_bins, labels=category_labels)
    elif category_values is not None:
        values_to_labels = dict(zip(category_values, category_labels))
        lake_lse_df['category'] = lake_lse_df[category_column_name].map(values_to_labels)
    
    non_significant_increase_percentages = []
    non_significant_decrease_percentages = []
    significant_increase_percentages = []
    significant_decrease_percentages = []
    for current_group in category_labels:
        current_group_df = lake_lse_df[lake_lse_df['category'] == current_group]
        current_group_df = current_group_df[current_group_df[trend_column_name].notna()]
        current_group_df = current_group_df[current_group_df[significant_trend_column_name].notna()]
        # if empty
        if current_group_df.empty:
            non_significant_increase_percentages.append(0)
            non_significant_decrease_percentages.append(0)
            significant_increase_percentages.append(0)
            significant_decrease_percentages.append(0)
            continue
        current_trend_percentages = calculate_statistics(current_group_df, trend_column_name, statistics_type='linear_trend_percentage')
        current_significant_trend_percentages = calculate_statistics(current_group_df, significant_trend_column_name, statistics_type='mk_significant_trend_percentage')
        non_significant_increase_percentages.append(current_trend_percentages['increase_percentage'] - current_significant_trend_percentages['significant_increase_percentage'])
        non_significant_decrease_percentages.append(current_trend_percentages['decrease_percentage'] - current_significant_trend_percentages['significant_decrease_percentage'])
        significant_increase_percentages.append(current_significant_trend_percentages['significant_increase_percentage'])
        significant_decrease_percentages.append(current_significant_trend_percentages['significant_decrease_percentage'])
    print(f'non_significant_increase_percentages: {non_significant_increase_percentages}')
    print(f'non_significant_decrease_percentages: {non_significant_decrease_percentages}')
    print(f'significant_increase_percentages: {significant_increase_percentages}')
    print(f'significant_decrease_percentages: {significant_decrease_percentages}')
    # normalize the percentages to sum to 100
    percentage_totals = [significant_increase_percentage+non_significant_increase_percentage+significant_decrease_percentage+non_significant_decrease_percentage for significant_increase_percentage, non_significant_increase_percentage, significant_decrease_percentage, non_significant_decrease_percentage in zip(significant_increase_percentages, non_significant_increase_percentages, significant_decrease_percentages, non_significant_decrease_percentages)]
    significant_increase_percentages = [significant_increase_percentage/percentage_total*100 if percentage_total else 0 for significant_increase_percentage, percentage_total in zip(significant_increase_percentages, percentage_totals)]
    non_significant_increase_percentages = [non_significant_increase_percentage/percentage_total*100 if percentage_total else 0 for non_significant_increase_percentage, percentage_total in zip(non_significant_increase_percentages, percentage_totals)]
    significant_decrease_percentages = [significant_decrease_percentage/percentage_total*100 if percentage_total else 0 for significant_decrease_percentage, percentage_total in zip(significant_decrease_percentages, percentage_totals)]
    non_significant_decrease_percentages = [non_significant_decrease_percentage/percentage_total*100 if percentage_total else 0 for non_significant_decrease_percentage, percentage_total in zip(non_significant_decrease_percentages, percentage_totals)]
    percentage_per_category_bar_plot(
        significant_increase=significant_increase_percentages,
        significant_decrease=significant_decrease_percentages,
        non_significant_increase=non_significant_increase_percentages,
        non_significant_decrease=non_significant_decrease_percentages,
        bar_width=bar_width,
        bar_span_color_2d_list=bar_span_color_2d_list,
        category_labels=category_labels,
        xticklabel_color_list=xticklabel_color_list,
        ax=ax
    )
    
def box_plot_of_different_groups(
    to_plot_df,
    to_plot_column_name,
    to_group_by_column_name,
    group_bins,
    group_labels,
    group_colors_box=None,
    group_colors_median=None,
    box_width=0.35,
    vmin=None,
    vmax=None,
    use_bootstrap=False,
    bootstrap_statistic=np.median,
    ax=None,
    xticklabel_rotation=45,
    show_plot=True
):
    if group_colors_box is None:
        patch_artist = False
    else:
        patch_artist = True
    if group_colors_box is not None and not isinstance(group_colors_box, mpl.cm.ScalarMappable):
        assert len(group_colors_box) == len(group_labels), 'The length of group_colors_box should be the same as the length of group_labels'
    if group_colors_median is not None:
        assert len(group_colors_median) == len(group_labels), 'The length of group_colors_median should be the same as the length of group_labels'
    
    if ax is None:
        fig, ax = plt.subplots()
    to_plot_df[to_group_by_column_name] = pd.cut(to_plot_df[to_group_by_column_name], bins=group_bins, labels=group_labels, ordered=True)
    if use_bootstrap:
        bootstrap_stats_dist = []
        for group in group_labels:
            current_data = to_plot_df[to_plot_column_name][to_plot_df[to_group_by_column_name] == group].to_numpy().flatten()
            print(current_data.shape)
            current_bootstrap_results = bootstrap((current_data, ), statistic=bootstrap_statistic, confidence_level=0.95, batch=25).bootstrap_distribution
            bootstrap_stats_dist.append(current_bootstrap_results)
        bp = ax.boxplot(bootstrap_stats_dist, 
                        labels=group_labels, 
                        widths=box_width, 
                        showfliers=False, 
                        patch_artist=patch_artist)
    else:
        data = [to_plot_df[to_plot_df[to_group_by_column_name] == group][to_plot_column_name].values 
            for group in group_labels]
        #remove nan in the data
        data = [current_data[~np.isnan(current_data)] for current_data in data]

        bp = ax.boxplot(data, positions=np.arange(len(group_labels)), widths=box_width, 
                showfliers=False, patch_artist=patch_artist)

        # Set the x-axis labels
        ax.set_xticks(np.arange(len(group_labels)))
        ax.set_xticklabels(group_labels)
        
    if group_colors_box is not None:
        if isinstance(group_colors_box, mpl.cm.ScalarMappable):
            medians_of_data = [np.median(current_data) for current_data in data]
            for i, box in enumerate(bp['boxes']):
                box.set(facecolor=group_colors_box.to_rgba(medians_of_data[i]))
        else:
            for i, box in enumerate(bp['boxes']):
                box.set(facecolor=group_colors_box[i])
        
    if group_colors_median is not None:
        for i, median in enumerate(bp['medians']):
            median.set(color=group_colors_median[i])
    if vmin is not None and vmax is not None:
        ax.set_ylim(vmin, vmax)
    ax.xaxis.set_tick_params(rotation=xticklabel_rotation)
    if show_plot:
        plt.show()
    
def histogram_of_different_groups(
    to_plot_df,
    to_plot_column_name,
    to_group_by_column_name,
    group_bins,
    group_labels,
    hist_bins,
    hist_colors,
    hist_alpha=0.5,
    hist_density=True,
    hist_type='step',
    ax=None
):
    if ax is None:
        fig, ax = plt.subplots()
    to_plot_df['category'] = pd.cut(to_plot_df[to_group_by_column_name], bins=group_bins, labels=group_labels)
    for group in group_labels:
        current_group_df = to_plot_df[to_plot_df['category'] == group]
        ax.hist(current_group_df[to_plot_column_name], bins=hist_bins, label=group, color=hist_colors[group], alpha=hist_alpha, density=hist_density, histtype=hist_type)
    #add legend
    ax.legend()
    plt.show()
    
    
def create_2d_cmap_discrete_four_corner(
    lower_left_corner_hex_color,
    lower_right_corner_hex_color,
    upper_right_corner_hex_color,
    upper_left_corner_hex_color,
    num_intervals_vertical,
    num_intervals_horizontal,
    center_hex_color=None,
    center_weight_adjust=1.0,
    return_matrix=False
):
    if center_hex_color is not None:
        center_rgba_color = mcolors.to_rgba(center_hex_color)
        center_rgba_color = np.array(center_rgba_color)
    lower_left_corner_rgba_color = mcolors.to_rgba(lower_left_corner_hex_color)
    lower_right_corner_rgba_color = mcolors.to_rgba(lower_right_corner_hex_color)
    upper_right_corner_rgba_color = mcolors.to_rgba(upper_right_corner_hex_color)
    upper_left_corner_rgba_color = mcolors.to_rgba(upper_left_corner_hex_color)
    #convert to np.array
    lower_left_corner_rgba_color = np.array(lower_left_corner_rgba_color)
    lower_right_corner_rgba_color = np.array(lower_right_corner_rgba_color)
    upper_left_corner_rgba_color = np.array(upper_left_corner_rgba_color)
    upper_right_corner_rgba_color = np.array(upper_right_corner_rgba_color)
    
    x = np.linspace(0, 1, num_intervals_horizontal)
    y = np.linspace(0, 1, num_intervals_vertical)
    xx, yy = np.meshgrid(x, y)
    
    bilinear_interpolation = (upper_left_corner_rgba_color[None, None, :] * (1 - xx[:, :, None]) * (1 - yy[:, :, None]) + 
                              upper_right_corner_rgba_color[None, None, :] * xx[:, :, None] * (1 - yy[:, :, None]) + 
                              lower_left_corner_rgba_color[None, None, :] * (1 - xx[:, :, None]) * yy[:, :, None] + 
                              lower_right_corner_rgba_color[None, None, :] * xx[:, :, None] * yy[:, :, None])
    
    if center_hex_color is not None:
        weights = np.sqrt((xx - 0.5)**2 + (yy - 0.5)**2)
        weights = (1 - weights / np.max(weights)) ** center_weight_adjust  
        # Normalize to range [0, 1]
        bilinear_interpolation = bilinear_interpolation * (1 - weights[:, :, None]) + center_rgba_color[None, None, :] * weights[:, :, None]
    bilinear_interpolation = np.clip(bilinear_interpolation, 0, 1)
    if return_matrix:
        return bilinear_interpolation
    else:
        return mcolors.ListedColormap(bilinear_interpolation.reshape(-1, 4))
    
def create_2d_colorbar_vertical_interpolation(
    lower_hex_colors,
    upper_hex_colors,
    num_intervals_vertical,
    return_matrix=False
):
    lower_rgba_colors = [mcolors.to_rgba(lower_hex_color) for lower_hex_color in lower_hex_colors]
    upper_rgba_colors = [mcolors.to_rgba(upper_hex_color) for upper_hex_color in upper_hex_colors]
    lower_rgba_colors = [np.array(lower_rgba_color) for lower_rgba_color in lower_rgba_colors]
    upper_rgba_colors = [np.array(upper_rgba_color) for upper_rgba_color in upper_rgba_colors]
    lower_rgba_colors = np.array(lower_rgba_colors)
    upper_rgba_colors = np.array(upper_rgba_colors)
    
    vertical_interpolation = np.array(
        [lower_rgba_colors + (upper_rgba_colors - lower_rgba_colors) * i / (num_intervals_vertical - 1) for i in range(num_intervals_vertical)]
    )
    vertical_interpolation = np.array(vertical_interpolation)
    if return_matrix:
        return vertical_interpolation
    else:
        return mcolors.ListedColormap(vertical_interpolation.reshape(-1, 4))
    
def create_2d_colorbar_from_rgba_matrix(
    rgba_matrix,
    return_matrix=False
):
    if return_matrix:
        return rgba_matrix
    else:
        return mcolors.ListedColormap(rgba_matrix.reshape(-1, 4))
    
def draw_2d_colorbar(
    cmap,
    num_intervals_horizontal,
    num_intervals_vertical,
    xtick_positions=None,
    xtick_labels=None,
    ytick_positions=None,
    ytick_labels=None,
    xlabel=None,
    ylabel=None,
    ax=None
):
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(np.arange(num_intervals_vertical*num_intervals_horizontal).reshape(num_intervals_vertical, num_intervals_horizontal),
                 cmap=cmap)
    #ax.invert_yaxis()
    # Reverse ytick_positions and ytick_labels
    ytick_positions = [num_intervals_vertical - 1 - ytick_position for ytick_position in ytick_positions]
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels, rotation=60, fontsize=TICK_LABEL_SIZE)
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ytick_labels, rotation=0, fontsize=TICK_LABEL_SIZE)
    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE)
    
def _find_grid_cell(x_intervals, y_intervals, point, origional='lower left'):
    x, y = point
    if origional not in ['lower left']:
        raise ValueError('Only lower left is supported for now')
    # Find the index of the interval that x falls into
    x_index = np.searchsorted(x_intervals, x, side='right') - 1
    if x_index < 0:
        x_index = 0
    if x_index >= len(x_intervals) - 1:
        x_index = len(x_intervals) - 2
    
    # Find the index of the interval that y falls into
    y_index = np.searchsorted(y_intervals, y, side='right') - 1
    if y_index < 0:
        y_index = 0
    if y_index >= len(y_intervals) - 1:
        y_index = len(y_intervals) - 2
    if origional in ['lower left', 'lower right']:
        y_index = len(y_intervals) - 2 - y_index
    return (x_index, y_index)

def generate_color_index(
    row,
    x_column_name,
    y_column_name,
    x_intervals,
    y_intervals
):
    point = (row[x_column_name], row[y_column_name])
    x_index, y_index = _find_grid_cell(
        x_intervals=x_intervals,
        y_intervals=y_intervals,
        point=point,
        origional='lower left'
    )
    color_index = y_index * (len(x_intervals) - 1) + x_index
    return int(color_index)


def gdf_plotting_using_2d_colobar(
    to_plot_gdf,
    x_column_name,
    y_column_name,
    x_intervals,
    y_intervals,
    cmap_2d=None,
    lower_left_corner_hex_color=None,
    lower_right_corner_hex_color=None,
    upper_right_corner_hex_color=None,
    upper_left_corner_hex_color=None,
    temporary_color_column_name='temporary_color_column',
    center_hex_color=None,
    center_weight_adjust=1.0,
    ax=None,
    fix_for_meridian=True,
    projection=ccrs.PlateCarree(),
    extent=None,
    gridlines=True,
    title=None,
    set_global=True,
    return_ax=True,
    return_cmap=True,
    plot_cmap=True,
    cmap_xtick_positions=None,
    cmap_xtick_labels=None,
    cmap_ytick_positions=None,
    cmap_ytick_labels=None,
    cmap_xlabel=None,
    cmap_ylabel=None,
    add_pie_percentage=False,
    pie_proportional_radius=False,
    additional_pie_data=None,
    additional_pie_colors=None,
    save_path=None,
    show_plot=True
):
    to_plot_gdf[temporary_color_column_name] = to_plot_gdf.apply(generate_color_index, axis=1, args=(x_column_name, y_column_name, x_intervals, y_intervals))
    if fix_for_meridian:
        to_plot_gdf = handle_antimeridian(to_plot_gdf)
    #to_plot_gdf = to_plot_gdf.to_crs(projection)
    num_intervals_horizontal = len(x_intervals) - 1
    num_intervals_vertical = len(y_intervals) - 1
    if cmap_2d is None:
        cmap = create_2d_cmap_discrete_four_corner(
            lower_left_corner_hex_color=lower_left_corner_hex_color,
            lower_right_corner_hex_color=lower_right_corner_hex_color,
            upper_right_corner_hex_color=upper_right_corner_hex_color,
            upper_left_corner_hex_color=upper_left_corner_hex_color,
            num_intervals_horizontal=num_intervals_horizontal,
            num_intervals_vertical=num_intervals_vertical,
            center_hex_color=center_hex_color,
            center_weight_adjust=center_weight_adjust
        )
    else:
        cmap=cmap_2d
    
    if extent is None:
        bounds = to_plot_gdf.total_bounds  # [minx, miny, maxx, maxy]
        print(list(bounds))
        print(type(list(bounds)))
        extent = [bounds[0], bounds[2], bounds[1], bounds[3]]  # [xmin, xmax, ymin, ymax]
    else:
        extent = [extent[0], extent[2], extent[1], extent[3]]
    lon_range = extent[1] - extent[0]
    lat_range = extent[3] - extent[2]
    aspect_ratio = lon_range / lat_range
    if ax is None:
        # Define the Cartopy CRS for the plot
        fig, ax = plt.subplots(1, 1, figsize=(5*aspect_ratio, 5), subplot_kw={'projection': projection})
    norm = mcolors.Normalize(vmin=0, vmax=(num_intervals_horizontal * num_intervals_vertical - 1))
    land_feature = cfeature.NaturalEarthFeature(
        'physical', 'land', '50m',
        edgecolor='face',
        facecolor='darkgrey'
    )

    ax.add_feature(land_feature, zorder=1.5)
    to_plot_gdf.plot(column=temporary_color_column_name, cmap=cmap, norm=norm, ax=ax, legend=False, transform=ccrs.PlateCarree(), edgecolor='none', zorder=2)
    #add rivers
    ax.add_feature(cfeature.RIVERS, zorder=2.1)
    #add ocean
    #ocean_feature = cfeature.NaturalEarthFeature('physical', 'ocean', '110m', edgecolor='none', facecolor=cfeature.COLORS['water'])
    #ax.add_feature(ocean_feature)
    #add coastline
    ax.add_feature(cfeature.COASTLINE)
    #add gridlines
    if gridlines:
        # Add gridlines with tick labels on the left and bottom
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--', xpadding=1, ypadding=1.5)
        gl.top_labels = True
        gl.right_labels = True
        gl.bottom_labels = False
        gl.left_labels = False
        gl.xlabel_style = {'size': AXIS_LABEL_SIZE, 'color': 'gray'}
        gl.ylabel_style = {'size': AXIS_LABEL_SIZE, 'color': 'gray'}
    if title is not None:
        ax.set_title(title)
    if set_global:
        ax.set_global()
    else:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    if plot_cmap:
        cax = inset_axes(ax, width="30%", height="30%", loc='lower left', bbox_to_anchor=(0.00, 0.14, 0.9, 1), bbox_transform=ax.transAxes)
        draw_2d_colorbar(
            cmap=cmap,
            num_intervals_horizontal=len(x_intervals) - 1,
            num_intervals_vertical=len(y_intervals) - 1,
            xtick_positions=cmap_xtick_positions,
            xtick_labels=cmap_xtick_labels,
            ytick_positions=cmap_ytick_positions,
            ytick_labels=cmap_ytick_labels,
            xlabel=cmap_xlabel,
            ylabel=cmap_ylabel,
            ax=cax
        )
        
    if add_pie_percentage:
        
        def calculate_directions(row):
            current_x = row[x_column_name]
            current_y = row[y_column_name]
            x_interval_middle = np.mean(x_intervals)
            y_interval_middle = np.mean(y_intervals)
            if current_x > x_interval_middle:
                if current_y > y_interval_middle:
                    return 'ur'
                elif current_y < y_interval_middle:
                    return 'lr'
            elif current_x < x_interval_middle:
                if current_y > y_interval_middle:
                    return 'ul'
                elif current_y < y_interval_middle:
                    return 'll'
        
        directions = to_plot_gdf.apply(calculate_directions, axis=1)
        ur_count = (directions == 'ur').sum()
        lr_count = (directions == 'lr').sum()
        ul_count = (directions == 'ul').sum()
        ll_count = (directions == 'll').sum()
        
        if not pie_proportional_radius:
            pie_cax = inset_axes(ax, width="35%", height="35%", loc='lower right', bbox_to_anchor=(0.00, 0.05, 0.9, 1), bbox_transform=ax.transAxes)
            pie_cax.pie([ur_count, ul_count, ll_count, lr_count], autopct='%1.0f%%', startangle=140, colors=[lower_left_corner_hex_color, lower_right_corner_hex_color, upper_right_corner_hex_color, upper_left_corner_hex_color], textprops=dict(color="black", fontsize=TICK_LABEL_SIZE))
        else:
            pie_cax = inset_axes(ax, width="32%", height="32%", loc='lower right', bbox_to_anchor=(0.00, 0.05, 0.87, 1), bbox_transform=ax.transAxes, axes_class = mpl.projections.get_projection_class('polar'))
            draw_pie_with_different_radius(datas=[ur_count, ul_count, ll_count, lr_count], colors=[upper_right_corner_hex_color, upper_left_corner_hex_color, lower_left_corner_hex_color, lower_right_corner_hex_color], ax=pie_cax)
    
    if additional_pie_data is not None and additional_pie_colors is not None:
        additional_pie_cax = inset_axes(ax, width="32%", height="32%", loc='lower right', bbox_to_anchor=(0.00, 0.05, 0.55, 1)
                                        , bbox_transform=ax.transAxes, axes_class = mpl.projections.get_projection_class('polar'))
        draw_pie_with_different_radius(datas=additional_pie_data, colors=additional_pie_colors, ax=additional_pie_cax)
    
    
    if save_path is not None:
        if save_path.endswith('.png'):
            plt.savefig(save_path, dpi=300)
        elif save_path.endswith('.pdf'):
            plt.savefig(save_path, format='pdf', bbox_inches='tight')
        elif save_path.endswith('.svg'):
            plt.savefig(save_path, format='svg', bbox_inches='tight')
        else:
            raise ValueError(f'Unsupported file format: {save_path}')
    if show_plot:
        plt.show()
    
    if return_ax and return_cmap:
        return ax, cmap
    elif return_ax:
        return ax
    elif return_cmap:
        return cmap
    
def violin_plots_of_different_groups(
    to_plot_df,
    to_plot_column_name,
    to_group_by_column_name,
    group_bins=None,
    group_labels=None,
    group_values=None,
    group_colors=None,
    show_means=True,
    mean_color='white',
    show_medians=True,
    median_color='white',
    ax=None,
    title='Violin plots of different groups',
    add_whiskers=True,
    violin_positions=None,
    violin_label_positions=None
):
    
    def adjacent_values(vals, q1, q3):
        upper_adjacent_value = q3 + (q3 - q1) * 1.5
        upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

        lower_adjacent_value = q1 - (q3 - q1) * 1.5
        lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
        return lower_adjacent_value, upper_adjacent_value


    def set_axis_style(ax, labels, violin_label_positions=None):
        if violin_label_positions is None:
            violin_label_positions = np.arange(1, len(labels) + 1)
        ax.set_xticks(violin_label_positions, labels=labels)
        ax.set_xlim(violin_label_positions[0]-0.75, violin_label_positions[-1]+0.75)
        ax.set_xlabel('Sample name')
        
        
    if group_bins is None and group_values is None:
        raise ValueError('Either group_bins or group_values should be provided.')
    if group_bins is not None and group_values is not None:
        raise ValueError('Only one of group_bins and group_values should be provided.')
    if group_bins is not None:
        to_plot_df['category'] = pd.cut(to_plot_df[to_group_by_column_name], bins=group_bins, labels=group_labels)
    elif group_values is not None:
        values_to_labels = dict(zip(group_values, group_labels))
        to_plot_df['category'] = to_plot_df[to_group_by_column_name].map(values_to_labels)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    data_arrays = []
    for group in group_labels:
        current_data = to_plot_df[to_plot_column_name][to_plot_df['category'] == group].to_numpy().flatten()
        #remove nans
        current_data = current_data[~np.isnan(current_data)]
        current_data = sorted(current_data)
        data_arrays.append(current_data)
        
    ax.set_title(title)
    if violin_positions is None:
        violin_positions = np.arange(1, len(group_labels) + 1)
    parts = ax.violinplot(data_arrays, widths=0.5, showmeans=False, showmedians=False, showextrema=False, positions=violin_positions)
    if isinstance(group_colors, mpl.cm.ScalarMappable): 
        medians = [np.median(data) for data in data_arrays]
        for pc, color in zip(parts['bodies'], group_colors.to_rgba(medians)):
            pc.set_facecolor(color)
            pc.set_edgecolor('black')
            pc.set_alpha(1)
    else:
        for pc, color in zip(parts['bodies'], group_colors):
            pc.set_facecolor(color)
            pc.set_edgecolor('black')
            pc.set_alpha(1)
    quartile1, medians, quartile3 = [], [], []
    for data in data_arrays:
        q1, median, q3 = np.percentile(data, [25, 50, 75])
        quartile1.append(q1)
        medians.append(median)
        quartile3.append(q3)
    whiskers = np.array([
        adjacent_values(sorted_array, q1, q3)
        for sorted_array, q1, q3 in zip(data_arrays, quartile1, quartile3)])
    whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

    inds = violin_positions
    if show_means:
        means = [np.mean(data) for data in data_arrays]
        print(f'means: {means}')
        ax.scatter(inds, means, marker='o', color=mean_color, s=30, zorder=3)
    if show_medians:
        print(f'medians: {medians}')
        ax.scatter(inds, medians, marker='o', color=median_color, s=30, zorder=3)
    ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=2.5)
    if add_whiskers:
        ax.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)
    set_axis_style(ax, group_labels, violin_label_positions)