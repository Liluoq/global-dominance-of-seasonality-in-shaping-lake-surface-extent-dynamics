import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs  # Import Cartopy coordinate reference systems
import cartopy.feature as cfeature
from matplotlib.colors import TwoSlopeNorm, SymLogNorm, Normalize, BoundaryNorm
import pandas as pd
from shapely.geometry import LineString
from shapely.ops import split
from shapely.geometry import GeometryCollection
import numpy as np
import shapely.affinity as affinity
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import sys
sys.path.append('/WORK/Codes/global_lake_area/my_spatial_analyze')
from grid_analyze import calculate_statistics_along_direction
from cartopy.mpl.geoaxes import GeoAxes
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

def split_antimeridian(geometry):
    if geometry.is_empty or geometry.bounds[0] > 180 or geometry.bounds[2] < -180:
        return [geometry]
    if geometry.bounds[0] < -180:
        geometry = affinity.translate(geometry, xoff=360)
    split_line = LineString([(180, -90), (180, 90)])
    split_geometries = split(geometry, split_line)
    
    result = []
    for geom in split_geometries.geoms if isinstance(split_geometries, GeometryCollection) else [split_geometries]:
        if geom.bounds[0] >= 180:
            geom = affinity.translate(geom, xoff=-360)
        elif geom.bounds[2] < -180:
            geom = affinity.translate(geom, xoff=360)
        result.append(geom)
    return result

def handle_antimeridian(gdf):
    exploded_geoms = []
    for geom in gdf['geometry']:
        split_geoms = split_antimeridian(geom)
        exploded_geoms.extend(split_geoms)
    
    new_gdf = gdf.loc[gdf.index.repeat([len(split_antimeridian(geom)) for geom in gdf['geometry']])].reset_index(drop=True)
    new_gdf['geometry'] = exploded_geoms
    
    return new_gdf

def plot_grid(grid_gdf, ax=None, projection=ccrs.PlateCarree(), 
              title='Grid Visualization', cmap=None, color_column=None, vmin=None, vmax=None, edgecolor='none',
              div_colorbar=True, use_log_scale_color=False, linthresh_for_log_color=None, use_discrete_color=False, discrete_bins=None, gridlines=False, 
              extent=None, add_rivers=False, add_ocean=False, set_global=False, fix_for_antimeridian=True, 
              significant_ratio_column=None, significant_ratio_threshold=0.5,
              significant_column=None, significant_threshold=0.05, draw_colorbar=True, return_colorbar_sm=False,
              save_path=None, show=True):
    """
    Plots a GeoDataFrame representing a grid, focusing only on the facecolors, and adds coastlines using Cartopy.
    
    Parameters:
    - grid_gdf: GeoDataFrame containing the grid to plot.
    - ax: Optional matplotlib axes object. If None, a new figure and axes will be created with Cartopy.
    - title: Title of the plot.
    - cmap: Colormap for the grid cells.
    - color_column: Column in GeoDataFrame to determine the color of the grid cells.
    - extent: Tuple of (xmin, xmax, ymin, ymax) to set the extent of the plot.
    
    Returns:
    - ax: The matplotlib axes containing the plot.
    """
    if use_log_scale_color and linthresh_for_log_color is None:
        linthresh_for_log_color = 0.1
    
    if significant_ratio_column is not None and significant_column is not None:
        raise ValueError('Only one of significant_ratio_column and significant_column can be specified.')
    
    if discrete_bins is not None:
        if isinstance(discrete_bins, list):
            discrete_bins = np.array(discrete_bins)
        elif isinstance(discrete_bins, np.ndarray):
            pass
        else:   
            raise ValueError('discrete_bins must be a list or a numpy array.')
    
    show = False
    # Create a plot if no axes are provided
    if extent == None:
        bounds = grid_gdf.total_bounds  # [minx, miny, maxx, maxy]
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
    else:
        fig = ax.figure
    
    if fix_for_antimeridian:
        grid_gdf = handle_antimeridian(grid_gdf)
    grid_gdf = grid_gdf.to_crs(projection)
    # Plotting the grid with Cartopy
    if color_column:
        
        if cmap is None:
            # Define the colormap
            cmap = plt.get_cmap('RdBu')
        else:
            cmap = plt.get_cmap(cmap)

        # Define the normalization
        if vmin is None:
            vmin = grid_gdf[color_column].min()
        if vmax is None:
            vmax = grid_gdf[color_column].max()
        if not use_discrete_color:
            if use_log_scale_color:
                if div_colorbar:
                    max_vlim = max(abs(vmin), abs(vmax))
                    norm = SymLogNorm(linthresh=linthresh_for_log_color, linscale=1, vmin=-max_vlim, vmax=max_vlim)
                else:
                    norm = SymLogNorm(linthresh=linthresh_for_log_color, linscale=1, vmin=vmin, vmax=vmax)
            else:
                if div_colorbar:
                    norm = TwoSlopeNorm(vmin=vmin, vcenter=(vmin+vmax)/2, vmax=vmax)
                else:
                    norm = Normalize(vmin=vmin, vmax=vmax)
        else:
            norm = BoundaryNorm(discrete_bins, ncolors=cmap.N, extend='both')

        # Color cells based on values in the color_column
        grid_gdf.plot(column=color_column, ax=ax, cmap=cmap, edgecolor=edgecolor, norm=norm, transform=projection)

        land_feature = cfeature.NaturalEarthFeature(
            'physical', 'land', '50m',
            edgecolor='face',
            facecolor='darkgrey'
        )

        ax.add_feature(land_feature, zorder=0.5)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm._A = []  # Fake up the array of the scalar mappable. Urgh...
        if draw_colorbar:
            fig.colorbar(sm, ax=ax, extend='both')
    else:
        # Plot all cells in the same color if no color_column is specified
        grid_gdf.plot(ax=ax, color='lightblue', edgecolor='none')
    
    if significant_ratio_column is not None or significant_column is not None:
        geometry_column = grid_gdf.geometry.name
        if significant_ratio_column is not None:
            sig_gdf = grid_gdf[grid_gdf[significant_ratio_column] > significant_ratio_threshold]
        elif significant_column is not None:
            sig_gdf = grid_gdf[grid_gdf[significant_column] < significant_threshold]    
        for idx, row in sig_gdf.iterrows():
            ax.scatter(row[geometry_column].centroid.x, row[geometry_column].centroid.y, marker='x', color='black', s=1, linewidths=0.75)
    
    
    
    if add_rivers:
        ax.add_feature(cfeature.RIVERS)
    if add_ocean:
        ocean_feature = cfeature.NaturalEarthFeature('physical', 'ocean', '110m', edgecolor='none', facecolor=cfeature.COLORS['water'])
        ax.add_feature(ocean_feature)
    # Add coastlines to the plot
    ax.coastlines()

    if gridlines:
        # Add gridlines with tick labels on the left and bottom
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--', xpadding=1, ypadding=1)
        gl.top_labels = True
        gl.right_labels = True
        gl.bottom_labels = False
        gl.left_labels = False
        gl.xlabel_style = {'size': 10, 'color': 'gray'}
        gl.ylabel_style = {'size': 10, 'color': 'gray'}
    
    # Setting title and labels
    if title is not None:
        ax.set_title(title)

    if extent:
        ax.set_extent(extent, crs=ccrs.PlateCarree())

    if set_global:
        ax.set_global()
    
    # Show plot if it is not part of a larger multi-plot
    if show:
        plt.show()
    if save_path:
        plt.savefig(save_path, dpi=300)

    if return_colorbar_sm:
        return ax, sm
    else:
        return ax

def plot_grid_with_panels(
    grid_gdf, ax=None, inset_axes_layout=None,
    main_plot_kwargs={}, 
    panel_columns=None,
    panel_stat_lists=[['sum']],
    panel_steps=1.0,
    panel_legend_lists=[['Sum']],
    panel_color_lists=[['black']],
    additional_panel_plot_kwargs={},
    additional_colorbar_kwargs={},
    save_path=None
):
    if ax is None:
        projection = main_plot_kwargs['projection']
        fig, ax = plt.subplots(1, 1, figsize=(24, 12))
        
    
    if inset_axes_layout is None:
        inset_axes_layout = {
            'main': {'width': '80%', 'height': '80%', 'loc': 'lower left', 
                     'bbox_to_anchor': (0.0, 0.0, 1.0, 1.0), 'bbox_transform': ax.transAxes, 'borderpad': 0.0},
            'top': {'width': '70%', 'height': '30%', 'loc': 'upper left', 
                    'bbox_to_anchor': (0.0, 0.0, 1.0, 1.0), 'bbox_transform': ax.transAxes, 'borderpad': 0.0},
            'right': {'width': '30%', 'height': '70%', 'loc': 'lower right', 
                      'bbox_to_anchor': (0.0, 0.0, 1.0, 1.0), 'bbox_transform': ax.transAxes, 'borderpad': 0.0},
        }
    #make ax invisible
    ax.axis('off')
    
    if 'colorbar_ax' in inset_axes_layout.keys():
        plot_colorbar = True
    else:
        plot_colorbar = False
    
    top_ax = inset_axes(ax, **inset_axes_layout['top'], bbox_transform=ax.transAxes)
    right_ax = inset_axes(ax, **inset_axes_layout['right'], bbox_transform=ax.transAxes)
    main_ax = inset_axes(ax, **inset_axes_layout['main'], bbox_transform=ax.transAxes, axes_class=GeoAxes, axes_kwargs={'projection': projection})
    ocean = cfeature.NaturalEarthFeature(
    'physical', 'ocean', '50m', edgecolor='face', facecolor='lightgrey')
    main_ax.add_feature(ocean)
    if plot_colorbar:
        colorbar_ax = inset_axes(ax, **inset_axes_layout['colorbar_ax'], bbox_transform=ax.transAxes)
    
    if plot_colorbar:
        _, sm = plot_grid(
            grid_gdf=grid_gdf, ax=main_ax, **main_plot_kwargs, return_colorbar_sm=True
        )
    else:
        plot_grid(
            grid_gdf=grid_gdf, ax=main_ax, **main_plot_kwargs
        )
    
    if panel_columns is None:
        panel_columns = [main_plot_kwargs['color_column']]
    
    for panel_ax, pos in zip([top_ax, right_ax], ['top', 'right']):
        for panel_column, panel_stat_list, panel_legend_list, panel_color_list in zip(panel_columns, panel_stat_lists, panel_legend_lists, panel_color_lists):
            for panel_stat, panel_legend, panel_color in zip(panel_stat_list, panel_legend_list, panel_color_list):
                if pos == 'top':
                    stat_along_longitude = calculate_statistics_along_direction(
                        gdf=grid_gdf, column=panel_column, axis='longitude', step=panel_steps, stat_type=panel_stat
                    )
                    panel_ax.plot(stat_along_longitude['bin_center'], stat_along_longitude[panel_stat], 
                            color=panel_color, label=panel_legend,
                            **additional_panel_plot_kwargs)
                    panel_ax.set_xlim([-180, 180])
                    panel_ax.xaxis.set_visible(False)
                    panel_ax.axhline(0, color='black', linewidth=0.5)
                    panel_ax.spines['bottom'].set_visible(False)
                    panel_ax.spines['top'].set_visible(False)
                    panel_ax.spines['left'].set_visible(False)
                    panel_ax.tick_params(labelleft=False, labelright=True, left=False, right=True)
                elif pos == 'right':
                    stat_along_latitude = calculate_statistics_along_direction(
                        gdf=grid_gdf, column=panel_column, axis='latitude', step=panel_steps, stat_type=panel_stat
                    )
                    panel_ax.plot(stat_along_latitude[panel_stat], stat_along_latitude['bin_center'], 
                            color=panel_color, label=panel_legend,
                            **additional_panel_plot_kwargs)
                    panel_ax.set_ylim([-90, 90])
                    panel_ax.yaxis.set_visible(False)
                    panel_ax.axvline(0, color='black', linewidth=0.5)
                    panel_ax.spines['left'].set_visible(False)
                    panel_ax.spines['right'].set_visible(False)
                    panel_ax.spines['bottom'].set_visible(False)
                    panel_ax.tick_params(labeltop=True, labelbottom=False, top=True, bottom=False)
                    
                panel_ax.legend()
          
    if plot_colorbar:
        colorbar_ax.figure.colorbar(sm, cax=colorbar_ax, **additional_colorbar_kwargs)
          
    if save_path:
        if save_path.endswith('.png'):
            plt.savefig(save_path, dpi=300)
        elif save_path.endswith('.pdf'):
            plt.savefig(save_path, format='pdf', bbox_inches='tight')
        elif save_path.endswith('.svg'):
            plt.savefig(save_path, format='svg', bbox_inches='tight')
        else:
            raise ValueError('save_path must end with .png or .pdf or .svg')