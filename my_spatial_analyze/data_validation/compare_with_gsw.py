import sys
import rasterio.warp 
sys.path.append('../../')
from my_unet_gdal.area_calculation import calculate_lake_area_grid_parallel, create_grid, filter_lakes_to_cells_and_create_layers, save_extent_as_shapefile, save_vector_layer_to_shapefile, rasterize_and_calculate_lake_area, clip_raster_to_extent, check_dict_overlap, check_and_reproject_vector_srs_to_raster
import rasterio
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os
import csv
from osgeo import gdal, ogr, osr
from multiprocessing import Pool
import random
import pandas as pd
from data_validation import plot_using_datashader, plot_using_scipy_gaussian_kde
import re
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import mpl_scatter_density # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


def mask_my_water_using_gsw(
    my_water_tif_path,
    gsw_tif_path,
    masked_my_water_output_tif_path,
    gsw_non_valid_value=0,
):
    print(f'Masking My Water using GSW for {my_water_tif_path}')
    output_folder = os.path.dirname(masked_my_water_output_tif_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    with rasterio.open(my_water_tif_path) as src:
        my_water = src.read(1)
        my_water_transform = src.transform
        my_water_crs = src.crs
        my_water_meta = src.meta.copy()
    with rasterio.open(gsw_tif_path) as src:
        gsw = src.read(1)
        gsw_transform = src.transform
        gsw_crs = src.crs
        gsw_meta = src.meta.copy()
        
    if my_water_crs != gsw_crs:
        raise ValueError('CRS of input GeoTIFFs do not match')
    
    if my_water_transform != gsw_transform:
        print(f'GeoTransforms of input GeoTIFFs do not match. Reprojecting GSW to match My Water')
        reprojected_gsw = np.zeros_like(my_water)
        rasterio.warp.reproject(
            source=gsw,
            destination=reprojected_gsw,
            src_transform=gsw_transform,
            src_crs=gsw_crs,
            dst_transform=my_water_transform,
            dst_crs=my_water_crs,
            resampling=rasterio.warp.Resampling.nearest
        ) 
        
    masked_my_water = np.where(gsw == gsw_non_valid_value, 0, my_water)
    
    out_meta = my_water_meta.copy()
    out_meta.update({"driver": "GTiff",
                     "height": masked_my_water.shape[0],
                     "width": masked_my_water.shape[1],
                     "transform": my_water_transform,
                     'crs': my_water_crs,
                     "dtype": masked_my_water.dtype,
                     "nodata": 0})
    
    with rasterio.open(masked_my_water_output_tif_path, "w", **out_meta) as dest:
        dest.write(masked_my_water, 1)
        
def mask_my_water_and_calculate_both_area(
    my_water_tif_paths,
    gsw_tif_paths,
    output_my_water_csv_paths,
    output_gsw_csv_paths,
    my_water_raster_clip_temporary_folders,
    gsw_raster_clip_temporary_folders,
    lake_shape_path,
    grid_size: float,
    temporary_vector_folder,
    filter_once=True,
    save_lake_rasterization=True,
    lake_id_field='Hylak_id',
    outside_value=-1,
    force_raster_binary_gt_threshold_masked_my_water=None,
    force_raster_binary_eq_value_masked_my_water=None,
    force_raster_binary_gt_threshold_gsw=None,
    force_raster_binary_eq_value_gsw=None,
    num_processes=1,
    remove_temporary=True,
    check_overlap=True,
    verbose=2
):
    
    filtered = False
    rasterized_already_saved = False

    if not len(my_water_tif_paths) == len(gsw_tif_paths) == len(output_my_water_csv_paths) == len(output_gsw_csv_paths):
        raise ValueError('Number of input paths must be the same')
    
    for my_water_tif_path, my_water_raster_clip_temporary_folder, output_my_water_csv_path, gsw_tif_path, gsw_raster_clip_temporary_folder, output_gsw_csv_path in zip(my_water_tif_paths, my_water_raster_clip_temporary_folders, output_my_water_csv_paths, gsw_tif_paths, gsw_raster_clip_temporary_folders, output_gsw_csv_paths):
        if verbose != 0:
            print(f"Processing raster: {my_water_tif_path} and {gsw_tif_path}")
        # open raster and vector data
        try:
            my_water_raster_ds = gdal.Open(my_water_tif_path) # open the raster data
            my_water_raster_srs = osr.SpatialReference(my_water_raster_ds.GetProjection()) # read the spatial reference system of the raster
            gsw_raster_ds = gdal.Open(gsw_tif_path) # open the raster data
            gsw_raster_srs = osr.SpatialReference(gsw_raster_ds.GetProjection()) # read the spatial reference system of the raster
            lake_ds = ogr.Open(lake_shape_path) # open the vector data
            lake_layer = lake_ds.GetLayer() # get the first layer of the vector data, which is supposed to be the lake layer
            lake_srs = lake_layer.GetSpatialRef() # read the spatial reference system of the vector data
        except Exception as e:
            # if there is an error opening the raster or vector data, print the error message and exit the program
            print(f"Error opening raster or vector data: {e}")
            sys.exit(1)
        # check if the spatial reference systems of the raster and vector data match
        # it is important that the raster and vector data have the same spatial reference system
        assert my_water_raster_srs.IsSame(lake_srs) and gsw_raster_srs.IsSame(lake_srs), "Raster and vector data must have the same spatial reference system."
        
        if filter_once and filtered:
            pass
        else:
            if verbose != 0:
                print(f"Filtering lakes to grid cells and creating in-memory layers...")
            # create a grid of cells with unique identifiers, which will be used to divide the lakes into subgroups 
            lake_layer_extent = lake_layer.GetExtent()
            minx, maxx, miny, maxy = lake_layer_extent # get the extent of the lake layer, order is xmin, xmax, ymin, ymax
            grid = create_grid(minx, maxx, miny, maxy, grid_size, verbose=verbose) # order is important
            cells_to_filtered_ds_and_layers = filter_lakes_to_cells_and_create_layers( # filter the lakes to the grid cells and create in-memory layers for each subgroups of lakes
                all_lakes_layer=lake_layer,
                grid=grid,
                verbose=verbose
            )
            # update the grid dictionary with the new extent of the grid cells
            grid_having_lakes = {}
            cells_having_lakes_to_filtered_ds_and_layers = {}
            for idx in grid.keys():
                current_ds, current_layer = cells_to_filtered_ds_and_layers[idx]
                if current_ds is not None and current_layer is not None:
                    current_layer_extent = current_layer.GetExtent()
                    current_minx, current_maxx, current_miny, current_maxy = current_layer_extent
                    current_new_grid_cell = (current_minx, current_maxx, current_miny, current_maxy)
                    grid_having_lakes[idx] = current_new_grid_cell
                    cells_having_lakes_to_filtered_ds_and_layers[idx] = (current_ds, current_layer)
            print(grid_having_lakes)
                            
            grid_cells_shp_temporary_folder = os.path.join(temporary_vector_folder, 'grid_cells_shp')
            if not os.path.exists(grid_cells_shp_temporary_folder):
                os.makedirs(grid_cells_shp_temporary_folder)
            for idx, extent in grid_having_lakes.items():
                current_grid_cell_shp_filename = f"grid_cell_{idx}.shp"
                current_grid_cell_shp_path = os.path.join(grid_cells_shp_temporary_folder, current_grid_cell_shp_filename)
                save_extent_as_shapefile(extent, current_grid_cell_shp_path, my_water_raster_srs.ExportToWkt(), verbose=verbose)
            
            filtered_lakes_shp_temporary_folder = os.path.join(temporary_vector_folder, 'filtered_lakes_shp')
            if not os.path.exists(filtered_lakes_shp_temporary_folder):
                os.makedirs(filtered_lakes_shp_temporary_folder)
            cells_having_lakes_to_filtered_lake_shp_paths = {}
            save_lake_rasterization_paths = {}
                
            for idx, (_, filtered_lake_layer) in cells_having_lakes_to_filtered_ds_and_layers.items():
                current_filtered_lakes_shp_filename = f"filtered_lakes_{idx}.shp"
                current_filtered_lakes_shp_path = os.path.join(filtered_lakes_shp_temporary_folder, current_filtered_lakes_shp_filename)
                cells_having_lakes_to_filtered_lake_shp_paths[idx] = current_filtered_lakes_shp_path
                if save_lake_rasterization and not rasterized_already_saved:
                    current_lake_rasterization_filename = f"filtered_lakes_rasterized_{idx}.tif"
                    current_lake_rasterization_path = os.path.join(filtered_lakes_shp_temporary_folder, current_lake_rasterization_filename)
                    save_lake_rasterization_paths[idx] = current_lake_rasterization_path
                else:
                    current_lake_rasterization_path = None
                save_vector_layer_to_shapefile(filtered_lake_layer, current_filtered_lakes_shp_path, verbose=verbose)
            
            filtered = True
        
        # create a list of arguments for the clip_raster_to_extent function and the rasterize_and_calculate_lake_area function
        my_water_raster_ds_path_basename = os.path.basename(my_water_tif_path)
        my_water_raster_ds_path_basename_noext = os.path.splitext(my_water_raster_ds_path_basename)[0]
        if not os.path.exists(my_water_raster_clip_temporary_folder):
            os.makedirs(my_water_raster_clip_temporary_folder)
        my_water_clipped_raster_paths = {idx: os.path.join(my_water_raster_clip_temporary_folder, f"{my_water_raster_ds_path_basename_noext}_clipped_{idx}.tif") for idx in grid_having_lakes.keys()}
        my_water_clip_args = [(my_water_tif_path, grid_cell, my_water_clipped_raster_paths[idx], verbose) for idx, grid_cell in grid_having_lakes.items()]

        gsw_raster_ds_path_basename = os.path.basename(gsw_tif_path)
        gsw_raster_ds_path_basename_noext = os.path.splitext(gsw_raster_ds_path_basename)[0]
        if not os.path.exists(gsw_raster_clip_temporary_folder):
            os.makedirs(gsw_raster_clip_temporary_folder)
        gsw_clipped_raster_paths = {idx: os.path.join(gsw_raster_clip_temporary_folder, f"{gsw_raster_ds_path_basename_noext}_clipped_{idx}.tif") for idx in grid_having_lakes.keys()}
        gsw_clip_args = [(gsw_tif_path, grid_cell, gsw_clipped_raster_paths[idx], verbose) for idx, grid_cell in grid_having_lakes.items()]
        
        mask_my_water_output_paths = {idx: os.path.join(my_water_raster_clip_temporary_folder, f"{my_water_raster_ds_path_basename_noext}_clipped_masked_{idx}.tif") for idx in grid_having_lakes.keys()}
        mask_my_water_using_gsw_args = [(my_water_clipped_raster_paths[idx], gsw_clipped_raster_paths[idx], mask_my_water_output_paths[idx]) for idx in grid_having_lakes.keys()]
        
        
        masked_my_water_calculate_lake_area_args = [
            (mask_my_water_output_paths[idx], cells_having_lakes_to_filtered_lake_shp_paths[idx], lake_id_field, outside_value, force_raster_binary_gt_threshold_masked_my_water, force_raster_binary_eq_value_masked_my_water, verbose, save_lake_rasterization_paths[idx]) for idx in cells_having_lakes_to_filtered_lake_shp_paths.keys()
        ]
        gsw_calculate_lake_area_args = [
            (gsw_clipped_raster_paths[idx], cells_having_lakes_to_filtered_lake_shp_paths[idx], lake_id_field, outside_value, force_raster_binary_gt_threshold_gsw, force_raster_binary_eq_value_gsw, verbose, save_lake_rasterization_paths[idx]) for idx in cells_having_lakes_to_filtered_lake_shp_paths.keys()
        ]
        calculate_area_function = rasterize_and_calculate_lake_area
            
        # parallel execution (num_process > 1)
        if num_processes > 1:
            with Pool(num_processes) as pool:
                pool.starmap(clip_raster_to_extent, my_water_clip_args)
            with Pool (num_processes) as pool:
                pool.starmap(clip_raster_to_extent, gsw_clip_args) 
            
            with Pool(num_processes) as pool:
                pool.starmap(mask_my_water_using_gsw, mask_my_water_using_gsw_args)
            
            random.shuffle(masked_my_water_calculate_lake_area_args)
            random.shuffle(gsw_calculate_lake_area_args)
            
            with Pool(num_processes) as pool:
                my_water_areas_list = pool.starmap(calculate_area_function, masked_my_water_calculate_lake_area_args) # contains a list of dictionaries that is rasterization field: area
                gsw_areas_list = pool.starmap(calculate_area_function, gsw_calculate_lake_area_args)
        # serial execution (num_process = 1)
        else:
            for clip_args in my_water_clip_args:
                clip_raster_to_extent(*clip_args)
            for clip_args in gsw_clip_args:
                clip_raster_to_extent(*clip_args)
            for mask_args in mask_my_water_using_gsw_args:
                mask_my_water_using_gsw(*mask_args)
            
            my_water_areas_list = []
            for args in masked_my_water_calculate_lake_area_args:
                my_water_areas_list.append(calculate_area_function(*args))
                
            gsw_areas_list = []
            for args in gsw_calculate_lake_area_args:
                gsw_areas_list.append(calculate_area_function(*args))
        # remove temporary files
        if remove_temporary:
            for clipped_raster_path in my_water_clipped_raster_paths.values():
                if verbose != 0:
                    print(f"Removing temporary file: {clipped_raster_path}")
                os.remove(clipped_raster_path)
            for clipped_raster_path in gsw_clipped_raster_paths.values():
                if verbose != 0:
                    print(f"Removing temporary file: {clipped_raster_path}")
                os.remove(clipped_raster_path)
            for masked_my_water_output_path in mask_my_water_output_paths.values():
                if verbose != 0:
                    print(f"Removing temporary file: {masked_my_water_output_path}")
                os.remove(masked_my_water_output_path)
        # check for overlap and print details
        if check_overlap: 
            if verbose != 0:
                print("Checking for overlap between lake areas...")
            check_dict_overlap(my_water_areas_list)
            check_dict_overlap(gsw_areas_list)
        #merge the lake areas for each grid cell into one dictionary
        merged_my_water_areas = {**{k: v for d in my_water_areas_list for k, v in d.items()}}
        merged_gsw_areas = {**{k: v for d in gsw_areas_list for k, v in d.items()}}
        
        with open(output_my_water_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)

            # Write the header
            writer.writerow(['Hylak_id', 'area'])

            # Write the data
            for hylak_id, area in merged_my_water_areas.items():
                writer.writerow([hylak_id, area])
                
        with open(output_gsw_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)

            # Write the header
            writer.writerow(['Hylak_id', 'area'])

            # Write the data
            for hylak_id, area in merged_gsw_areas.items():
                writer.writerow([hylak_id, area])
        
        rasterized_already_saved = True
        # change save_lake_rasterization_paths to key:None
        save_lake_rasterization_paths = {key: None for key in save_lake_rasterization_paths.keys()}
    
    return None
        
def concatenate_area_csvs(
    csv_paths_to_concatenate,
    concatenated_save_path
):
    all_data = pd.DataFrame()
    # Loop through each file in the directory
    for filepath in csv_paths_to_concatenate:
        filename = os.path.basename(filepath)
        if filename.endswith('_area.csv'):
            # Extract the basin_id and start_date using regex
            match = re.match(r'(\d+)_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})_.*_area\.csv', filename)
            if match:
                basin_id, start_date, end_date = match.groups()
                
                # Read the CSV file
                df = pd.read_csv(filepath)
                
                # Add a column for the start_date with the area as the value
                df['Start_Date'] = start_date  # Add the start_date as a new column for pivoting
                
                # Append this data to the all_data DataFrame
                all_data = pd.concat([all_data, df], ignore_index=True)
    # Pivot the DataFrame to get the desired format
    pivot_df = all_data.pivot_table(index='Hylak_id', columns='Start_Date', values='area', aggfunc='sum')

    # Ensure Lake_ID remains as a regular column
    pivot_df.reset_index(inplace=True)  # This now keeps Lake_ID as a column after resetting the index

    output_concatenated_csv_folder = os.path.dirname(concatenated_save_path)
    if not os.path.exists(output_concatenated_csv_folder):
        os.makedirs(output_concatenated_csv_folder)
    # Save the final DataFrame to a new CSV file
    pivot_df.to_csv(concatenated_save_path, index=False)
        
def compare_gsw_with_my_water(
    gsw_concatenated_area_csv_path,
    my_water_gsw_masked_concatenated_area_csv_path,
    area_columns,
    lake_id_column_name='Hylak_id',
    unit_scale=1e-6,
    ax=None,
    title=None
):
    #change global font size to 12
    plt.rcParams.update({'font.size': 14})
    gsw_concatenated_df = pd.read_csv(gsw_concatenated_area_csv_path)
    my_water_gsw_masked_concatenated_df = pd.read_csv(my_water_gsw_masked_concatenated_area_csv_path)
    gsw_area_columns = [f'gsw_{area_column}' for area_column in area_columns]
    my_water_area_columns = [f'my_water_{area_column}' for area_column in area_columns]
    gsw_concatenated_df = gsw_concatenated_df.rename(columns={area_column: gsw_area_column for area_column, gsw_area_column in zip(area_columns, gsw_area_columns)})
    my_water_gsw_masked_concatenated_df = my_water_gsw_masked_concatenated_df.rename(columns={area_column: my_water_area_column for area_column, my_water_area_column in zip(area_columns, my_water_area_columns)})
    merged_df = pd.merge(gsw_concatenated_df, my_water_gsw_masked_concatenated_df, on=lake_id_column_name)
    gsw_areas_array_2d = merged_df[gsw_area_columns].to_numpy()
    my_water_areas_array_2d = merged_df[my_water_area_columns].to_numpy()
    gsw_areas_array = merged_df[gsw_area_columns].to_numpy().flatten()
    my_water_areas_array = merged_df[my_water_area_columns].to_numpy().flatten()
    
    # mask 0 as nan
    nan_mask = np.logical_and(gsw_areas_array != 0, my_water_areas_array != 0)
    gsw_areas_array = gsw_areas_array[nan_mask]
    my_water_areas_array = my_water_areas_array[nan_mask]
    nan_mask_2d = np.logical_and(gsw_areas_array_2d != 0, my_water_areas_array_2d != 0)
    print(nan_mask_2d.shape)
    gsw_areas_array_2d = np.where(nan_mask_2d, gsw_areas_array_2d, np.nan)
    my_water_areas_array_2d = np.where(nan_mask_2d, my_water_areas_array_2d, np.nan)
    
    
    gsw_areas_array = gsw_areas_array * unit_scale
    my_water_areas_array = my_water_areas_array * unit_scale
    gsw_areas_array_2d = gsw_areas_array_2d * unit_scale
    my_water_areas_array_2d = my_water_areas_array_2d * unit_scale
    
    assert len(gsw_areas_array) == len(my_water_areas_array), 'Length of GSW and My Water areas must be the same'
    assert gsw_areas_array_2d.shape == my_water_areas_array_2d.shape, 'Shape of GSW and My Water areas must be the same'
    
    num_obs = len(gsw_areas_array)
    # create square ax
    # "Viridis-like" colormap with white background
    white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
        (0, '#ffffff'),
        (1e-3, '#EDEDE9'),
        (0.05, '#404388'),
        (0.4, '#2a788e'),
        (0.6, '#21a784'),
        (0.8, '#78d151'),
        (1, '#fde624'),
    ], N=256)

    def using_mpl_scatter_density(fig, x, y, ax=None):
        if ax is None:
            ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
        else:
            fig = ax.figure
        density = ax.scatter_density(x, y, cmap=white_viridis, dpi=40)
        fig.colorbar(density, ax=ax, label='Data point count')
        return ax

    if ax is None:
        fig = plt.figure()
        ax = using_mpl_scatter_density(fig=fig, x=gsw_areas_array, y=my_water_areas_array)
    else:
        ax = using_mpl_scatter_density(fig=ax.figure, x=gsw_areas_array, y=my_water_areas_array, ax=ax)
    # add 45 degree line
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    min_val = min(xmin, ymin)
    max_val = max(xmax, ymax)
    ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', alpha=0.25)
    r2 = r2_score(gsw_areas_array, my_water_areas_array)
    # calculate median pbias
    median_abs_pbias = np.nanmedian(np.abs((my_water_areas_array - gsw_areas_array)) / gsw_areas_array) * 100
    median_pbias = np.nanmedian((my_water_areas_array - gsw_areas_array) / gsw_areas_array) * 100
    # calculate NSE (Nash-Sutcliffe Efficiency) for each lake using 2d arrays, and get median, each row is a lake
    print(gsw_areas_array_2d.shape)
    print(my_water_areas_array_2d.shape)
    nse = 1 - np.nansum((my_water_areas_array_2d - gsw_areas_array_2d) ** 2, axis=1) / np.nansum((gsw_areas_array_2d - np.reshape(np.repeat(np.nanmean(gsw_areas_array_2d, axis=1), gsw_areas_array_2d.shape[1]), (-1, gsw_areas_array_2d.shape[1]))) ** 2, axis=1)
    median_nse = np.nanmedian(nse)
    # calculateeee Root Mean Standard Deviation Ratio (RSR) for each lake using 2d arrays, and get median, each row is a lake
    rsr = np.sqrt(np.nansum((my_water_areas_array_2d - gsw_areas_array_2d) ** 2, axis=1) / np.nansum((gsw_areas_array_2d - np.reshape(np.repeat(np.nanmean(gsw_areas_array_2d, axis=1), gsw_areas_array_2d.shape[1]), (-1, gsw_areas_array_2d.shape[1]))) ** 2, axis=1))
    median_rsr = np.nanmedian(rsr)
    # calculate rRMSE (relative Root Mean Square Error) for each lake using 2d arrays, and get median, each row is a lake
    rrmse = np.sqrt(np.nansum((my_water_areas_array_2d - gsw_areas_array_2d) ** 2, axis=1) / np.nansum(gsw_areas_array_2d ** 2, axis=1))
    median_rrmse = np.nanmedian(rrmse)
    # Annotate the plot with R-square, median pbias, and N without a box
    ax.text(0.05, 0.95, f'$R^2$ = {r2:.2f}\nN = {num_obs}\nMedian PBias = {median_pbias:.0f}%\nMedian Abs. PBias = {median_abs_pbias:.0f}%', transform=ax.transAxes, 
                    verticalalignment='top')
    
    ax.set_xlabel('Monthly area (GSW)')
    ax.set_ylabel('Monthly area (This study)')
    if title is not None:
        ax.set_title(title)
    #set both axis to be log scale
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_aspect('equal', adjustable='box')
    
    
        
if __name__ == '__main__':
    CALCULATE_AREA = True
    
    processed_basin_id_list = [
        1020000010, 8020000010, 4020000010, 5020000010, 2020000010, 9020000010, 
        7020000010, 6020000010, 3020000010, 1020011530, 8020008900, 4020006940, 
        5020015660, 2020003440, 7020014250, 6020006540, 3020003790, 1020018110, 
        8020010700, 4020015090, 5020037270, 2020018240, 7020021430, 6020008320, 
        3020005240, 1020021940, 8020020760, 4020024190, 5020049720, 2020024230, 
        7020024600, 6020014330, 3020008670, 1020027430, 8020022890, 4020034510, 
        2020033490, 7020038340, 6020017370, 1020034170, 8020032840, # lack 3020009320
        2020041390, 7020046750, 6020021870, 1020035180, 8020044560, 5020082270, 
        2020057170, 7020047840, 6020029280, 2020065840, 7020065090, 2020071190, 
        4020050210, 3020024310, 4020050220, 1020040190, 4020050290, 4020050470
    ]
    
    basin_id_list = [
        1020000010, 8020000010, 4020000010, 5020000010, 2020000010, 9020000010, 
        7020000010, 6020000010, 3020000010, 1020011530, 8020008900, 4020006940, 
        5020015660, 2020003440, 7020014250, 6020006540, 3020003790, 1020018110, 
        8020010700, 4020015090, 5020037270, 2020018240, 7020021430, 6020008320, 
        3020005240, 1020021940, 8020020760, 4020024190, 5020049720, 2020024230, 
        7020024600, 6020014330, 3020008670, 1020027430, 8020022890, 4020034510, 
        2020033490, 7020038340, 6020017370, 1020034170, 8020032840, 3020009320,
        2020041390, 7020046750, 6020021870, 1020035180, 8020044560, 5020082270, 
        2020057170, 7020047840, 6020029280, 2020065840, 7020065090, 2020071190, 
        4020050210, 3020024310, 4020050220, 1020040190, 4020050290, 4020050470
    ]
    
    if CALCULATE_AREA:
        my_water_tif_path_pattern = '/WORK/Data/global_lake_area/mosaic_tifs/{basin_id}/{basin_id}_{start_date}_{end_date}_water_mosaic.tif'
        gsw_tif_path_pattern = '/WORK/Data/global_lake_area/gsw_images/mosaic/{basin_id}/{basin_id}_gsw_30m_{start_date}_{end_date}.tif'
        output_my_water_csv_path_pattern = '/WORK/Data/global_lake_area/area_csvs/gsw_masked_my_water_area/{basin_id}/{basin_id}_{start_date}_{end_date}_gsw_masked_my_water_area.csv'
        output_gsw_csv_path_pattern = '/WORK/Data/global_lake_area/area_csvs/gsw_area/{basin_id}/{basin_id}_{start_date}_{end_date}_gsw_area.csv'
        my_water_raster_clip_temporary_folder_pattern = '/WORK/Data/global_lake_area/raster_clip_temporary_for_gsw_masked_my_water/{basin_id}/{start_date}_{end_date}'
        gsw_raster_clip_temporary_folder_pattern = '/WORK/Data/global_lake_area/raster_clip_temporary_for_gsw/{basin_id}/{start_date}_{end_date}'
        
        temporary_vector_folder_pattern = '/WORK/Data/global_lake_area/temporary_vector_for_gsw_masked_my_water/{basin_id}'
        
        my_water_concatenated_save_path = '/WORK/Data/global_lake_area/area_csvs/gsw_masked_my_water_area_concatenated/{basin_id}_gsw_masked_my_water_area_concatenated.csv'
        gsw_concatenated_save_path = '/WORK/Data/global_lake_area/area_csvs/gsw_area_concatenated/{basin_id}_gsw_area_concatenated.csv'
        for basin_id in basin_id_list:
            if basin_id in processed_basin_id_list:
                continue
            input_lake_shp_folder = '/WORK/Data/global_lake_area/lake_shps/HydroLAKES_updated_using_GLAKES/per_basin_no_contained_buffered'
            input_lake_shp_file_name = f'hylak_buffered_updated_no_contained_{basin_id}.shp'
            input_lake_shp_path = os.path.join(input_lake_shp_folder, input_lake_shp_file_name)
            
            start_date = '2001-01-01'
            end_date = '2022-01-01'
            date_fmt = '%Y-%m-%d'
            
            start_date = datetime.strptime(start_date, date_fmt)
            end_date = datetime.strptime(end_date, date_fmt)
            my_water_tif_paths = []
            gsw_tif_paths = []
            my_water_raster_clip_temporary_folders = []
            gsw_raster_clip_temporary_folders = []
            output_my_water_csv_paths = []
            output_gsw_csv_paths = []
            current_date = start_date
            while current_date < end_date:
                current_date_str = current_date.strftime(date_fmt)
                next_month_str = (current_date + relativedelta(months=1)).strftime(date_fmt)
                current_my_water_tif_path = my_water_tif_path_pattern.format(basin_id=basin_id, start_date=current_date_str, end_date=next_month_str)
                current_gsw_tif_path = gsw_tif_path_pattern.format(basin_id=basin_id, start_date=current_date_str, end_date=next_month_str)
                current_output_my_water_csv_path = output_my_water_csv_path_pattern.format(basin_id=basin_id, start_date=current_date_str, end_date=next_month_str)
                current_output_gsw_csv_path = output_gsw_csv_path_pattern.format(basin_id=basin_id, start_date=current_date_str, end_date=next_month_str)
                if not os.path.exists(os.path.dirname(current_output_my_water_csv_path)):
                    os.makedirs(os.path.dirname(current_output_my_water_csv_path))
                if not os.path.exists(os.path.dirname(current_output_gsw_csv_path)):
                    os.makedirs(os.path.dirname(current_output_gsw_csv_path))
                current_my_water_raster_clip_temporary_folder = my_water_raster_clip_temporary_folder_pattern.format(basin_id=basin_id, start_date=current_date_str, end_date=next_month_str)
                current_gsw_raster_clip_temporary_folder = gsw_raster_clip_temporary_folder_pattern.format(basin_id=basin_id, start_date=current_date_str, end_date=next_month_str)
                
                # if need reproject lake shapefile, add '_reprojected' to the lake shp name
                lake_shp_name_if_need_reprojection = input_lake_shp_file_name.replace('.shp', '_reprojected.shp')
                lake_shp_path_if_need_reprojection = os.path.join(input_lake_shp_folder, lake_shp_name_if_need_reprojection)
                reprojected = check_and_reproject_vector_srs_to_raster(
                    raster_path=current_my_water_tif_path,
                    vector_path=input_lake_shp_path,
                    output_path=lake_shp_path_if_need_reprojection,
                    verbose=2
                )
                if reprojected:
                    input_lake_shp_path = lake_shp_path_if_need_reprojection
                
                my_water_tif_paths.append(current_my_water_tif_path)
                gsw_tif_paths.append(current_gsw_tif_path)
                my_water_raster_clip_temporary_folders.append(current_my_water_raster_clip_temporary_folder)
                gsw_raster_clip_temporary_folders.append(current_gsw_raster_clip_temporary_folder)
                output_my_water_csv_paths.append(current_output_my_water_csv_path)
                output_gsw_csv_paths.append(current_output_gsw_csv_path)
                
                current_date = current_date + relativedelta(months=1)
                
            current_temporary_vector_folder = temporary_vector_folder_pattern.format(basin_id=basin_id)
            
            mask_my_water_and_calculate_both_area(
                my_water_tif_paths=my_water_tif_paths,
                gsw_tif_paths=gsw_tif_paths,
                output_my_water_csv_paths=output_my_water_csv_paths,
                output_gsw_csv_paths=output_gsw_csv_paths,
                my_water_raster_clip_temporary_folders=my_water_raster_clip_temporary_folders,
                gsw_raster_clip_temporary_folders=gsw_raster_clip_temporary_folders,
                lake_shape_path=input_lake_shp_path,
                grid_size=128000,
                temporary_vector_folder=current_temporary_vector_folder,
                force_raster_binary_eq_value_masked_my_water=1,
                force_raster_binary_eq_value_gsw=2,
                num_processes=20
            )
            
            current_my_water_concatenated_save_path = my_water_concatenated_save_path.format(basin_id=basin_id)
            current_gsw_concatenated_save_path = gsw_concatenated_save_path.format(basin_id=basin_id)
            
            concatenate_area_csvs(
                csv_paths_to_concatenate=output_my_water_csv_paths,
                concatenated_save_path=current_my_water_concatenated_save_path
            )
            
            concatenate_area_csvs(
                csv_paths_to_concatenate=output_gsw_csv_paths,
                concatenated_save_path=current_gsw_concatenated_save_path
            )