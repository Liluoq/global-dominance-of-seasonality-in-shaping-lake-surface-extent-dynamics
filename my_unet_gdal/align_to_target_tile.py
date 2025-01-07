from osgeo import gdal, ogr, osr
import os
import glob
import sys
from .reproject_to_target_tile import divide_into_tiles
from multiprocessing import Pool
gdal.UseExceptions()

# Function to calculate output bounds from a raster
def calculate_output_bounds(raster_path):
    """Calculate the geographic bounds of a raster file.

    This function opens a raster file and calculates its geographic bounds (minX, minY, maxX, maxY)
    based on its geotransform and dimensions.

    Args:
        raster_path (str): Path to the input raster file.

    Returns:
        tuple: A tuple containing (minX, minY, maxX, maxY) coordinates defining the raster bounds.

    Raises:
        RuntimeError: If unable to open the raster file.
    """
    try:
        raster = gdal.Open(raster_path)
    except RuntimeError as e:
        print('Unable to open {}'.format(raster_path))
        print(e)
        sys.exit(1)
    gt = raster.GetGeoTransform()
    
    minX = gt[0]
    maxY = gt[3]
    maxX = minX + gt[1] * raster.RasterXSize
    minY = maxY + gt[5] * raster.RasterYSize
    
    return (minX, minY, maxX, maxY)

# Function to align one raster to the bounds of another
def align_rasters(base_raster_path, to_align_raster_path, aligned_output_path, gdal_resample_alg=gdal.GRA_Bilinear, output_dtype=gdal.GDT_Int16):
    """Align one raster to match the bounds and resolution of a base raster.

    This function warps (resamples) a raster to match the spatial properties of a base raster,
    including its extent, resolution, and projection.

    Args:
        base_raster_path (str): Path to the reference raster that defines the target alignment.
        to_align_raster_path (str): Path to the raster that needs to be aligned.
        aligned_output_path (str): Path where the aligned raster will be saved.
        gdal_resample_alg (int, optional): GDAL resampling algorithm to use. Defaults to gdal.GRA_Bilinear.
        output_dtype (int, optional): GDAL data type for output raster. Defaults to gdal.GDT_Int16.

    Raises:
        RuntimeError: If unable to open input rasters or perform the alignment operation.
    """
    output_bounds = calculate_output_bounds(base_raster_path)
    
    # Open the base raster to get the resolution
    try:
        base_raster = gdal.Open(base_raster_path)
    except RuntimeError as e:
        print('Unable to open {}'.format(base_raster_path))
        print(e)
        sys.exit(1)
    geotransform = base_raster.GetGeoTransform()
    x_res = geotransform[1]
    y_res = -geotransform[5]
    
    # Perform the alignment
    try:
        warp_options = gdal.WarpOptions(xRes=x_res,
                        yRes=y_res,
                        resampleAlg=gdal_resample_alg,
                        format='GTiff',
                        outputBounds=output_bounds,
                        dstSRS=base_raster.GetProjection(),
                        outputType=output_dtype,
                        creationOptions=[
                                        'COMPRESS=LZW',  # Use LZW compression
                                        'PREDICTOR=2',   # Optimal for images with floating point or integer values
                                        'TILED=YES'      # Organize the file in tiles rather than strips
                                        ]
                        )
        gdal.Warp(aligned_output_path, to_align_raster_path, options=warp_options)
    except RuntimeError as e:
        print('Failed to align {}'.format(to_align_raster_path))
        print(e)
        sys.exit(1)
    
# Function to combine the bands of the aligned rasters into one GeoTIFF
def combine_bands(raster_paths, output_combined_path, output_dtype=gdal.GDT_Int16):
    """
    Combines multiple single-band rasters into a multi-band GeoTIFF file.

    Args:
        raster_paths (list): List of paths to the input rasters to be combined.
        output_combined_path (str): Path where the combined multi-band raster will be saved.
        output_dtype (int, optional): GDAL data type for output raster. Defaults to gdal.GDT_Int16.

    Raises:
        RuntimeError: If unable to create VRT or convert to GeoTIFF.
    """
    # Create a VRT that stacks the bands
    vrt_options = gdal.BuildVRTOptions(separate=True)
    try:
        vrt = gdal.BuildVRT('/tmp/combined.vrt', raster_paths, options=vrt_options)
    except RuntimeError as e:
        print('Failed to create VRT')
        print(e)
        sys.exit(1)
    
    # Convert the VRT to a GeoTIFF
    try:
        translate_options = gdal.TranslateOptions(format='GTiff',
                                                  outputType=output_dtype,
                                                  creationOptions=[
                                                    'COMPRESS=LZW',  # Use LZW compression
                                                    'PREDICTOR=2',   # Optimal for images with floating point or integer values
                                                    'TILED=YES'      # Organize the file in tiles rather than strips
                                                  ])
        gdal.Translate(output_combined_path, vrt, options=translate_options)
    except RuntimeError as e:
        print('Failed to convert VRT to GeoTIFF')
        print(e)
        sys.exit(1)
    
    # Cleanup
    vrt = None
    
def align_and_combine_bands(base_raster_path, to_align_raster_path, aligned_output_path, combine_raster_paths, output_combined_path, gdal_resample_alg=gdal.GRA_Bilinear, output_dtype=gdal.GDT_Int16):
    """
    Aligns one raster to a base raster and combines multiple bands into a single GeoTIFF.

    Args:
        base_raster_path (str): Path to the base raster that other rasters will be aligned to.
        to_align_raster_path (str): Path to the raster that needs to be aligned.
        aligned_output_path (str): Path where the aligned raster will be saved.
        combine_raster_paths (list): List of paths to rasters that will be combined into bands.
        output_combined_path (str): Path where the final multi-band GeoTIFF will be saved.
        gdal_resample_alg (int, optional): GDAL resampling algorithm. Defaults to gdal.GRA_Bilinear.
        output_dtype (int, optional): GDAL data type for output raster. Defaults to gdal.GDT_Int16.

    Returns:
        None
    """
    align_rasters(base_raster_path, to_align_raster_path, aligned_output_path, gdal_resample_alg, output_dtype)
    combine_bands(combine_raster_paths, output_combined_path, output_dtype)
    return None

def align_and_combine_bands_parallel(base_raster_paths, to_align_raster_paths, aligned_output_paths, combine_raster_paths_list, output_combined_paths, gdal_resample_alg=gdal.GRA_Bilinear, output_dtype=gdal.GDT_Int16, parallel_cores=4):
    """
    Aligns and combines bands for multiple raster sets in parallel.

    Args:
        base_raster_paths (list): List of paths to base rasters that others will be aligned to.
        to_align_raster_paths (list): List of paths to rasters that need to be aligned.
        aligned_output_paths (list): List of paths where aligned rasters will be saved.
        combine_raster_paths_list (list): List of lists, where each inner list contains paths to rasters to combine.
        output_combined_paths (list): List of paths where final multi-band GeoTIFFs will be saved.
        gdal_resample_alg (int, optional): GDAL resampling algorithm. Defaults to gdal.GRA_Bilinear.
        output_dtype (int, optional): GDAL data type for output rasters. Defaults to gdal.GDT_Int16.
        parallel_cores (int, optional): Number of CPU cores to use for parallel processing. Defaults to 4.

    Returns:
        list: Results from parallel processing of align_and_combine_bands operations.
    """
    with Pool(parallel_cores) as p:
        results = p.starmap(align_and_combine_bands, zip(base_raster_paths, to_align_raster_paths, aligned_output_paths, combine_raster_paths_list, output_combined_paths, [gdal_resample_alg]*len(base_raster_paths), [output_dtype]*len(base_raster_paths)))
    return results