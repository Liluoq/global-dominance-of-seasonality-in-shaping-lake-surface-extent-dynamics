from osgeo import gdal, ogr, osr
import os
import glob
import sys
from .reproject_to_target_tile import divide_into_tiles
from multiprocessing import Pool
gdal.UseExceptions()

# Function to calculate output bounds from a raster
def calculate_output_bounds(raster_path):
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
    align_rasters(base_raster_path, to_align_raster_path, aligned_output_path, gdal_resample_alg, output_dtype)
    combine_bands(combine_raster_paths, output_combined_path, output_dtype)
    return None

def align_and_combine_bands_parallel(base_raster_paths, to_align_raster_paths, aligned_output_paths, combine_raster_paths_list, output_combined_paths, gdal_resample_alg=gdal.GRA_Bilinear, output_dtype=gdal.GDT_Int16, parallel_cores=4):
    with Pool(parallel_cores) as p:
        results = p.starmap(align_and_combine_bands, zip(base_raster_paths, to_align_raster_paths, aligned_output_paths, combine_raster_paths_list, output_combined_paths, [gdal_resample_alg]*len(base_raster_paths), [output_dtype]*len(base_raster_paths)))
    return results