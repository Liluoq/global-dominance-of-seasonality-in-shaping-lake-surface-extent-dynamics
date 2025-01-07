from osgeo import gdal, ogr, osr
import os
import glob
import sys
from multiprocessing import Pool
import geopandas as gpd
from shapely.geometry import Polygon
gdal.UseExceptions()

def tile_intersects_land(tile_bounds, boundary_gdf):
    """
    Check if the tile intersects with the land boundary.
    
    Args:
        tile_bounds (tuple): A tuple containing the bounds of the tile in the form (xmin, ymin, xmax, ymax).
        boundary_gdf (geopandas.GeoDataFrame): A GeoDataFrame representing the land boundary. It should have the same spatial reference system (SRS) as the tile.
    
    Returns:
        bool: True if the tile intersects with the land boundary, False otherwise.
    """
    tile_geom = Polygon([
        (tile_bounds[0], tile_bounds[3]),
        (tile_bounds[0], tile_bounds[1]),
        (tile_bounds[2], tile_bounds[1]),
        (tile_bounds[2], tile_bounds[3])
    ])
    return boundary_gdf.intersects(tile_geom).any()

def divide_into_tiles(input_tif, tile_size_x, tile_size_y, output_folder, boundary_path=None, verbose=0, w_buffer=0, h_buffer=0):
    """
    Divides a large GeoTIFF into smaller tiles.
    """
    if verbose not in [0, 1, 2]:
        raise ValueError("verbose must be 0, 1, or 2")
    os.makedirs(output_folder, exist_ok=True)
    input_tif_basename = os.path.basename(input_tif)
    input_tif_basename_noext = os.path.splitext(input_tif_basename)[0]
    ds = gdal.Open(input_tif)
    gt = ds.GetGeoTransform()
    width = ds.RasterXSize
    height = ds.RasterYSize
    print(gt)
    tif_srs = osr.SpatialReference(wkt=ds.GetProjection())
    if boundary_path is not None:
        boundary_gdf = gpd.read_file(boundary_path)
        if boundary_gdf.crs.to_wkt() != tif_srs.ExportToWkt():
            boundary_gdf = boundary_gdf.to_crs(tif_srs.ExportToWkt())
    
    if verbose != 0:
        print(f"Input GeoTIFF size: {width} x {height}, trying to create tiles of size {tile_size_x} x {tile_size_y}")
    for i in range(0, width, tile_size_x):
        for j in range(0, height, tile_size_y):
            w = min(i + tile_size_x, width) - i
            h = min(j + tile_size_y, height) - j
            minx = gt[0] + i * gt[1] + j * gt[2]
            maxx = gt[0] + (i + w) * gt[1] + (j + h) * gt[2]
            miny = gt[3] + i * gt[4] + j * gt[5]
            maxy = gt[3] + (i + w) * gt[4] + (j + h) * gt[5]
            if boundary_path is not None:
                if not tile_intersects_land((minx, miny, maxx, maxy), boundary_gdf):
                    if verbose != 0:
                        print(f"Tile {i}, {j} does not intersect with the land boundary, skipping")
                    continue
            try:
                if verbose == 2:
                    print(f"Creating tile for index {i}, {j}")
                gdal.Translate(
                    os.path.join(output_folder, f"{input_tif_basename_noext}_tile_{i}_{j}.tif"),
                    ds,
                    srcWin=[i, j, w + w_buffer, h + h_buffer],
                    format="GTiff",
                )
            except RuntimeError as e:
                print(f"Failed to create tile for {i}, {j}, with width {w + w_buffer} and height {h + h_buffer}")
                print(e)
                sys.exit(1)

def get_target_srs_resolution(target_sample_path, return_wkt=False):
    """
    Get the target SRS and resolution from a sample GeoTIFF.
    Returns tgt_srs, x_res, y_res
    """
    try:
        target_sample_ds = gdal.Open(target_sample_path)
    except RuntimeError as e:
        print('Unable to open {}'.format(target_sample_path))
        print(e)
        sys.exit(1)

    try:
        tgt_srs = osr.SpatialReference(wkt=target_sample_ds.GetProjection())
        tgt_geotransform = target_sample_ds.GetGeoTransform()
        x_res = tgt_geotransform[1]
        y_res = -tgt_geotransform[5]
    except RuntimeError as e:
        print('Unable to get target projection and geotransform')
        print(e)
        sys.exit(1)
    if return_wkt:
        return tgt_srs.ExportToWkt(), x_res, y_res
    else:
        return tgt_srs, x_res, y_res

def resample_tile(tile, output_folder, tgt_srs_wkt, x_res, y_res, resample_alg=gdal.GRA_Bilinear, output_dtype=gdal.GDT_Int16, verbose=0):
    """

    """
    os.makedirs(output_folder, exist_ok=True)
    tgt_srs = osr.SpatialReference()
    tgt_srs.ImportFromWkt(tgt_srs_wkt)
    output_tile = os.path.join(output_folder, os.path.basename(tile))
    if verbose == 2:
        warp_options = gdal.WarpOptions(format='GTiff', outputType=output_dtype, creationOptions=['COMPRESS=LZW', 'PREDICTOR=2', 'TILED=YES'], 
                                        dstSRS=tgt_srs, xRes=x_res, yRes=y_res, resampleAlg=resample_alg, callback=gdal.TermProgress)
    else:
        warp_options = gdal.WarpOptions(format='GTiff', outputType=output_dtype, creationOptions=['COMPRESS=LZW', 'PREDICTOR=2', 'TILED=YES'], 
                                        dstSRS=tgt_srs, xRes=x_res, yRes=y_res, resampleAlg=resample_alg)
    try:
        if verbose == 2:
            print(f"Resampling tile {tile}")
        ds = gdal.Warp(output_tile, tile, options=warp_options)
        ds = None  # Close and save the output dataset
        os.remove(tile)  # Remove the original tile
    except RuntimeError as e:
        print(f"Failed to resample tile {tile}")
        print(e)
        sys.exit(1)
    return output_tile

def resample_tiles(input_folder, output_folder, tgt_srs, x_res, y_res, resample_alg=gdal.GRA_Bilinear, output_dtype=gdal.GDT_Int16, verbose=0, parallel=False, parallel_cores=4):
    """
    Resamples each tile in the input folder to the target SRS and saves to the output folder.
    """
    if verbose not in [0, 1, 2]:
        raise ValueError("verbose must be 0, 1, or 2")
    tiles = glob.glob(os.path.join(input_folder, "*.tif"))
    os.makedirs(output_folder, exist_ok=True)
    if verbose != 0:
        print(f"Resampling {len(tiles)} tiles from {input_folder} to {output_folder} with target SRS {tgt_srs} and resolution {x_res}, {y_res}")
    tgt_srs_wkt = tgt_srs.ExportToWkt()
    args = [(tile, output_folder, tgt_srs_wkt, x_res, y_res, resample_alg, output_dtype, verbose) for tile in tiles]
    
    if parallel:
        with Pool(parallel_cores) as p:
            p.map(resample_tile, args)
    else:
        for arg in args:
            resample_tile(*arg)

def mosaic_tiles(input_folder, output_tif, input_tile_file_basename=None, remove_tile=True, output_dtype=gdal.GDT_CFloat32, verbose=0, temporary_vrt_prefix=None):
    """
    Mosaics all tiles in the input folder back into a single GeoTIFF.
    """
    if verbose not in [0, 1, 2]:
        raise ValueError("verbose must be 0, 1, or 2")
    try:
        if verbose != 0:
            print(f"Mosaicing tiles from {input_folder} into {output_tif}")
        tiles = glob.glob(os.path.join(input_folder, f"*{input_tile_file_basename}*.tif"))
        if temporary_vrt_prefix is None:
            temp_vrt_path = os.path.join(input_folder, "temporary.vrt")
        else:
            temp_vrt_path = os.path.join(input_folder, f"{temporary_vrt_prefix}_temporary.vrt")
        #vrt_options = gdal.BuildVRTOptions(resampleAlg=gdal.GRA_Bilinear)
        vrt = gdal.BuildVRT(temp_vrt_path, tiles)
        translate_options = gdal.TranslateOptions(format='GTiff',
                                                  outputType=output_dtype,
                                                  creationOptions=[
                                                    'COMPRESS=LZW',  # Use LZW compression
                                                    'PREDICTOR=2',   # Optimal for images with floating point or integer values
                                                    'TILED=YES'      # Organize the file in tiles rather than strips
                                                  ])
        gdal.Translate(output_tif, vrt, options=translate_options)
        # Remove the resampled tiles after successful mosaicing
        if remove_tile:
            for tile in tiles:
                try:
                    os.remove(tile) 
                except Exception as e:
                    print(f"An error occurred while removing the tile {tile}: {e}")
    except Exception as e:
        print(f"An error occurred when mosaicing: {e}")
    finally:
        vrt=None
        if verbose != 0:
            print(f"Removing temporary VRT file {temp_vrt_path}")
        os.remove(temp_vrt_path)  # Clean up temporary VRT file



if __name__ == "__main__":
    # Example usage:
    
    wkdir = '/WORK/SSD_Data/global_lake_area'
    if wkdir == None:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
    else:
        os.chdir(wkdir)
    
    input_geo_tiff = "./modis_images_500m/test_reproject500m/8020000010_modis_awei_500m_2001-08-01_2001-09-01.tif"
    tile_size_x, tile_size_y = 1024, 1024  # Define tile size
    tiles_output_folder = "./modis_images_500m/temperary_tiles"
    resampled_tiles_folder = "./modis_images_500m/resampled_temperary_tiles"
    output_mosaic_tif = "./modis_images_resample_30m/test/8020000010_modis_awei_30m_resampled_2001-08-01_2001-09-01.tif"
    target_sample_path = "./modis_images_resample_30m/test_30m_from_gee/modis_30m/8020000010_modis_awei_30m_2001-02-01_2001-03-01-0000000000-0000000000.tif"
    tgt_srs, x_res, y_res = get_target_srs_resolution(target_sample_path)

    # Step 1: Divide into tiles
    divide_into_tiles(input_geo_tiff, tile_size_x, tile_size_y, tiles_output_folder, verbose=2)

    # Step 2: Resample tiles
    resample_tiles(tiles_output_folder, resampled_tiles_folder, tgt_srs, x_res, y_res, resample_alg=gdal.GRA_NearestNeighbour, verbose=2, parallel=True)

    # Step 3: Mosaic tiles
    #mosaic_tiles(resampled_tiles_folder, output_mosaic_tif, verbose=2)