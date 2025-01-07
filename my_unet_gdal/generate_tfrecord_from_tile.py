from .reproject_to_target_tile import divide_into_tiles, get_target_srs_resolution, resample_tile
from .align_to_target_tile import align_rasters
from osgeo import gdal
import os
import numpy as np
import sys
import tensorflow as tf
import json
import shutil
import glob
from multiprocessing import Pool
gdal.UseExceptions()

def slice_geotiff_with_gdal(input_file, window_size=128, overlap=64, verbose=0):
    """
    Slice the GeoTIFF into windows using GDAL.
    
    Parameters
    ----------
    input_file : str
        Path to the input GeoTIFF file
    """
    try:
        dataset = gdal.Open(input_file)
    except RuntimeError as e:
        print('Unable to open {}'.format(input_file))
        print(e)
        sys.exit(1)
    
    windows = []
    positions = []
    if verbose != 0:
        print(f'Slicing {input_file} into windows of size {window_size} with overlap {overlap}')
    
    band_arrays = []
    for b in range(1, dataset.RasterCount + 1):
        try:
            band = dataset.GetRasterBand(b)
        except RuntimeError as e:
            print('Unable to read the raster band')
            print(e)
            sys.exit(1)
        
        try:
            band_array = band.ReadAsArray()
            band_arrays.append(band_array)
        except RuntimeError as e:
            print(f'Failed to read band array for {input_file}')
            print(e)
            sys.exit(1)
    
    for xoff in range(0, band.XSize - window_size + 1, window_size - overlap):
        if verbose == 2:
            print(f'Creating window for xoff {xoff}')
        for yoff in range(0, band.YSize - window_size + 1, window_size - overlap):
            window = np.stack([band_array[yoff:yoff+window_size, xoff:xoff+window_size] for band_array in band_arrays], axis=-1)
            if window.shape == (window_size, window_size, dataset.RasterCount):  # Ensure window is the correct size
                windows.append(window)
                positions.append((xoff, yoff))
            else:
                print(f'Window for xoff {xoff} and yoff {yoff} has the wrong shape {window.shape}')
                raise ValueError('Window has the wrong shape')
    if verbose == 1:
        print('Size of windows:', len(windows))
    return windows, positions

print('ATTENTION: The order of band names must be the same as the order of the bands in the geotiff files.')
print('Otherwise, the bands will be mixed up (fatal error).')
def windows_to_tfrecord(windows, output_file, band_names:list, verbose=0, compression_type="GZIP", dtype=np.float16):
    """
    Convert windows to TFRecord.
    
    Parameters
    ----------
    windows : list
        List of windows (each window is a numpy array of shape (window_size, window_size, num_bands))
    output_file : str
        Path to the output TFRecord file
    band_names : list
        Must be exactly the same order as the bands in the windows, i.e., in the geotiff files.
        Otherwise, the bands will be mixed up (fatal error).
    """
    if verbose != 0:
        print(f'Converting windows to TFRecord and saving to {output_file}')
    
    tf_options = tf.io.TFRecordOptions(compression_type=compression_type)
    if compression_type == "GZIP":
        output_file = output_file + '.gz'
    elif compression_type is None:
        pass
    else:
        raise ValueError("Compression type not supported")
    with tf.io.TFRecordWriter(output_file, options=tf_options) as writer:
        window_cnt = 0
        for window in windows:
            if verbose == 2:
                print(f'Converting window {window_cnt} to TFRecord {output_file}')
            # Ensure the window is in the correct format (dtype)
            window = window.astype(dtype)
            feature = {}
            for i, band_name in enumerate(band_names):
                data = tf.io.serialize_tensor(window[:,:,i])
                feature[band_name] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[data.numpy()]))
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

def reproject_and_convert_to_tfrecord(tile_path, temp_folder, output_folder, band_names:list, tgt_srs_wkt, x_res, y_res, resample_alg, output_dtype='int16', to_combine_raster=None, window_size=128, 
                                      window_overlap=64, verbose=0, remove_files=True, compression_type="GZIP"):
    """Take a tile (or just a GeoTiff), reproject it, slice it into windows, and convert the windows to TFRecord.

    Args:
        tile_path (str): input tile path
        resample_temp_folder (str): Temporary folder for resampled tile, removed after converting to TFRecord if remove_files is True
        output_folder (str): Output folder for TFRecord and window positions
        tgt_srs (gdal spatial reference system obj): Target spatial reference system for resampling
        x_res (float): Target x resolution for resampling
        y_res (float): Target y resolution for resampling
        resample_alg (gdal resample algorithm): Algorithm for resampling
        verbose (int, optional): Defaults to 0.
    """
    if output_dtype not in ['int16']:
        raise ValueError('output_dtype must be one of [int16]')
    if output_dtype == 'int16':
        output_gdal_dtype = gdal.GDT_Int16
        output_tf_dtype = np.int16
    if verbose != 0:
        print(f'Reprojecting and converting {tile_path} to TFRecord')
    #define output paths
    file_basename = os.path.basename(tile_path)
    file_baseneme_noext = os.path.splitext(file_basename)[0]
    resample_temp_folder = os.path.join(temp_folder, 'resampled')
    if not os.path.exists(resample_temp_folder):
        os.makedirs(resample_temp_folder, exist_ok=True)
    if to_combine_raster == None:
        output_tf_path = os.path.join(output_folder, file_baseneme_noext + '_resampled.tfrecord')
        output_window_pos_path = os.path.join(output_folder, file_baseneme_noext + '_resampled_window_positions.json')
        #resample the tile
        args = (tile_path, resample_temp_folder, tgt_srs_wkt, x_res, y_res, resample_alg, output_gdal_dtype, verbose)
        resampled_tile_path = resample_tile(*args)
        #save resampled metadata (srs_wkt and geotransform) to json file, for later use in reconstructing the prediction image to GeoTiff
        try:
            resampled_tile = gdal.Open(resampled_tile_path)
        except RuntimeError as e:
            print('Unable to open {}'.format(resampled_tile_path))
            print(e)
            sys.exit(1)
        resampled_srs_wkt = resampled_tile.GetProjection()
        resampled_geotransform = resampled_tile.GetGeoTransform()
        resampled_metadata = {
            'srs_wkt': resampled_srs_wkt,
            'geotransform': resampled_geotransform
        }
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)
        with open(os.path.join(output_folder, file_baseneme_noext + '_resampled_proj_metadata.json'), 'w') as f:
            json.dump(resampled_metadata, f)
        resampled_tile = None
        #slice resampled tile into windows and convert to tfrecord
        windows, window_positions = slice_geotiff_with_gdal(resampled_tile_path, window_size=window_size, overlap=window_overlap, verbose=verbose)
    else:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)
        output_tf_path = os.path.join(output_folder, file_baseneme_noext + '_resampled_combined.tfrecord')
        output_window_pos_path = os.path.join(output_folder, file_baseneme_noext + '_resampled_combined_window_positions.json')
        #resample the tile
        args = (tile_path, resample_temp_folder, tgt_srs_wkt, x_res, y_res, resample_alg, output_gdal_dtype, verbose)
        resampled_tile_path = resample_tile(*args)
        #align and combine bands
        aligned_temp_folder = os.path.join(temp_folder, 'aligned')
        if not os.path.exists(aligned_temp_folder):
            os.makedirs(aligned_temp_folder, exist_ok=True)
        aligned_output_path = os.path.join(aligned_temp_folder, file_baseneme_noext + '_resampled_aligned.tif')
        align_rasters(
            base_raster_path=resampled_tile_path,
            to_align_raster_path=to_combine_raster,
            aligned_output_path=aligned_output_path,
            gdal_resample_alg=resample_alg,
            output_dtype=output_gdal_dtype
        )
        try:
            resampled_tile = gdal.Open(resampled_tile_path)
        except RuntimeError as e:
            print('Unable to open {}'.format(resampled_tile_path))
            print(e)
            sys.exit(1)
        resampled_srs_wkt = resampled_tile.GetProjection()
        resampled_geotransform = resampled_tile.GetGeoTransform()
        resampled_metadata = {
            'srs_wkt': resampled_srs_wkt,
            'geotransform': resampled_geotransform
        }
        with open(os.path.join(output_folder, file_baseneme_noext + '_resampled_combined_proj_metadata.json'), 'w') as f:
            json.dump(resampled_metadata, f)
        resampled_tile = None
        #slice resampled tile into windows and convert to tfrecord
        windows_base, window_positions = slice_geotiff_with_gdal(resampled_tile_path, window_size=window_size, overlap=window_overlap, verbose=verbose)
        windows_to_combine, _ = slice_geotiff_with_gdal(aligned_output_path, window_size=window_size, overlap=window_overlap, verbose=verbose)
        assert len(windows_base) == len(windows_to_combine), 'Number of windows in base and to combine rasters must be the same'
        windows = [np.concatenate([windows_base[i], windows_to_combine[i]], axis=-1) for i in range(len(windows_base))]
        
    windows_to_tfrecord(windows, output_tf_path, band_names=band_names, verbose=verbose, compression_type=compression_type, dtype=output_tf_dtype)
    with open(output_window_pos_path, 'w') as f:
        json.dump(window_positions, f)
    #remove resampled tile file after converting to tfrecord
    if remove_files:
        os.remove(resampled_tile_path)
        if to_combine_raster is not None:
            os.remove(aligned_output_path)

def reproject_and_convert_to_tfrecord_by_tile(
    input_tif_path, 
    temporary_tile_folder,
    tile_size_x,
    tile_size_y,
    tile_width_buffer,
    tile_height_buffer,
    output_folder, 
    band_names:list,
    tgt_srs_wkt, 
    x_res, 
    y_res, 
    resample_alg, 
    output_dtype='int16',
    to_combine_raster=None,
    window_size=128, 
    window_overlap=64, 
    verbose=0, 
    num_processes=None,
    remove_files=True,
    compression_type="GZIP",
    boundary_path=None
    ):
    """
    Reproject and convert to TFRecord by tile.

    Args:
        input_tif_path (str): Path to the input TIFF file.
        temporary_tile_folder (str): Path to the temporary tile folder.
        tile_size_x (int): Size of the tiles in the x-direction.
        tile_size_y (int): Size of the tiles in the y-direction.
        tile_width_buffer (int): Width buffer for each tile.
        tile_height_buffer (int): Height buffer for each tile.
        output_folder (str): Path to the output folder.
        band_names (list): List of band names.
        tgt_srs_wkt (str): Well-known text (WKT) representation of the target spatial reference system.
        x_res (float): X resolution for reprojection.
        y_res (float): Y resolution for reprojection.
        resample_alg (str): Resampling algorithm to use.
        output_dtype (str, optional): Output data type. Defaults to 'int16'.
        to_combine_raster (str, optional): Path to the raster to combine with. Defaults to None.
        window_size (int, optional): Size of the sliding window. Defaults to 128.
        window_overlap (int, optional): Overlap of the sliding window. Defaults to 64.
        verbose (int, optional): Verbosity level. Defaults to 0.
        num_processes (int, optional): Number of processes to use for parallel processing. Defaults to None.
        remove_files (bool, optional): Whether to remove temporary files. Defaults to True.
        compression_type (str, optional): Compression type for the TFRecord. Defaults to "GZIP".
    """
    
    input_tif_basename = os.path.basename(input_tif_path)
    input_tif_base_noext = os.path.splitext(input_tif_basename)[0]
    temporary_original_tile_folder = os.path.join(temporary_tile_folder, 'original')
    temporary_resampled_tile_folder = os.path.join(temporary_tile_folder, 'resampled')

    if os.path.exists(temporary_original_tile_folder):
        shutil.rmtree(temporary_original_tile_folder)
    os.makedirs(temporary_original_tile_folder)

    if os.path.exists(temporary_resampled_tile_folder):
        shutil.rmtree(temporary_resampled_tile_folder)
    os.makedirs(temporary_resampled_tile_folder)
    divide_into_tiles(
        input_tif=input_tif_path,
        tile_size_x=tile_size_x,
        tile_size_y=tile_size_y,
        output_folder=temporary_original_tile_folder,
        boundary_path=boundary_path,
        w_buffer=tile_width_buffer,
        h_buffer=tile_height_buffer,
        verbose=verbose
    )
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)
    tile_paths = glob.glob(os.path.join(temporary_original_tile_folder, f'{input_tif_base_noext}_tile_*.tif'))
    if num_processes is None:
        for tile_path in tile_paths:
            reproject_and_convert_to_tfrecord(
                tile_path=tile_path,
                resample_temp_folder=temporary_resampled_tile_folder,
                output_folder=output_folder,
                band_names=band_names,
                tgt_srs_wkt=tgt_srs_wkt,
                x_res=x_res,
                y_res=y_res,
                resample_alg=resample_alg,
                output_dtype=output_dtype,
                to_combine_raster=to_combine_raster,
                window_size=window_size,
                window_overlap=window_overlap,
                verbose=verbose,
                remove_files=remove_files,
                compression_type=compression_type
            )
    else:
        if not isinstance(num_processes, int):
            raise ValueError('num_processes must be an integer')
        if num_processes < 1:
            raise ValueError('num_processes must be greater than 1')
        args = [
            (tile_path, temporary_resampled_tile_folder, output_folder, band_names, tgt_srs_wkt, x_res, y_res, resample_alg, output_dtype, to_combine_raster, window_size, window_overlap, verbose, remove_files, compression_type)
            for tile_path in tile_paths
        ]
        with Pool(num_processes) as p:
            p.starmap(reproject_and_convert_to_tfrecord, args)
    
    if remove_files:
        #remove temporary folders
        shutil.rmtree(temporary_original_tile_folder)
        shutil.rmtree(temporary_resampled_tile_folder)
    

if __name__ == '__main__':
    
    # Set the working directory to the directory of the script
    wkdir = '/WORK/Data/global_lake_area'
    if wkdir == None:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
    else:
        os.chdir(wkdir)
    
    input_tif = './modis_images_500m/test/8020000010_modis_awei_500m_original_projection_2001-02-01_2001-03-01.tif'
    temporary_tile_folder = './temporary_files/temporary_tiles'
    output_folder = './temporary_files/temporary_tfrecord_tiles'
    target_srs_path = './modis_images_resample_30m/test_30m_from_gee/modis_30m/8020000010_modis_awei_30m_2001-02-01_2001-03-01-0000000000-0000000000.tif'
    tgt_srs_wkt, x_res, y_res = get_target_srs_resolution(target_srs_path, return_wkt=True)
    #os.remove('/Codes/modis_images_tfrecord/temp/original/8020000010_modis_awei_500m_original_projection_2001-02-01_2001-03-01_tile_4096_3072.tif')
    
    reproject_and_convert_to_tfrecord_by_tile(
        input_tif_path=input_tif,
        temporary_tile_folder=temporary_tile_folder,
        tile_size_x=1024,
        tile_size_y=1024,
        tile_width_buffer=16,
        tile_height_buffer=16,
        output_folder=output_folder,
        tgt_srs_wkt=tgt_srs_wkt,
        x_res=x_res,
        y_res=y_res,
        resample_alg=gdal.GRA_Bilinear,
        window_size=128,
        window_overlap=64,
        window_band_name='MODIS_AWEI',
        verbose=1,
        num_processes=10,
        remove_files=False
    )
    #os.rmdir(temporary_tile_folder)