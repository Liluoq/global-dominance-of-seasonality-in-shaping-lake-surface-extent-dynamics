from osgeo import gdal
import os
import numpy as np
import sys
import tensorflow as tf
import json
from multiprocessing import Pool
from typing import List
gdal.UseExceptions()

def parse_predicted_windows_from_tfrecord(tfrecord_path, dtype=tf.float16, band_name='Water', verbose=0, compression_type='GZIP'):
    """
    Parse the predicted windows from a TFRecord file.
    
    Parameters
    ----------
    tfrecord_path : str
        Path to the TFRecord file
    """
    def parse_tfrecord_to_window(example_proto):
        feature_description = {
            band_name: tf.io.FixedLenFeature([], tf.string)
        }
        parsed_features = tf.io.parse_single_example(example_proto, feature_description)
        window = tf.io.parse_tensor(parsed_features[band_name], out_type=dtype)
        return window
    if verbose != 0:
        print(f'Parsing windows from {tfrecord_path}')
    predictions_dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type=compression_type).map(parse_tfrecord_to_window)
    predicted_windows = [prediction.numpy() for prediction in predictions_dataset]
    return predicted_windows

def reconstruct_image(predicted_windows, window_positions, window_size=128, verbose=0):
    """
    Reconstruct the original image from the segmented windows without needing to specify output_shape.
    
    Parameters
    ----------
    predicted_windows : list
        List of predicted windows (a window is a numpy array of shape (window_size, window_size))
    window_positions : list
        List of window positions (a window position is a tuple (xoff, yoff))
    window_size : int
        Size of the windows
    overlap : int
        Overlap between windows
    """
    # Determine the output_shape based on the window positions and window size
    max_x, max_y = max(pos[0] for pos in window_positions), max(pos[1] for pos in window_positions)
    output_shape = (max_y + window_size, max_x + window_size)
    
    reconstructed_image = np.zeros(output_shape)
    count_map = np.zeros(output_shape)  # To keep track of how many times a pixel is covered

    if verbose != 0:
        print(f'Reconstructing image with shape {output_shape} from {len(predicted_windows)} windows')
    for window, (xoff, yoff) in zip(predicted_windows, window_positions):
        window = window.squeeze()  # Remove the channel dimension
        reconstructed_image[yoff:yoff+window_size, xoff:xoff+window_size] += window
        count_map[yoff:yoff+window_size, xoff:xoff+window_size] += 1

    # Avoid division by zero
    count_map[count_map == 0] = 1
    reconstructed_image /= count_map  # Average overlapping regions
    
    return reconstructed_image

def save_reconstructed_image_as_geotiff(reconstructed_image, output_path, srs_wkt, transform, dtype=gdal.GDT_Int16, verbose=0):
    """
    Save the reconstructed image as a GeoTIFF file with the specified spatial reference system
    and transformation metadata.

    :param reconstructed_image: 2D numpy array of the reconstructed image.
    :param output_path: Path where the output GeoTIFF will be saved.
    :param srs_wkt: Spatial reference system from the original GeoTIFF.
    :param transform: GeoTransform from the original GeoTIFF.
    """
    if dtype in [tf.int8, tf.int16]:
        dtype = gdal.GDT_Int16
    # Create a driver for GeoTIFF
    driver = gdal.GetDriverByName('GTiff')
    
    # Create the new file using the dimensions and data type of the reconstructed image
    try:
        out_dataset = driver.Create(output_path, reconstructed_image.shape[1], reconstructed_image.shape[0], 1, dtype, options=['COMPRESS=LZW', 'PREDICTOR=2', 'TILED=YES'])
    except RuntimeError as e:
        print(f'Unable to create {output_path}')
        print(e)
        sys.exit(1)
    
    # Set the SRS and GeoTransform to the output dataset
    out_dataset.SetProjection(srs_wkt)
    out_dataset.SetGeoTransform(transform)
    
    # Write the reconstructed image to the output dataset as the first band
    try:
        out_band = out_dataset.GetRasterBand(1)
    except RuntimeError as e:
        print(f'Unable to get the first band of {output_path}')
        print(e)
        sys.exit(1)
    if verbose != 0:
        print(f'Saving reconstructed image to {output_path}')
    out_band.WriteArray(reconstructed_image)
    
    # Optionally, set the band name if required
    # out_band.SetDescription('Band Name Here')
    
    # Save and close the datasets
    out_band.FlushCache()
    out_dataset = None
    
def reconstruct_tif_from_predicted_tfrecord(
    predicted_tfrecord_path: str,
    output_tif: str,
    predicted_window_size: int,
    predicted_dtype: tf.dtypes.DType,
    predicted_band_name: str,
    window_position_path: str,
    tgt_projection_metadata_path: str,
    verbose: int = 0,
    compression_type='GZIP'
) -> None:
    """
    Reconstruct a GeoTIFF image from predicted windows stored in a TFRecord file.

    Parameters
    ----------
    predicted_tfrecord_path : str
        Path to the TFRecord file containing the predicted windows.
    output_tif : str
        Path to save the reconstructed GeoTIFF image.
    predicted_window_size : int
        Size of the predicted windows.
    predicted_dtype : tf.dtypes.DType
        Data type of the predicted windows.
    predicted_band_name : str
        Name of the band in the predicted windows.
    window_position_path : str
        Path to the JSON file containing the window positions.
    tgt_projection_metadata_path : str
        Path to the JSON file containing the target projection metadata.
    verbose : int, optional
        Verbosity level (0: no messages, 1: print messages), by default 0
    """
    predicted_windows = parse_predicted_windows_from_tfrecord(
        tfrecord_path=predicted_tfrecord_path,
        dtype=predicted_dtype,
        band_name=predicted_band_name,
        verbose=verbose,
        compression_type=compression_type
    )
    
    # Load window positions from JSON
    with open(window_position_path, 'r') as f:
        window_positions = json.load(f)
    
    reconstructed_image = reconstruct_image(
        predicted_windows=predicted_windows,
        window_positions=window_positions,
        window_size=predicted_window_size,
        verbose=verbose
    )
    
    with open(tgt_projection_metadata_path, 'r') as f:
        tgt_projection_metadata = json.load(f)
    tgt_srs_wkt = tgt_projection_metadata['srs_wkt']
    tgt_transform = tgt_projection_metadata['geotransform']
    
    save_reconstructed_image_as_geotiff(
        reconstructed_image=reconstructed_image,
        output_path=output_tif,
        srs_wkt=tgt_srs_wkt,
        transform=tgt_transform,
        dtype=predicted_dtype,
        verbose=verbose
    )
    
def reconstruct_tif_from_predicted_tfrecord_parallel(
    predicted_tfrecord_paths: List[str],
    output_tif_paths: List[str],
    predicted_window_size: int,
    predicted_dtype: tf.dtypes.DType,
    predicted_band_name: str,
    window_position_paths: List[str],
    tgt_projection_metadata_paths: List[str],
    verbose: int = 0,
    compression_type='GZIP',
    num_processes: int = 4
) -> None:
    """
    Reconstruct GeoTIFF images from predicted windows stored in TFRecord files in parallel.

    Parameters
    ----------
    predicted_tfrecord_paths : List[str]
        List of paths to the TFRecord files containing the predicted windows.
    output_tif_paths : List[str]
        List of paths to save the reconstructed GeoTIFF images.
    predicted_window_size : int
        Size of the predicted windows.
    predicted_dtype : tf.dtypes.DType
        Data type of the predicted windows.
    predicted_band_name : str
        Name of the band in the predicted windows.
    window_position_folder : str
        Folder containing the JSON files with the window positions.
    tgt_projection_metadata_folder : str
        Folder containing the JSON files with the target projection metadata.
    verbose : int, optional
        Verbosity level (0: no messages, 1: print messages), by default 0
    num_processes : int, optional
        Number of processes to use for parallel processing, by default 4
    """
    if verbose != 0:
        print(f'Reconstructing {len(predicted_tfrecord_paths)} GeoTIFF images from predicted windows in parallel')
    with Pool(num_processes) as pool:
        pool.starmap(
            reconstruct_tif_from_predicted_tfrecord,
            zip(
                predicted_tfrecord_paths,
                output_tif_paths,
                [predicted_window_size] * len(predicted_tfrecord_paths),
                [predicted_dtype] * len(predicted_tfrecord_paths),
                [predicted_band_name] * len(predicted_tfrecord_paths),
                window_position_paths,
                tgt_projection_metadata_paths,
                [verbose] * len(predicted_tfrecord_paths),
                [compression_type] * len(predicted_tfrecord_paths)
            )
        )