import generate_tfrecord_from_tile
import unet_prediction
import reconstruct_tile_from_prediction
from reproject_to_target_tile import mosaic_tiles
from reproject_to_target_tile import get_target_srs_resolution
import os
import sys
import glob
import tensorflow as tf
from osgeo import gdal
# Path: Codes/my_unet_gdal/combined.py

def from_input_tif_to_predicted_tif(
    model: tf.keras.Model,
    input_tif: str,
    tile_size_x: int,
    tile_size_y: int,
    tile_width_buffer: int,
    tile_height_buffer: int,
    tgt_srs_wkt: str,
    x_res: float,
    y_res: float,
    temporary_tiles_folder: str,
    temporary_tfrecord_tiles_folder: str,
    temporary_predicted_tiles_folder: str,
    temporary_reconstructed_tiles_folder: str,
    output_tif: str,
    input_band_name: str,
    input_dtype: tf.dtypes.DType,
    window_size: int,
    window_overlap: int,
    output_band_name: str,
    output_dtype: tf.dtypes.DType,
    batch_size: int,
    num_processes: int = None,
    verbose: int = 0,
    reprojection_resample_alg= gdal.GRA_Bilinear,
    remove_temporary_files: bool = True # this parameter won't be passed to the functions called in this function, temporary files generated in this function will be deleted after the function is finished if this parameter is set True
):
    """
    Converts an input TIFF file to a predicted TIFF file using a trained model.

    Args:
        model (tf.keras.Model): The trained model to use for prediction.
        input_tif (str): The path to the input TIFF file.
        tile_size_x (int): The width of each tile in pixels.
        tile_size_y (int): The height of each tile in pixels.
        tile_width_buffer (int): The buffer size to add to the width of each tile.
        tile_height_buffer (int): The buffer size to add to the height of each tile.
        tgt_srs_wkt (str): The well-known text (WKT) representation of the target spatial reference system (SRS).
        x_res (float): The desired X resolution of the output tiles in the target SRS.
        y_res (float): The desired Y resolution of the output tiles in the target SRS.
        temporary_tiles_folder (str): The path to the folder where temporary tiles will be stored.
        temporary_tfrecord_tiles_folder (str): The path to the folder where temporary TFRecord tiles will be stored.
        temporary_predicted_tiles_folder (str): The path to the folder where temporary predicted tiles will be stored.
        temporary_reconstructed_tiles_folder (str): The path to the folder where temporary reconstructed tiles will be stored.
        output_tif (str): The path to the output TIFF file.
        input_band_name (str): The name of the input band in the TIFF file.
        input_dtype (tf.dtypes.DType): The data type of the input band.
        window_size (int): The size of the sliding window for prediction.
        window_overlap (int): The overlap between adjacent windows.
        output_band_name (str): The name of the output band in the predicted TIFF file.
        output_dtype (tf.dtypes.DType): The data type of the output band.
        batch_size (int): The batch size for prediction.
        verbose (int, optional): The verbosity level. Defaults to 0.
        reprojection_resample_alg (int, optional): The resampling algorithm to use for reprojection. Defaults to gdal.GRA_Bilinear.
        remove_temporary_files (bool, optional): Whether to remove temporary files after the function is finished. Defaults to True.
    """
    if verbose != 0:
        print(f"Reprojecting and converting input TIFF to TFRecord tiles...: {input_tif}")
    generate_tfrecord_from_tile.reproject_and_convert_to_tfrecord_by_tile(
        input_tif_path=input_tif,
        temporary_tile_folder=temporary_tiles_folder,
        tile_size_x=tile_size_x,
        tile_size_y=tile_size_y,
        tile_width_buffer=tile_width_buffer,
        tile_height_buffer=tile_height_buffer,
        output_folder=temporary_tfrecord_tiles_folder,
        tgt_srs_wkt=tgt_srs_wkt,
        x_res=x_res,
        y_res=y_res,
        resample_alg=reprojection_resample_alg,
        window_size=window_size,
        window_overlap=window_overlap,
        window_band_name=input_band_name,
        verbose=verbose,
        num_processes=num_processes,
        remove_files=False
    )
    
    
    input_base_noext = os.path.splitext(os.path.basename(input_tif))[0]
    if verbose != 0:
        print(f"input_base_noext: {input_base_noext}")
    tfrecord_paths = glob.glob(os.path.join(temporary_tfrecord_tiles_folder, f'{input_base_noext}_tile_*_resampled.tfrecord'))
    if verbose != 0:
        print(f"tfrecord_path example: {tfrecord_paths[0]}")
    predicted_tfrecord_paths = [os.path.join(temporary_predicted_tiles_folder, f'{os.path.splitext(os.path.basename(tfrecord_path))[0]}_predicted.tfrecord') for tfrecord_path in tfrecord_paths]
    if verbose != 0:
        print(f"predicted_tfrecord_path example: {predicted_tfrecord_paths[0]}")
    
    if verbose != 0:
        print(f"Predicting and saving to TFRecord...: {input_tif}")
    for tfrecord_path, predicted_tfrecord_path in zip(tfrecord_paths, predicted_tfrecord_paths):
        unet_prediction.predict_and_save_to_tfrecord(
            model=model,
            tfrecord_path=tfrecord_path,
            input_band_name=input_band_name,
            input_dtype=input_dtype,
            batch_size=batch_size,
            output_band_name=output_band_name,
            output_tfrecord_path=predicted_tfrecord_path,
            output_dtype=output_dtype,
            verbose=verbose
        )
        
    reconstructed_tif_paths = [os.path.join(temporary_reconstructed_tiles_folder, f'{os.path.splitext(os.path.basename(predicted_tfrecord_path))[0]}_reconstructed.tif') for predicted_tfrecord_path in predicted_tfrecord_paths]
    if verbose != 0:
        print(f"reconstructed_tif_path example: {reconstructed_tif_paths[0]}")
    if verbose != 0:
        print(f"Reconstructing tiles...: {input_tif}")
    for tfrecord_path, predicted_tfrecord_path, reconstructed_tif_path in zip(tfrecord_paths, predicted_tfrecord_paths, reconstructed_tif_paths):
        window_position_path = os.path.join(temporary_tfrecord_tiles_folder, f'{os.path.splitext(os.path.basename(tfrecord_path))[0]}_window_positions.csv')
        resample_metadata_path = os.path.join(temporary_tfrecord_tiles_folder, f'{os.path.splitext(os.path.basename(tfrecord_path))[0]}_proj_metadata.csv')
        reconstruct_tile_from_prediction.reconstruct_tif_from_predicted_tfrecord(
            predicted_tfrecord_path=predicted_tfrecord_path,
            output_tif=reconstructed_tif_path,
            predicted_window_size=window_size,
            predicted_window_overlap=window_overlap,
            predicted_band_name=output_band_name,
            predicted_dtype=output_dtype,
            window_position_path=window_position_path,
            tgt_projection_metadata_path=resample_metadata_path,
            verbose=verbose
        )
    
    if verbose != 0:
        print(f"Mosaicking tiles...: {input_tif}")
    mosaic_tiles(
        input_folder=temporary_reconstructed_tiles_folder,
        output_tif=output_tif,
        input_tile_file_basename=f'{input_base_noext}*_reconstructed.tif',
        verbose=verbose
    )
    if verbose != 0:
        print(f"Finished processing {input_tif} to {output_tif}")
    
if __name__ == '__main__':
    # Set the working directory to the directory of the script
    wkdir = '/WORK/Data/global_lake_area'
    if wkdir == None:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
    else:
        os.chdir(wkdir)
    
    # Load the trained model
    model_path = '../../Codes/global_lake_area/trained_unet/8020000010/cp-0035.ckpt'
    model = tf.keras.models.load_model(model_path)
    
    # Define the paths to the input and output files
    input_tif = './modis_images_500m/test/8020000010_modis_awei_500m_original_projection_2001-02-01_2001-03-01.tif'
    output_tif = '/predicted_tif/8020000010_modis_awei_30m_predicted_2001-02-01_2001-03-01.tif'

    target_srs_path = './modis_images_resample_30m/test_30m_from_gee/modis_30m/8020000010_modis_awei_30m_2001-02-01_2001-03-01-0000000000-0000000000.tif'
    # Define the parameters for the conversion
    tile_size_x = 1024
    tile_size_y = 1024
    tile_width_buffer = 16
    tile_height_buffer = 16
    
    tgt_srs_wkt, x_res, y_res = get_target_srs_resolution(target_srs_path, return_wkt=True)
    print(f"tgt_srs_wkt: {tgt_srs_wkt}")
    temporary_tiles_folder = './temporary_files/temporary_tiles'
    temporary_tfrecord_tiles_folder = './temporary_files/temporary_tfrecord_tiles'
    temporary_predicted_tiles_folder = './temporary_files/temporary_predicted_tiles'
    temporary_reconstructed_tiles_folder = './temporary_files/temporary_reconstructed_tiles'
    input_band_name = 'MODIS_AWEI'
    input_dtype = tf.float16
    window_size = 128
    window_overlap = 0
    output_band_name = 'Predicted_Water'
    output_dtype = tf.int8
    batch_size = 128
    verbose = 1

    # Call the function to convert the input TIFF to a predicted TIFF
    from_input_tif_to_predicted_tif(
        model=model,
        input_tif=input_tif,
        tile_size_x=tile_size_x,
        tile_size_y=tile_size_y,
        tile_width_buffer=tile_width_buffer,
        tile_height_buffer=tile_height_buffer,
        tgt_srs_wkt=tgt_srs_wkt,
        x_res=x_res,
        y_res=y_res,
        temporary_tiles_folder=temporary_tiles_folder,
        temporary_tfrecord_tiles_folder=temporary_tfrecord_tiles_folder,
        temporary_predicted_tiles_folder=temporary_predicted_tiles_folder,
        temporary_reconstructed_tiles_folder=temporary_reconstructed_tiles_folder,
        output_tif=output_tif,
        input_band_name=input_band_name,
        input_dtype=input_dtype,
        window_size=window_size,
        window_overlap=window_overlap,
        output_band_name=output_band_name,
        output_dtype=output_dtype,
        batch_size=batch_size,
        num_processes=4,
        verbose=verbose
    )