import tensorflow as tf
from typing import Optional
import sys
from typing import List
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import Future

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

def load_dataset(tfrecord_path: str, band_names: list, band_scaling_factors:list, dtype: tf.dtypes.DType, batch_size: Optional[int] = 128, compression_type='GZIP', num_processes=1) -> tf.data.Dataset:
    """
    Load dataset from TFRecord file.

    Args:
        tfrecord_path (str): Path to the TFRecord file.
        band_name (str): Name of the band feature in the TFRecord file.
        dtype (tf.dtypes.DType): Data type of the parsed window tensor.
        batch_size (int, optional): Batch size for the dataset. Defaults to 128.

    Returns:
        tf.data.Dataset: Loaded dataset.

    """
    def _parse_function(example_proto):
        """
        Parse function for parsing TFRecord examples.

        Args:
            example_proto (tf.train.Example): Input example.

        Returns:
            tf.Tensor: Parsed window tensor.

        """
        features = {
            band_name: tf.io.FixedLenFeature([], tf.string) for band_name in band_names  # Define the feature description for parsing
        }
        try:
            # Parse the example
            parsed_features = tf.io.parse_single_example(example_proto, features)

            # Deserialize the bands
            for band_name in band_names:
                parsed_features[band_name] = tf.io.parse_tensor(parsed_features[band_name], out_type=dtype)
            parsed_window = tf.stack([parsed_features[band_name] for band_name in band_names], axis=0)  # Stack the bands to create a window
            parsed_window = tf.transpose(parsed_window, perm=[1, 2, 0])  # Transpose the window to match the shape of the input to the model
            scaling_factors = tf.constant(band_scaling_factors, dtype=tf.float32)
            scaled_parsed_window = tf.cast(parsed_window, tf.float32) * scaling_factors  # Cast the window to the correct data type
            return scaled_parsed_window
        except tf.errors.InvalidArgumentError as e:
            # Handle invalid argument error
            print(f"Invalid argument error: {e}")
            sys.exit(1)

    try:
        dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type=compression_type)  # Create a dataset from the TFRecord file
        if num_processes > 1:
            dataset = dataset.map(_parse_function, num_parallel_calls=num_processes)  # Apply the parse function to each example in the dataset
            dataset = dataset.batch(batch_size, num_parallel_calls=num_processes)  # Batch the dataset
        else:
            dataset = dataset.map(_parse_function)  # Apply the parse function to each example in the dataset
            dataset = dataset.batch(batch_size)  # Batch the dataset
        return dataset
    except tf.errors.NotFoundError as e:
        # Handle file not found error
        print(f"File not found error: {e}")
        sys.exit(1)
    except tf.errors.UnknownError as e:
        # Handle unknown error
        print(f"Unknown error: {e}")
        sys.exit(1)

def predict(model: tf.keras.Model, dataset: tf.data.Dataset, num_processes=1, max_queue_size_per_process=4) -> tf.Tensor:
    """
    Perform predictions using a model on a dataset.

    Args:
        model (tf.keras.Model): The model to use for predictions.
        dataset (tf.data.Dataset): The dataset to perform predictions on.

    Returns:
        tf.Tensor: The predictions made by the model.

    """
    if num_processes > 1:
        predictions = model.predict(dataset, use_multiprocessing=True, workers=num_processes, max_queue_size=num_processes*max_queue_size_per_process)
    else:
        predictions = model.predict(dataset)  # No need for a loop, dataset is already batched
    return predictions

def save_predictions_to_tfrecord(predictions: List[tf.Tensor], output_band_name: str, output_file: str, output_dtype:tf.int8, compression_type='GZIP') -> None:
    """
    Saves the predictions to a TFRecord file.

    Args:
        predictions (List[tf.Tensor]): List of predictions.
        output_band_name (str): Name of the output band in the TFRecord file.
        output_file (str): Output file path.

    Returns:
        None
    """
    
    if compression_type == "GZIP":    
        output_file = output_file + '.gz'
    elif compression_type is None:
        pass
    else:
        raise ValueError("Compression type not supported")
    
    with tf.io.TFRecordWriter(output_file, options=tf.io.TFRecordOptions(compression_type=compression_type)) as writer:
        for prediction in predictions:  # Iterate through each prediction
            # Convert prediction to binary format, if not already
            prediction_binary = tf.cast(prediction > 0.5, output_dtype)
            
            # Ensure prediction is serialized properly
            serialized_prediction = tf.io.serialize_tensor(prediction_binary)
            feature = {
                output_band_name: tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_prediction.numpy()]))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
            
def predict_and_save_to_tfrecord(
    model: tf.keras.Model,
    tfrecord_path: str,
    input_band_names: list,
    input_band_scaling_factors: list,
    input_dtype: tf.dtypes.DType,
    batch_size: int,
    output_band_name: str,
    output_file: str,
    output_dtype: tf.dtypes.DType,
    verbose: int = 0,
    compression_type='GZIP',
    num_processes: int = 1,
    max_queue_size_per_process=4
) -> None:
    """
    Perform predictions using a model on a dataset, save the predictions to a TFRecord file.

    Args:
        model: The model to use for predictions.
        tfrecord_path (str): Path to the TFRecord file.
        input_band_name (str): Name of the band feature in the TFRecord file.
        dtype (tf.dtypes.DType): Data type of the parsed window tensor.
        batch_size (int, optional): Batch size for the dataset. Defaults to 128.
        output_band_name: Name of the output band in the TFRecord file.
        output_file: Output file path.
        verbose (int, optional): Verbosity level. Defaults to 0.

    Returns:
        None

    """
    
    if verbose != 0:
        print(f"Performing predictions and saving results from input: {tfrecord_path}")
    dataset = load_dataset(tfrecord_path, input_band_names, input_band_scaling_factors, input_dtype, batch_size, compression_type=compression_type, num_processes=num_processes)  # Load the dataset
    predictions = predict(model, dataset, num_processes=num_processes, max_queue_size_per_process=max_queue_size_per_process)  # Predict using the model
    save_predictions_to_tfrecord(
        predictions=predictions,
        output_band_name=output_band_name,
        output_file=output_file,
        output_dtype=output_dtype,
        compression_type=compression_type
        )  # Save the predictions to a TFRecord file
    
def predict_and_save_to_tfrecord_async(
        model: tf.keras.Model,
        tfrecord_path: str,
        input_band_names: list,
        input_band_scaling_factors: list,
        input_dtype: tf.dtypes.DType,
        batch_size: int,
        output_band_name: str,
        output_file: str,
        output_dtype: tf.dtypes.DType,
        verbose: int = 0,
        compression_type='GZIP',
        num_processes: int = 1,
        max_queue_size_per_process=4
    ) -> Future:
        if verbose != 0:
            print(f"Performing predictions and saving results from input: {tfrecord_path}")
        dataset = load_dataset(tfrecord_path, input_band_names, input_band_scaling_factors, input_dtype, batch_size, compression_type=compression_type, num_processes=num_processes)
        predictions = predict(model, dataset, num_processes=num_processes, max_queue_size_per_process=max_queue_size_per_process)
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            # Return the Future object
            return executor.submit(
                save_predictions_to_tfrecord,
                predictions=predictions,
                output_band_name=output_band_name,
                output_file=output_file,
                output_dtype=output_dtype,
                compression_type=compression_type
            )
    
    
if __name__ == '__main__':
    import os
    # Set the working directory to the directory of the script
    wkdir = '/WORK/Data/global_lake_area'
    if wkdir == None:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
    else:
        os.chdir(wkdir)
        
    test_model_path = f'../../Codes/global_lake_area/trained_unet/8020000010/cp-0035.ckpt'
    model = tf.keras.models.load_model(test_model_path)
    tfrecord_path = f'./modis_image_tfrecords/test/8020000010_modis_awei_500m_original_projection_2001-02-01_2001-03-01_tile_3072_2048_resampled.tfrecord.gz'
    predicted_tfrecord_path = f'./predicted_tfrecords/test/8020000010_modis_awei_500m_original_projection_2001-02-01_2001-03-01_tile_3072_2048_resampled_predicted.tfrecord.gz'
    predict_and_save_to_tfrecord(
        model=model,
        tfrecord_path=tfrecord_path,
        input_band_name='MODIS_AWEI',
        input_dtype=tf.float16,
        batch_size=1,
        output_band_name='Predicted_water',
        output_file=predicted_tfrecord_path,
        output_dtype=tf.int8,
        verbose=1
    )