import selfee as se
import json
import tensorflow as tf
import geemap
import ee
import re
import time
import gzip
import numpy as np
import os
import random
import shutil
import datetime
import urllib3
import socket
import functools
from tqdm.keras import TqdmCallback
from dateutil.relativedelta import relativedelta
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import models 
from tensorflow.keras import metrics
from tensorflow.keras import optimizers
from tensorflow.keras.layers import BatchNormalization
import matplotlib.pyplot as plt
import my_unet_definition.model as my_unet
import my_unet_definition.evaluation_metrics as my_metrics

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

def read_samples(training_pattern, eval_pattern, local_sample_folder, input_bands, input_band_scaling_factors:list, response, kernel_size, batch_size, only_sizes=False, compression_type=None):
    kernel_shape = [kernel_size, kernel_size]
    features = input_bands + response
    columns = [
    tf.io.FixedLenFeature(shape=kernel_shape, dtype=tf.float32) for k in features
    ]
    features_dict = dict(zip(features, columns))
    
    def parse_tfrecord(example_proto):
        return tf.io.parse_single_example(example_proto, features_dict)
    
    def to_tuple(inputs):
        inputs_list = [inputs.get(key) for key in features]
        stacked = tf.stack(inputs_list, axis=0)
        # Convert from CHW to HWC
        stacked = tf.transpose(stacked, [1, 2, 0])
        # Scale the input bands
        scaling_factors = tf.constant(input_band_scaling_factors, dtype=tf.float32)
        scaled = tf.cast(stacked[:,:,:len(input_bands)], tf.float32) * scaling_factors
        return scaled, stacked[:,:,len(input_bands):]
    
    def predicate(inputs):
        inputs_list = [inputs.get(key) for key in features]
        stacked = tf.stack(inputs_list, axis = 0)
        stacked = tf.transpose(stacked, [1, 2, 0])
        output = stacked[:,:,len(input_bands):]
        return tf.math.reduce_sum(output) < 99999    
    
    def get_dataset_size():
        file_list = [s for s in os.popen(f'ls {local_sample_folder}').read().split('\n') if s]
        training_files = [local_sample_folder + s for s in file_list if (training_pattern in s) and ('tfrecord' in s)]
        dataset = tf.data.TFRecordDataset(training_files, compression_type=compression_type)
        dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.filter(predicate)
        c = 0
        for element in dataset.as_numpy_iterator():
            c += 1
        training_size = c

        file_list = [s for s in os.popen(f'ls {local_sample_folder}').read().split('\n') if s]
        eval_files = [local_sample_folder + s for s in file_list if (eval_pattern in s) and ('tfrecord' in s)]
        dataset = tf.data.TFRecordDataset(eval_files, compression_type=compression_type)
        dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.filter(predicate)
        c = 0
        for element in dataset.as_numpy_iterator():
            c += 1
        eval_size = c
        print(f'training size: {training_size}, eval_size: {eval_size}')
        return training_size, eval_size
    
    def get_dataset(pattern):
        file_list = [s for s in os.popen(f'ls {local_sample_folder}').read().split('\n') if s]
        files = [local_sample_folder + s for s in file_list if (pattern in s) and ('tfrecord' in s)]
        dataset = tf.data.TFRecordDataset(files, compression_type=compression_type).prefetch(tf.data.AUTOTUNE)
        dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
        if 'filtered' in local_sample_folder:
            pass
        else:
            dataset = dataset.filter(predicate)
        dataset = dataset.map(to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset
    
    def get_training_dataset(shuffle_buffer_size):
        dataset = get_dataset(training_pattern)
        dataset = dataset.shuffle(shuffle_buffer_size).batch(batch_size).repeat()
        return dataset
    
    def get_eval_dataset():
        dataset = get_dataset(eval_pattern)
        dataset = dataset.batch(1).repeat()
        return dataset
    training_size, eval_size = get_dataset_size()
    print(f'training size: {training_size}, eval_size: {eval_size}')
    shuffle_buffer_size = int(training_size/30)
    if only_sizes:
        return training_size, eval_size
    else:
        return get_training_dataset(shuffle_buffer_size), get_eval_dataset(), training_size, eval_size

def read_test_samples(test_pattern, local_sample_folder, input_bands, input_band_scaling_factors, response, kernel_size, compression_type=None):
    kernel_shape = [kernel_size, kernel_size]
    features = input_bands + response
    columns = [
    tf.io.FixedLenFeature(shape=kernel_shape, dtype=tf.float32) for k in features
    ]
    features_dict = dict(zip(features, columns))
    
    def parse_tfrecord(example_proto):
        return tf.io.parse_single_example(example_proto, features_dict)
    
    def to_tuple(inputs):
        inputs_list = [inputs.get(key) for key in features]
        stacked = tf.stack(inputs_list, axis=0)
        # Convert from CHW to HWC
        stacked = tf.transpose(stacked, [1, 2, 0])
        # Scale the input bands
        scaling_factors = tf.constant(input_band_scaling_factors, dtype=tf.float32)
        scaled = tf.cast(stacked[:,:,:len(input_bands)], tf.float32) * scaling_factors
        return scaled, stacked[:,:,len(input_bands):]
    
    def predicate(inputs):
        inputs_list = [inputs.get(key) for key in features]
        stacked = tf.stack(inputs_list, axis = 0)
        stacked = tf.transpose(stacked, [1, 2, 0])
        output = stacked[:,:,len(input_bands):]
        return tf.math.reduce_sum(output) < 99999    
    
    def get_dataset(pattern):
        file_list = [s for s in os.popen(f'ls {local_sample_folder}').read().split('\n') if s]
        files = [local_sample_folder + s for s in file_list if (pattern in s) and ('tfrecord' in s)]
        dataset = tf.data.TFRecordDataset(files, compression_type=compression_type).prefetch(tf.data.AUTOTUNE)
        dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.filter(predicate)
        dataset = dataset.map(to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset
    
    def get_test_dataset():
        dataset = get_dataset(test_pattern)
        dataset = dataset.batch(1)
        return dataset
    return get_test_dataset()

def draw_unet_train_figure(history, save_path):
    def moving_average(points, window_size=0):
        smoothed_points = []
        for point in points:
            if len(smoothed_points) < window_size:
                smoothed_points.append(point)
            else:
                smoothed_points.append(1/(window_size+1)*(point + np.sum([smoothed_points[-(i+1)] for i in range(window_size)])))
        return smoothed_points
    acc = moving_average(history['binary_accuracy'])
    val_acc = moving_average(history['val_binary_accuracy'])
    loss = moving_average(history['loss'])
    val_loss = moving_average(history['val_loss'])
    iou_coef = moving_average(history['IoU_coef'])
    val_iou_coef = moving_average(history['val_IoU_coef'])

    epochs = range(1, len(acc) + 1)
    plt.figure(figsize=(20, 10))
    plt.subplot(3,1,1)
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)
    plt.xticks(np.arange(0, len(epochs)+1, 25))  # Change here
    plt.yticks(np.arange(0, 1.05, 0.05))
    plt.gca().xaxis.grid(True, which='major', linestyle='--', linewidth=0.5)
    plt.gca().yaxis.grid(True, which='major', linestyle='--', linewidth=0.5)
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(5))  # Change here

    plt.subplot(3,1,2)
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)
    plt.xticks(np.arange(0, len(epochs)+1, 25))  # Change here
    plt.yticks(np.arange(0, max(max(loss), max(val_loss))+0.05, 0.05))
    plt.gca().xaxis.grid(True, which='major', linestyle='--', linewidth=0.5)
    plt.gca().yaxis.grid(True, which='major', linestyle='--', linewidth=0.5)
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(5))  # Change here

    plt.subplot(3,1,3)
    plt.plot(epochs, iou_coef, 'r', label='Training IoU')
    plt.plot(epochs, val_iou_coef, 'b', label='Validation IoU')
    plt.title('Training and validation IoU')
    plt.legend()
    plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)
    plt.xticks(np.arange(0, len(epochs)+1, 25))  # Change here
    plt.yticks(np.arange(0, 1.05, 0.05))
    plt.gca().xaxis.grid(True, which='major', linestyle='--', linewidth=0.5)
    plt.gca().yaxis.grid(True, which='major', linestyle='--', linewidth=0.5)
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(5))  # Change here

    plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close()
    return None

def unet_train(training_pattern, eval_pattern, local_sample_folder, input_bands, input_band_scaling_factors:list, response,
               kernel_size, epochs, batch_size, save_models_and_figures_folder, keep_ratio=1):
    
    def deserialize_example(serialized_example, features_dict):
        """
        Parse the serialized data so we get a dict with our data.
        """
        return tf.io.parse_single_example(serialized_example, features_dict)

    def adapted_predicate(serialized_example, features_dict, input_bands):
        """
        Deserialize and apply the filtering logic to each example.
        """
        example = deserialize_example(serialized_example, features_dict)
        inputs_list = [example[key] for key in features_dict.keys()]
        
        stacked = tf.stack(inputs_list, axis=0)
        stacked = tf.transpose(stacked, [1, 2, 0])
        output = stacked[:, :, len(input_bands):]
        return tf.math.reduce_sum(output) < 99999

    def filter_and_save_dataset(original_pattern, compressed_sample_folder, destination_folder, features_dict, input_bands, keep_ratio=1):
        """
        Filters out samples based on the predicate and saves the result as uncompressed TFRecord files.
        """
        file_list = [s for s in os.popen(f'ls {compressed_sample_folder}').read().split('\n') if s]
        file_list = [compressed_sample_folder + s for s in file_list if (original_pattern in s) and ('tfrecord' in s)]
        for file_path in file_list:
            raw_dataset = tf.data.TFRecordDataset(file_path, compression_type='GZIP')
            filtered_dataset = raw_dataset.filter(
                lambda x: adapted_predicate(x, features_dict, input_bands) and random.random() < keep_ratio
            )
            
            # Construct the output file path
            file_name = os.path.basename(file_path)
            output_path = os.path.join(destination_folder, file_name.replace(".gz", ""))
            
            # Serialize and save the filtered dataset
            with tf.io.TFRecordWriter(output_path) as writer:
                for raw_record in filtered_dataset:
                    writer.write(raw_record.numpy())
    
    kernel_shape = [kernel_size, kernel_size]
    features = input_bands + response
    columns = [
    tf.io.FixedLenFeature(shape=kernel_shape, dtype=tf.float32) for k in features
    ]
    features_dict = dict(zip(features, columns))
    
    local_uncompressed_sample_folder = local_sample_folder + 'uncompressed_filtered/'
    if not os.path.exists(local_uncompressed_sample_folder):
        os.makedirs(local_uncompressed_sample_folder, exist_ok=True)
    print(f'Filtering and saving training dataset to {local_uncompressed_sample_folder}')
    filter_and_save_dataset(training_pattern, local_sample_folder, local_uncompressed_sample_folder, features_dict, input_bands, keep_ratio)
    print(f'Filtering and saving evaluation dataset to {local_uncompressed_sample_folder}')
    filter_and_save_dataset(eval_pattern, local_sample_folder, local_uncompressed_sample_folder, features_dict, input_bands, keep_ratio)
    
    training, evaluation, training_size, eval_size = read_samples(
        training_pattern=training_pattern, eval_pattern=eval_pattern, local_sample_folder=local_uncompressed_sample_folder,
        input_bands=input_bands, input_band_scaling_factors=input_band_scaling_factors, response=response, kernel_size=kernel_size, batch_size=batch_size,
        compression_type=None
    )
    
    if not os.path.exists(save_models_and_figures_folder):
        os.makedirs(save_models_and_figures_folder)
    m = my_unet.attentionunet(input_shape=(kernel_size, kernel_size, len(input_bands)))
    m.compile(
        optimizer=optimizers.Adam(),
        loss=my_metrics.IoU_loss,
        metrics=[metrics.binary_accuracy, my_metrics.IoU_coef]
    )
    checkpoint_path = save_models_and_figures_folder + "cp-{epoch:04d}.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    period=5
    )
    
    tqdm_callback = TqdmCallback()
    
    history = m.fit(
                    x=training, 
                    epochs=epochs, 
                    steps_per_epoch=int(training_size / batch_size), 
                    validation_data=evaluation,
                    validation_steps=eval_size,
                    callbacks=[cp_callback, tqdm_callback],
                    verbose = 0,
                    workers = 8,
                    use_multiprocessing=True,
                    max_queue_size = 25,
                    )
    history = history.history
    with open(save_models_and_figures_folder + 'history.txt', 'w+') as file:
        file.write(json.dumps(history))
    if os.path.exists(local_uncompressed_sample_folder):
        print('Removing uncompressed samples')
        shutil.rmtree(local_uncompressed_sample_folder)
    def moving_average(points, window_size=0):
        smoothed_points = []
        for point in points:
            if len(smoothed_points) < window_size:
                smoothed_points.append(point)
            else:
                smoothed_points.append(1/(window_size+1)*(point + np.sum([smoothed_points[-(i+1)] for i in range(window_size)])))
        return smoothed_points
    acc = moving_average(history['binary_accuracy'])
    val_acc = moving_average(history['val_binary_accuracy'])
    loss = moving_average(history['loss'])
    val_loss = moving_average(history['val_loss'])
    iou_coef = moving_average(history['IoU_coef'])
    val_iou_coef = moving_average(history['val_IoU_coef'])

    epochs = range(1, len(acc) + 1)
    plt.figure(figsize=(20, 10))
    plt.subplot(3,1,1)
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.subplot(3,1,2)
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.subplot(3,1,3)
    plt.plot(epochs, iou_coef, 'r', label='Training IoU')
    plt.plot(epochs, val_iou_coef, 'b', label='Validation IoU')
    plt.title('Training and validation IoU')
    plt.legend()
    plt.savefig(save_models_and_figures_folder + 'loss_and_accuracy.pdf', format="pdf", bbox_inches="tight")
    
    return m
    
def unet_train_per_feature(collection_size, local_sample_folder, save_models_and_figures_folder, 
                           training_pattern='training', eval_pattern='eval', input_bands=['AWEI_MODIS'], response=['Water_GSW'],
                           kernel_size=128, epochs=150, batch_size=32, start_feature_index=0, end_feature_index=int(1e10)):
    local_sample_folders = [local_sample_folder + str(i) + '/' for i in range(collection_size)]
    save_models_and_figures_folders = [save_models_and_figures_folder + str(i) + '/' for i in range(collection_size)]
    while start_feature_index < end_feature_index and start_feature_index < collection_size:
        print(f'Training UNet in Feature {start_feature_index}')
        lsf = local_sample_folders[start_feature_index]
        smaff = save_models_and_figures_folders[start_feature_index]
        unet_train(training_pattern=training_pattern, eval_pattern=eval_pattern,
                   local_sample_folder=lsf, input_bands=input_bands, response=response, kernel_size=kernel_size,
                   epochs=epochs, batch_size=batch_size, save_models_and_figures_folder=smaff)
        start_feature_index += 1
    return None

def unet_evaluate_on_test(model_path, test_pattern, test_sample_folder, input_bands=['AWEI_MODIS'], response=['Water_GSW'], kernel_size=128):
    test = read_test_samples(test_pattern, test_sample_folder, input_bands=input_bands, response=response, kernel_size=kernel_size)
    m = tf.keras.models.load_model(model_path)
    loss, accuracy = m.evaluate(test)
    return loss, accuracy

#new version
def do_prediction(m, input_bands:list, output_band:str, input_image_base, output_image_name, local_input_folder, local_output_folder, bucket, gcs_folder, gee_folder, kernel_size, sa_index, 
                  compress=True, download=False, upload=False):
    """Perform inference on exported imagery, upload to Earth Engine.
    """
    sa_json = se.KEYPATHS[sa_index]
    if not os.path.exists(local_input_folder):
        os.mkdir(local_input_folder)
    if not os.path.exists(local_output_folder):
        os.mkdir(local_output_folder)
    kernel_shape = [kernel_size, kernel_size]
    kernel_buffer = [int(x*0.5) for x in kernel_shape]
    os.system(f'gcloud auth activate-service-account --key-file {sa_json}')
    if(download == True):
        print('Looking for TFRecord files...')
        
        # Get a list of all the files in the output bucket.
        filesList = os.popen(f'gsutil ls gs://{bucket}/{gcs_folder}').read()
        filesList = [f for f in filesList.split('\n') if f]
        # Get only the files generated by the image export.
        exportFilesList = [s for s in filesList if (input_image_base in s and ('tfrecord' in s or 'json' in s))]
        for file in exportFilesList:
            os.system(f'gsutil cp {file} {local_input_folder}')
            if(file.endswith('json')):
                cloud_json_file = file
    exportFilesList = os.popen(f'ls {local_input_folder}').read()
    exportFilesList = [local_input_folder + s for s in exportFilesList.split('\n') if(s and input_image_base in s and ('tfrecord' in s or 'json' in s))]
    # Get the list of image files and the JSON mixer file.
    imageFilesList = []
    jsonFile = None
    for f in exportFilesList:
        if f.endswith('.tfrecord.gz'):
            imageFilesList.append(f)
        elif f.endswith('.json'):
            jsonFile = f

    # Make sure the files are in the right order.
    imageFilesList.sort()

    from pprint import pprint
    #pprint(imageFilesList)
    #print(jsonFile)

    import json
    # Load the contents of the mixer file to a JSON object.
    with open(jsonFile) as f:
        mixer = json.load(f)
    #pprint(mixer)
    patches = mixer['totalPatches']

    # Get set up for prediction.
    x_buffer = int(kernel_buffer[0] / 2)
    y_buffer = int(kernel_buffer[1] / 2)

    buffered_shape = [
        kernel_shape[0] + kernel_buffer[0],
        kernel_shape[1] + kernel_buffer[1]]

    imageColumns = [
        tf.io.FixedLenFeature(shape=buffered_shape, dtype=tf.float32)
        for k in input_bands
    ]

    imageFeaturesDict = dict(zip(input_bands, imageColumns))

    def parse_image(example_proto):
        return tf.io.parse_single_example(example_proto, imageFeaturesDict)

    def toTupleImage(inputs):
        inputsList = [inputs.get(key) for key in input_bands]
        stacked = tf.stack(inputsList, axis=0)
        stacked = tf.transpose(stacked, [1, 2, 0])
        return stacked

    # Create a dataset from the TFRecord file(s) in Cloud Storage.
    imageDataset = tf.data.TFRecordDataset(imageFilesList, compression_type='GZIP')
    imageDataset = imageDataset.map(parse_image, num_parallel_calls=12)
    imageDataset = imageDataset.map(toTupleImage)
    imageDataset = imageDataset.batch(1)
    

    # Perform inference.
    print('Running predictions...')
    predictions = m.predict(imageDataset, steps=patches, verbose=1)
    for i in range(len(predictions)):
        predictions[i][predictions[i] >= 0.5] = 1
        predictions[i][predictions[i] < 0.5] = 0
    print('Writing predictions...')


    out_image_file = local_output_folder + output_image_name + '.tfrecord'

    writer = tf.io.TFRecordWriter(out_image_file)
    patches = 0
    print(f'Writing {out_image_file}')
    for predictionPatch in predictions:
        predictionPatch = predictionPatch.astype(np.uint8)
        predictionPatch = predictionPatch[
            x_buffer:x_buffer+kernel_size, y_buffer:y_buffer+kernel_size]
        # Create an example.
            
        example = tf.train.Example(
        features=tf.train.Features(
            feature={
            output_band: tf.train.Feature(
                float_list=tf.train.FloatList(
                    value=predictionPatch.flatten()))
            }
        )
        )
        # Write the example.
        writer.write(example.SerializeToString())
        patches += 1

    writer.close()
    
    if compress == True:
        print(f'compressing {out_image_file}')
        compressed_out_image_file = f'{out_image_file}.gz'
        with open(out_image_file, 'rb') as f_in:
            with gzip.open(compressed_out_image_file, 'wb') as f_out:
                f_out.writelines(f_in)
        os.remove(out_image_file)
    
    predicted_json_file = local_output_folder + output_image_name + 'mixer.json'
    os.system(f'cp {jsonFile} {predicted_json_file}')
    
    upload_task_id = None
    if upload == True:
    # Start the upload.
        gee_image_asset = gee_folder + output_image_name
        cloud_out_image_file = 'gs://' + bucket + gcs_folder + output_image_name + '.TFRecord'
        os.system(f'gsutil cp {out_image_file} {cloud_out_image_file}')
        os.system(f'earthengine --service_account_file {sa_json} upload image --asset_id={gee_image_asset} {cloud_out_image_file} {cloud_json_file}')
        se.auth_gee_service_account(sa_index)
        upload_task_id = ee.data.getTaskList()[0]['id']
    return upload_task_id

def do_prediction_modis_awei(model, input_image_base, output_image_name, local_input_folder, local_output_folder, bucket, gcs_folder, gee_folder, sa_index = None, 
                 compress=True, download=False, upload=False):
    input_bands = ['AWEI_MODIS']
    output_band = 'Predicted_Water'
    kernel_size = 128
    upload_task_id = do_prediction(
        m=model,
        input_bands=input_bands, output_band=output_band,
        input_image_base=input_image_base, output_image_name=output_image_name,
        local_input_folder=local_input_folder, local_output_folder=local_output_folder,
        bucket=bucket, gcs_folder=gcs_folder, gee_folder=gee_folder,
        kernel_size=kernel_size,
        sa_index=sa_index,
        compress=compress,
        download=download,
        upload=upload
    )
    return upload_task_id

def do_prediction_modis_awei_monthly(model_path, start_date:str, end_date:str, input_image_base, output_image_base, local_input_folder, local_output_folder, bucket, gcs_folder, gee_folder, sa_index=None, compress=True, download=False, upload=False):
    date_format = '%Y-%m-%d'
    start_date = datetime.datetime.strptime(start_date, date_format).date()
    end_date = datetime.datetime.strptime(end_date, date_format).date()
    current_date = start_date
    upload_task_ids = []
    model = tf.keras.models.load_model(model_path)
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    while current_date < end_date:
        current_input_image_base = input_image_base + f'{current_date.strftime(date_format)}_{(current_date + relativedelta(months=1)).strftime(date_format)}'
        current_output_image_base = output_image_base + f'{current_date.strftime(date_format)}_{(current_date + relativedelta(months=1)).strftime(date_format)}'
        current_upload_task_id = do_prediction_modis_awei(
            model=model,
            input_image_base=current_input_image_base,
            output_image_name=current_output_image_base,
            local_input_folder=local_input_folder,
            local_output_folder=local_output_folder,
            bucket=bucket, gcs_folder=gcs_folder, gee_folder=gee_folder,
            sa_index=sa_index, compress=True, download=download, upload=upload
        )
        upload_task_ids.append(current_upload_task_id)
        
        current_date = current_date + relativedelta(months=1)

def upload_tfrecord_gz_to_gee(file_name_base, local_folder, local_task_info_folder, bucket, gcs_folder, gee_asset, gee_file_name, sa_index:int, auth=False, test=False):
    if(auth):
        se.auth_gcs_service_account(sa_index)
    if not os.path.exists(local_task_info_folder):
        os.makedirs(local_task_info_folder)
    compressed_image_file_name = f'{file_name_base}.tfrecord.gz'
    decompressed_image_file_name = f'{file_name_base}.tfrecord'
    json_file_name = f'{file_name_base}mixer.json'
    local_compressed_image_path = f'{local_folder}{compressed_image_file_name}'
    local_decompressed_image_path = f'{local_folder}{decompressed_image_file_name}'
    local_json_path = f'{local_folder}{json_file_name}'
    gcs_decompressed_image_path = f'gs://{bucket}/{gcs_folder}{decompressed_image_file_name}'
    gcs_json_path = f'gs://{bucket}/{gcs_folder}{json_file_name}'
    
    print(f'decompressing {compressed_image_file_name} locally')
    with gzip.open(local_compressed_image_path, 'rb') as gzip_file:
        local_compressed_image_content = gzip_file.read()
    with open(local_decompressed_image_path, 'wb') as decompressed_file:
        decompressed_file.write(local_compressed_image_content)
    
    success_flag = False
    while not success_flag:
        try:
            print(f'uploading {compressed_image_file_name} and {json_file_name} from local to gcs')
            os.system(f'gsutil cp -J {local_decompressed_image_path} {gcs_decompressed_image_path}')
            os.system(f'gsutil cp {local_json_path} {gcs_json_path}')
            print(f'removing {local_decompressed_image_path}')
            os.system(f'rm {local_decompressed_image_path}')
            
            gee_image_path = f'{gee_asset}{gee_file_name}'
            sa_json = se.KEYPATHS[sa_index]
            print(f'uploading {decompressed_image_file_name} and {json_file_name} from gcs to gee')
            if not test:
                task_id = os.popen(f'earthengine --service_account_file {sa_json} upload image --pyramiding_policy MODE --asset_id {gee_image_path} {gcs_decompressed_image_path} {gcs_json_path}').read()
                task_id = task_id.split(' ')[-1].replace('\n', '')
            else:
                task_id = 'test-no-task-id'
            success_flag = True
        except socket.gaierror:
            print('socket error, retrying')
            time.sleep(random.randint(60, 180))
        except urllib3.exceptions.NewConnectionError:
            print('connection error, retrying')
            time.sleep(random.randint(60, 180))
        except urllib3.exceptions.MaxRetryError:
            print('max retry error, retrying')
            time.sleep(random.randint(60, 180))
    
    local_task_info_path = f'{local_task_info_folder}{file_name_base}info.json'
    task_info_json = {
        'local_compressed_image_path': local_compressed_image_path,
        'local_json_path': local_json_path,
        'gcs_decompressed_image_path': gcs_decompressed_image_path,
        'gcs_json_path': gcs_json_path,
        'gee_image_path': gee_image_path,
        'upload_task_id': task_id,
        'sa_index': sa_index
    }
    with open(local_task_info_path, 'w') as f:
        json.dump(task_info_json, f)
    return local_task_info_path

def upload_monthly_tfrecord_gz_to_gee(start_date, end_date, file_name_base, local_folder, local_task_info_folder, bucket, gcs_folder, gee_asset, gee_file_name_base, sa_index:int, auth=False, test=False):
    date_format = '%Y-%m-%d'
    start_date = datetime.datetime.strptime(start_date, date_format).date()
    end_date = datetime.datetime.strptime(end_date, date_format).date()
    current_date = start_date
    task_count = 0
    completed_task = 0
    local_task_info_paths = []
    while current_date < end_date:
        current_file_name_base = file_name_base + current_date.strftime(date_format) + '_' + (current_date + relativedelta(months=1)).strftime(date_format)
        current_gee_file_name = gee_file_name_base + current_date.strftime(date_format) + '_' + (current_date + relativedelta(months=1)).strftime(date_format)
        print(f'uploading {current_file_name_base} to gee')
        current_local_task_info_path = upload_tfrecord_gz_to_gee(
            file_name_base=current_file_name_base,
            local_folder=local_folder,
            local_task_info_folder=local_task_info_folder,
            bucket=bucket,
            gcs_folder=gcs_folder,
            gee_asset=gee_asset,
            gee_file_name=current_gee_file_name,
            sa_index=sa_index,
            auth=auth, test=test
        )
        local_task_info_paths.append(current_local_task_info_path)
        task_count += 1
        current_date = current_date + relativedelta(months=1)
        if(task_count % 5 == 0):
            with open(local_task_info_paths[task_count - 4], 'r') as f:
                ahead_5th_task_info = json.load(f)
            ahead_5th_task_id = ahead_5th_task_info['upload_task_id']
            # possible states: READY, RUNNING, FAILED, COMPLETED
            ahead_5th_task_state = ee.data.getTaskStatus(ahead_5th_task_id)[0]['state']
            while(ahead_5th_task_state == 'READY'):
                print(f'Waiting: {ahead_5th_task_id}({task_count - 4}) now is still {ahead_5th_task_state}')
                time.sleep(60)
                ahead_5th_task_state = ee.data.getTaskStatus(ahead_5th_task_id)[0]['state']
        if(completed_task < task_count):
            with open(local_task_info_paths[completed_task], 'r') as f:
                completed_task_info = json.load(f)
            completed_task_id = completed_task_info['upload_task_id']
            completed_task_state = ee.data.getTaskStatus(completed_task_id)[0]['state']
            while((completed_task_state == 'FAILED' or completed_task_state == 'COMPLETED')):
                completed_gcs_decompressed_image_path = completed_task_info['gcs_decompressed_image_path']
                completed_gcs_json_path = completed_task_info['gcs_json_path']
                print(f'{completed_task_id}({completed_task}) is {completed_task_state}, try removing related files in gcs')
                print(f'removing {completed_gcs_decompressed_image_path} {completed_gcs_json_path}')
                os.system(f'gsutil rm {completed_gcs_decompressed_image_path}')
                os.system(f'gsutil rm {completed_gcs_json_path}')
                completed_task += 1
                print(f'***********************{task_count} {len(local_task_info_paths)}')
                if not completed_task < task_count:
                    break
                with open(local_task_info_paths[completed_task], 'r') as f:
                    completed_task_info = json.load(f)
                completed_task_id = completed_task_info['upload_task_id']
                completed_task_state = ee.data.getTaskStatus(completed_task_id)[0]['state']
    while completed_task < task_count:
        with open(local_task_info_paths[completed_task], 'r') as f:
            completed_task_info = json.load(f)
        completed_task_id = completed_task_info['upload_task_id']
        completed_task_state = ee.data.getTaskStatus(completed_task_id)[0]['state']
        if(completed_task_state == 'FAILED' or completed_task_state == 'COMPLETED'):
            completed_gcs_decompressed_image_path = completed_task_info['gcs_decompressed_image_path']
            completed_gcs_json_path = completed_task_info['gcs_json_path']
            print(f'{completed_task_id}({completed_task}) is {completed_task_state}, try removing related files in gcs')
            print(f'removing {completed_gcs_decompressed_image_path} {completed_gcs_json_path}')
            os.system(f'gsutil rm {completed_gcs_decompressed_image_path}')
            os.system(f'gsutil rm {completed_gcs_json_path}')
            completed_task += 1
        else:
            print(f'Waiting: {completed_task_id} is still {completed_task_state}')
            time.sleep(60)
    return None

def upload_single_local_to_gcs(local_folder, file_name, bucket, gcs_folder, sa_index=None):
    local_path = local_folder + file_name
    gcs_path = "gs://" + bucket + '/' + gcs_folder + file_name
    if(sa_index):
        se.auth_gcs_service_account(sa_index)
    print(f"Uploading {local_path} to {gcs_path}")
    os.system(f'gsutil cp {local_path} {gcs_path}')
    return None

def upload_multiple_local_to_gcs(local_foler, file_base, file_format, bucket, gcs_folder, sa_index=None):
    if(sa_index):
        se.auth_gcs_service_account(sa_index)
    filenames = [s for s in os.popen(f'ls {local_foler}').read().split('\n') if (file_base in s and s and s.endswith(file_format))]
    for file_name in filenames:
        upload_single_local_to_gcs(local_folder=local_foler, file_name=file_name, bucket=bucket, gcs_folder=gcs_folder)
    return None

def upload_single_tif_gcs_to_gee(bucket, gcs_folder, gcs_file_name, gee_asset, gee_file_name, sa_index=0):
    gcs_path = "gs://" + bucket + "/" + gcs_folder + gcs_file_name
    gee_asset_id = gee_asset + gee_file_name
    sa_json = se.KEYPATHS[sa_index]
    print(f'earthengine --service_account_file {sa_json} upload image --asset_id={gee_asset_id} {gcs_path}')
    os.system(f'earthengine --service_account_file {sa_json} upload image --asset_id={gee_asset_id} {gcs_path}')
    
def upload_multiple_gcs_to_gee(bucket, gcs_folder, file_base, gee_asset, file_format, sa_index=0):
    se.auth_gcs_service_account(sa_index)
    if file_format == 'tif':
        files_list = [s for s in os.popen(f'gsutil ls gs://{bucket}/{gcs_folder}').read().split('\n') if (s and file_base in s and s.endswith(file_format))]
        filename_list = [s.split('/')[-1] for s in files_list]
        filename_underscore_list = [re.sub('\.', '_', s) for s in filename_list]
        filename_underscore_list = [re.sub('_tif', '', s) for s in filename_underscore_list]
        for file_name, file_name_underscore in zip(filename_list, filename_underscore_list):
            upload_single_tif_gcs_to_gee(
                bucket=bucket,
                gcs_folder=gcs_folder,
                gcs_file_name=file_name,
                gee_asset=gee_asset,
                gee_file_name=file_name_underscore            
            )
    return None

def image_generation(start_time, end_time, kernel_size, mode, input_bands:list, input_bands_type='int16', response=['Water_GSW'], custom_projection_wkt=None, custom_scale=500, return_mask=False, test_display=False, gsw_max_start_date='2021-12-01', gsw_max_end_date='2022-01-01'):
    
    if mode not in ['modis_500m_gsw_projection', 'modis_500m_original_projection', 'modis_custom_projection', 'modis_30m_gsw_projection', 'joint_neighborhood', 'joint_neighborhood_custom_projection', 'gsw', 'gsw_custom_projection']:
        raise ValueError('mode must be one of the following: modis_500m_gsw_projection, modis_500m_original_projection, modis_custom_projection, modis_30m_gsw_projection, joint_neighborhood, joint_neighborhood_custom_projection, gsw, gsw_custom_projection')
    
    if mode in ['modis_custom_projection', 'joint_neighborhood_custom_projection'] and (custom_projection_wkt == None or custom_scale == None):
        raise ValueError('custom_projection_wkt and custom_scale must be set for joint_neighborhood and joint_neighborhood_custom_projection')
    
    if not set(input_bands).issubset(["B", "G", "R", "NIR", "SWIR1", "SWIR2", "AWEI", "SensorZenith", "SolarZenith", 'GSW_Occurrence', 'GSW_Recurrence']):
        raise ValueError('input_bands must be a subset of ["B", "G", "R", "NIR", "SWIR1", "SWIR2", "AWEI", "SensorZenith", "SolarZenith", "GSW_Occurrence", "GSW_Recurrence"]')
    
    if input_bands_type not in ['int16']:
        raise ValueError('input_bands_type must be one of the following: int16')
    
    modis_bands = ["sur_refl_b03", "sur_refl_b04", "sur_refl_b01", "sur_refl_b02", "sur_refl_b06", "sur_refl_b07", "SensorZenith", "SolarZenith"]
    modis_bands_new = ["B_MODIS", "G_MODIS", "R_MODIS", "NIR_MODIS", "SWIR1_MODIS", "SWIR2_MODIS", "SensorZenith", "SolarZenith"]
    common_bands = ["B", "G", "R", "NIR", "SWIR1", "SWIR2", "SensorZenith", "SolarZenith"]
    
    list = ee.List.repeat(1, kernel_size)
    lists = ee.List.repeat(list, kernel_size)
    kernel = ee.Kernel.fixed(kernel_size, kernel_size, lists)
    
    custom_projection = ee.Projection(custom_projection_wkt)
    
    def bitwiseExtract(value, fromBit, toBit = None):
        if (toBit == None): toBit = fromBit
        maskSize = ee.Number(1).add(toBit).subtract(fromBit)
        mask = ee.Number(1).leftShift(maskSize).subtract(1)
        return value.rightShift(fromBit).bitwiseAnd(mask)
    
    img_filter = ee.Filter([ee.Filter.date(start_time, end_time)])
    if start_time <= gsw_max_start_date:
        gsw_img_filter = img_filter
        print('using regular filter for gsw, start date:', start_time)
    else:
        gsw_img_filter = ee.Filter([ee.Filter.date(gsw_max_start_date, gsw_max_end_date)]) 
        print('using max filter for gsw, start date:', gsw_max_start_date)    
    
    gsw = ee.ImageCollection('JRC/GSW1_4/MonthlyHistory').filter(gsw_img_filter).first()
    if mode in ['joint_neighborhood_custom_projection', 'gsw_custom_projection']:
        gsw = gsw.reproject(custom_projection, scale=custom_scale)
        if mode == 'gsw_custom_projection':
            return gsw.rename(response), None
    gsw = gsw.updateMask(gsw).subtract(1).unmask(999999).rename(response)
    if mode == 'gsw':
        return gsw, None
    
    #These two bands in GEE is signed-int8 type - Attention
    if 'GSW_Occurrence' in input_bands:
        gsw_occurrence = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select('occurrence')
        if mode in ['joint_neighborhood_custom_projection', 'modis_custom_projection']:
            gsw_occurrence = gsw_occurrence.reproject(custom_projection, scale=custom_scale)
        gsw_occurrence = gsw_occurrence.unmask(0).rename('GSW_Occurrence')
    
    if 'GSW_Recurrence' in input_bands:
        gsw_recurrence = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select('recurrence')
        if mode in ['joint_neighborhood_custom_projection', 'modis_custom_projection']:
            gsw_recurrence = gsw_recurrence.reproject(custom_projection, scale=custom_scale)
        gsw_recurrence = gsw_recurrence.unmask(0).rename('GSW_Recurrence')
    
    def modisPreprocess(img):
        qa = img.select("state_1km")
        cloud_state = bitwiseExtract(qa, 0, 1)
        cloudshadow_state = bitwiseExtract(qa, 2)
        cirrus_state = bitwiseExtract(qa, 8, 9)
        mask = cloud_state.eq(0)\
                .And(cloudshadow_state.eq(0))\
                .And(cirrus_state.eq(0))
        if mode == 'modis_500m_gsw_projection':
            img = img.select(modis_bands, modis_bands_new).updateMask(mask).reproject(gsw.projection(), scale=500)
        elif mode == 'modis_500m_original_projection':
            img = img.select(modis_bands, modis_bands_new).updateMask(mask)
        elif mode == 'modis_custom_projection':
            img = img.select(modis_bands, modis_bands_new).updateMask(mask).reproject(custom_projection, scale=custom_scale)
        elif mode == 'modis_30m_gsw_projection':
            img = img.select(modis_bands, modis_bands_new).updateMask(mask).reproject(gsw.projection(), scale=30)
        elif mode == 'joint_neighborhood':
            img = img.select(modis_bands, modis_bands_new).updateMask(mask).reproject(gsw.projection(), scale=30)
        elif mode == 'joint_neighborhood_custom_projection':
            img = img.select(modis_bands, modis_bands_new).updateMask(mask).reproject(custom_projection, scale=custom_scale)
        return img
    
    modis = ee.ImageCollection("MODIS/061/MOD09GA")\
          .filter(img_filter)\
          .map(modisPreprocess)
          
    
    modis_median_30m = modis.median().select(modis_bands_new, common_bands)
    if mode in ['modis_custom_projection', 'joint_neighborhood_custom_projection']:
        modis_median_30m = modis_median_30m.reproject(custom_projection, scale=custom_scale)
    
    if return_mask:
        #currently assume that the mask is the same for all bands
        return modis_median_30m.select('B').mask()
    
    modis_median_30m = modis_median_30m.unmask(0)
    
    if 'AWEI' in input_bands:
        def AWEI(img):
            band2 = img.select("G")
            band4 = img.select("NIR")
            band5 = img.select("SWIR1")
            band7 = img.select("SWIR2")
            return band2.subtract(band5).multiply(4).subtract( band4.multiply(0.25).add( band7.multiply(2.75) ) )
        modis_median_awei_30m = AWEI(modis_median_30m).rename('AWEI') #unmask(0) to avoid situations when MODIS has too much missing values
        modis_median_30m = modis_median_30m.addBands(modis_median_awei_30m)
    if 'GSW_Occurrence' in input_bands:
        modis_median_30m = modis_median_30m.addBands(gsw_occurrence)
    if 'GSW_Recurrence' in input_bands:
        modis_median_30m = modis_median_30m.addBands(gsw_recurrence)
    modis_median_30m = modis_median_30m.select(input_bands)
    if input_bands_type == 'int16':
        modis_median_30m = modis_median_30m.int16()
    if mode != 'joint_neighborhood' and mode != 'joint_neighborhood_custom_projection':
        return modis_median_30m, None
    
    joined_median_30m = gsw.addBands(modis_median_30m)

    neighborhood_joined_30m = joined_median_30m.neighborhoodToArray(kernel)
    Map = geemap.Map()
    if(test_display):
        modis_visualization = {
            'min': -0.0100,
            'max': 0.5000,
            'bands': ['NIR_MODIS', 'R_MODIS', 'G_MODIS']
        }
        awei_visualization = {
            'min': -0.5,
            'max': 0.0
        }
        gsw_raw_visualization = {
            'bands': ['water'],
            'min': 0.0,
            'max': 2.0,
            'palette': ['ffffff', 'fffcb8', '0905ff']
        }

        Map.addLayer(ee.ImageCollection('JRC/GSW1_4/MonthlyHistory').filter(gsw_img_filter).first(), gsw_raw_visualization, 'gsw_raw')
        Map.addLayer(gsw, {'palette': ['black', 'red']}, 'gsw')
        Map.addLayer(modis_median_30m, modis_visualization, 'modis_30m')
        Map.addLayer(modis_median_awei_30m, awei_visualization, 'modis_awei_30m')
    
    if mode == 'joint_neighborhood' or mode == 'joint_neighborhood_custom_projection':
        return neighborhood_joined_30m, Map
     
def export_modis_mask_to_asset(start_date, end_date, gee_asset, gee_file_name, region, kernel_size=128):
    modis_median_awei_30m_mask, Map = image_generation(start_time=start_date, end_time=end_date, kernel_size=kernel_size, mode='modis_awei_mask')
    task = ee.batch.Export.image.toAsset(
                image = modis_median_awei_30m_mask,
                description = gee_file_name,
                assetId = gee_asset+gee_file_name,
                region = region,
                scale = 30,
                maxPixels = 1e13
            )
    print(f"Exporting {gee_file_name} to {gee_asset}")
    task.start()
    return None

def unet_sample_generation_time_period(region, start_time, end_time, generate_samples, train_size, eval_size, test_size, shard_size, 
                                    training_base, eval_base, test_base, starting_seed, export_to, input_bands:list, response=['Water_GSW'],
                                    train_shard_num=20, eval_shard_num=20, test_shard_num=20,
                                    custom_projection_wkt=None, custom_scale=None, mode='joint_neighborhood', 
                                    local_task_id_folder=None, local_task_id_base=None, drive_folder=None, bucket=None, gcs_folder=None, kernel_size=128, scale=30):
    
    valid_export_to = ['drive', 'gcs']
    if export_to not in valid_export_to:
        raise ValueError(f'export_to must be one of {valid_export_to}')
    if export_to == 'drive' and not drive_folder:
        raise ValueError('drive_folder must be specified if export_to is drive')
    if export_to == 'gcs' and not (bucket and gcs_folder):
        raise ValueError('bucket and gcs_folder must be specified if export_to is gcs')
    
    test_display = not generate_samples
    neighborhood_joined_awei_30m, Map = image_generation(
        start_time=start_time,
        end_time=end_time,
        kernel_size=kernel_size,
        mode=mode,
        input_bands=input_bands,
        response=response,
        custom_projection_wkt=custom_projection_wkt,
        custom_scale=custom_scale,
        test_display=test_display
    )

    train_shards = int(train_size/shard_size)
    eval_shards = int(eval_size/shard_size)
    test_shards = int(test_size/shard_size)
    
    training_tasks = {}
    eval_tasks = {}
    test_tasks = {}
    samples = ee.FeatureCollection([])
    for i in range(int(train_shards)):
        sample = neighborhood_joined_awei_30m.sample(**{
            'numPixels': shard_size,
            'scale': scale,
            'region': region,
            'seed': i+starting_seed,
            'tileScale': 16
        })
        samples = samples.merge(sample)
        if((i+1) % train_shard_num == 0):
            g = int((i+1)/train_shard_num)
            desc = training_base + "_" + str(g)
            if export_to == 'drive':
                task = ee.batch.Export.table.toDrive(**{
                    'collection': samples,
                    'description': desc,
                    'folder': drive_folder,
                    'fileNamePrefix': desc,
                    'fileFormat': 'TFRecord',
                })
            elif export_to == 'gcs':
                task = ee.batch.Export.table.toCloudStorage(**{
                    'collection': samples,
                    'description': desc,
                    'bucket': bucket,
                    'fileNamePrefix': gcs_folder + desc,
                    'fileFormat': 'TFRecord',
                })
            if(generate_samples):
                task.start()
                training_tasks[desc] = task.id
            samples = ee.FeatureCollection([])

    samples = ee.FeatureCollection([])
    for i in range(int(eval_shards)):
        sample = neighborhood_joined_awei_30m.sample(**{
            'numPixels': shard_size,
            'scale': scale,
            'region': region,
            'seed': i+int(train_shards)+starting_seed,
            'tileScale': 16
        })
        samples = samples.merge(sample)
        if((i+1) % eval_shard_num == 0):
            g = int((i+1)/eval_shard_num)
            desc = eval_base + "_" + str(g)
            if export_to == 'drive':
                task = ee.batch.Export.table.toDrive(**{
                    'collection': samples,
                    'description': desc,
                    'folder': drive_folder,
                    'fileNamePrefix': desc,
                    'fileFormat': 'TFRecord',
                })
            elif export_to == 'gcs':
                task = ee.batch.Export.table.toCloudStorage(**{
                    'collection': samples,
                    'description': desc,
                    'bucket': bucket,
                    'fileNamePrefix': gcs_folder + desc,
                    'fileFormat': 'TFRecord',
                })
            if(generate_samples):
                task.start()
                eval_tasks[desc] = task.id
            samples = ee.FeatureCollection([])
            
    samples = ee.FeatureCollection([])
    for i in range(int(test_shards)):
        sample = neighborhood_joined_awei_30m.sample(**{
            'numPixels': shard_size,
            'scale': scale,
            'region': region,
            'seed': i+int(train_shards)+int(eval_shards)+starting_seed,
            'tileScale': 16
        })
        samples = samples.merge(sample)
        if((i+1) % test_shard_num == 0):
            g = int((i+1)/test_shard_num)
            desc = test_base + "_" + str(g)
            if export_to == 'drive':
                task = ee.batch.Export.table.toDrive(**{
                    'collection': samples,
                    'description': desc,
                    'folder': drive_folder,
                    'fileNamePrefix': desc,
                    'fileFormat': 'TFRecord',
                })
            elif export_to == 'gcs':
                task = ee.batch.Export.table.toCloudStorage(**{
                    'collection': samples,
                    'description': desc,
                    'bucket': bucket,
                    'fileNamePrefix': gcs_folder + desc,
                    'fileFormat': 'TFRecord',
                })
            if(generate_samples):
                task.start()
                test_tasks[desc] = task.id
            samples = ee.FeatureCollection([])
            
    if test_display:
        return Map
    else:
        if local_task_id_folder and local_task_id_base:
            if not os.path.exists(local_task_id_folder):
                os.makedirs(local_task_id_folder)
            task_id_files = [local_task_id_folder + local_task_id_base + post for post in ['_training.txt', '_eval.txt', '_test.txt']]
            task_id_dicts = [training_tasks, eval_tasks, test_tasks]
            mode = 'w+'
            for task_id_dict, task_id_file in zip(task_id_dicts, task_id_files):
                with open(task_id_file, mode=mode) as f:
                    print(f'writing {task_id_file}')
                    f.write(json.dumps(task_id_dict))
            return task_id_files
        else:
            return None
    

#def unet_sample_generation(region, start_year, end_year):
    
def gcs_download_files(gcs_bucket, gcs_folder, local_folder, pattern=''):
    gcs_path = 'gs://' + gcs_bucket + '/' + gcs_folder
    file_list = os.popen("gsutil ls {}".format(gcs_path)).read()
    file_list = [f for f in file_list.split('\n') if (f and (pattern in f))]
    for file in file_list:
        os.system("gsutil cp {} {}".format(file, local_folder))
    return file_list

def gcs_download_folder(gcs_bucket, gcs_folder, local_folder):
    if not os.path.exists(local_folder):
        os.makedirs(local_folder)
    gcs_path = 'gs://' + gcs_bucket + '/' + gcs_folder
    print(os.popen('curl ifconfig.me').read())
    print(f'gsutil -m cp -r {gcs_path} {local_folder}')
    os.system(f'gsutil -m cp -r {gcs_path} {local_folder}')

def gcs_download_samples_per_feature(collection_size, gcs_bucket, gcs_folder, local_sample_folder, start_feature_index=0, end_feature_index=int(1e10)):
    local_sample_folders = [local_sample_folder + str(i) + '/' for i in range(collection_size)]
    gcs_folders = [gcs_folder + str(i) + '/' for i in range(collection_size)]
    while start_feature_index < collection_size and start_feature_index < end_feature_index:
        gcs_download_folder(
            gcs_bucket=gcs_bucket,
            gcs_folder=gcs_folders[start_feature_index],
            local_folder=local_sample_folders[start_feature_index]
        )
        start_feature_index += 1
    return None

def gcs_download_exported_modis_awei_per_feature(collection_size, gcs_bucket, gcs_folder, local_image_folder, start_feature_index=0, end_feature_index=int(1e10)):
    local_image_folders = [local_image_folder + str(i) + '/' for i in range(collection_size)]
    gcs_folders = [gcs_folder + str(i) + '/' for i in range(collection_size)]
    while start_feature_index < collection_size and start_feature_index < end_feature_index:
        gcs_download_folder(
            gcs_bucket=gcs_bucket,
            gcs_folder=gcs_folders[start_feature_index],
            local_folder=local_image_folders[start_feature_index]
        )
        start_feature_index += 1
    return None
    

def tasks_all_completed(task_id_file):
    with open(task_id_file) as f:
        task_dict = json.load(f)
    task_states = [task['state'] for task in ee.data.getTaskStatus(list(task_dict.values()))]
    all_completed = functools.reduce(lambda a, b: a and b, [True for s in task_states if s == 'COMPLETED'])
    return all_completed

def doExport(image, export_image_base, bucket, gcs_folder, local_task_id_folder, local_task_id_base, kernel_shape, kernel_buffer, region, scale, fileFormat = 'TFRecord'):

    task = ee.batch.Export.image.toCloudStorage(
        image = image,
        description = export_image_base,
        bucket = bucket,
        fileNamePrefix = gcs_folder + export_image_base,
        region = region.getInfo()['coordinates'],
        scale = scale,
        fileFormat = fileFormat,
        maxPixels = 1e11,
        formatOptions = {
            'patchDimensions': kernel_shape,
            'kernelSize': kernel_buffer,
            'compressed': True,
            'maxFileSize': 104857600
        }
    )
    task.start()
    task_id = {}
    task_id[export_image_base] = task.id
    if not os.path.exists(local_task_id_folder):
        os.makedirs(local_task_id_folder)
    task_id_file_path = local_task_id_folder + local_task_id_base + '.txt'
    with open(task_id_file_path, 'w+') as f:
      f.write(json.dumps(task_id))
    return task_id_file_path

def modis_image_export(region, start_time, end_time, export_image_base, bucket, gcs_folder, local_task_id_folder, local_task_id_base, kernel_size, scale, fileFormat='TFRecord'):
    kernel_shape = [kernel_size, kernel_size]
    kernel_buffer = [int(x*0.5) for x in kernel_shape]
    modis_median_awei, Map = image_generation(start_time=start_time, end_time=end_time, kernel_size=kernel_size, mode='modis_awei', test_display=False)
    task_id_file_path = doExport(
        image=modis_median_awei,
        region=region,
        export_image_base=export_image_base,
        bucket=bucket, gcs_folder=gcs_folder,
        local_task_id_folder=local_task_id_folder, local_task_id_base=local_task_id_base,
        kernel_shape=kernel_shape, kernel_buffer=kernel_buffer,
        scale=scale
    )
    return task_id_file_path

def monthly_modis_image_export(region, start_time, end_time, export_image_base, bucket, gcs_folder, local_task_id_folder, local_task_id_base, kernel_size=128, scale=30, fileFormat='TFRecord'):
    date_format = '%Y-%m-%d'
    start_time = datetime.datetime.strptime(start_time, date_format).date()
    end_time = datetime.datetime.strptime(end_time, date_format).date()
    current_time = start_time
    task_id_file_paths = []
    while current_time < end_time:
        current_start_time = current_time
        current_end_time = current_time + relativedelta(months=1)
        current_export_image_base = export_image_base + f'_{current_start_time.strftime(date_format)}_{current_end_time.strftime(date_format)}'
        current_local_task_id_base = local_task_id_base + f'_{current_start_time.strftime(date_format)}_{current_end_time.strftime(date_format)}'
        task_id_file_path = modis_image_export(
            region=region,
            start_time=current_start_time.strftime(date_format), end_time=current_end_time.strftime(date_format),
            export_image_base=current_export_image_base,
            bucket=bucket, gcs_folder=gcs_folder,
            local_task_id_folder=local_task_id_folder, local_task_id_base=current_local_task_id_base,
            kernel_size=kernel_size, scale=scale, fileFormat=fileFormat
        )
        print(f'Exporting {current_export_image_base}')
        task_id_file_paths.append(task_id_file_path)
        current_time = current_time + relativedelta(months=1)
    return task_id_file_paths

def mosaic_fused_result(input_gee_asset:str, filename_pattern:str, start_date:str, n_grids = 123):
    """Deprecated

    Args:
        input_gee_asset (str): _description_
        filename_pattern (str): _description_
        start_date (str): _description_
        n_grids (int, optional): _description_. Defaults to 123.

    Returns:
        _type_: _description_
    """
    image_list = ee.List([])
    date_format = '%Y-%m-%d'
    start_date = datetime.datetime.strptime(start_date, date_format)
    for i in range(n_grids):
        current_asset_id = f'{input_gee_asset}{str(i)}/{filename_pattern}_{str(i)}_{start_date.strftime(date_format)}_{(start_date + relativedelta(months=1)).strftime(date_format)}'
        current_image = ee.Image(current_asset_id)
        image_list = image_list.add(current_image)
    image_collection = ee.ImageCollection(image_list)
    mosaic_image = image_collection.mosaic()
    return mosaic_image

def export_monthly_mosaic_fused_result_to_asset(
    input_gee_asset:str, 
    filename_pattern:str, 
    output_gee_asset:str, 
    output_filename_pattern:str, 
    start_date:str, 
    end_date:str, 
    export_region, 
    scale=30, 
    n_grids=123
    ):
    date_format = '%Y-%m-%d'
    start_date = datetime.datetime.strptime(start_date, date_format)
    end_date = datetime.datetime.strptime(end_date, date_format)
    while start_date < end_date:
        mosaic_image = mosaic_fused_result(input_gee_asset, filename_pattern, start_date.strftime(date_format), n_grids)
        current_export_asset_id = f'{output_gee_asset}{output_filename_pattern}_{start_date.strftime(date_format)}_{(start_date + relativedelta(months=1)).strftime(date_format)}'
        print('Exporting: ', current_export_asset_id)
        task = ee.batch.Export.image.toAsset(
            image=mosaic_image,
            description=f'{output_filename_pattern}_{start_date.strftime(date_format)}_{(start_date + relativedelta(months=1)).strftime(date_format)}',
            assetId=current_export_asset_id,
            scale=scale,
            region=export_region,
            maxPixels=1e13
        )
        task.start()
        start_date = start_date + relativedelta(months=1)

def export_monthly_gap_filled_mosaic_to_asset(
    input_mosaic_gee_asset:str,
    input_right_gaps_asset:str,
    input_bottom_gaps_asset:str,
    file_name_pattern:str,
    start_date:str,
    end_date:str,
    output_gee_asset:str,
    output_file_name_pattern:str,
    export_region,
    scale=30
):
    date_format = '%Y-%m-%d'
    start_date = datetime.datetime.strptime(start_date, date_format)
    end_date = datetime.datetime.strptime(end_date, date_format)
    while start_date < end_date:
        current_mosaic_id = f'{input_mosaic_gee_asset}{file_name_pattern}_mosaic_{start_date.strftime(date_format)}_{(start_date + relativedelta(months=1)).strftime(date_format)}'
        current_right_gaps_id = f'{input_right_gaps_asset}{file_name_pattern}_right_gaps_mosaic_{start_date.strftime(date_format)}_{(start_date + relativedelta(months=1)).strftime(date_format)}'
        current_bottom_gaps_id = f'{input_bottom_gaps_asset}{file_name_pattern}_bottom_gaps_mosaic_{start_date.strftime(date_format)}_{(start_date + relativedelta(months=1)).strftime(date_format)}'
        current_mosaic = ee.Image(current_mosaic_id)
        current_mask_not = current_mosaic.mask().Not()
        current_right_gaps = ee.Image(current_right_gaps_id)
        current_bottom_gaps = ee.Image(current_bottom_gaps_id)
        mosaic_image = current_mosaic.gt(0.5).unmask(0).where(current_right_gaps.gt(0.5).updateMask(current_mask_not), 1).where(current_bottom_gaps.gt(0.5).updateMask(current_mask_not), 1)
        current_export_asset_id = f'{output_gee_asset}{output_file_name_pattern}_{start_date.strftime(date_format)}_{(start_date + relativedelta(months=1)).strftime(date_format)}'
        print(f'reading {current_mosaic_id}, {current_right_gaps_id}, {current_bottom_gaps_id}')
        print('Exporting: ', current_export_asset_id)
        task = ee.batch.Export.image.toAsset(
            image=mosaic_image,
            description=f'{output_file_name_pattern}_{start_date.strftime(date_format)}_{(start_date + relativedelta(months=1)).strftime(date_format)}',
            assetId=current_export_asset_id,
            scale=scale,
            region=export_region,
            maxPixels=1e13
        )
        task.start()
        start_date = start_date + relativedelta(months=1)
    return None
        
def constrain_mosaic_on_gsw_max_extent(mosaic_gee_asset:str, mosaic_file_name_pattern:str, constrained_mosaic_gee_asset:str, constrained_mosaic_file_name_pattern:str, start_date:str, export_region, scale=30):
    date_format = '%Y-%m-%d'
    start_date = datetime.datetime.strptime(start_date, date_format)
    mosaic_gee_asset_id = f'{mosaic_gee_asset}{mosaic_file_name_pattern}_{start_date.strftime(date_format)}_{(start_date + relativedelta(months=1)).strftime(date_format)}'
    constrained_mosaic_gee_asset_id = f'{constrained_mosaic_gee_asset}{constrained_mosaic_file_name_pattern}_{start_date.strftime(date_format)}_{(start_date + relativedelta(months=1)).strftime(date_format)}'
    
    gsw_max_extent = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('max_extent')
    mosaic_image = ee.Image(mosaic_gee_asset_id)
    constrained = mosaic_image.gt(0.5).And(gsw_max_extent.gt(0.5))
    print('Exporting: ', f'{constrained_mosaic_file_name_pattern}_{start_date.strftime(date_format)}_{(start_date + relativedelta(months=1)).strftime(date_format)}')
    task = ee.batch.Export.image.toAsset(
        image=constrained,
        description=f'{constrained_mosaic_file_name_pattern}_{start_date.strftime(date_format)}_{(start_date + relativedelta(months=1)).strftime(date_format)}',
        assetId=constrained_mosaic_gee_asset_id,
        scale=scale,
        region=export_region,
        maxPixels=1e13
    )
    task.start()
    return None

def constrain_mosaic_on_gsw_max_extent_monthly(mosaic_gee_asset:str, mosaic_file_name_pattern:str, constrained_mosaic_gee_asset:str, 
                                               constrained_mosaic_file_name_pattern:str, start_date:str, end_date:str, export_region, scale=30):
    date_format = '%Y-%m-%d'
    start_date = datetime.datetime.strptime(start_date, date_format)
    end_date = datetime.datetime.strptime(end_date, date_format)
    while start_date < end_date:
        constrain_mosaic_on_gsw_max_extent(mosaic_gee_asset, mosaic_file_name_pattern, constrained_mosaic_gee_asset, constrained_mosaic_file_name_pattern, start_date.strftime(date_format), export_region, scale)
        start_date = start_date + relativedelta(months=1)
    return None
    
#for accuracy assessment
def modis_gsw_comask_water_area(modis_based_asset, modis_based_file_name_pattern, output_gcs_bucket, output_gcs_folder, 
                                output_gcs_filename_pattern, start_date:str, 
                                hydrolakes_to_image_asset_id=None, region=None, hydrolakes_dissolved_geometry=None, scale=30):
    if(hydrolakes_to_image_asset_id is not None and region is not None):
        assert hydrolakes_to_image_asset_id is not None and region is not None and hydrolakes_dissolved_geometry is None, 'Cannot specify both hydrolakes_to_image_asset_id&region and hydrolakes_dissolved_geometry' 
    elif(hydrolakes_dissolved_geometry is not None):
        assert hydrolakes_to_image_asset_id is None and region is None and hydrolakes_dissolved_geometry is not None, 'Cannot specify both hydrolakes_to_image_asset_id&region and hydrolakes_dissolved_geometry'
    date_format = '%Y-%m-%d'
    #define date_format_gsw for dates like "2016_06"
    date_format_gsw = '%Y_%m'
    start_date = datetime.datetime.strptime(start_date, date_format)
    modis_based_asset_id = f'{modis_based_asset}{modis_based_file_name_pattern}_{start_date.strftime(date_format)}_{(start_date + relativedelta(months=1)).strftime(date_format)}'
    gsw_asset_id = f'JRC/GSW1_4/MonthlyHistory/{start_date.strftime(date_format_gsw)}'
    
    modis_based_image = ee.Image(modis_based_asset_id).gt(0.5)
    modis_based_water_mask_caused_by_gap = modis_based_image.mask()
    gsw = ee.Image(gsw_asset_id)
    modis_based_image_mask, map = image_generation(start_time=start_date.strftime(date_format), end_time=(start_date + relativedelta(months=1)).strftime(date_format), kernel_size=128, mode='modis_awei_mask', test_display=False)
    gsw_mask = gsw.neq(0)
    modis_based_water = modis_based_image.updateMask(gsw_mask).updateMask(modis_based_image_mask).updateMask(modis_based_water_mask_caused_by_gap)

    gsw_water = gsw.eq(2).updateMask(gsw_mask).updateMask(modis_based_image_mask).updateMask(modis_based_water_mask_caused_by_gap)
    
    if hydrolakes_to_image_asset_id is not None and region is not None and hydrolakes_dissolved_geometry is None:
        hydrolakes_to_image = ee.Image(hydrolakes_to_image_asset_id).gt(0.5)
        modis_based_area = ee.Image.pixelArea().updateMask(modis_based_water).updateMask(hydrolakes_to_image).reduceRegion(
            reducer = ee.Reducer.sum(),
            geometry = region,
            scale = scale,
            maxPixels = 1e13
        )
        gsw_area = ee.Image.pixelArea().updateMask(gsw_water).updateMask(hydrolakes_to_image).reduceRegion(
            reducer = ee.Reducer.sum(),
            geometry = region,
            scale = scale,
            maxPixels = 1e13
        )
    elif hydrolakes_to_image_asset_id is None and region is None and hydrolakes_dissolved_geometry is not None:
        modis_based_area = ee.Image.pixelArea().updateMask(modis_based_water).reduceRegion(
            reducer = ee.Reducer.sum(),
            geometry = hydrolakes_dissolved_geometry,
            scale = scale,
            maxPixels = 1e13
        )
        gsw_area = ee.Image.pixelArea().updateMask(gsw_water).reduceRegion(
            reducer = ee.Reducer.sum(),
            geometry = hydrolakes_dissolved_geometry,
            scale = scale,
            maxPixels = 1e13
        )
    
    modis_based_area = ee.FeatureCollection(ee.Feature(None, modis_based_area))
    gsw_area = ee.FeatureCollection(ee.Feature(None, gsw_area))
    merged_area = modis_based_area.merge(gsw_area)
    
    print('Exporting: ', output_gcs_filename_pattern + '_' + start_date.strftime(date_format) + '_' + (start_date + relativedelta(months=1)).strftime(date_format))
    task = ee.batch.Export.table.toCloudStorage(
        collection=merged_area,
        bucket=output_gcs_bucket,
        fileNamePrefix=output_gcs_folder + output_gcs_filename_pattern + '_' + start_date.strftime(date_format) + '_' + (start_date + relativedelta(months=1)).strftime(date_format),
        fileFormat='CSV',
        description=output_gcs_filename_pattern + '_' + start_date.strftime(date_format) + '_' + (start_date + relativedelta(months=1)).strftime(date_format)
    )
    task.start()
    
    #for area, tp in zip([modis_based_area, gsw_area], ['modis', 'gsw']):
    #    print('Exporting: ', output_gcs_filename_pattern + '_' + tp + start_date.strftime(date_format) + '_' + (start_date + relativedelta(months=1)).strftime(date_format) + '_modis_based_area')
    #    task = ee.batch.Export.table.toCloudStorage(
    #        collection=area,
    #        bucket=output_gcs_bucket,
    #        fileNamePrefix=output_gcs_folder + output_gcs_filename_pattern + '_' + tp + '_' + start_date.strftime(date_format) + '_' + (start_date + relativedelta(months=1)).strftime(date_format),
    #        fileFormat='CSV',
    #        description=output_gcs_filename_pattern + '_' + tp + '_' + start_date.strftime(date_format) + '_' + (start_date + relativedelta(months=1)).strftime(date_format)
    #    )
    #    task.start()
    return None

#for accuracy assessment
def modis_gsw_comask_water_area_monthly(modis_based_asset, modis_based_file_name_pattern, output_gcs_bucket, output_gcs_folder, 
                                        output_gcs_filename_pattern, start_date:str, end_date:str, hydrolakes_to_image_asset_id=None, region=None, hydrolakes_dissolved_geometry=None, scale=30):
    date_format = '%Y-%m-%d'
    start_date = datetime.datetime.strptime(start_date, date_format)
    end_date = datetime.datetime.strptime(end_date, date_format)
    while start_date < end_date:
        current_start_date = start_date
        modis_gsw_comask_water_area(
            modis_based_asset=modis_based_asset,
            modis_based_file_name_pattern=modis_based_file_name_pattern,
            output_gcs_bucket=output_gcs_bucket,
            output_gcs_folder=output_gcs_folder,
            output_gcs_filename_pattern=output_gcs_filename_pattern,
            start_date=current_start_date.strftime(date_format),
            hydrolakes_to_image_asset_id=hydrolakes_to_image_asset_id,
            region=region, 
            hydrolakes_dissolved_geometry=hydrolakes_dissolved_geometry,
            scale=scale
        )
        start_date = start_date + relativedelta(months=1)
    return None

def modis_s1s2l8_comask_water_area(
    modis_based_asset, 
    modis_based_file_name_pattern,
    s1s2l8_asset,
    s1s2l8_file_name_pattern,
    output_gcs_bucket, 
    output_gcs_folder, 
    output_gcs_filename_pattern, 
    start_date:str, 
    hydrolakes_to_image_asset_id=None, 
    region=None, 
    hydrolakes_dissolved_geometry=None, 
    scale=30):
    #check if the exporting region set is valid, either hydrolakes_to_image_asset_id&region or hydrolakes_dissolved_geometry is allowed
    if(hydrolakes_to_image_asset_id is not None and region is not None):
        assert hydrolakes_to_image_asset_id is not None and region is not None and hydrolakes_dissolved_geometry is None, 'Cannot specify both hydrolakes_to_image_asset_id&region and hydrolakes_dissolved_geometry' 
    elif(hydrolakes_dissolved_geometry is not None):
        assert hydrolakes_to_image_asset_id is None and region is None and hydrolakes_dissolved_geometry is not None, 'Cannot specify both hydrolakes_to_image_asset_id&region and hydrolakes_dissolved_geometry'
    date_format = '%Y-%m-%d'
    start_date = datetime.datetime.strptime(start_date, date_format)
    modis_based_asset_id = f'{modis_based_asset}{modis_based_file_name_pattern}_{start_date.strftime(date_format)}_{(start_date + relativedelta(months=1)).strftime(date_format)}'
    s1s2l8_asset_id = f'{s1s2l8_asset}{s1s2l8_file_name_pattern}_{start_date.strftime(date_format)}_{(start_date + relativedelta(months=1)).strftime(date_format)}'
    
    modis_based_image = ee.Image(modis_based_asset_id).gt(0.5)
    modis_based_water_mask_caused_by_gap = modis_based_image.mask()
    s1s2l8_image = ee.Image(s1s2l8_asset_id).gt(0.5)
    modis_based_image_mask, map = image_generation(start_time=start_date.strftime(date_format), end_time=(start_date + relativedelta(months=1)).strftime(date_format), kernel_size=128, mode='modis_awei_mask', test_display=False)
    s1s2l8_image_mask = s1s2l8_image.mask()
    modis_based_water = modis_based_image.updateMask(s1s2l8_image_mask).updateMask(modis_based_image_mask).updateMask(modis_based_water_mask_caused_by_gap)
    s1s2l8_water = s1s2l8_image.updateMask(s1s2l8_image_mask).updateMask(modis_based_image_mask).updateMask(modis_based_water_mask_caused_by_gap)
    
    if hydrolakes_to_image_asset_id is not None and region is not None and hydrolakes_dissolved_geometry is None:
        hydrolakes_to_image = ee.Image(hydrolakes_to_image_asset_id).gt(0.5)
        modis_based_area = ee.Image.pixelArea().updateMask(modis_based_water).updateMask(hydrolakes_to_image).reduceRegion(
            reducer = ee.Reducer.sum(),
            geometry = region,
            scale = scale,
            maxPixels = 1e13
        )
        s1s2l8_area = ee.Image.pixelArea().updateMask(s1s2l8_water).updateMask(hydrolakes_to_image).reduceRegion(
            reducer = ee.Reducer.sum(),
            geometry = region,
            scale = scale,
            maxPixels = 1e13
        )
    elif hydrolakes_to_image_asset_id is None and region is None and hydrolakes_dissolved_geometry is not None:
        modis_based_area = ee.Image.pixelArea().updateMask(modis_based_water).reduceRegion(
            reducer = ee.Reducer.sum(),
            geometry = hydrolakes_dissolved_geometry,
            scale = scale,
            maxPixels = 1e13
        )
        s1s2l8_area = ee.Image.pixelArea().updateMask(s1s2l8_water).reduceRegion(
            reducer = ee.Reducer.sum(),
            geometry = hydrolakes_dissolved_geometry,
            scale = scale,
            maxPixels = 1e13
        )
    else:
        print('Wrong setting')
    
    modis_based_area = ee.FeatureCollection(ee.Feature(None, modis_based_area))
    s1s2l8_area = ee.FeatureCollection(ee.Feature(None, s1s2l8_area))
    merged_area = modis_based_area.merge(s1s2l8_area)
    
    print('Exporting: ', output_gcs_filename_pattern + '_' + start_date.strftime(date_format) + '_' + (start_date + relativedelta(months=1)).strftime(date_format))
    task = ee.batch.Export.table.toCloudStorage(
        collection=merged_area,
        bucket=output_gcs_bucket,
        fileNamePrefix=output_gcs_folder + output_gcs_filename_pattern + '_' + start_date.strftime(date_format) + '_' + (start_date + relativedelta(months=1)).strftime(date_format),
        fileFormat='CSV',
        description=output_gcs_filename_pattern + '_' + start_date.strftime(date_format) + '_' + (start_date + relativedelta(months=1)).strftime(date_format)
    )
    task.start()
    
def modis_s1s2l8_comask_water_area_monthly(
    modis_based_asset, 
    modis_based_file_name_pattern, 
    s1s2l8_asset,
    s1s2l8_file_name_pattern,
    output_gcs_bucket, 
    output_gcs_folder, 
    output_gcs_filename_pattern, 
    start_date:str, 
    end_date:str, 
    hydrolakes_to_image_asset_id=None, 
    region=None, 
    hydrolakes_dissolved_geometry=None, scale=30):
    """
    """
    date_format = '%Y-%m-%d'
    start_date = datetime.datetime.strptime(start_date, date_format)
    end_date = datetime.datetime.strptime(end_date, date_format)
    while start_date < end_date:
        current_start_date = start_date
        modis_s1s2l8_comask_water_area(
            modis_based_asset=modis_based_asset,
            modis_based_file_name_pattern=modis_based_file_name_pattern,
            s1s2l8_asset=s1s2l8_asset,
            s1s2l8_file_name_pattern=s1s2l8_file_name_pattern,
            output_gcs_bucket=output_gcs_bucket,
            output_gcs_folder=output_gcs_folder,
            output_gcs_filename_pattern=output_gcs_filename_pattern,
            start_date=current_start_date.strftime(date_format),
            hydrolakes_to_image_asset_id=hydrolakes_to_image_asset_id,
            region=region, 
            hydrolakes_dissolved_geometry=hydrolakes_dissolved_geometry,
            scale=scale
        )
        start_date = start_date + relativedelta(months=1)
    return None
    
def s1s2l8_gsw_comask_water_area(
    s1s2l8_based_asset, 
    s1s2l8_based_file_name_pattern, 
    output_gcs_bucket, 
    output_gcs_folder, 
    output_gcs_filename_pattern, 
    start_date:str, 
    hydrolakes_to_image_asset_id=None, 
    region=None, 
    hydrolakes_dissolved_geometry=None, 
    scale=30
):
    if(hydrolakes_to_image_asset_id is not None and region is not None):
        assert hydrolakes_to_image_asset_id is not None and region is not None and hydrolakes_dissolved_geometry is None, 'Cannot specify both hydrolakes_to_image_asset_id&region and hydrolakes_dissolved_geometry' 
    elif(hydrolakes_dissolved_geometry is not None):
        assert hydrolakes_to_image_asset_id is None and region is None and hydrolakes_dissolved_geometry is not None, 'Cannot specify both hydrolakes_to_image_asset_id&region and hydrolakes_dissolved_geometry'
    date_format = '%Y-%m-%d'
    #define date_format_gsw for dates like "2016_06"
    date_format_gsw = '%Y_%m'
    start_date = datetime.datetime.strptime(start_date, date_format)
    s1s2l8_based_asset_id = f'{s1s2l8_based_asset}{s1s2l8_based_file_name_pattern}_{start_date.strftime(date_format)}_{(start_date + relativedelta(months=1)).strftime(date_format)}'
    gsw_asset_id = f'JRC/GSW1_4/MonthlyHistory/{start_date.strftime(date_format_gsw)}'
    
    s1s2l8_based_image = ee.Image(s1s2l8_based_asset_id).gt(0.5)
    gsw = ee.Image(gsw_asset_id)
    s1s2l8_based_image_mask = s1s2l8_based_image.mask()
    gsw_mask = gsw.neq(0)
    s1s2l8_based_water = s1s2l8_based_image.updateMask(gsw_mask).updateMask(s1s2l8_based_image_mask)
    gsw_water = gsw.eq(2).updateMask(gsw_mask).updateMask(s1s2l8_based_image_mask)
    
    if hydrolakes_to_image_asset_id is not None and region is not None and hydrolakes_dissolved_geometry is None:
        hydrolakes_to_image = ee.Image(hydrolakes_to_image_asset_id).gt(0.5)
        s1s2l8_based_area = ee.Image.pixelArea().updateMask(s1s2l8_based_water).updateMask(hydrolakes_to_image).reduceRegion(
            reducer = ee.Reducer.sum(),
            geometry = region,
            scale = scale,
            maxPixels = 1e13
        )
        gsw_area = ee.Image.pixelArea().updateMask(gsw_water).updateMask(hydrolakes_to_image).reduceRegion(
            reducer = ee.Reducer.sum(),
            geometry = region,
            scale = scale,
            maxPixels = 1e13
        )
    elif hydrolakes_to_image_asset_id is None and region is None and hydrolakes_dissolved_geometry is not None:
        s1s2l8_based_area = ee.Image.pixelArea().updateMask(s1s2l8_based_water).reduceRegion(
            reducer = ee.Reducer.sum(),
            geometry = hydrolakes_dissolved_geometry,
            scale = scale,
            maxPixels = 1e13
        )
        gsw_area = ee.Image.pixelArea().updateMask(gsw_water).reduceRegion(
            reducer = ee.Reducer.sum(),
            geometry = hydrolakes_dissolved_geometry,
            scale = scale,
            maxPixels = 1e13
        )
    
    s1s2l8_based_area = ee.FeatureCollection(ee.Feature(None, s1s2l8_based_area))
    gsw_area = ee.FeatureCollection(ee.Feature(None, gsw_area))
    merged_area = s1s2l8_based_area.merge(gsw_area)
    
    print('Exporting: ', output_gcs_filename_pattern + '_' + start_date.strftime(date_format) + '_' + (start_date + relativedelta(months=1)).strftime(date_format))
    task = ee.batch.Export.table.toCloudStorage(
        collection=merged_area,
        bucket=output_gcs_bucket,
        fileNamePrefix=output_gcs_folder + output_gcs_filename_pattern + '_' + start_date.strftime(date_format) + '_' + (start_date + relativedelta(months=1)).strftime(date_format),
        fileFormat='CSV',
        description=output_gcs_filename_pattern + '_' + start_date.strftime(date_format) + '_' + (start_date + relativedelta(months=1)).strftime(date_format)
    )
    task.start()
    
def s1s2l8_gsw_comask_water_area_monthly(
    s1s2l8_based_asset, 
    s1s2l8_based_file_name_pattern, 
    output_gcs_bucket, 
    output_gcs_folder, 
    output_gcs_filename_pattern, 
    start_date:str, 
    end_date:str, 
    hydrolakes_to_image_asset_id=None, 
    region=None, 
    hydrolakes_dissolved_geometry=None, 
    scale=30
):
    date_format = '%Y-%m-%d'
    start_date = datetime.datetime.strptime(start_date, date_format)
    end_date = datetime.datetime.strptime(end_date, date_format)
    while start_date < end_date:
        current_start_date = start_date
        s1s2l8_gsw_comask_water_area(
            s1s2l8_based_asset=s1s2l8_based_asset,
            s1s2l8_based_file_name_pattern=s1s2l8_based_file_name_pattern,
            output_gcs_bucket=output_gcs_bucket,
            output_gcs_folder=output_gcs_folder,
            output_gcs_filename_pattern=output_gcs_filename_pattern,
            start_date=current_start_date.strftime(date_format),
            hydrolakes_to_image_asset_id=hydrolakes_to_image_asset_id,
            region=region, 
            hydrolakes_dissolved_geometry=hydrolakes_dissolved_geometry,
            scale=scale
        )
        start_date = start_date + relativedelta(months=1)
    return None

def gsw_modis_non_missing_data_area(
    output_gcs_bucket, 
    output_gcs_folder, 
    output_gcs_filename_pattern, 
    start_date:str,
    region=None,
    scale=30
):
    date_format = '%Y-%m-%d'
    #define date_format_gsw for dates like "2016_06"
    date_format_gsw = '%Y_%m'
    start_date = datetime.datetime.strptime(start_date, date_format)
    modis_based_image_mask, map = image_generation(start_time=start_date.strftime(date_format), end_time=(start_date + relativedelta(months=1)).strftime(date_format), kernel_size=128, mode='modis_awei_mask', test_display=False)
    gsw_asset_id = f'JRC/GSW1_4/MonthlyHistory/{start_date.strftime(date_format_gsw)}'
    gsw = ee.Image(gsw_asset_id)
    gsw_mask = gsw.neq(0)
    
    gsw_non_missing_area = ee.Image.pixelArea().updateMask(gsw_mask).reduceRegion(
            reducer = ee.Reducer.sum(),
            geometry = region,
            scale = scale,
            maxPixels = 1e13
        )
    modis_non_missing_area = ee.Image.pixelArea().updateMask(modis_based_image_mask).reduceRegion(
            reducer = ee.Reducer.sum(),
            geometry = region,
            scale = scale,
            maxPixels = 1e13
        )
    
    gsw_non_missing_area = ee.FeatureCollection(ee.Feature(None, gsw_non_missing_area))
    modis_non_missing_area = ee.FeatureCollection(ee.Feature(None, modis_non_missing_area))
    merged_area = gsw_non_missing_area.merge(modis_non_missing_area)
    
    print('Exporting: ', output_gcs_filename_pattern + '_' + start_date.strftime(date_format) + '_' + (start_date + relativedelta(months=1)).strftime(date_format))
    task = ee.batch.Export.table.toCloudStorage(
        collection=merged_area,
        bucket=output_gcs_bucket,
        fileNamePrefix=output_gcs_folder + output_gcs_filename_pattern + '_' + start_date.strftime(date_format) + '_' + (start_date + relativedelta(months=1)).strftime(date_format),
        fileFormat='CSV',
        description=output_gcs_filename_pattern + '_' + start_date.strftime(date_format) + '_' + (start_date + relativedelta(months=1)).strftime(date_format)
    )
    task.start()
    
def gsw_modis_non_missing_data_area_monthly(
    output_gcs_bucket, 
    output_gcs_folder, 
    output_gcs_filename_pattern, 
    start_date:str, 
    end_date:str, 
    region=None, 
    scale=30
):
    date_format = '%Y-%m-%d'
    start_date = datetime.datetime.strptime(start_date, date_format)
    end_date = datetime.datetime.strptime(end_date, date_format)
    while start_date < end_date:
        current_start_date = start_date
        gsw_modis_non_missing_data_area(
            output_gcs_bucket=output_gcs_bucket,
            output_gcs_folder=output_gcs_folder,
            output_gcs_filename_pattern=output_gcs_filename_pattern,
            start_date=current_start_date.strftime(date_format),
            region=region,
            scale=scale
        )
        start_date = start_date + relativedelta(months=1)
    return None

def gsw_modis_missing_data_area(
    output_gcs_bucket, 
    output_gcs_folder, 
    output_gcs_filename_pattern, 
    start_date:str,
    region=None,
    scale=30
):
    date_format = '%Y-%m-%d'
    #define date_format_gsw for dates like "2016_06"
    date_format_gsw = '%Y_%m'
    start_date = datetime.datetime.strptime(start_date, date_format)
    modis_based_image_mask, map = image_generation(start_time=start_date.strftime(date_format), end_time=(start_date + relativedelta(months=1)).strftime(date_format), kernel_size=128, mode='modis_awei_mask', test_display=False)
    gsw_asset_id = f'JRC/GSW1_4/MonthlyHistory/{start_date.strftime(date_format_gsw)}'
    gsw = ee.Image(gsw_asset_id)
    gsw_mask = gsw.neq(0)
    
    gsw_missing_area = ee.Image.pixelArea().updateMask(gsw_mask.Not()).reduceRegion(
            reducer = ee.Reducer.sum(),
            geometry = region,
            scale = scale,
            maxPixels = 1e13
        )
    modis_missing_area = ee.Image.pixelArea().updateMask(modis_based_image_mask.Not()).reduceRegion(
            reducer = ee.Reducer.sum(),
            geometry = region,
            scale = scale,
            maxPixels = 1e13
        )
    
    gsw_missing_area = ee.FeatureCollection(ee.Feature(None, gsw_missing_area))
    modis_missing_area = ee.FeatureCollection(ee.Feature(None, modis_missing_area))
    merged_area = gsw_missing_area.merge(modis_missing_area)
    
    print('Exporting: ', output_gcs_filename_pattern + '_' + start_date.strftime(date_format) + '_' + (start_date + relativedelta(months=1)).strftime(date_format))
    task = ee.batch.Export.table.toCloudStorage(
        collection=merged_area,
        bucket=output_gcs_bucket,
        fileNamePrefix=output_gcs_folder + output_gcs_filename_pattern + '_' + start_date.strftime(date_format) + '_' + (start_date + relativedelta(months=1)).strftime(date_format),
        fileFormat='CSV',
        description=output_gcs_filename_pattern + '_' + start_date.strftime(date_format) + '_' + (start_date + relativedelta(months=1)).strftime(date_format)
    )
    task.start()
    
def gsw_modis_missing_data_area_monthly(
    output_gcs_bucket, 
    output_gcs_folder, 
    output_gcs_filename_pattern, 
    start_date:str, 
    end_date:str, 
    region=None, 
    scale=30
):
    date_format = '%Y-%m-%d'
    start_date = datetime.datetime.strptime(start_date, date_format)
    end_date = datetime.datetime.strptime(end_date, date_format)
    while start_date < end_date:
        current_start_date = start_date
        gsw_modis_missing_data_area(
            output_gcs_bucket=output_gcs_bucket,
            output_gcs_folder=output_gcs_folder,
            output_gcs_filename_pattern=output_gcs_filename_pattern,
            start_date=current_start_date.strftime(date_format),
            region=region,
            scale=scale
        )
        start_date = start_date + relativedelta(months=1)
    return None
    

def export_modis_based_and_gsw_fused_water_extents(modis_based_id_pattern, start_date, fused_id_pattern, export_region, scale=30):
    date_format = '%Y-%m-%d'
    #define date_format_gsw for dates like "2016_06"
    date_format_gsw = '%Y_%m'
    start_date = datetime.datetime.strptime(start_date, date_format)
    modis_based_asset_id = f'{modis_based_id_pattern}_{start_date.strftime(date_format)}_{(start_date + relativedelta(months=1)).strftime(date_format)}'
    modis_based_water = ee.Image(modis_based_asset_id).gt(0.5).unmask(0)
    gsw_asset_id = f'JRC/GSW1_4/MonthlyHistory/{start_date.strftime(date_format_gsw)}'
    gsw = ee.Image(gsw_asset_id)
    modis_based_image_mask, map = image_generation(start_time=start_date.strftime(date_format), end_time=(start_date + relativedelta(months=1)).strftime(date_format), kernel_size=128, mode='modis_awei_mask', test_display=False)
    gsw_mask = gsw.neq(0)
    gsw_water = gsw.eq(2)
    final_mask = modis_based_image_mask.Or(gsw_mask)
    fused_water = modis_based_water.Or(gsw_water).updateMask(final_mask).unmask(-1)
    
    task = ee.batch.Export.image.toAsset(
        image=fused_water,
        description=f'{fused_id_pattern}_{start_date.strftime(date_format)}_{(start_date + relativedelta(months=1)).strftime(date_format)}'.split('/')[-1],
        assetId=f'{fused_id_pattern}_{start_date.strftime(date_format)}_{(start_date + relativedelta(months=1)).strftime(date_format)}',
        region=export_region,
        scale=scale,
        maxPixels=1e13
    )
    print(f'Exporting {fused_id_pattern}_{start_date.strftime(date_format)}_{(start_date + relativedelta(months=1)).strftime(date_format)}')
    task.start()
    
def export_modis_based_and_gsw_fused_water_extents_monthly(modis_based_id_pattern, fused_id_pattern, start_date, end_date, export_region, scale=30):
    date_format = '%Y-%m-%d'
    start_date = datetime.datetime.strptime(start_date, date_format)
    end_date = datetime.datetime.strptime(end_date, date_format)
    while start_date < end_date:
        export_modis_based_and_gsw_fused_water_extents(modis_based_id_pattern, start_date.strftime(date_format), fused_id_pattern, export_region, scale)
        start_date = start_date + relativedelta(months=1)
    return None

def export_to_drive_surface_water_storage_retrieval_variables(surface_water_gee_id_pattern:str, start_date:str, 
                                                              lakeatlas_to_image_id:str, lake_catchment_to_image_id, hydrobasins_id:str, 
                                                              drive_folder:str, output_description:str, file_format:str, test=False):
    end_date = (datetime.datetime.strptime(start_date, '%Y-%m-%d') + relativedelta(months=1)).strftime('%Y-%m-%d')
    era5_land = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR').filterDate(start_date, end_date).first()
    lakeatlas_to_image = ee.Image(lakeatlas_to_image_id)
    lake_lakeatlas_to_image = lakeatlas_to_image.neq(2)
    reservoir_lakeatlas_to_image = lakeatlas_to_image.eq(2)
    lake_catchment_to_image = ee.Image(lake_catchment_to_image_id).neq(2)
    hydrobasins = ee.FeatureCollection(hydrobasins_id)
    surface_water_gee_id = surface_water_gee_id_pattern + '_' + start_date + '_' + end_date
    surface_water = ee.Image(surface_water_gee_id).gt(0.5)
    surface_water_missing_values = surface_water.eq(-1)
    
    hydrobasins_runoff = era5_land.select('runoff_sum').multiply(11.132**2).rename('runoff_sum_mkm2').updateMask(lake_catchment_to_image).reduceRegions(
        collection = hydrobasins,
        reducer = ee.Reducer.sum(),
        scale = 11132
    ).select(['NEXT_DOWN', 'HYBAS_ID', 'sum'],
             ['NEXT_DOWN', 'HYBAS_ID', 'runoff_sum_mkm2'])
    
    hydrobasins_runoff_evaporation = era5_land.select('total_evaporation_sum').multiply(11.132**2).rename('total_evaporation_sum_mkm2').updateMask(lake_lakeatlas_to_image).updateMask(surface_water).reduceRegions(
        collection = hydrobasins_runoff,
        reducer = ee.Reducer.sum(),
        scale = 11132
    ).select(['NEXT_DOWN', 'HYBAS_ID', 'runoff_sum_mkm2', 'sum'],
             ['NEXT_DOWN', 'HYBAS_ID', 'runoff_sum_mkm2', 'total_evaporation_sum_mkm2'])
    
    hydrobasins_runoff_evaporation_precipitation = era5_land.select('total_precipitation_sum').multiply(11.132**2).rename('total_precipitation_sum_mkm2').updateMask(lake_lakeatlas_to_image).updateMask(surface_water).reduceRegions(
        collection = hydrobasins_runoff_evaporation,
        reducer = ee.Reducer.sum(),
        scale = 11132
    ).select(['NEXT_DOWN', 'HYBAS_ID', 'runoff_sum_mkm2', 'total_evaporation_sum_mkm2', 'sum'],
             ['NEXT_DOWN', 'HYBAS_ID', 'runoff_sum_mkm2', 'total_evaporation_sum_mkm2', 'total_precipitation_sum_mkm2'])

    if(test == True):
        print(ee.Feature(hydrobasins_runoff_evaporation_precipitation.first()).getInfo())
        return None

    hydrobasins_runoff_evaporation_precipitation_lakearea = ee.Image.pixelArea().rename('lake_area_m2').updateMask(lake_lakeatlas_to_image).updateMask(surface_water).reduceRegions(
        collection = hydrobasins_runoff_evaporation_precipitation,
        reducer = ee.Reducer.sum(),
        scale = 30
    ).select(['NEXT_DOWN', 'HYBAS_ID', 'runoff_sum_mkm2', 'total_evaporation_sum_mkm2', 'total_precipitation_sum_mkm2', 'sum'],
             ['NEXT_DOWN', 'HYBAS_ID', 'runoff_sum_mkm2', 'total_evaporation_sum_mkm2', 'total_precipitation_sum_mkm2', 'lake_area_m2'])
    
    hydrobasins_runoff_evaporation_precipitation_lakearea_reservoirarea = ee.Image.pixelArea().rename('reservoir_area_m2').updateMask(reservoir_lakeatlas_to_image).updateMask(surface_water).reduceRegions(
        collection = hydrobasins_runoff_evaporation_precipitation_lakearea,
        reducer = ee.Reducer.sum(),
        scale = 30
    ).select(['NEXT_DOWN', 'HYBAS_ID', 'runoff_sum_mkm2', 'total_evaporation_sum_mkm2', 'total_precipitation_sum_mkm2', 'lake_area_m2', 'sum'],
             ['NEXT_DOWN', 'HYBAS_ID', 'runoff_sum_mkm2', 'total_evaporation_sum_mkm2', 'total_precipitation_sum_mkm2', 'lake_area_m2', 'reservoir_area_m2'])
    
    hydrobasins_runoff_evaporation_precipitation_lakearea_reservoirarea_lakemissingarea = ee.Image.pixelArea().rename('lake_missing_area_m2').updateMask(lake_lakeatlas_to_image).updateMask(surface_water_missing_values).reduceRegions(
        collection = hydrobasins_runoff_evaporation_precipitation_lakearea_reservoirarea,
        reducer = ee.Reducer.sum(),
        scale = 30
    ).select(['NEXT_DOWN', 'HYBAS_ID', 'runoff_sum_mkm2', 'total_evaporation_sum_mkm2', 'total_precipitation_sum_mkm2', 'lake_area_m2', 'reservoir_area_m2', 'sum'],
             ['NEXT_DOWN', 'HYBAS_ID', 'runoff_sum_mkm2', 'total_evaporation_sum_mkm2', 'total_precipitation_sum_mkm2', 'lake_area_m2', 'reservoir_area_m2', 'lake_missing_area_m2'])
    
    hydrobasins_runoff_evaporation_precipitation_lakearea_reservoirarea_lakemissingarea_reservoirmissingarea = ee.Image.pixelArea().rename('reservoir_missing_area_m2').updateMask(reservoir_lakeatlas_to_image).updateMask(surface_water_missing_values).reduceRegions(
        collection = hydrobasins_runoff_evaporation_precipitation_lakearea_reservoirarea_lakemissingarea,
        reducer = ee.Reducer.sum(),
        scale = 30
    ).select(['NEXT_DOWN', 'HYBAS_ID', 'runoff_sum_mkm2', 'total_evaporation_sum_mkm2', 'total_precipitation_sum_mkm2', 'lake_area_m2', 'reservoir_area_m2', 'lake_missing_area_m2', 'sum'],
             ['NEXT_DOWN', 'HYBAS_ID', 'runoff_sum_mkm2', 'total_evaporation_sum_mkm2', 'total_precipitation_sum_mkm2', 'lake_area_m2', 'reservoir_area_m2', 'lake_missing_area_m2', 'reservoir_missing_area_m2'])
    
    properties_to_select = ['lake_area_m2', 
                            'reservoir_area_m2', 
                            'lake_missing_area_m2', 
                            'reservoir_missing_area_m2', 
                            'runoff_sum_mkm2', 
                            'total_evaporation_sum_mkm2', 
                            'total_precipitation_sum_mkm2',
                            'NEXT_DOWN',
                            'HYBAS_ID'
                            ]
    
    collection_to_export = hydrobasins_runoff_evaporation_precipitation_lakearea_reservoirarea_lakemissingarea_reservoirmissingarea.select(properties_to_select)
    print(f'Exporting to Google Drive {output_description}')
    task = ee.batch.Export.table.toDrive(
        collection = collection_to_export,
        description = output_description,
        folder = drive_folder,
        fileFormat = file_format
    )
    task.start()
    return None

def export_to_drive_surface_water_storage_retrieval_variables_monthly(
    surface_water_gee_id_pattern:str,
    start_date:str, end_date:str,
    lakeatlas_to_image_id:str, 
    lake_catchment_to_image_id:str, 
    hydrobasins_id:str, 
    drive_folder:str, 
    output_description_pattern:str, 
    file_format:str, 
    test=False
    ):
    date_format = '%Y-%m-%d'
    start_date = datetime.datetime.strptime(start_date, date_format)
    end_date = datetime.datetime.strptime(end_date, date_format)
    while start_date < end_date:
        output_description = f'{output_description_pattern}_{start_date.strftime(date_format)}_{(start_date + relativedelta(months=1)).strftime(date_format)}'
        export_to_drive_surface_water_storage_retrieval_variables(
            surface_water_gee_id_pattern=surface_water_gee_id_pattern,
            start_date=start_date.strftime(date_format),
            lakeatlas_to_image_id=lakeatlas_to_image_id,
            lake_catchment_to_image_id=lake_catchment_to_image_id,
            hydrobasins_id=hydrobasins_id,
            drive_folder=drive_folder,
            output_description=output_description,
            file_format=file_format,
            test=test
        )
        start_date = start_date + relativedelta(months=1)
    return None

def monthly_images_to_bands_to_asset(
    image_gee_asset_id_pattern:str,
    start_date:str, end_date:str,
    out_image_gee_asset_id:str,
    out_band_name_pattern:str,
    output_region, scale=30
):
    date_format = '%Y-%m-%d'
    start_date = datetime.datetime.strptime(start_date, date_format)
    end_date = datetime.datetime.strptime(end_date, date_format)
    image_list = ee.List([])
    date_list = []
    while start_date < end_date:
        current_date = start_date.strftime(date_format)
        date_list.append(current_date)
        current_image_gee_asset_id = f'{image_gee_asset_id_pattern}_{start_date.strftime(date_format)}_{(start_date + relativedelta(months=1)).strftime(date_format)}'
        current_image = ee.Image(current_image_gee_asset_id).rename(current_date)
        image_list = image_list.add(current_image)
        start_date = start_date + relativedelta(months=1)
    image_collection = ee.ImageCollection(image_list)
    images_to_bands = image_collection.toBands()
    
    band_names = [f'{i}_{date_list[i]}' for i in range(len(date_list))]
    new_band_names = [f'{out_band_name_pattern}_{date_list[i]}' for i in range(len(date_list))]
    
    images_to_bands = images_to_bands.select(band_names, new_band_names)
    
    print(f'Exporting {out_image_gee_asset_id}')
    task = ee.batch.Export.image.toAsset(
        description=out_image_gee_asset_id.split('/')[-1],
        assetId=out_image_gee_asset_id,
        image=images_to_bands,
        scale=scale,
        region=output_region
    )
    task.start()
    return None

def monthly_areas_comparison_modis_gsw_s1s2l8_toDrive(
    start_date:str,
    modis_asset_id_pattern:str,
    s1s2l8_asset_id_pattern:str,
    export_folder:str,
    export_filename_pattern:str,
    export_region,
    export_file_format='CSV'
):
    """Create and start tasks that calculates the comparable (ie, used union mask) monthly areas of modis- gsw- and s1s2l8-based lake surface extent
    """
    date_format = '%Y-%m-%d'
    start_date = datetime.datetime.strptime(start_date, date_format)
    date_format_gsw = '%Y_%m'
    gsw_asset_id = f'JRC/GSW1_4/MonthlyHistory/{start_date.strftime(date_format_gsw)}'
    modis_asset_id = modis_asset_id_pattern.format(start_date.strftime(date_format), (start_date + relativedelta(months=1)).strftime(date_format))
    s1s2l8_asset_id = s1s2l8_asset_id_pattern.format(start_date.strftime(date_format), (start_date + relativedelta(months=1)).strftime(date_format))
    gsw = ee.Image(gsw_asset_id)
    modis = ee.Image(modis_asset_id)
    s1s2l8 = ee.Image(s1s2l8_asset_id)
    
    #get the mask of gsw, modis, and s1s2l8 based results
    #Note: for modis, because the U-Net outputs all NaN as 0, we need to use the image_generation function to get the mask
    gsw_noData = gsw.eq(0)
    modis_noData, map = image_generation(start_time=start_date.strftime(date_format), end_time=(start_date + relativedelta(months=1)).strftime(date_format), kernel_size=128, mode='modis_awei_mask', test_display=False)
    s1s2l8_noData = s1s2l8.mask()
    gsw_water = gsw.eq(2)
    modis_water = modis.gt(0.5)
    s1s2l8_water = s1s2l8.gt(0.5)
    
    #calculate the cross-masked areas for validation and comparison
        #for gsw and modis
    gsw_area_gsw_modis_masked = ee.Image.pixelArea().updateMask(gsw_water).updateMask(modis_noData).updateMask(gsw_noData).reduceRegion(
        reducer = ee.Reducer.sum(),
        geometry = export_region,
        scale = 30,
        maxPixels = 1e13
    ).rename(['area'], ['gsw_area_gsw_modis_masked'])
    modis_area_gsw_modis_masked = ee.Image.pixelArea().updateMask(modis_water).updateMask(modis_noData).updateMask(gsw_noData).reduceRegion(
        reducer = ee.Reducer.sum(),
        geometry = export_region,
        scale = 30,
        maxPixels = 1e13
    ).rename(['area'], ['modis_area_gsw_modis_masked'])
        #for gsw and s1s2l8
    gsw_area_gsw_s1s2l8_masked = ee.Image.pixelArea().updateMask(gsw_water).updateMask(s1s2l8_noData).updateMask(gsw_noData).reduceRegion(
        reducer = ee.Reducer.sum(),
        geometry = export_region,
        scale = 30,
        maxPixels = 1e13
    ).rename(['area'], ['gsw_area_gsw_s1s2l8_masked'])
    s1s2l8_area_gsw_s1s2l8_masked = ee.Image.pixelArea().updateMask(s1s2l8_water).updateMask(s1s2l8_noData).updateMask(gsw_noData).reduceRegion(
        reducer = ee.Reducer.sum(),
        geometry = export_region,
        scale = 30,
        maxPixels = 1e13
    ).rename(['area'], ['s1s2l8_area_gsw_s1s2l8_masked'])
        #for modis and s1s2l8
    modis_area_modis_s1s2l8_masked = ee.Image.pixelArea().updateMask(modis_water).updateMask(s1s2l8_noData).updateMask(modis_noData).reduceRegion(
        reducer = ee.Reducer.sum(),
        geometry = export_region,
        scale = 30,
        maxPixels = 1e13
    ).rename(['area'], ['modis_area_modis_s1s2l8_masked'])
    s1s2l8_area_modis_s1s2l8_masked = ee.Image.pixelArea().updateMask(s1s2l8_water).updateMask(s1s2l8_noData).updateMask(modis_noData).reduceRegion(
        reducer = ee.Reducer.sum(),
        geometry = export_region,
        scale = 30,
        maxPixels = 1e13
    ).rename(['area'], ['s1s2l8_area_modis_s1s2l8_masked'])
    
    combined_areas = gsw_area_gsw_modis_masked\
                        .combine(modis_area_gsw_modis_masked)\
                        .combine(gsw_area_gsw_s1s2l8_masked)\
                        .combine(s1s2l8_area_gsw_s1s2l8_masked)\
                        .combine(modis_area_modis_s1s2l8_masked)\
                        .combine(s1s2l8_area_modis_s1s2l8_masked)
    areas_feature_collection = ee.FeatureCollection(ee.Feature(None, combined_areas))
    task = ee.batch.Export.table.toDrive(
        collection=areas_feature_collection,
        folder=export_folder,
        description=export_filename_pattern.format(start_date.strftime(date_format), (start_date + relativedelta(months=1)).strftime(date_format)),
        fileFormat=export_file_format
    )
    task.start()
    print(f'Exporting {export_filename_pattern.format(start_date.strftime(date_format), (start_date + relativedelta(months=1)).strftime(date_format))}')
    return None
    
def export_modis_gsw_fused_areas_of_lakes_from_hydrolakes_to_gcs(
    modis_gsw_fused_asset_id_pattern:str,
    start_date:str,
    hydrolakes_asset_id:str,
    export_gcs_bucket:str,
    export_gcs_folder:str,
    export_gcs_filename_pattern:str,
    area_property_name:str,
    additional_properties_to_select:list,
    scale=30,
    file_format='CSV',
    export_geometry=False
):
    date_format = '%Y-%m-%d'
    start_date = datetime.datetime.strptime(start_date, date_format)
    hydrolakes = ee.FeatureCollection(hydrolakes_asset_id)
    modis_gsw_fused = ee.Image(modis_gsw_fused_asset_id_pattern.format(start_date.strftime(date_format), (start_date + relativedelta(months=1)).strftime(date_format))).gt(0.5)
    properties_before_select = ['sum'] + additional_properties_to_select
    properties_to_select = [area_property_name] + additional_properties_to_select
    hydrolakes_areas_and_properties_selected = ee.Image.pixelArea().updateMask(modis_gsw_fused).reduceRegions(
        collection=hydrolakes,
        reducer=ee.Reducer.sum(),
        scale=scale
    ).select(properties_before_select, properties_to_select)
    if not export_geometry:
        hydrolakes_areas_and_properties_selected = hydrolakes_areas_and_properties_selected.map(lambda x: ee.Feature(x).setGeometry(None))
    task = ee.batch.Export.table.toCloudStorage(
        collection=hydrolakes_areas_and_properties_selected,
        description=export_gcs_filename_pattern.format(start_date.strftime(date_format), (start_date + relativedelta(months=1)).strftime(date_format)),
        bucket=export_gcs_bucket,
        fileNamePrefix=export_gcs_folder + export_gcs_filename_pattern.format(start_date.strftime(date_format), (start_date + relativedelta(months=1)).strftime(date_format)),
        fileFormat=file_format
    )
    task.start()
    print(f'Exporting {export_gcs_filename_pattern.format(start_date.strftime(date_format), (start_date + relativedelta(months=1)).strftime(date_format))}')
    return None
    
def monthly_export_modis_gsw_fused_areas_of_lakes_from_hydrolakes_to_gcs(
    modis_gsw_fused_asset_id_pattern:str,
    start_date:str,
    end_date:str,
    hydrolakes_asset_id:str,
    export_gcs_bucket:str,
    export_gcs_folder:str,
    export_gcs_filename_pattern:str,
    area_property_name:str,
    additional_properties_to_select:list,
    scale=30,
    file_format='CSV',
    export_geometry=False
):
    date_format = '%Y-%m-%d'
    start_date = datetime.datetime.strptime(start_date, date_format)
    end_date = datetime.datetime.strptime(end_date, date_format)
    while start_date < end_date:
        export_modis_gsw_fused_areas_of_lakes_from_hydrolakes_to_gcs(
            modis_gsw_fused_asset_id_pattern=modis_gsw_fused_asset_id_pattern,
            start_date=start_date.strftime(date_format),
            hydrolakes_asset_id=hydrolakes_asset_id,
            export_gcs_bucket=export_gcs_bucket,
            export_gcs_folder=export_gcs_folder,
            export_gcs_filename_pattern=export_gcs_filename_pattern,
            area_property_name=area_property_name,
            additional_properties_to_select=additional_properties_to_select,
            scale=scale,
            file_format=file_format,
            export_geometry=export_geometry
        )
        start_date = start_date + relativedelta(months=1)
    return None

def export_ERA5Land_monthly_aggregated_variables_per_lake_and_catchment_to_gcs(
    start_date:str,
    lake_catchment_feature_collection_asset_id:str,
    hydrolakes_feature_collection_asset_id:str,
    modis_gsw_fused_asset_id_pattern:str,
    selected_era5_variables_for_lake:list,
    selected_era5_variables_for_catchment:list,
    export_gcs_bucket:str,
    export_gcs_folder_hydrolakes:str,
    export_gcs_filename_pattern_hydrolakes:str,
    export_gcs_folder_lake_catchment:str,
    export_gcs_filename_pattern_lake_catchment:str,
    scale=11132,
    file_format='CSV',
    export_geometry=False,
    export_hydrolakes=True,
    export_lake_catchment=True
):
    """Export monthly aggregated ERA5Land variables per lake catchment to Google Cloud Storage

    Args:
        start_date (str): start date of the monthly aggregated ERA5Land variables
        lake_catchment_feature_collection_asset_id (str): _description_
        selected_era5_variables (list): _description_
        export_gcs_bucket (str): _description_
        export_gcs_folder (str): _description_
        export_gcs_filename_pattern (str): _description_
        scale (int, optional): _description_. Defaults to 11132.
        file_format (str, optional): _description_. Defaults to 'CSV'.
        export_geometry (bool, optional): _description_. Defaults to False.
    """
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    hydrolakes_fc = ee.FeatureCollection(hydrolakes_feature_collection_asset_id)
    lake_catchment_fc = ee.FeatureCollection(lake_catchment_feature_collection_asset_id)
    era5_land = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR').filterDate(start_date.strftime('%Y-%m-%d'), (start_date + relativedelta(months=1)).strftime('%Y-%m-%d')).first()
    modis_gsw_fused = ee.Image(modis_gsw_fused_asset_id_pattern.format(start_date.strftime('%Y-%m-%d'), (start_date + relativedelta(months=1)).strftime('%Y-%m-%d'))).gt(0.5)
    hydrolakes_era5_variables_added = era5_land\
                            .select(selected_era5_variables_for_lake)\
                            .rename([f'{i}_lake_mean_m' for i in selected_era5_variables_for_lake])\
                            .reduceRegions(
                                reducer=ee.Reducer.mean(),
                                collection=hydrolakes_fc,
                                scale=scale
                            ).select(['Hylak_id'] + [f'{i}_lake_mean_m' for i in selected_era5_variables_for_lake])
    if not export_geometry:
        hydrolakes_era5_variables_added = hydrolakes_era5_variables_added.map(lambda x: ee.Feature(x).setGeometry(None))
    lake_catchment_fc_era5_variables_added = era5_land\
                                            .select(selected_era5_variables_for_catchment)\
                                            .rename([f'{i}_catchment_sum_m3' for i in selected_era5_variables_for_catchment])\
                                            .multiply(ee.Image.pixelArea())\
                                            .reduceRegions(
                                                reducer=ee.Reducer.sum(),
                                                collection=lake_catchment_fc,
                                                scale=scale
                                            ).select(['Hylak_id'] + [f'{i}_catchment_sum_m3' for i in selected_era5_variables_for_catchment])
    if not export_geometry:
        lake_catchment_fc_era5_variables_added = lake_catchment_fc_era5_variables_added.map(lambda x: ee.Feature(x).setGeometry(None))
    
    if export_hydrolakes:
        task = ee.batch.Export.table.toCloudStorage(
            collection=hydrolakes_era5_variables_added,
            description=export_gcs_filename_pattern_hydrolakes.format(start_date.strftime('%Y-%m-%d'), (start_date + relativedelta(months=1)).strftime('%Y-%m-%d')),
            bucket=export_gcs_bucket,
            fileNamePrefix=export_gcs_folder_hydrolakes + export_gcs_filename_pattern_hydrolakes.format(start_date.strftime('%Y-%m-%d'), (start_date + relativedelta(months=1)).strftime('%Y-%m-%d')),
            fileFormat=file_format
        )
        task.start()
        print(f'Exporting {export_gcs_filename_pattern_hydrolakes.format(start_date.strftime("%Y-%m-%d"), (start_date + relativedelta(months=1)).strftime("%Y-%m-%d"))}')
    
    if export_lake_catchment:
        task = ee.batch.Export.table.toCloudStorage(
            collection=lake_catchment_fc_era5_variables_added,
            description=export_gcs_filename_pattern_lake_catchment.format(start_date.strftime('%Y-%m-%d'), (start_date + relativedelta(months=1)).strftime('%Y-%m-%d')),
            bucket=export_gcs_bucket,
            fileNamePrefix=export_gcs_folder_lake_catchment + export_gcs_filename_pattern_lake_catchment.format(start_date.strftime('%Y-%m-%d'), (start_date + relativedelta(months=1)).strftime('%Y-%m-%d')),
            fileFormat=file_format
        )
        task.start()
        print(f'Exporting {export_gcs_filename_pattern_lake_catchment.format(start_date.strftime("%Y-%m-%d"), (start_date + relativedelta(months=1)).strftime("%Y-%m-%d"))}')
    
    return None

def monthly_export_ERA5Land_monthly_aggregated_variables_per_lake_and_catchment_to_gcs(
    start_date:str,
    end_date:str,
    lake_catchment_feature_collection_asset_id:str,
    hydrolakes_feature_collection_asset_id:str,
    modis_gsw_fused_asset_id_pattern:str,
    selected_era5_variables_for_lake:list,
    selected_era5_variables_for_catchment:list,
    export_gcs_bucket:str,
    export_gcs_folder_hydrolakes:str,
    export_gcs_filename_pattern_hydrolakes:str,
    export_gcs_folder_lake_catchment:str,
    export_gcs_filename_pattern_lake_catchment:str,
    scale=11132,
    file_format='CSV',
    export_geometry=False,
    export_hydrolakes=True,
    export_lake_catchment=True
):
    """Export monthly aggregated ERA5Land variables per lake catchment to Google Cloud Storage

    Args:
        start_date (str): _description_
        end_date (str): _description_
        lake_catchment_feature_collection_asset_id (str): _description_
        hydrolakes_feature_collection_asset_id (str): _description_
        modis_gsw_fused_asset_id_pattern (str): _description_
        selected_era5_variables_for_lake (list): _description_
        selected_era5_variables_for_catchment (list): _description_
        export_gcs_bucket (str): _description_
        export_gcs_folder_hydrolakes (str): _description_
        export_gcs_filename_pattern_hydrolakes (str): _description_
        export_gcs_folder_lake_catchment (str): _description_
        export_gcs_filename_pattern_lake_catchment (str): _description_
        scale (int, optional): _description_. Defaults to 11132.
        file_format (str, optional): _description_. Defaults to 'CSV'.
        export_geometry (bool, optional): _description_. Defaults to False.
    """
    
    date_format = '%Y-%m-%d'
    start_date = datetime.datetime.strptime(start_date, date_format)
    end_date = datetime.datetime.strptime(end_date, date_format)
    while start_date < end_date:
        export_ERA5Land_monthly_aggregated_variables_per_lake_and_catchment_to_gcs(
            start_date=start_date.strftime(date_format),
            lake_catchment_feature_collection_asset_id=lake_catchment_feature_collection_asset_id,
            hydrolakes_feature_collection_asset_id=hydrolakes_feature_collection_asset_id,
            modis_gsw_fused_asset_id_pattern=modis_gsw_fused_asset_id_pattern,
            selected_era5_variables_for_lake=selected_era5_variables_for_lake,
            selected_era5_variables_for_catchment=selected_era5_variables_for_catchment,
            export_gcs_bucket=export_gcs_bucket,
            export_gcs_folder_hydrolakes=export_gcs_folder_hydrolakes,
            export_gcs_filename_pattern_hydrolakes=export_gcs_filename_pattern_hydrolakes,
            export_gcs_folder_lake_catchment=export_gcs_folder_lake_catchment,
            export_gcs_filename_pattern_lake_catchment=export_gcs_filename_pattern_lake_catchment,
            scale=scale,
            file_format=file_format,
            export_geometry=export_geometry,
            export_hydrolakes=export_hydrolakes,
            export_lake_catchment=export_lake_catchment
        )
        start_date = start_date + relativedelta(months=1)
    return None