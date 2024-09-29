import unetgee as ug
from batch_processing.load_config_module import load_config_module
import argparse
import tensorflow as tf
import json
from my_unet_definition.evaluation_metrics import IoU_coef, IoU_loss
from update_training_record import update_training_record

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained U-Net model(s) on test data')

    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')

    args = parser.parse_args()
    config_path = args.config
    config = load_config_module(config_path)

    hybas_id = config.HYBAS_ID
    final_decision = config.FINAL_DECISION
    model_path = config.MODEL_PATH
    model_training_record_path = config.MODEL_TRAINING_RECORD_PATH
    test_pattern = config.TEST_PATTERN
    local_sample_folder = config.LOCAL_SAMPLE_FOLDER
    input_bands = config.INPUT_BANDS
    input_band_scaling_factors = config.INPUT_BAND_SCALING_FACTORS
    response = config.RESPONSE
    kernel_size = config.KERNEL_SIZE
    compression_type_of_test_data = config.COMPRESSION_TYPE_OF_TEST_DATA

    test_data = ug.read_test_samples(
        test_pattern=test_pattern,
        local_sample_folder=local_sample_folder,
        input_bands=input_bands,
        input_band_scaling_factors=input_band_scaling_factors,
        response=response,
        kernel_size=kernel_size,
        compression_type=compression_type_of_test_data
    )

    model = tf.keras.models.load_model(model_path, custom_objects={'IoU_coef': IoU_coef, 'IoU_loss': IoU_loss})
    
    test_loss, test_acc, test_iou = model.evaluate(test_data)

    with open(model_training_record_path, 'r') as f:
        training_record = json.load(f)
    train_acc = training_record['binary_accuracy'][final_decision-1]
    eval_acc = training_record['val_binary_accuracy'][final_decision-1]
    train_iou = training_record['IoU_coef'][final_decision-1]
    eval_iou = training_record['val_IoU_coef'][final_decision-1]
    
    update_training_record(
        model_name=hybas_id,
        test_acc=test_acc,
        test_IoU=test_iou,
        training_acc=train_acc,
        eval_acc=eval_acc,
        training_IoU=train_iou,
        eval_IoU=eval_iou
    )
