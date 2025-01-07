# Description: Configuration file for batch processing

BASIN_ID = str(8020000010)

# Date related parameters
START_DATE = "2018-01-01"
END_DATE = "2024-01-01"
DATE_FORMAT = "%Y-%m-%d"

# Code directory/path related parameters
CODE_DIR = "/WORK/Codes/global_lake_area/batch_processing"
BATCH_FULL_SCRIPT = "batch_full.py"
BATCH_TFRECORD_GENERATION_SCRIPT = "batch_tfrecord_generation.py"
BATCH_UNET_PREDICTION_SCRIPT = "batch_unet_prediction.py"
BATCH_PREDICTION_RECONSTRUCTION_SCRIPT = "batch_prediction_reconstruction.py"
BATCH_MOSAIC_SCRIPT = "batch_mosaic.py"
TRAINING_RECORDS_PATH = '../training_records.csv'

# Data directory/path related parameters
WKDIR = "/WORK/SSD_Data/global_lake_area"

GSW_OCCURRENCE_AND_RECURRENCE_PATH = f"gsw_occurrence_and_recurrence_laea_30m/mosaic/gsw_occurrence_and_recurrence_30m_laea_{BASIN_ID}_mosaic.tif"
BOUNDARY_SHP_PATH = f"hybas_lev02_shp/hybas_lev02_{BASIN_ID}.shp"
UNPREDICTED_TFRECORD_OUTPUT_FOLDER = f"unpredicted_tfrecords/{BASIN_ID}"
INPUT_MODIS_500M_FOLDER = f"modis_8bands_laea_500m/{BASIN_ID}"
INPUT_MODIS_500M_NAME_PATTERN = f"{BASIN_ID}" + "_modis_500m_{current_date_str}_{next_month_str}*.tif"
PREDICTED_OUTPUT_FOLDER = f"predicted_tfrecords/{BASIN_ID}"
RECONSTUCTED_OUTPUT_FOLDER = f"reconstructed_tifs/{BASIN_ID}"
MOSAIC_OUTPUT_FOLDER = f"mosaic_tifs/{BASIN_ID}"

# Model parameters
MODEL_DIR = "/WORK/Codes/global_lake_area"
MODEL_SELECTION = 310
FORMATTED_MODEL_SELECTION = f"{MODEL_SELECTION:04d}"
MODEL_PATH = f"{MODEL_DIR}/trained_unet_10bands_att_laea_no_zenith/{BASIN_ID}/cp-{FORMATTED_MODEL_SELECTION}.ckpt"

# Processing parameters
TILE_SIZE_X = str(512)
TILE_SIZE_Y = str(512)
RESAMPLE_ALG = 'nearest'
PREDICTION_BATCH_SIZE = str(64)
PREDICTION_MAX_QUEUE_SIZE_PER_PROCESS = str(2)
VERBOSE = str(1)
NUM_PROCESSES_CLEAR_FINISH_TAG = str(1)
NUM_PROCESSES_TFRECORD_GENERATION = str(6)
NUM_PROCESSES_UNET_PREDICTION = str(2)
PREDICTION_MULTI_TF_SESSIONS = True
PREDICTION_NUM_TF_SESSIONS = str(2)
NUM_PROCESSES_PREDICTION_RECONSTRUCTION = str(1)
NUM_PROCESSES_MOSAIC = str(1)
WAIT = str(30)

PREDICTION_ASYNC_SAVE = False 
 
