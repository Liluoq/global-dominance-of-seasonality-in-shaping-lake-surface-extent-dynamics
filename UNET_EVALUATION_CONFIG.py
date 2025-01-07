

HYBAS_ID = 4020050470
FINAL_DECISION = 145
FORMATTED_FINAL_DECISION = f"{FINAL_DECISION:04d}"
MODEL_PATH = f'/WORK/Codes/global_lake_area/trained_unet_10bands_att_laea_no_zenith/{HYBAS_ID}/cp-{FORMATTED_FINAL_DECISION}.ckpt'
MODEL_TRAINING_RECORD_PATH = f'/WORK/Codes/global_lake_area/trained_unet_10bands_att_laea_no_zenith/{HYBAS_ID}/history.txt'
TEST_PATTERN = 'test'
LOCAL_SAMPLE_FOLDER = f'/WORK/SSD_Data/global_lake_area/unet_sample_10bands_laea_per_basin/10bands_LAEA_samples_{HYBAS_ID}/'
INPUT_BANDS = ['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2', 'GSW_Occurrence', 'GSW_Recurrence']
INPUT_BAND_SCALING_FACTORS = [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.01, 0.01]
RESPONSE = ['Water_GSW']
KERNEL_SIZE = 128
COMPRESSION_TYPE_OF_TEST_DATA = 'GZIP'