# Description: Configuration file for area calculation batch processing

# Path: batch_processing/AREA_CALCULATION_CONFIG.py

# Set the working directory to the directory of the script
WKDIR = '/WORK/Data/global_lake_area'
CODE_DIR = '/WORK/Codes/global_lake_area/batch_processing'
BATCH_AREA_SCRIPT = 'batch_area_calculation.py'

# Basin ID
BASIN_ID = str(9020000010)

# Start date and end date
START_DATE = '2001-01-01'
END_DATE = '2024-01-01'
DATE_FMT = '%Y-%m-%d'

# Folders relative to WKDIR
INPUT_TIF_FOLDER = f'mosaic_tifs/{BASIN_ID}'
INPUT_TIF_FILE_NAME_PATTERN = '{basin_id}_{start_date}_{end_date}_water_mosaic.tif'
OUTPUT_CSV_FOLDER = f'area_csvs/{BASIN_ID}'
OUTPUT_CSV_FILE_NAME_PATTERN = '{basin_id}_{start_date}_{end_date}_area.csv'
OUTPUT_CONCATENATED_CSV_PATH = f'area_csvs/concatenated/{BASIN_ID}_concatenated.csv'
INPUT_LAKE_SHP_FOLDER = 'lake_shps/HydroLAKES_updated_using_GLAKES/per_basin_no_contained_buffered'
INPUT_LAKE_SHP_FILE_NAME = f'hylak_buffered_updated_no_contained_{BASIN_ID}.shp'
RASTER_CLIP_TEMPORARY_FOLDER = f'raster_clip_temporary/{BASIN_ID}'
TEMPORARY_VECTOR_FOLDER = f'temporary_vector/{BASIN_ID}'

# Parameters for the calculation
FORCE_RASTER_BINARY_EQ_VALUE = None
LAKE_ID_FIELD = 'Hylak_id'
GRID_SIZE = 128000
METHOD = 'rasterize_vector'
OUTSIDE_VALUE = int(-1)
NUM_PROCESSES = 14
CHECK_OVERLAP = True
FILTER_ONCE = True
VERBOSE = 2
SAVE_LAKE_RASTERIZATION = True