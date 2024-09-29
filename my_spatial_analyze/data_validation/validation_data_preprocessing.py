import pandas as pd
import numpy as np
import os
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime
from dateutil.relativedelta import relativedelta
import re

def lixingdong_data_preprocessing(
    my_lakes_path,
    lixingdong_datasets_folder='/WORK/Data/global_lake_area/validation_data/xingdongLi_2019/datasets',
    lixingdong_filename_pattern='{}_Water_level-storage_changes.tab',
    lixingdong_lakes_positions='/WORK/Data/global_lake_area/validation_data/xingdongLi_2019/xingdongLi_lake_pos.csv',
    preprocessed_save_folder='/WORK/Data/global_lake_area/validation_data/xingdongLi_2019/processed_datasets',
):

    my_lakes_gdf = pd.read_pickle(my_lakes_path)
    my_lakes_gdf.set_geometry('geometry', inplace=True)

    pos_df = pd.read_csv(lixingdong_lakes_positions)

    lake_names = []
    # iterate over the rows of the DataFrame
    for index, row in pos_df.iterrows():
        # Extract the lake name and coordinates
        lake_name = row['Lake_name']
        current_lixingdong_filename = lixingdong_filename_pattern.format(lake_name)
        current_lixingdong_path = os.path.join(lixingdong_datasets_folder, current_lixingdong_filename)
        current_lixingdong_df = pd.read_table(current_lixingdong_path, sep='\t')
        current_lixingdong_df['Date/Time'] = pd.to_datetime(current_lixingdong_df['Date/Time']).dt.strftime('%Y-%m-01')
        current_lixingdong_row = current_lixingdong_df.groupby('Date/Time').mean().T.reset_index(drop=True).iloc[0,:].to_frame().T
        
        lat = row['Latitude']
        lon = row['Longitude']
        current_point = Point(lon, lat)
        current_my_lake_row = my_lakes_gdf[my_lakes_gdf.contains(current_point)]
        if current_my_lake_row.empty:
            print(f'Lake {lake_name} not found in my_lakes dataset')
            continue
        if len(current_my_lake_row) > 1:
            #keep only the row with smallest Hylak_id and make it a 1 row DataFrame
            current_my_lake_row = current_my_lake_row.loc[current_my_lake_row['Hylak_id'].idxmin()]
            current_my_lake_row = current_my_lake_row.to_frame().T
            print(current_my_lake_row)
            
        if len(current_lixingdong_row.columns) != 0:
            lake_names.append(lake_name)
        my_lakes_start_date = '2001-01-01'
        my_lakes_end_date = '2024-01-01'
        current_lixingdong_columns = current_lixingdong_row.columns.tolist()
        current_lixingdong_start_date = current_lixingdong_columns[0]
        current_lixingdong_end_date = current_lixingdong_columns[-1]
        my_lakes_start_date = datetime.strptime(my_lakes_start_date, '%Y-%m-%d')
        my_lakes_end_date = datetime.strptime(my_lakes_end_date, '%Y-%m-%d')
        current_lixingdong_start_date = datetime.strptime(current_lixingdong_start_date, '%Y-%m-%d')
        current_lixingdong_end_date = datetime.strptime(current_lixingdong_end_date, '%Y-%m-%d')
        construct_date = max(my_lakes_start_date, current_lixingdong_start_date)
        select_dates = []
        while construct_date < min(my_lakes_end_date, current_lixingdong_end_date):
            selected_date = construct_date.strftime('%Y-%m-01')
            select_dates.append(selected_date)
            construct_date += relativedelta(months=1)
        select_dates = [date for date in select_dates if date in current_lixingdong_columns]
        my_lakes_frozen_columns = [f'frozen_{date}' for date in select_dates]
        current_hylak_id = current_my_lake_row['Hylak_id'].values[0]
        #if current_hylak_id == 149:
        #    break
        current_my_lake_row = current_my_lake_row[select_dates + my_lakes_frozen_columns]
        
        current_lixingdong_row = current_lixingdong_row[select_dates]
        
        current_my_lake_save_path = os.path.join(preprocessed_save_folder, f'{lake_name}_my_lake.csv')
        current_lixingdong_save_path = os.path.join(preprocessed_save_folder, f'{lake_name}_lixingdong.csv')
        
        current_my_lake_row.to_csv(current_my_lake_save_path, index=False)
        current_lixingdong_row.to_csv(current_lixingdong_save_path, index=False)
        
        print(lake_names)
        return lake_names
    
def GREALM_data_preprocessing(
    my_lakes_path,
    GREALM_datasets_folder='/WORK/Data/global_lake_area/validation_data/GREALM/unprocessed_data',
    preprocessed_save_folder='/WORK/Data/global_lake_area/validation_data/GREALM/processed_datasets',
    correction_flag_column_name='water_level_corrected'
):
    if not os.path.exists(preprocessed_save_folder):
        os.makedirs(preprocessed_save_folder)
    
    def _convert_longitude(longitude):
        if longitude > 180:
            return longitude - 360
        else:
            return longitude

    GREALM_column_names = [
            "Satellite mission name", "Satellite repeat cycle", "Date",
            "Hour of day", "Minutes of hour", "Target height variation",
            "Estimated error of target height variation", "Mean along track Ku-band backscatter coefficient",
            "Wet tropospheric correction", "Ionosphere correction", "Dry tropospheric correction",
            "Instrument operating mode 1", "Instrument operating mode 2", "Flag for potential frozen surface",
            "Target height variation in EGM2008 datum", "Flag for data source"
        ]
    
    GREALM_column_names_oi = [
        "Date", "Target height variation in EGM2008 datum"
    ]

    def _read_GREALM_data(GREALM_filepath, GREALM_column_names):
        # Read and parse the file
        data_lines = []
        with open(GREALM_filepath, 'r') as file:
            lines = file.readlines()
            for i, line in enumerate(lines):
                # Extract metadata using regular expressions
                if "Latitude and longitude" in line:
                    lat_lon_match = re.search(r"(\d+\.\d+)\s+(\d+\.\d+)", line)
                    if lat_lon_match:
                        latitude = float(lat_lon_match.group(1))
                        longitude = float(lat_lon_match.group(2))
                        longitude = _convert_longitude(longitude)
                
                # Skip comment lines and empty lines before line 51
                if i < 50:
                    continue
                
                # Collect data lines starting from line 51
                if i >= 50:
                    data_lines.append(line.strip())

        # Create the DataFrame
        data = [line.split() for line in data_lines]
        df = pd.DataFrame(data, columns=GREALM_column_names)

        # Convert numeric columns to appropriate types
        df = df.apply(pd.to_numeric, errors='ignore')
        
        current_lake_pnt = Point(longitude, latitude)
        return df, current_lake_pnt
    
    my_lakes_gdf = pd.read_pickle(my_lakes_path)
    my_lakes_gdf.set_geometry('geometry', inplace=True)
    
    GREALM_filenames = os.listdir(GREALM_datasets_folder)
    GREALM_filepaths = [os.path.join(GREALM_datasets_folder, filename) for filename in GREALM_filenames]
    
    for GREALM_filepath in GREALM_filepaths:
        current_GREALM_df, current_lake_pnt = _read_GREALM_data(GREALM_filepath, GREALM_column_names)
        invalid_values_to_replace = [99999999, 999.99, 9999.99]
        current_GREALM_df.replace(invalid_values_to_replace, np.nan, inplace=True)
        current_GREALM_df.dropna(inplace=True)
        current_GREALM_df = current_GREALM_df[GREALM_column_names_oi]
        GREALM_date_column_name = GREALM_column_names_oi[0]
        current_GREALM_df[GREALM_date_column_name] = current_GREALM_df[GREALM_date_column_name].apply(lambda x: datetime.strptime(str(int(x)), '%Y%m%d').strftime('%Y-%m-01'))
        pivot_current_GREALM_df = current_GREALM_df.pivot_table(columns=GREALM_date_column_name, values=GREALM_column_names_oi[1], aggfunc='mean')
        current_my_lake_row = my_lakes_gdf[my_lakes_gdf.contains(current_lake_pnt)]
        if len(current_my_lake_row) > 1:
            print('Multiple lakes found for a single lake point')
            continue
        elif len(current_my_lake_row) == 0:
            print(f'Lake not found in my_lakes dataset')
            continue

        current_my_lake_level_corrected_flag = current_my_lake_row['water_level_corrected'].values[0]
        if not current_my_lake_level_corrected_flag:
            print(f'Lake {current_my_lake_row["Hylak_id"].values[0]} does not have corrected water level data')
            continue
        current_lake_id = current_my_lake_row['Hylak_id'].values[0]
        
        GREALM_water_level_columns = pivot_current_GREALM_df.columns.tolist()
        my_lakes_start_date = '2001-01-01'
        my_lakes_end_date = '2024-01-01'
        my_lakes_water_level_columns = []
        my_lakes_start_date = datetime.strptime(my_lakes_start_date, '%Y-%m-%d')
        my_lakes_end_date = datetime.strptime(my_lakes_end_date, '%Y-%m-%d')
        construct_date = my_lakes_start_date
        while construct_date < my_lakes_end_date:
            selected_date = construct_date.strftime('%Y-%m-01')
            my_lakes_water_level_columns.append(selected_date)
            construct_date += relativedelta(months=1)
        
        common_water_level_columns = [column for column in my_lakes_water_level_columns if column in GREALM_water_level_columns]
        my_lakes_frozen_columns = [f'frozen_{date}' for date in common_water_level_columns]
        
        pivot_current_GREALM_df = pivot_current_GREALM_df[common_water_level_columns]
        current_my_lake_row = current_my_lake_row[common_water_level_columns + my_lakes_frozen_columns]
        
        current_my_lake_save_path = os.path.join(preprocessed_save_folder, f'{current_lake_id}_my_lake.csv')
        current_GREALM_save_path = os.path.join(preprocessed_save_folder, f'{current_lake_id}_GREALM.csv')
        
        current_my_lake_row.to_csv(current_my_lake_save_path, index=False)
        pivot_current_GREALM_df.to_csv(current_GREALM_save_path, index=False)