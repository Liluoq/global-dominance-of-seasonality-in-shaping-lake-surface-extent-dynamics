import basin_wise_analysis
import geopandas as gpd
import pandas as pd
import numpy as np
import os
import sys

if __name__ == '__main__':
    hydrobasins_shp_path_pattern = '/WORK/Data/global_lake_area/hydrobasins/separated/hybas_{}_lev02_v1c.shp'
    region_abbreviation_list = ['af', 'ar', 'as', 'au', 'eu', 'gr', 'na', 'sa', 'si']
    output_merged_pkl_path = '/WORK/Data/global_lake_area/hydrobasins/merged/hydrobasins_lev02.pkl'
    
    basin_wise_analysis.merge_hydrobasins_of_multiple_regions(
        hydrobasins_shp_path_pattern=hydrobasins_shp_path_pattern,
        region_abbreviation_list=region_abbreviation_list,
        output_merged_pkl_path=output_merged_pkl_path
    )