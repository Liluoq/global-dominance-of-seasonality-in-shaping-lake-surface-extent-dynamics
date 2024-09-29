import geopandas as gpd
import pandas as pd
import numpy as np
import os
import argparse

def glakes_update_hydrolakes_all():
    glakes_folder = '/WORK/Data/global_lake_area/lake_shps/GLAKES/GLAKES'
    glakes_filenames = [
        'GLAKES_af.shp',
        'GLAKES_as.shp',
        'GLAKES_eu.shp',
        'GLAKES_na1.shp',
        'GLAKES_na2.shp',
        'GLAKES_oc.shp',
        'GLAKES_sa.shp'
    ]
    hydrolakes_folder = '/WORK/Data/global_lake_area/lake_shps/HydroLAKES_original/HydroLAKES_polys_v10_shp'
    hydrolakes_filename = 'HydroLAKES_polys_v10.shp'
    
    hydrolakes_save_folder = '/WORK/Data/global_lake_area/lake_shps/HydroLAKES_updated_using_GLAKES/all'
    hydrolakes_updated_filename = 'HydroLAKES_polys_v10_updated_using_GLAKES.shp'
    if not os.path.exists(hydrolakes_save_folder):
        os.makedirs(hydrolakes_save_folder)
    
    print('Reading files...')
    glakes_gdfs = [
        gpd.read_file(os.path.join(glakes_folder, glakes_filename)) for glakes_filename in glakes_filenames
    ]
    print('Concatenating...')
    glakes_gdf_all = pd.concat(glakes_gdfs, ignore_index=True)
    
    hydrolakes_gdf_all = gpd.read_file(os.path.join(hydrolakes_folder, hydrolakes_filename))
    print('Joining...')
    joined = gpd.sjoin(glakes_gdf_all, hydrolakes_gdf_all, how='inner', op='intersects')
    print('Dissolving...')
    dissolved_glakes_gdf = joined.dissolve(by='Hylak_id')
    
    updated_hydrolakes_gdf_all = hydrolakes_gdf_all.copy()
    print('Updating geometries...')
    for index, row in dissolved_glakes_gdf.iterrows():
        current_hylak_id = index
        current_new_geom = row['geometry']
        updated_hydrolakes_gdf_all.loc[updated_hydrolakes_gdf_all['Hylak_id'] == current_hylak_id, 'geometry'] = current_new_geom
        
    largest_lake_area_idx = updated_hydrolakes_gdf_all.groupby('geometry')['Lake_area'].idxmax()

    # Step 2: Filter the GeoDataFrame to keep only the rows with the largest Lake_area for each geometry
    updated_hydrolakes_gdf_all_filtered = updated_hydrolakes_gdf_all.loc[largest_lake_area_idx]

    updated_hydrolakes_gdf_all_filtered.to_file(os.path.join(hydrolakes_save_folder, hydrolakes_updated_filename))

def hydrolakes_divide_by_basin():
    hydrolakes_all_folder = '/WORK/Data/global_lake_area/lake_shps/HydroLAKES_original/HydroLAKES_polys_v10_shp'
    hydrolakes_all_filename = 'HydroLAKES_polys_v10.shp'
    hydrolakes_pnt_folder = '/WORK/Data/global_lake_area/lake_shps/HydroLAKES_original/HydroLAKES_points_v10_shp'
    hydrolakes_pnt_filename = 'HydroLAKES_points_v10.shp'
    hybas_folder = '/WORK/Data/global_lake_area/lake_shps/hybas'
    hybas_filename = 'hybas_lev02_v1c_merged_lake_count_added_gt100.shp'
    
    hydrolakes_save_folder = '/WORK/Data/global_lake_area/lake_shps/HydroLAKES_original/per_basin'
    if not os.path.exists(hydrolakes_save_folder):
        os.makedirs(hydrolakes_save_folder)
    hydrolakes_save_filename_pattern = 'hylak_original_unbuffered_{hybas_id}.shp'
    
    print(f'Reading {hydrolakes_all_filename}')
    hydrolakes_all_gdf = gpd.read_file(os.path.join(hydrolakes_all_folder, hydrolakes_all_filename))
    print(f'Reading {hydrolakes_pnt_filename}')
    hydrolakes_pnt_gdf = gpd.read_file(os.path.join(hydrolakes_pnt_folder, hydrolakes_pnt_filename))
    print(f'Reading {hybas_filename}')
    hybas_gdf = gpd.read_file(os.path.join(hybas_folder, hybas_filename))
    
    if hydrolakes_all_gdf.crs != hydrolakes_pnt_gdf.crs:
        print('CRS mismatch for hydrolakes_all and hydrolakes_pnt.')
        print(f'hydrolakes_all CRS: {hydrolakes_all_gdf.crs}')
        print(f'hydrolakes_pnt CRS: {hydrolakes_pnt_gdf.crs}')
        print('Reprojecting...')
        hydrolakes_pnt_gdf = hydrolakes_pnt_gdf.to_crs(hydrolakes_all_gdf.crs)
        
    if hydrolakes_all_gdf.crs != hybas_gdf.crs:
        print('CRS mismatch for hydrolakes_all and hybas.')
        print(f'hydrolakes_all CRS: {hydrolakes_all_gdf.crs}')
        print(f'hybas CRS: {hybas_gdf.crs}')
        print('Reprojecting...')
        hybas_gdf = hybas_gdf.to_crs(hydrolakes_all_gdf.crs)
        
    for index, row in hybas_gdf.iterrows():
        current_hybas_geom = row['geometry']
        hybas_id = row['HYBAS_ID']
        print(f'Processing basin {hybas_id}...')
        all_hydrolakes_pnt_within = hydrolakes_pnt_gdf[hydrolakes_pnt_gdf.within(current_hybas_geom)]
        all_hydrolakes_within_hylak_id = all_hydrolakes_pnt_within['Hylak_id'].to_numpy()
        all_hydrolakes_within = hydrolakes_all_gdf[hydrolakes_all_gdf['Hylak_id'].isin(all_hydrolakes_within_hylak_id)].copy()
        print(f'{len(all_hydrolakes_within)} lakes within basin {hybas_id}.')
        save_path = os.path.join(hydrolakes_save_folder, hydrolakes_save_filename_pattern.format(hybas_id=hybas_id))
        all_hydrolakes_within.to_file(save_path)
    

def glakes_update_hydrolakes_per_basin():
    use_centroid_for_hylak = True
    glakes_folder = '/WORK/Data/global_lake_area/lake_shps/GLAKES/GLAKES'
    glakes_filenames = [
        'GLAKES_af.shp',
        'GLAKES_as.shp',
        'GLAKES_eu.shp',
        'GLAKES_na1.shp',
        'GLAKES_na2.shp',
        'GLAKES_oc.shp',
        'GLAKES_sa.shp'
    ]
    hydrolakes_folder = '/WORK/Data/global_lake_area/lake_shps/HydroLAKES_original/per_basin'
    updated_hydrolakes_save_folder = '/WORK/Data/global_lake_area/lake_shps/HydroLAKES_updated_using_GLAKES/per_basin'
    if not os.path.exists(updated_hydrolakes_save_folder):
        os.makedirs(updated_hydrolakes_save_folder)
    hybas_id_list = [
        1020000010, 1020011530, 1020018110, 1020021940, 1020027430, 1020034170, 1020035180, 1020040190,
        2020000010, 2020003440, 2020018240, 2020024230, 2020033490, 2020041390, 2020057170, 2020065840, 2020071190,
        3020000010, 3020003790, 3020005240, 3020008670, 3020009320, 3020024310,
        4020000010, 4020006940, 4020015090, 4020024190, 4020034510, 4020050210, 4020050220, 4020050290, 4020050470,
        5020000010, 5020015660, 5020037270, 5020049720, 5020082270, 
        6020000010, 6020006540, 6020008320, 6020014330, 6020017370, 6020021870, 6020029280,
        7020000010, 7020014250, 7020021430, 7020024600, 7020038340, 7020046750, 7020047840, 7020065090,
        8020000010, 8020008900, 8020010700, 8020020760, 8020022890, 8020032840, 8020044560,
        9020000010
    ]
    hylak_per_basin_filenames = [
        f'hylak_buffered_{hybas_id}.shp' for hybas_id in hybas_id_list
    ]
    updated_hylak_per_basin_filenames = [
        f'hylak_unbuffered_{hybas_id}_updated.shp' for hybas_id in hybas_id_list
    ]
    
    glakes_gdfs = [
        gpd.read_file(os.path.join(glakes_folder, glakes_filename)) for glakes_filename in glakes_filenames
    ]
    glakes_gdf_all = pd.concat(glakes_gdfs, ignore_index=True)
    
    for hylak_per_basin_filename, updated_hylak_per_basin_filename in zip(hylak_per_basin_filenames, updated_hylak_per_basin_filenames):
        if not os.path.exists(os.path.join(hydrolakes_folder, hylak_per_basin_filename)):
            print(f'{hylak_per_basin_filename} does not exist.')
            continue
        print(f'Processing {hylak_per_basin_filename}...')
        hylak_per_basin_gdf = gpd.read_file(os.path.join(hydrolakes_folder, hylak_per_basin_filename))
        updated_hylak_per_basin_gdf = hylak_per_basin_gdf.copy()
        if use_centroid_for_hylak:
            hylak_per_basin_gdf['geometry'] = hylak_per_basin_gdf.centroid 
        
        if glakes_gdf_all.crs != hylak_per_basin_gdf.crs:
            print(f'CRS mismatch for {hylak_per_basin_filename}.')
            print(f'GLAKES CRS: {glakes_gdf_all.crs}')
            print(f'Hydrolakes CRS: {hylak_per_basin_gdf.crs}')
            print('Reprojecting...')
            glakes_gdf_all = glakes_gdf_all.to_crs(hylak_per_basin_gdf.crs)
        joined = gpd.sjoin(glakes_gdf_all, hylak_per_basin_gdf, how='inner', predicate='intersects').reset_index(drop=True)
        largest_lake_area_idx = joined.groupby('Lake_id')['Lake_area'].idxmax()
        no_overlap_joined = joined.loc[largest_lake_area_idx]
        
        dissolved_glakes_gdf = no_overlap_joined.dissolve(by='Hylak_id')
        
        for index, row in dissolved_glakes_gdf.iterrows():
            current_hylak_id = index
            current_new_geom = row['geometry']
            updated_hylak_per_basin_gdf.loc[updated_hylak_per_basin_gdf['Hylak_id'] == current_hylak_id, 'geometry'] = current_new_geom
            
        updated_hylak_per_basin_gdf.to_file(os.path.join(updated_hydrolakes_save_folder, updated_hylak_per_basin_filename))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='update_all')
    
    args = parser.parse_args()
    mode = args.mode
    
    if mode == 'update_all':
        glakes_update_hydrolakes_all()
    elif mode == 'divide_by_basin':
        hydrolakes_divide_by_basin()
    elif mode == 'update_per_basin':
        glakes_update_hydrolakes_per_basin()