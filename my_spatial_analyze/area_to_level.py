import netCDF4 as nc
import numpy as np

def get_coefs_using_hylak_id(hylak_ids:np.ndarray, nc_path):
    data = nc.Dataset(nc_path)
    f_hA = np.array(data.variables['f_hA'][:])
    all_hylak_id = np.array(data.variables['lake_id'][:])
    selection = np.isin(all_hylak_id, hylak_ids)
    coefs = f_hA[selection, 0:2]
    selected_hylak_ids = all_hylak_id[selection]
    return selected_hylak_ids, coefs
    
def convert_area_to_level(area, coef_a, coef_b, unit_scale=1e-6):
    level = (area*unit_scale/coef_a)**(1/coef_b)
    return level
    
def convert_area_df_to_level_df(area_df, area_columns, globathy_nc_path, id_column_name, columns_to_drop=[]):
    height_df = area_df.copy()
    hybas_ids = area_df[id_column_name].to_numpy()
    selected_hylak_ids, coefs = get_coefs_using_hylak_id(hylak_ids=hybas_ids, nc_path=globathy_nc_path)
    for index, row in height_df.iterrows():
        current_hylak_id = row[id_column_name]
        current_hylak_id_index = np.where(selected_hylak_ids == current_hylak_id)[0]
        if current_hylak_id_index.size != 0:
            current_coefs = coefs[current_hylak_id_index][0]
            for i, area_column in enumerate(area_columns):
                current_area = row[area_column]
                current_height = convert_area_to_level(current_area, current_coefs[0], current_coefs[1])
                height_df.loc[index, area_column] = current_height
        else:
            raise ValueError('The current hylak id is not found in the globathy dataset.')
    if columns_to_drop:
        height_df.drop(columns=columns_to_drop, inplace=True)
    return height_df

if __name__ == '__main__':
    data = nc.Dataset('/WORK/Data/global_lake_area/nc_data/globathy/GLOBathy_hAV_relationships.nc')
    print(data.variables)
    f_hA = data.variables['f_hA']
    print(f_hA[1, :])