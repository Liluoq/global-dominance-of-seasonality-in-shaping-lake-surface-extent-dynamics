import dask.dataframe as dd
import dask_geopandas as dgpd
import numpy as np
import pdb

def attach_permafrost_type(
    my_lakes_ddf: dgpd.GeoDataFrame,
    hydrolakes_pnt_dgdf: dgpd.GeoDataFrame,
    permafrost_dgdf: dgpd.GeoDataFrame,
    permafrost_extent_column_name='EXTENT',
    lake_id_column='Hylak_id',
    save_path=None
):
    """Attach permafrost type information to lake data.

    This function performs a spatial join between permafrost data and lake points,
    then merges the result with the main lake dataset to attach permafrost type
    information to each lake.

    Args:
        my_lakes_ddf (dgpd.GeoDataFrame): Main lake dataset as a Dask DataFrame
        hydrolakes_pnt_dgdf (dgpd.GeoDataFrame): Point representation of lakes as a Dask GeoDataFrame
        permafrost_dgdf (dgpd.GeoDataFrame): Permafrost data as a Dask GeoDataFrame
        permafrost_extent_column_name (str, optional): Name of column containing permafrost extent. 
            Defaults to 'EXTENT'.
        lake_id_column (str, optional): Name of lake ID column. Defaults to 'Hylak_id'.
        save_path (str, optional): Path to save resulting DataFrame. Defaults to None.

    Returns:
        dask.dataframe.DataFrame: Lake data with attached permafrost type information
    """

    permafrost_dgdf = permafrost_dgdf[[permafrost_extent_column_name, 'geometry']]
    hydrolakes_in_permafrost = permafrost_dgdf.sjoin(hydrolakes_pnt_dgdf, how='inner', predicate='contains')[[lake_id_column, permafrost_extent_column_name]]
    my_lakes_with_permafrost = my_lakes_ddf.merge(hydrolakes_in_permafrost, on=lake_id_column, how='left')
    if save_path:
        my_lakes_with_permafrost.to_csv(save_path, index=False, single_file=True)
    return my_lakes_with_permafrost

if __name__ == '__main__':
    my_lakes_csv_path = '/WORK/Data/global_lake_area/area_csvs/lakes/csv/lakes_all_with_aridity_index.csv'
    hydrolakes_pnt_path = '/WORK/Data/global_lake_area/hydroATLAS/hydroLAKES/HydroLAKES_points_v10_shp/HydroLAKES_points_v10.gpkg'
    permafrost_path = '/WORK/Data/global_lake_area/auxiliary_data_for_analysis/permafrost/permaice.shp'
    attatched_permafrost_type_lakes_path = '/WORK/Data/global_lake_area/area_csvs/lakes/csv/lakes_all_with_aridity_and_permafrost_type.csv'
    my_lakes_ddf = dd.read_csv(my_lakes_csv_path)
    hydrolakes_pnt_dgdf = dgpd.read_file(hydrolakes_pnt_path, npartitions=4)
    permafrost_dgdf = dgpd.read_file(permafrost_path, npartitions=4)
    permafrost_crs = permafrost_dgdf.crs
    hydrolakes_pnt_dgdf = hydrolakes_pnt_dgdf.to_crs(permafrost_crs)
    attach_permafrost_type(
        my_lakes_ddf=my_lakes_ddf,
        hydrolakes_pnt_dgdf=hydrolakes_pnt_dgdf,
        permafrost_dgdf=permafrost_dgdf,
        save_path=attatched_permafrost_type_lakes_path
    )
    