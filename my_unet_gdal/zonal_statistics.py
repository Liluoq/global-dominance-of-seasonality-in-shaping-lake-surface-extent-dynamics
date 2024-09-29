from osgeo import gdal, ogr, osr
import numpy as np
import os
import sys
gdal.UseExceptions()

def reproject_vector(input_vec_path: str, output_vec_path: str, tgt_srs_wkt: str, verbose: int = 0, gdal_driver_name: str = 'ESRI Shapefile') -> None:
    """
    Reprojects a vector file to a target spatial reference system (SRS).

    Args:
        input_vec_path (str): The path to the input vector file.
        output_vec_path (str): The path to the output vector file.
        tgt_srs_wkt (str): The well-known text (WKT) representation of the target SRS.
        verbose (int, optional): Verbosity level. Defaults to 0.
        gdal_driver_name (str, optional): The name of the GDAL driver to use. Defaults to 'ESRI Shapefile'.

    Raises:
        Exception: If an error occurs while opening the input vector file, importing the target SRS,
                   deleting the output vector file, creating the output vector file, creating a field
                   in the output layer, or processing features.

    """
    try:
        driver = ogr.GetDriverByName(gdal_driver_name)
        input_vec = driver.Open(input_vec_path, 0)
        input_lyr = input_vec.GetLayer()
    except Exception as e:
        print("An error occurred while opening the input vector file:", str(e))
        sys.exit(1)
    
    try:
        tgt_srs = osr.SpatialReference()
        tgt_srs.ImportFromWkt(tgt_srs_wkt)
    except Exception as e:
        print("An error occurred while importing the target spatial reference:", str(e))
        sys.exit(1)

    if os.path.exists(output_vec_path):
        try:
            driver.DeleteDataSource(output_vec_path)
        except Exception as e:
            print("An error occurred while deleting the output vector file:", str(e))
            sys.exit(1)
    try:
        output_vec = driver.CreateDataSource(output_vec_path)
        output_lyr = output_vec.CreateLayer(input_lyr.GetName(), srs=tgt_srs, geom_type=input_lyr.GetGeomType())
    except Exception as e:
        print("An error occurred while creating the output vector file:", str(e))
        sys.exit(1)
    
    try:
        input_layer_defn = input_lyr.GetLayerDefn()
        for i in range(input_layer_defn.GetFieldCount()):
            field_defn = input_layer_defn.GetFieldDefn(i)
            output_lyr.CreateField(field_defn)
    except Exception as e:
        print("An error occurred while creating a field in the output layer:", str(e))
        sys.exit(1)
        
    try:
        output_lyr_defn = output_lyr.GetLayerDefn()
        for feature in input_lyr:
            geom = feature.GetGeometryRef()
            geom.TransformTo(tgt_srs)
            
            output_feature = ogr.Feature(output_lyr_defn)
            output_feature.SetGeometry(geom)
            
            for i in range(output_lyr_defn.GetFieldCount()):
                output_feature.SetField(output_lyr_defn.GetFieldDefn(i).GetNameRef(), feature.GetField(i))
                
            output_lyr.CreateFeature(output_feature)
            output_feature = None
    except Exception as e:
        print("An error occurred while processing features:", str(e))
        sys.exit(1)
        
    input_vec = None
    output_vec = None
    
def zonal_statistics_as_single_region(raster_path: str, vector_path: str, statistic_type: str = 'sum') -> float:
    """
    Calculates zonal statistics for a raster dataset within the polygons of a vector dataset.

    Parameters:
    raster_path (str): The file path of the raster dataset. (should only contain one band)
    vector_path (str): The file path of the vector dataset.
    statistic_type (str, optional): The type of statistic to calculate. Default is 'sum'.

    Returns:
    float: The zonal statistic value.

    Raises:
    ValueError: If an invalid statistic type is provided.

    """
    try:
        # Open the vector file
        vector_ds = ogr.Open(vector_path)
        layer = vector_ds.GetLayer()
    except Exception as e:
        print("An error occurred while opening the vector file:", str(e))
        sys.exit(1)    
    
    try:
        # Open the raster file
        raster_ds = gdal.Open(raster_path)
        raster_band = raster_ds.GetRasterBand(1)
        raster_array = raster_band.ReadAsArray()
    except Exception as e:
        print("An error occurred while opening the raster file:", str(e))
        sys.exit(1)
    
    try:
        # Create a mask dataset
        mem_drv = gdal.GetDriverByName('MEM')
        mask_ds = mem_drv.Create('', raster_ds.RasterXSize, raster_ds.RasterYSize, 1, gdal.GDT_Int8)
    except Exception as e:
        print("An error occurred while creating the mask dataset:", str(e))
        sys.exit(1)
    mask_ds.SetProjection(raster_ds.GetProjection())
    mask_ds.SetGeoTransform(raster_ds.GetGeoTransform())
    
    band = mask_ds.GetRasterBand(1)
    band.Fill(-1)  # Fill the raster with -1 to identify areas outside polygons
    burn_value = 1  # A list of burn values to use
    rasterize_options = {'AllTouched': 1, 'BurnValues': [burn_value]}
    try:
        # Rasterize the layer
        gdal.RasterizeLayer(mask_ds, [1], layer, options=rasterize_options)
    except Exception as e:
        print("An error occurred while rasterizing the layer:", str(e))
        sys.exit(1)
    
    mask_array = band.ReadAsArray()
    
    zone_mask = mask_array == burn_value
    if statistic_type == 'sum':
        # Calculate the sum of raster values within the zone mask
        zonal_statistic = np.sum(raster_array[zone_mask])
    else:
        print("Invalid statistic type. Please use 'sum'.")
        raise ValueError

    return zonal_statistic

def zonal_statistics_as_multiple_regions(raster_path: str, vector_path: str, vector_group_field: str, statistic_type: str = 'sum') -> dict:
    """
    Calculate zonal statistics for multiple regions.

    Args:
        raster_path (str): The file path of the raster dataset. (should only contain one band)
        vector_path (str): The file path of the vector dataset.
        vector_group_field (str): The attribute field in the vector dataset that defines the regions.
        statistic_type (str, optional): The type of statistic to calculate. Defaults to 'sum'.

    Returns:
        dict: A dictionary containing the zonal statistics for each region.

    Raises:
        ValueError: If an invalid statistic type is provided.

    """
    try:
        # Open the vector file
        vector_ds = ogr.Open(vector_path)
        layer = vector_ds.GetLayer()
    except Exception as e:
        print("An error occurred while opening the vector file:", str(e))
        sys.exit(1)    
    
    try:
        # Open the raster file
        raster_ds = gdal.Open(raster_path)
        raster_band = raster_ds.GetRasterBand(1)
        raster_array = raster_band.ReadAsArray()
    except Exception as e:
        print("An error occurred while opening the raster file:", str(e))
        sys.exit(1)
    
    try:
        # Create a mask dataset
        mem_drv = gdal.GetDriverByName('MEM')
        mask_ds = mem_drv.Create('', raster_ds.RasterXSize, raster_ds.RasterYSize, 1, gdal.GDT_Int8)
    except Exception as e:
        print("An error occurred while creating the mask dataset:", str(e))
        sys.exit(1)
    mask_ds.SetProjection(raster_ds.GetProjection())
    mask_ds.SetGeoTransform(raster_ds.GetGeoTransform())
    
    band = mask_ds.GetRasterBand(1)
    band.Fill(-1)  # Fill the raster with -1 to identify areas outside polygons
    
    try:
        # Rasterize the layer
        burn_value = -1
        rasterize_options = {'AllTouched': 1, 'BurnValues': [burn_value], 'Attribute': vector_group_field}
        gdal.RasterizeLayer(mask_ds, [1], layer, options=rasterize_options)
        mask_array = band.ReadAsArray()
    except Exception as e:
        print("An error occurred while rasterizing the layer:", str(e))
        sys.exit(1)
    
    unique_zones = np.unique(mask_array[mask_array != burn_value])
    zonal_statistics_dict = {}
    for unique_zone in unique_zones:
        zone_mask = mask_array == unique_zone
        if statistic_type == 'sum':
            zonal_statistic = np.sum(raster_array[zone_mask])
            zonal_statistics_dict[unique_zone] = zonal_statistic
        else:
            print("Invalid statistic type. Please use 'sum'.")
            raise ValueError
    
    return zonal_statistics_dict