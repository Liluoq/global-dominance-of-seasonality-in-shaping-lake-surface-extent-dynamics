from osgeo import gdal, ogr, osr
import numpy as np
import time
import sys
import rtree
import csv
import random
import os
from multiprocessing import Pool
from typing import Dict, Tuple

gdal.UseExceptions()

def create_grid(minx, maxx, miny, maxy, step, verbose=0):
    """
    Create a grid of cells with unique identifiers.

    Parameters:
    minx (float): The minimum x-coordinate of the grid.
    maxx (float): The maximum x-coordinate of the grid.
    miny (float): The minimum y-coordinate of the grid.
    maxy (float): The maximum y-coordinate of the grid.
    step (float): The step size for each cell in the grid.

    Returns:
    dict: A dictionary containing the grid cells with unique identifiers.
          Each grid cell is represented by a tuple of (minx, maxx, miny, maxy).
    """
    idx = 0
    grid = {}
    for x in np.arange(minx, maxx, step):
        for y in np.arange(miny, maxy, step):
            grid[idx] = (x, x + step, y, y + step)
            idx += 1
    if verbose == 2:
        print(f"Grid created with {len(grid)} cells.")
        print(f"Grid cell size: {step} x {step}")
        print(f"Grid cell extent: {minx} {maxx} {miny} {maxy}")
        print(f'Grids: {grid}')
    return grid

def create_in_memory_layer(source_layer, new_layer_name):
    """
    Create an in-memory layer and copy field definitions from the source layer.

    Parameters:
        source_layer (ogr.Layer): The source layer from which to copy field definitions.
        new_layer_name (str): The name of the new layer to be created.

    Returns:
        tuple: A tuple containing the in-memory data source and the new layer.

    """
    try:
        driver = ogr.GetDriverByName('MEMORY')
        dataSource = driver.CreateDataSource('memData')
        # Get spatial reference and geometry type from the source layer
        srs = source_layer.GetSpatialRef()
        geom_type = source_layer.GetLayerDefn().GetGeomType()
        # Create the new layer
        layer = dataSource.CreateLayer(new_layer_name, geom_type=geom_type, srs=srs)
    except Exception as e:
        print(f"Error creating in-memory layer {new_layer_name} from source: {e}")
        sys.exit(1)
    
    # Copy field definitions from the source layer to the new layer
    try:
        layer_defn = source_layer.GetLayerDefn()
        for i in range(layer_defn.GetFieldCount()):
            field_defn = layer_defn.GetFieldDefn(i)
            layer.CreateField(field_defn)
    except Exception as e:
        print(f"Error copying field definitions: {e}")
        sys.exit(1)
    
    return dataSource, layer

def filter_lakes_to_cells_and_create_layers(all_lakes_layer, grid, verbose=0):
    """
    Assign lakes to grid cells and create in-memory layers for each grid cell.

    Parameters:
    - all_lakes_layer: The layer containing all the lakes.
    - grid: A dictionary representing the grid cells.

    Returns:
    - cell_to_filtered_lakes_layer: A dictionary mapping each grid cell index to a tuple containing the in-memory layer and dataset.

    """
    if verbose != 0:
        print(f"Filtering lakes to grid cells and creating in-memory layers...")
    cell_to_filtered_lakes = {idx: [] for idx in grid.keys()}
    try:
        all_lakes_layer.ResetReading()
        for feature in all_lakes_layer:
            centroid = feature.GetGeometryRef().Centroid()
            x, y = centroid.GetX(), centroid.GetY()
            for idx, (minx, maxx, miny, maxy) in grid.items():
                if minx <= x <= maxx and miny <= y <= maxy:
                    cell_to_filtered_lakes[idx].append(feature.Clone())
                    break
        all_lakes_layer.ResetReading()
    except Exception as e:
        print(f"Error filtering lakes to grid cells: {e}")
        sys.exit(1)

    cell_to_filtered_lakes_layer = {}
    try:
        total_number_of_features = sum([len(features) for features in cell_to_filtered_lakes.values()])
        if verbose != 0:
            print(f"Total number of features: {total_number_of_features}")
        for idx, features in cell_to_filtered_lakes.items():
            if features:
                ds, new_layer = create_in_memory_layer(all_lakes_layer, f'Cell_{idx}')
                for feature in features:
                    new_layer.CreateFeature(feature)
                cell_to_filtered_lakes_layer[idx] = (ds, new_layer)
                if verbose != 0:
                    print(f"Cell {idx}: {len(features)} features")
            else:
                cell_to_filtered_lakes_layer[idx] = (None, None)
                if verbose != 0:
                    print(f"Cell {idx}: No features")
    except Exception as e:
        print(f"Error creating in-memory filtered layers: {e}")
        sys.exit(1)

    return cell_to_filtered_lakes_layer

def clip_raster_to_extent(src_raster_path, target_extent, output_raster_path, verbose=0):
    """
    Clips a raster to a specified extent and saves the clipped raster to a new file.

    Parameters:
    src_raster_path (str): The file path of the source raster.
    target_extent (tuple): The extent to which the raster should be clipped. It should be a tuple of four values representing [minX, maxX, minY, maxY].
    output_raster_path (str): The file path where the clipped raster should be saved.

    Returns:
    None
    """
    
    # Open the source raster
    try:
        src_raster = gdal.Open(src_raster_path)
        src_srs = src_raster.GetProjection()
    except Exception as e:
        print(f"Error opening source raster: {e}")
        sys.exit(1)
    
    try:
        minx, maxx, miny, maxy = target_extent
        translate_options = gdal.TranslateOptions(format='GTiff',
                                                    creationOptions=[
                                                        'COMPRESS=LZW',  # Use LZW compression
                                                        'PREDICTOR=2',   # Optimal for images with floating point or integer values
                                                        'TILED=YES',     # Organize the file in tiles rather than strips
                                                    ],
                                                    projWin=[minx, maxy, maxx, miny],
                                                    projWinSRS=src_srs,
                                                    noData=-9999)
        # Use gdal.Translate to clip the raster
        if verbose != 0:
            print(f"Clipping raster {src_raster_path} to extent: {target_extent}, and save to {output_raster_path}")
        
        gdal.Translate(
            output_raster_path,
            src_raster,
            options=translate_options
        )
    except Exception as e:
        print(f"Error clipping raster: {e}")
        sys.exit(1)

    src_raster = None  # Close the source raster
    
def rasterize_and_calculate_lake_area(water_classification_raster_path, lakes_shp_path, lake_id_field, 
                                      outside_value=-1, force_raster_binary_gt_threshold=None, force_raster_binary_eq_value=None, 
                                      verbose=0,
                                      save_lake_rasterization_path=None):
    """
    Rasterizes a lake layer and calculates the area of each lake based on a water classification raster.

    Args:
        water_classification_raster_ds (gdal.Dataset): The water classification raster dataset.
        lake_layer (ogr.Layer): The lake layer to be rasterized.
        lake_id_field (str): The attribute field used for rasterization.
        outside_value (int, optional): The value to be assigned to pixels outside the lake layer. Defaults to -1.

    Returns:
        dict: A dictionary containing the calculated area of each lake, where the keys are the lake IDs and the values are the corresponding areas.
    """
    if force_raster_binary_gt_threshold is not None and force_raster_binary_eq_value is not None:
        raise ValueError("force_raster_binary_gt_threshold and force_raster_binary_eq_value cannot be used together.")
    
    if verbose != 0:
        print(f'Calculating lake areas from water classification raster: {water_classification_raster_path}, using lake shapefile: {lakes_shp_path}')
    try:
        water_classification_raster_ds = gdal.Open(water_classification_raster_path)
    except Exception as e:
        print(f"Error opening water classification raster: {e}")
        sys.exit(1)
    try:
        geo_transform = water_classification_raster_ds.GetGeoTransform()
        pixel_area = abs(geo_transform[1] * geo_transform[5])
        if verbose != 0:
            print(f"Pixel area: {pixel_area}")
        rasterized_lake_ds = gdal.GetDriverByName('MEM').Create('', water_classification_raster_ds.RasterXSize, water_classification_raster_ds.RasterYSize, 1, gdal.GDT_Int32)
        rasterized_lake_ds.SetProjection(water_classification_raster_ds.GetProjection())
        rasterized_lake_ds.SetGeoTransform(geo_transform)
    except Exception as e:
        print(f"Error creating rasterized lake dataset: {e}")
        sys.exit(1)
    
    try:
        band = rasterized_lake_ds.GetRasterBand(1)
        band.SetNoDataValue(outside_value)
        band.Fill(outside_value)
    except Exception as e:
        print(f"Error setting raster band properties: {e}")
        sys.exit(1)
    
    try:
        lake_ds = ogr.Open(lakes_shp_path)
        lake_layer = lake_ds.GetLayer()
        if verbose != 0:
            layer_name = lake_layer.GetName()
            print(f"Rasterizing lake layer (name: {layer_name}) using field: {lake_id_field}, outside value: {outside_value}, number of features: {lake_layer.GetFeatureCount()}")
        gdal.RasterizeLayer(rasterized_lake_ds, [1], lake_layer, options=[f'ATTRIBUTE={lake_id_field}'])
        if save_lake_rasterization_path is not None:
            if verbose != 0:
                print(f"Saving rasterized lake layer to: {save_lake_rasterization_path}")
            geotiff_driver = gdal.GetDriverByName('GTiff')
            to_save_ds = geotiff_driver.CreateCopy(save_lake_rasterization_path, rasterized_lake_ds)
            to_save_ds = None
    except Exception as e:
        print(f"Error rasterizing layer: {e}")
        sys.exit(1)
    
    try:
        lake_id_array = band.ReadAsArray()
        water_classification_array = water_classification_raster_ds.GetRasterBand(1).ReadAsArray()
        if force_raster_binary_gt_threshold is not None:
            water_classification_array = water_classification_array >= force_raster_binary_gt_threshold
        elif force_raster_binary_eq_value is not None:
            water_classification_array = water_classification_array == force_raster_binary_eq_value
    except Exception as e:
        print(f"Error reading arrays from raster datasets: {e}")
        sys.exit(1)
    
    unique_lake_ids = np.unique(lake_id_array[lake_id_array != outside_value])
    print(len(unique_lake_ids))
    lake_id_to_index = {lake_id: idx for idx, lake_id in enumerate(unique_lake_ids)}
    lake_areas = np.zeros(len(unique_lake_ids))
    # Iterate through each pixel
    if verbose != 0:
        print(f"Calculating lake areas for {water_classification_raster_path} and {lakes_shp_path}")
    assert lake_id_array.shape == water_classification_array.shape, "Lake ID array and water classification array must have the same shape."
    
    start_time = time.time()
    # Vectorize the area calculation
    mask = (lake_id_array != outside_value) & (water_classification_array == 1)
    np.add.at(lake_areas, [lake_id_to_index[lid] for lid in lake_id_array[mask]], pixel_area)
    end_time = time.time()

    if verbose != 0:
        print(f"Time taken to calculate lake areas: {end_time - start_time:.2f} seconds for {len(lake_areas)} lakes.")
        
    # Invert the lake_id_to_index mapping to get index_to_lake_id
    index_to_lake_id = {index: lake_id for lake_id, index in lake_id_to_index.items()}

    # Initialize an empty dictionary to hold the final results
    final_lake_areas = {}

    # Populate the dictionary with lake_id as key and the corresponding area as value
    for index, area in enumerate(lake_areas):
        lake_id = index_to_lake_id[index]
        final_lake_areas[lake_id] = area
    
    return final_lake_areas

def build_spatial_index_from_layer(layer):
    index = rtree.index.Index()
    for fid, feature in enumerate(layer):
        geom = feature.GetGeometryRef()
        index.insert(fid, geom.GetEnvelope())
    return index

def build_spatial_index_from_ids_and_geoms(ids, geoms, interleaved=False):
    index = rtree.index.Index(interleaved=interleaved)
    for idx, geom in zip(ids, geoms):
        envelope = geom.GetEnvelope()  # Get the bounding box as a tuple (minX, maxX, minY, maxY)
        # Debugging print to check envelope values
        if envelope[0] > envelope[1] or envelope[2] > envelope[3]:
            raise ValueError(f"Invalid envelope for geometry with ID {idx}: {envelope}")
        index.insert(idx, envelope)
    return index

def vectorize_and_calculate_lake_area(water_classification_raster_path, lakes_shp_path, lake_id_field, outside_value=-1, lake_index=None, force_raster_binary_threshold=None, verbose=0, save_vectorized_raster_path=None):
    if verbose != 0:
        print(f'(Vectorization approach) Calculating lake areas from water classification raster: {water_classification_raster_path}, using lake shapefile: {lakes_shp_path}')
    # Read raster data    
    try:
        water_classification_raster_ds = gdal.Open(water_classification_raster_path)
    except Exception as e:
        print(f"Error opening water classification raster: {e}")
        sys.exit(1)
    # Read Vector data
    try:
        lake_ds = ogr.Open(lakes_shp_path)
        lake_layer = lake_ds.GetLayer()
    except Exception as e:
        print(f"Error opening lake shapefile: {e}")
        sys.exit(1)
    # Create a memory layer for rasterized lake data
    try:
        geo_transform = water_classification_raster_ds.GetGeoTransform()
        rasterized_lake_ds = gdal.GetDriverByName('MEM').Create('', water_classification_raster_ds.RasterXSize, water_classification_raster_ds.RasterYSize, 1, gdal.GDT_Int32)
        rasterized_lake_ds.SetProjection(water_classification_raster_ds.GetProjection())
        rasterized_lake_ds.SetGeoTransform(geo_transform)
    except Exception as e:
        print(f"Error creating rasterized lake dataset: {e}")
        sys.exit(1)
    # Initialize the rasterized lake raster band
    try:
        band = rasterized_lake_ds.GetRasterBand(1)
        band.SetNoDataValue(outside_value)
        band.Fill(outside_value)
    except Exception as e:
        print(f"Error setting raster band properties: {e}")
        sys.exit(1)
    # Rasterize the lake layer
    try:
        lake_ds = ogr.Open(lakes_shp_path)
        lake_layer = lake_ds.GetLayer()
        burn_values = [1]
        if verbose != 0:
            layer_name = lake_layer.GetName()
            print(f"Rasterizing lake layer (name: {layer_name}) using burn values: {burn_values}, outside value: {outside_value}, number of features: {lake_layer.GetFeatureCount()}")
        gdal.RasterizeLayer(rasterized_lake_ds, [1], lake_layer, burn_values=burn_values)
    except:
        print(f"Error rasterizing layer: {e}")
        sys.exit(1)
    # Mask raster data that is outside the lake layer
    try:
        lake_mask = band.ReadAsArray()
        water_classification_array = water_classification_raster_ds.GetRasterBand(1).ReadAsArray()
        if force_raster_binary_threshold is not None:
            water_classification_array = water_classification_array >= force_raster_binary_threshold
        water_classification_array[lake_mask == outside_value] = 0
    except Exception as e:
        print(f"Error reading arrays from raster datasets: {e}")
        sys.exit(1)
    # Create a new raster for the masked water classification
    masked_water_raster = gdal.GetDriverByName('MEM').Create('', water_classification_raster_ds.RasterXSize, water_classification_raster_ds.RasterYSize, 1, gdal.GDT_Float32)
    masked_water_raster.SetProjection(water_classification_raster_ds.GetProjection())
    masked_water_raster.SetGeoTransform(water_classification_raster_ds.GetGeoTransform())
    masked_water_band = masked_water_raster.GetRasterBand(1)
    masked_water_band.WriteArray(water_classification_array)
    masked_water_band.SetNoDataValue(np.nan)
    masked_water_band.FlushCache()
    # Vectorize the masked water classification raster
    vectorized_water_ds = ogr.GetDriverByName('Memory').CreateDataSource('vectorized_water')
    vectorized_water_layer = vectorized_water_ds.CreateLayer('vectorized_water', srs=lake_layer.GetSpatialRef())
    if verbose != 0:
        print("Vectorizing the masked water classification raster...")
    gdal.Polygonize(masked_water_band, None, vectorized_water_layer, -1, [], callback=None)
    
    # Save vectorized layer to a file
    if save_vectorized_raster_path:
        save_vector_layer_to_shapefile(vectorized_water_layer, save_vectorized_raster_path)
    # Build spatial index for the lake layer
    
    # Precompute and cache lake geometries and ids
    hylak_ids = [lake.GetField(lake_id_field) for lake in lake_layer]
    if not all(isinstance(hylak_id, int) for hylak_id in hylak_ids):
        print("Lake ID field must contain integer values.")
        sys.exit(1)
    min_hylak_id = min(hylak_ids)
    max_hylak_id = max(hylak_ids)
    lake_index_offset = min_hylak_id
    lake_areas = np.zeros(max_hylak_id - min_hylak_id + 1)
    lake_geometries = [None] * (max_hylak_id - min_hylak_id + 1)
    lake_layer.ResetReading()
    for lake_feature in lake_layer:
        hylak_id = lake_feature.GetField(lake_id_field)
        lake_geom = lake_feature.GetGeometryRef().Clone()
        lake_geometries[hylak_id - lake_index_offset] = lake_geom
    lake_layer.ResetReading()

    lake_geometries_no_none = [geom for geom in lake_geometries if geom is not None]
    assert len(lake_geometries_no_none) == len(hylak_ids), "Lake geometries and IDs must have the same length."

    # Precompute water centroids and process intersections
    if verbose != 0:
        print("Calculating lake areas...")
        
    vectorized_water_layer.ResetReading()
    num_water_bodies = vectorized_water_layer.GetFeatureCount()
    water_bodies_centroid = [water_body.GetGeometryRef().Centroid() for water_body in vectorized_water_layer]
    water_bodies_area = np.array([water_body.GetGeometryRef().GetArea() for water_body in vectorized_water_layer])
    water_bodies_index = np.array(range(num_water_bodies))
    water_bodies_not_allocated = np.ones(num_water_bodies, dtype=bool)
    
    for hylak_id, lake_geom in zip(hylak_ids, lake_geometries_no_none):
        lake_geom_envelope = lake_geom.GetEnvelope()
        # Find indices where water bodies have not been allocated yet
        for idx in np.where(water_bodies_not_allocated)[0]:  # This directly gives us the indices
            water_centroid = water_bodies_centroid[idx]
            if lake_geom_envelope[0] <= water_centroid.GetX() <= lake_geom_envelope[1] and lake_geom_envelope[2] <= water_centroid.GetY() <= lake_geom_envelope[3]:
                if lake_geom.Contains(water_centroid):
                    lake_areas[hylak_id - lake_index_offset] += water_bodies_area[idx]
                    water_bodies_not_allocated[idx] = False
    
    vectorized_water_layer.ResetReading()
    
    lake_areas_dict = {hylak_id: lake_areas[hylak_id - lake_index_offset] for hylak_id in hylak_ids}
    if verbose != 0:
        print(f"Calculated areas for {len(lake_areas_dict)} lakes.")
    
    return lake_areas_dict
    

def check_and_reproject_vector_srs_to_raster(raster_path, vector_path, output_path, verbose=0):
    try:
        raster_ds = gdal.Open(raster_path)
        raster_srs = osr.SpatialReference()
        raster_srs.ImportFromWkt(raster_ds.GetProjection())
        
        vector_ds = ogr.Open(vector_path)
        vector_layer = vector_ds.GetLayer()
        vector_srs = vector_layer.GetSpatialRef()
    except Exception as e:
        print(f"Error opening raster or vector data: {e}")
        sys.exit(1)
        
    if not raster_srs.IsSame(vector_srs):
        if verbose != 0:
            print("Spatial references do not match, reprojecting...")
        
        # Create the coordinate transformation
        coord_transform = osr.CoordinateTransformation(vector_srs, raster_srs)
        
        # Get the ESRI Shapefile driver
        try:
            driver = ogr.GetDriverByName('ESRI Shapefile')
        except Exception as e:
            print(f"Error getting ESRI Shapefile driver: {e}")
            sys.exit(1)
        
        # Create the output shapefile
        if os.path.exists(output_path):
            if verbose != 0:
                print(f"Output shapefile already exists, deleting...")
            driver.DeleteDataSource(output_path)
        try:
            out_ds = driver.CreateDataSource(output_path)
        except Exception as e:
            print(f"Error creating output shapefile: {e}")
            sys.exit(1)
        
        # Create the output layer
        try:
            out_layer = out_ds.CreateLayer('reprojected_layer', geom_type=vector_layer.GetGeomType(), srs=raster_srs)
        except Exception as e:
            print(f"Error creating output layer: {e}")
            sys.exit(1)
        
        # Copy the fields from the input layer to the output layer
        in_layer_defn = vector_layer.GetLayerDefn()
        for i in range(0, in_layer_defn.GetFieldCount()):
            field_defn = in_layer_defn.GetFieldDefn(i)
            out_layer.CreateField(field_defn)
        
        # Reproject each feature
        for feature in vector_layer:
            geom = feature.GetGeometryRef().Clone()
            geom.Transform(coord_transform)
            new_feature = ogr.Feature(out_layer.GetLayerDefn())
            new_feature.SetGeometry(geom)
            for i in range(feature.GetFieldCount()):
                new_feature.SetField(i, feature.GetField(i))
            out_layer.CreateFeature(new_feature)
            new_feature = None
        
        print(f"Reprojected vector layer saved to: {output_path}")
        
        # Cleanup
        out_ds = None
        return True
    else:
        if verbose != 0:
            print("Spatial references match; no reprojecting needed.")
        return False

def check_dict_overlap(dict_list):
    # Track seen keys and their areas
    seen_keys = {}

    # Track overlaps
    overlaps = {}

    for d in dict_list:
        for key, area in d.items():
            if key in seen_keys:
                # If the key is already seen, add it to overlaps
                if key not in overlaps:
                    overlaps[key] = [seen_keys[key]]  # Add the first occurrence
                overlaps[key].append(area)  # Add the current occurrence
            else:
                # Track the key and its area
                seen_keys[key] = area

    # Check for overlaps and print details
    if overlaps:
        print("Overlap detected. Review the following lake areas:")
        for key, areas in overlaps.items():
            print(f"{key}: {areas}")
    else:
        print("No overlaps detected.")

def save_extent_as_shapefile(extent, output_file, srs_wkt, verbose=0):
    """
    Create a shapefile from an extent using a specified WKT for the spatial reference.

    Parameters:
        extent (tuple): The extent tuple (minx, maxx, miny, maxy).
        output_file (str): Path to the output shapefile.
        srs_wkt (str): Well-Known Text string defining the spatial reference system.
    """
    if verbose != 0:
        print(f"Creating shapefile {output_file} from extent {extent}...")
    try:
        # Set up the spatial reference from a WKT
        srs = osr.SpatialReference()
        srs.ImportFromWkt(srs_wkt)

        # Create the shapefile
        driver = ogr.GetDriverByName('ESRI Shapefile')
        if driver is None:
            raise RuntimeError("Shapefile driver not available.")

        # Check if the file exists; if so, delete it
        if ogr.Open(output_file, 0) is not None:
            driver.DeleteDataSource(output_file)

        ds = driver.CreateDataSource(output_file)
        if ds is None:
            raise RuntimeError("Creation of output file failed.")

        layer = ds.CreateLayer(output_file, srs, ogr.wkbPolygon)
        if layer is None:
            raise RuntimeError("Layer creation failed.")

        # Define and create the geometry from the extent
        ring = ogr.Geometry(ogr.wkbLinearRing)
        minx, maxx, miny, maxy = extent
        ring.AddPoint(minx, miny)
        ring.AddPoint(minx, maxy)
        ring.AddPoint(maxx, maxy)
        ring.AddPoint(maxx, miny)
        ring.AddPoint(minx, miny)

        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)

        # Create a new feature
        feature_def = layer.GetLayerDefn()
        feature = ogr.Feature(feature_def)
        feature.SetGeometry(poly)

        # Add the feature to the layer
        if layer.CreateFeature(feature) != 0:
            raise RuntimeError("Failed to create feature in shapefile.")

        feature = None
        ds = None
        print(f"Shapefile {output_file} created successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")

def save_vector_layer_to_shapefile(layer, output_path, verbose=0):
    """
    Save an OGR layer to a new Shapefile.

    Parameters:
        layer (ogr.Layer): The input vector layer to save.
        output_path (str): The path where the new Shapefile will be saved.
    """
    if verbose != 0:
        print(f"Saving layer to shapefile: {output_path}")
    try:
        driver = ogr.GetDriverByName('ESRI Shapefile')
        if os.path.exists(output_path):
            driver.DeleteDataSource(output_path)  # Delete any existing data source

        # Create a new data source at the specified path
        data_source = driver.CreateDataSource(output_path)
        new_layer = data_source.CreateLayer(layer.GetName(), geom_type=layer.GetGeomType())

        # Copy fields from the old layer to the new layer
        layer_defn = layer.GetLayerDefn()
        for i in range(layer_defn.GetFieldCount()):
            new_layer.CreateField(layer_defn.GetFieldDefn(i))

        # Copy features from the old layer to the new layer
        for feature in layer:
            new_layer.CreateFeature(feature.Clone())

        data_source = None  # Ensure data is written and resources are released
        print(f"Layer saved to {output_path}")
    except Exception as e:
        print(f"Failed to save layer to shapefile: {e}")

def calculate_lake_area_grid_parallel(water_classification_raster_ds_paths: list, lake_shape_path: str, 
                                      raster_clip_temporary_folders: str, temporary_vector_folder: str, 
                                      output_csv_paths: list, lake_id_field: str, grid_size: float, 
                                      method: str, outside_value: int = -1, 
                                      force_raster_binary_threshold=None, force_raster_binary_eq_value=None, 
                                      num_processes: int = 1, check_overlap: bool = False, filter_once: bool = True, 
                                      verbose=0, remove_temporary=True, save_lake_rasterization=False) -> list:
    """
    Rasterizes and calculates the area of lakes within each grid cell in parallel.

    Args:
        water_classification_raster_ds_path (str): The file path of the water classification raster dataset.
        lake_shape_path (str): The file path of the lake shapefile.
        raster_clip_temporary_folder (str): The folder path to store the clipped raster files.
        lake_id_field (str): The field name to use for rasterization.
        grid_size (float): The size of each grid cell.
        outside_value (int, optional): The value to assign to cells outside the lake boundaries. Defaults to -1.
        num_processes (int, optional): The number of processes to use for parallel execution. Defaults to 1.
        check_overlap (bool, optional): Flag indicating whether to check for overlap between lake areas. Defaults to False.

    Returns:
        dict: A dictionary containing the merged lake areas for each grid cell.
    """
    if method not in ['rasterize_vector', 'vectorize_raster']:
        print("Method must be either 'rasterize_vector' or 'vectorize_raster'.")
        sys.exit(1)
    
    filtered = False
    rasterized_already_saved = False
    merged_lake_areas_list = []
    assert len(water_classification_raster_ds_paths) == len(output_csv_paths), "The number of raster datasets and output CSV paths must be the same."
    for water_classification_raster_ds_path, raster_clip_temporary_folder, output_csv_path in zip(water_classification_raster_ds_paths, raster_clip_temporary_folders, output_csv_paths):
        if verbose != 0:
            print(f"Processing raster: {water_classification_raster_ds_path}")
        # open raster and vector data
        try:
            water_classification_raster_ds = gdal.Open(water_classification_raster_ds_path) # open the raster data
            water_classification_raster_srs = osr.SpatialReference(water_classification_raster_ds.GetProjection()) # read the spatial reference system of the raster
            lake_ds = ogr.Open(lake_shape_path) # open the vector data
            lake_layer = lake_ds.GetLayer() # get the first layer of the vector data, which is supposed to be the lake layer
            lake_srs = lake_layer.GetSpatialRef() # read the spatial reference system of the vector data
        except Exception as e:
            # if there is an error opening the raster or vector data, print the error message and exit the program
            print(f"Error opening raster or vector data: {e}")
            sys.exit(1)
        # check if the spatial reference systems of the raster and vector data match
        # it is important that the raster and vector data have the same spatial reference system
        assert water_classification_raster_srs.IsSame(lake_srs), "Raster and vector data must have the same spatial reference system."
        
        if filter_once and filtered:
            pass
        else:
            if verbose != 0:
                print(f"Filtering lakes to grid cells and creating in-memory layers...")
            # create a grid of cells with unique identifiers, which will be used to divide the lakes into subgroups 
            lake_layer_extent = lake_layer.GetExtent()
            minx, maxx, miny, maxy = lake_layer_extent # get the extent of the lake layer, order is xmin, xmax, ymin, ymax
            grid = create_grid(minx, maxx, miny, maxy, grid_size, verbose=verbose) # order is important
            cells_to_filtered_ds_and_layers = filter_lakes_to_cells_and_create_layers( # filter the lakes to the grid cells and create in-memory layers for each subgroups of lakes
                all_lakes_layer=lake_layer,
                grid=grid,
                verbose=verbose
            )
            # update the grid dictionary with the new extent of the grid cells
            grid_having_lakes = {}
            cells_having_lakes_to_filtered_ds_and_layers = {}
            for idx in grid.keys():
                current_ds, current_layer = cells_to_filtered_ds_and_layers[idx]
                if current_ds is not None and current_layer is not None:
                    current_layer_extent = current_layer.GetExtent()
                    current_minx, current_maxx, current_miny, current_maxy = current_layer_extent
                    current_new_grid_cell = (current_minx, current_maxx, current_miny, current_maxy)
                    grid_having_lakes[idx] = current_new_grid_cell
                    cells_having_lakes_to_filtered_ds_and_layers[idx] = (current_ds, current_layer)
            print(grid_having_lakes)
                            
            grid_cells_shp_temporary_folder = os.path.join(temporary_vector_folder, 'grid_cells_shp')
            if not os.path.exists(grid_cells_shp_temporary_folder):
                os.makedirs(grid_cells_shp_temporary_folder)
            for idx, extent in grid_having_lakes.items():
                current_grid_cell_shp_filename = f"grid_cell_{idx}.shp"
                current_grid_cell_shp_path = os.path.join(grid_cells_shp_temporary_folder, current_grid_cell_shp_filename)
                save_extent_as_shapefile(extent, current_grid_cell_shp_path, water_classification_raster_srs.ExportToWkt(), verbose=verbose)
            
            filtered_lakes_shp_temporary_folder = os.path.join(temporary_vector_folder, 'filtered_lakes_shp')
            if not os.path.exists(filtered_lakes_shp_temporary_folder):
                os.makedirs(filtered_lakes_shp_temporary_folder)
            cells_having_lakes_to_filtered_lake_shp_paths = {}
            save_lake_rasterization_paths = {}
                
            for idx, (_, filtered_lake_layer) in cells_having_lakes_to_filtered_ds_and_layers.items():
                current_filtered_lakes_shp_filename = f"filtered_lakes_{idx}.shp"
                current_filtered_lakes_shp_path = os.path.join(filtered_lakes_shp_temporary_folder, current_filtered_lakes_shp_filename)
                cells_having_lakes_to_filtered_lake_shp_paths[idx] = current_filtered_lakes_shp_path
                if save_lake_rasterization and not rasterized_already_saved:
                    current_lake_rasterization_filename = f"filtered_lakes_rasterized_{idx}.tif"
                    current_lake_rasterization_path = os.path.join(filtered_lakes_shp_temporary_folder, current_lake_rasterization_filename)
                    save_lake_rasterization_paths[idx] = current_lake_rasterization_path
                else:
                    current_lake_rasterization_path = None
                save_vector_layer_to_shapefile(filtered_lake_layer, current_filtered_lakes_shp_path, verbose=verbose)
            
            filtered = True
        
        # create a list of arguments for the clip_raster_to_extent function and the rasterize_and_calculate_lake_area function
        water_classification_raster_ds_path_basename = os.path.basename(water_classification_raster_ds_path)
        water_classification_raster_ds_path_basename_noext = os.path.splitext(water_classification_raster_ds_path_basename)[0]
        if not os.path.exists(raster_clip_temporary_folder):
            os.makedirs(raster_clip_temporary_folder)
        clipped_raster_paths = {idx: os.path.join(raster_clip_temporary_folder, f"{water_classification_raster_ds_path_basename_noext}_clipped_{idx}.tif") for idx in grid_having_lakes.keys()}
        clip_args = [(water_classification_raster_ds_path, grid_cell, clipped_raster_paths[idx], verbose) for idx, grid_cell in grid_having_lakes.items()]
        if method == 'rasterize_vector':
            calculate_lake_area_args = [
                (clipped_raster_paths[idx], cells_having_lakes_to_filtered_lake_shp_paths[idx], lake_id_field, outside_value, force_raster_binary_threshold, force_raster_binary_eq_value, verbose, save_lake_rasterization_paths[idx]) for idx in cells_having_lakes_to_filtered_lake_shp_paths.keys()
            ]
            calculate_area_function = rasterize_and_calculate_lake_area
        elif method == 'vectorize_raster':
            lake_index = None
            calculate_lake_area_args = [
                (clipped_raster_paths[idx], cells_having_lakes_to_filtered_lake_shp_paths[idx], lake_id_field, outside_value, lake_index, force_raster_binary_threshold, verbose) for idx in cells_having_lakes_to_filtered_lake_shp_paths.keys()
            ]
            calculate_area_function = vectorize_and_calculate_lake_area
        # parallel execution (num_process > 1)
        if num_processes > 1:
            with Pool(num_processes) as pool:
                pool.starmap(clip_raster_to_extent, clip_args) 
            
            random.shuffle(calculate_lake_area_args)
            
            with Pool(num_processes) as pool:
                lake_areas_list = pool.starmap(calculate_area_function, calculate_lake_area_args) # contains a list of dictionaries that is rasterization field: area
        # serial execution (num_process = 1)
        else:
            for clip_arg in clip_args:
                clip_raster_to_extent(*clip_arg)
            
            lake_areas_list = []
            for calculate_lake_area_arg in calculate_lake_area_args:
                lake_areas_list.append(calculate_area_function(*calculate_lake_area_arg))
        # remove temporary files
        if remove_temporary:
            for clipped_raster_path in clipped_raster_paths.values():
                if verbose != 0:
                    print(f"Removing temporary file: {clipped_raster_path}")
                os.remove(clipped_raster_path)
        # check for overlap and print details
        if check_overlap: 
            if verbose != 0:
                print("Checking for overlap between lake areas...")
            check_dict_overlap(lake_areas_list)
        #merge the lake areas for each grid cell into one dictionary
        merged_lake_areas = {**{k: v for d in lake_areas_list for k, v in d.items()}}
        
        with open(output_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)

            # Write the header
            writer.writerow(['Hylak_id', 'area'])

            # Write the data
            for hylak_id, area in merged_lake_areas.items():
                writer.writerow([hylak_id, area])
        
        merged_lake_areas_list.append(merged_lake_areas)
        rasterized_already_saved = True
        # change save_lake_rasterization_paths to key:None
        save_lake_rasterization_paths = {key: None for key in save_lake_rasterization_paths.keys()}
    
    return merged_lake_areas_list
