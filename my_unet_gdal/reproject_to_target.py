from osgeo import gdal
from osgeo import osr
import sys
gdal.UseExceptions()

def reproject_to_target(input_path, output_path, tgt_srs, x_res, y_res, resample_alg=gdal.GRA_Bilinear, verbose=False):
    """
    Reproject the input to the output with target srs and resolution
    """
    # Open the source raster
    try:
        src_ds = gdal.Open(input_path)
        if verbose:
            print('Success to open raster {}'.format(input_path))
            input_geotransform = src_ds.GetGeoTransform()
            input_proj = src_ds.GetProjection()
            print('Input GeoTransform: ', input_geotransform)
            print('Input Projection: ', input_proj)
    except RuntimeError as e:
        print('Unable to open {}'.format(input_path))
        print(e)
        sys.exit(1)

    # Use gdal.Warp to reproject and resample the source raster
    warp_options = gdal.WarpOptions(format='GTiff', dstSRS=tgt_srs, xRes=x_res, yRes=y_res, creationOptions=['COMPRESS=LZW', 'PREDICTOR=2', 'TILED=YES'],
                                    resampleAlg=resample_alg, callback=gdal.TermProgress)
    try:
        tgt_ds = gdal.Warp(output_path, src_ds, options=warp_options)
        if verbose:
            print('Success to reproject {}'.format(input_path))
            output_geotransform = tgt_ds.GetGeoTransform()
            output_proj = tgt_ds.GetProjection()
            print('Output GeoTransform: ', output_geotransform)
            print('Output Projection: ', output_proj)
    except RuntimeError as e:
        print('Reprojection failed')
        print(e)
        sys.exit(1)

    tgt_ds = None  # Close and save the output dataset