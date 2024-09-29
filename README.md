# Global Lake Surface Extent Dynamics

This repository contains all necessary codes for producing datasets and reproducing results for the manuscript "Global dominance of seasonality in shaping lake surface extent dynamics" (in review). Because the code for this manuscript is computationally intensive and requires a complex runtime environment, we have prepared a Docker image that should be set up on a local high-performance computer to run these analyses.

## Environment Configuration

Docker is used to reproduce all related contents in this manuscript, ensuring identical runtime environments and saving time. The corresponding image can be pulled following instructions in the [global-lake-area-runner](https://hub.docker.com/repository/docker/luoqili/global-lake-area-runner/general) DockerHub repository.

In VS Code, the [Remote Development](https://code.visualstudio.com/docs/remote/remote-overview) extension is required to use Docker as the runtime environment. Download the **_`/code/global_lake_area`_** directory in its entirety. Modify the "mount" parameter in the **_`./.devcontainer/devcontainer.json`_** file by replacing _`path/to`_ with the actual paths according to your file system. For example, on the authors' machine, these paths are:  

> _`{"source": "D:\\OneDrive\\Research\\global_lake_area", "target": "/WORK/Codes", "type": "bind"}`_ (the downloaded GitHub repository should be decompressed to a folder, and this folder should be bind to the _`/WORK/Codes`_ folder in the virtual environment)
> _`{"source": "E:\\data\\", "target": "/WORK/Data", "type": "bind"}`_ (this folder contains data downloaded from the Zenodo repository mentioned below, and is necessary for reproducing the figures and key numbers)
> _`{"source": "D:\\data\\", "target": "/WORK/SSD_Data", "type": "bind"}`_ (this folder can be ignored when only reproducing quantitative figures and numbers in the manuscript; to ignore this, set it to a empty folder)

The directory structures of the downloaded code and data should be maintained, so that the paths in the code do not have to be modified accordingly. 

Once completed, open the **_`global_lake_area`_** folder in VS Code and click _`Reopen in container`_. After a successful build, you'll be able to run scripts in the pre-configured environment.

## Data Availability

All data can be generated using corresponding scripts. Final datasets that are used for analysis are separately hosted in a [Zenodo repository](https://zenodo.org/records/13846979?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjExMjAxMGZkLWI1ZmUtNDUxNi1iZjM0LTIxY2UyYjAxODJkOSIsImRhdGEiOnt9LCJyYW5kb20iOiJkNjM5MDZhODc5MDQzYWQ4ZTBiN2ZlM2I4MmUzM2U5OSJ9.Q0iP4WkIHhZbwBbjHJekwybTDHUHTvqmwLvzQmE76bp2isuZEblJ2pIIIlfRG5RSnWlENMVbAGBZ3-y_8x1eLA) for peer review .

The `lakes_all_with_all_additional_attributes.pkl` and corresponding `.csv` file should be the complete version of monthly lake surface extent for ~1.4 million lakes globally. Other similar files in obtaining quantitive results should be its subset and therefore can be replaced by it.

## Locations of quantitative results in the codes

For best reproducibility, codes in this repository should be downloaded to a _`global_lake_area`_ parent directory, all descriptions below follow this convention.

1. **Fig. 1**: No quantitative data.

2. **Fig. 2**: Raw fig are generated in _`global_lake_area/my_spatial_analyze/data_analyze/extreme_analysis/low_water_extreme_plotting.ipynb`_. Fonts, layout, and sizes are further polished in Adobe Illustrator.

3. **Fig. 3**: Raw fig are generated in _`global_lake_area/my_spatial_analyze/data_analyze/grid_wise_analysis/gird_wise_plotting.ipynb`_. Fonts, layout, and sizes are further polished in Adobe Illustrator.

4. **Fig. 4**: Raw fig are generated in _`global_lake_area/my_spatial_analyze/data_analyze/time_series_analysis/time_series_plotting.ipynb`_. Fonts, layout, and sizes are further polished in Adobe Illustrator.

5. **Fig. 5**: Raw fig are generated in _`global_lake_area/my_spatial_analyze/data_analyze/extreme_analysis/low_water_extreme_plotting.ipynb`_. Fonts, layout, and sizes are further polished in Adobe Illustrator.

6. **Extended Data Fig. 1**: Raw fig are generated in _`global_lake_area/my_spatial_analyze/data_validation`_. Fonts, layout, and sizes are further polished in Adobe Illustrator.

7. **Extended Data Fig. 2**: Raw fig are generated in _`global_lake_area/my_spatial_analyze/lake_wise_plotting.ipynb`_ and _`global_lake_area/my_spatial_analyze/data_analyze/correlation_analysis/plotting.ipynb`_. Fonts, layout, and sizes are further polished in Adobe Illustrator.

8. **Extended Data Fig. 3**: No quantitative data.

9. **Extended Data Fig. 4**: No quantitative data.

10. **Extended Data Fig. 5**: Raw fig are generated in _`global_lake_area/my_plotting`_. Fonts, layout, and sizes are further polished in Adobe Illustrator.

11. **Supplementary Fig. 1**: Raw fig are generated in _`global_lake_area/my_spatial_analyze/basin_sise_analysis`_. Fonts, layout, and sizes are further polished in Adobe Illustrator.

12. **Supplementary Fig. 2**: Raw fig are generated in _`global_lake_area/my_spatial_analyze/lake_wise_plotting.ipynb`_. Fonts, layout, and sizes are further polished in Adobe Illustrator.

13. **Supplementary Fig. 3**: No quantitative data.

14. **Supplementary Fig. 4**: Raw fig are generated in _`global_lake_area/my_spatial_analyze/main_grid.py`_. Fonts, layout, and sizes are further polished in Adobe Illustrator.

15. **Supplementary Fig. 5**: Raw fig are generated in _`global_lake_area/my_spatial_analyze/data_validation`_. Fonts, layout, and sizes are further polished in Adobe Illustrator.

16. **Supplementary Table 1**: Data are obtained in the _`global_lake_area/training_records.csv`_ as described in the "File Description and Usage" section.

17. **Others**: numbers and percentages can be mostly found in _`global_lake_area/my_spatial_analyze/data_analyze/extreme_analysis/low_water_extreme_plotting.ipynb`_. Others can be found following the "File Description and Usage" section.

## File Description and Usage

A brief description for each file is provided below, with necessary, detailed usage added in the files.

1. **_`global_lake_area/`_** (folder)

    - **_`unetgee.py`_**: Contains functions for GEE authentication, U-Net sample generation, training, validation, MODIS and GSW raster export, and U-Net prediction.
    - **_`unet_train.py`_**: Calls the _`unet_train`_ function in _`unetgee.py`_, acting as a command-line interface for U-Net training.
    - **_`UNET_TRAIN_CONFIG.py`_**: Configuration settings for a single U-Net training.
    - **_`update_config_unet_train_run.py`_**: Used for batch U-Net training, updating _`UNET_TRAIN_CONFIG.py`_ and calling _`unet_train.py`_.
    - **_`training_records.csv`_**: Contains metadata for the U-Net models, such as sample sizes and model metrics.
    - **_`update_training_record.py`_**: Used for updating the _`training_records.csv`_ file.
    - **_`unet_samples_generate_per_basin.ipynb`_**: Notebook for exporting samples for U-Net training for each basin.
    - **_`unet_sample_size_count.ipynb`_**: Calculates sample sizes for U-Net training, evaluation, and validation; updates the _`training_records.csv`_ file.
    - **_`unet_evaluation.py`_**: Similar to _`unet_train.py`_, calculates performance metrics for each U-Net model and updates the _`training_records.csv`_ file.
    - **_`UNET_EVALUATION_CONFIG.py`_**: Configuration settings for a single U-Net evaluation.
    - **_`unet_evaluation_update_config_and_run.py`_**: Similar to _`update_config_unet_train_run.py`_, used for batch performance metrics calculation.
    - **_`selfee.py`_**: Defines service-account-related methods for automated authentication to solve network issues.
    - **_`projection_wkt_generation.ipynb`_**: Constructs customized Lambert Azimuth Equal Area (LAEA) projections for each basin in BasinATLAS lev02 product.
    - **_`hydrolakes_filter_by_bas.ipynb`_**: Exports lake boundaries for U-Net sample generation (not for final area calculation).
    - **_`gsw_export.ipynb`_**: Exports the occurrence and recurrence of GSW.
    - **_`gsw_occurrence_and_recurrence_mosaic.py`_**: Mosaics tiled GSW occurrence and recurrence maps.
    - **_`export_modis_and_gsw_image.ipynb`_**: Used for exporting MODIS and GSW images in LAEA projections and correct resolutions.
    - **_`draw_unet_train_history.py`_**: Draws training and validation curves for each U-Net model.
    - **_`add_final_decision_to_records.py`_**: Records the manually-selected optimal epoch to the _`training_records.csv`_ file.

---

2. **_`global_lake_area/.devcontainer`_** (folder)

Contains the _`devcontainer.json`_ file that defines the container-based runtime environment for this manuscript.

---

3. **_`global_lake_area/my_unet_definition`_** (folder)

    - **_`__init__.py`_**: Makes this folder a module.
    - **_`model.py`_**: Contains implementation of U-Net models (_`attentionunet`_ was used).
    - **_`evaluation_metrics.py`_**: Contains performance metrics and loss functions used (Intersection over Union, IoU).

---

4. **_`global_lake_area/my_unet_gdal`_** (folder)

    - **_`__init__.py`_**: Makes this folder a module.
    - **_`reproject_to_target.py`_**: Deprecated.
    - **_`combined.py`_**: Deprecated.
    - **_`zonal_statistics.py`_**: Deprecated.
    - **_`reproject_to_target_tile.py`_**: Contains functions for clipping, reprojecting, and mosaicing large geotiff files.
    - **_`generate_tfrecord_from_tile.py`_**: Contains functions for reprojecting, resampling, and converting geotiff files to tfrecord format.
    - **_`align_to_target_tile.py`_**: Contains functions for geographically aligning and combining two rasters.
    - **_`unet_predictions.py`_**: Contains functions for using trained U-Net models to process converted tfrecords.
    - **_`reconstruct_tile_from_prediction.py`_**: Contains functions for converting serialized tfrecord files back to geotiff tiles.
    - **_`area_calculation.py`_**: Contains functions for calculating areas from rasters using vector data as boundaries.
    - **_`quick_plotting.py`_**: Contains functions for drawing PNGs and GIFs from geotiff files.
    - **_`quick_plotting_runner.py`_**: Command-line interface for _`quick_plotting.py`_, takes coordinates in LAEA as input.

---

5. **_`global_lake_area/batch_processing`_** (folder)

    - **_`__init__.py`_**: Makes this folder a module.
    - **_`batch_tfrecord_generation.py`_**: Command-line interface for batch generation of MODIS-converted tfrecord files.
    - **_`batch_unet_prediction.py`_**: Command-line interface for batch prediction using U-Net.
    - **_`batch_prediction_reconstruction.py`_**: Command-line interface for batch reconstruction of water mask maps.
    - **_`batch_mosaic.py`_**: Mosaics water mask tiles into a large geotiff file.
    - **_`batch_full.py`_**: Combines multiple batch processing steps into one command-line interface.
    - **_`asynchronous_batch.py`_**: Asynchronously calls _`batch_full.py`_ to maximize usage of available computing resources.
    - **_`BATCH_CONFIG.py`_**: Configurations for _`batch_full.py`_ and _`asynchronous_batch.py`_.
    - **_`batch_area_calculation.py`_**: Command-line interface for batch area calculation from mosaiced water mask maps.
    - **_`AREA_CALCULATION_CONFIG.py`_**: Configurations for _`batch_area_calculation.py`_ (monthly lake surface extent results).
    - **_`MISSING_DATA_AREA_CALCULATION_CONFIG.py`_**: Configurations for _`batch_area_calculation.py`_ (monthly cloud contamination ratio results).
    - **_`MASKED_MY_WATER_AREA_CALCULATION_CONFIG.py`_**: Configurations for _`batch_area_calculation.py`_ (GSW-masked water mask map results).
    - **_`GSWR_AREA_CALCULATION_CONFIG.py`_**: Configurations for _`batch_area_calculation.py`_ (GSW image results for validation).
    - **_`area_calculation_update_and_run.py`_**: Updates config files automatically and runs _`batch_area_calculation.py`_.
    - **_`load_config_module.py`_**: Used for reading config files written in _`.py`_ format.

---

6. **_`global_lake_area/my_plotting`_** (folder)

Contains scripts for plotting the performances of U-Net models trained in this study.

---

7. **_`global_lake_area/my_spatial_analyze`_** (folder)

    - **_`__init__.py`_**: Makes this folder a module.
    - **_`area_postprocessing.py`_**: Contains functions for post-processing lake surface water extracted from U-Net-generated water mask maps.
    - **_`lake_wise_area_postprocessor.py`_**: Command-line interface for lake-wise postprocessing of lake surface extent time series.
    - **_`LAKE_WISE_AREA_POSTPROCESSING_CONFIG.py`_**: Configurations for _`lake_wise_area_postprocessor.py`_.
    - **_`lake_wise_area_postprocess_update_and_run.py`_**: Updates config files automatically and runs _`lake_wise_area_postprocessor.py`_.
    - **_`lake_wise_lse_analyze.py`_**: Contains functions used for lake-wise plotting.
    - **_`lake_wise_plotting.ipynb`_**: Plotting for SI Fig. 2.
    - **_`visualization.py`_**: Contains functions related to grid-wise plotting.
    - **_`main_grid.py`_**: Explorative grid-wise plotting (deprecated).
    - **_`lake_concatenator.py`_**: Combines lake-wise time series of lake surface extent in each basin into one large file for global 1.4 million lakes.
    - **_`glake_update_hydrolakes.py`_**: Contains functions and command-line interface for updating HydroLAKES using GLAKES.
    - **_`hylak_buffering.py`_**: Removes duplicated lakes and creates buffer zones for GLAKES-updated HydroLAKES.
    - **_`gsw_image_mosaic.py`_**: Command-line interface for mosaicing tiled GSW images into one large geotiff file for validation.
    - **_`grid_concatenator.py`_**: Deprecated.
    - **_`grid_analyze.py`_**: Contains functions that perform a subset of grid-level analysis.
    - **_`cloud_cover_ratio_calculater.py`_**: Command-line interface for calculating cloud cover ratios based on boundary size and monthly MODIS cloud-contamination area.
    - **_`basin_lse_calculation.py`_**: Deprecated.
    - **_`attach_geometry_and_generate_grid.py`_**: Contains functions that create grid from global (or regional) lakes and calculate corresponding statistics.
    - **_`area_to_volume.py`_**: Deprecated.
    - **_`area_to_level.py`_**: Deprecated.
    - **_`AREA_TO_LEVEL_CONFIG.py`_**: Deprecated.
    - **_`area_to_level_batch_converter.py`_**: Deprecated.
    - **_`./data_analyze`_** (folder)
        - **_`./basin_wise_analysis`_** (folder)
            - **_`basin_wise_analysis.py`_**: Contains functions for basin-wise analysis and plotting.
            - **_`basin_wise_plotting.ipynb`_**: Plotting basin-wise figures (including reservoir contribution).
            - **_`basinatlas_statistics_calculator.py`_**: Command-line interface for calculating statistics for BasinATLAS.
            - **_`hydrobasins_merger.py`_**: Merges multiple shp files of HydroBASINS.
            - **_`hydrobasins_statistics_calculator.py`_**: Command-line interface for calculating statistics for HydroBASINS.
        - **_`./climate_analysis`_** (folder)
            - **_`attach_aridity_index.py`_**: Adds aridity index from LakeATLAS to the time series csv of lake surface extent.
        - **_`./correlation_analysis`_** (folder)
            - **_`plotting.ipynb`_**: Plots median relative changes in seasonality by lake size (part of Extended Data Fig. 1).
            - **_`correlation_plots.py`_**: Contains functions for plotting the relationship between multiple variables.
        - **_`./extreme_analysis`_** (folder)
            - **_`area_extreme_analysis.py`_**: Contains functions that identify seasonality-induced low-water extremes and perform other analyses.
            - **_`low_water_extreme_analysis.ipynb`_**: Adds extreme-related columns to the time series csv of lake surface extent.
            - **_`low_water_extreme_plotting.ipynb`_**: Plots seasonality-induced low-water extremes and seasonality-dominance.
        - **_`./grid_wise_analysis`_** (folder): Contains plotting of changes in seasonality.
        - **_`./permafrost_analysis`_** (folder): Adds permafrost type column to the time series csv of lake surface extent.
        - **_`./time_series_analysis`_** (folder): Plots long-term trends.
    - **_`./data_validation`_** (folder): Validates our data using GSW estimates and altimetry-based water levels.

---

8. **_`global_lake_area/projection_wkt`_** (folder)

Contains LAEA projections used in this manuscript.
