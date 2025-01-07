# Global Lake Surface Extent Dynamics

This repository contains the complete codebase for producing datasets and reproducing results from the manuscript "Global dominance of seasonality in shaping lake surface extent dynamics". Due to the computationally intensive nature of the analysis and complex runtime environment requirements, we provide a Docker image for deployment on local high-performance computing systems.

**Our code is available in two formats:**  

**1. Code Ocean:** Provides single-click reproducibility for generating quantitative figures and key numbers from the manuscript.    Quantitative figures are named correspondingly, and key numbers can be found in `/results/low_water_extreme_plotting.ipynb`. Note that Extended Data Fig.1 and 2 is excluded due to quota limitations.

> **_Estimated runtime on Code Ocean: ~ 15 minutes_**

**2. [GitHub](https://github.com/Liluoq/global-dominance-of-seasonality-in-shaping-lake-surface-extent-dynamics):** Contains the full version of code and provides a docker image that guarantees identical development environment for reproducibility. **(Full version for detailed examination)**

> **_Estimated time for full reproducibility: several months_**

---

**Important Notes:**
- **Code Ocean Users:** Simply click the `Reproducible Run` button to execute the analysis. All results will be available in the `/results` directory.

- **GitHub Users:** Please follow the detailed setup instructions provided below to run the analysis locally.

## Reproducing Results (using GitHub version)

### System Requirements

Some of the code requires loading large datasets into RAM. A minimum of 64 GB RAM is required (tested on Windows and Linux).

**Note: Insufficient RAM may cause the program to crash.**

### Folder preparation

This manuscript's code must be run within a Docker container to ensure a consistent environment. Since folders from your host machine will be mounted in the container, you'll need to download and organize the code and data in specific directories (**necessary modification of the paths in scripts is also required if error occurs since codes are modified to fit with Code Ocean**).

1. Choose a location on your local machine with at least 50 GB of available storage space. We'll refer to this location as _`your_path`_. Note that the path convention uses backslashes ('\\') for Windows and forward slashes ('/') for Linux and MacOS.
2. Create two sub-folders: _`your_path\\code`_ and _`your_path\\data`_
3. Follow the instructions below to download the necessary code and data files.

### Docker Installation  

Docker is required to reproduce the contents of this manuscript, as it ensures a consistent runtime environment and streamlines the setup process. The Docker image can be pulled from the [global-lake-area-runner](https://hub.docker.com/repository/docker/luoqili/global-lake-area-runner/general) DockerHub repository.

> For Docker Desktop/Engine installation, please refer to the official documentation:
>> [Windows](https://docs.docker.com/desktop/install/windows-install/)  
>> [MacOS](https://docs.docker.com/desktop/install/mac-install/)  
>> [Linux](https://docs.docker.com/engine/install/)

To automatically download the Docker image, run the following command in your terminal (use Terminal for MacOS/Linux or PowerShell/Terminal for Windows):
> `docker pull luoqili/global-lake-area-runner:v1.0`

### Container Creation

#### Install VS Code and Extensions
1. Download VS Code from their [official website](https://code.visualstudio.com/download).
2. Install the [Remote Development](https://code.visualstudio.com/docs/remote/remote-overview) extension in VS Code, which is required to run Docker containers as development environments.

#### Open Project Folder in VS Code
1. Launch VS Code.
2. Click File > Open Folder.
3. Navigate to and select _`your_path\\code\\global_lake_area`_, which contains all the project code.

#### Reopen in Container
1. Verify that the _`.devcontainer`_ folder appears in the left panel. If not, review the previous steps.
2. Edit the _`.devcontainer/devcontainer.json`_ file to configure the correct mounting paths. Modify the "mount" parameter by replacing _`your_path`_ with your actual path in these locations:
    > _`{"source": "your_path\\code", "target": "/WORK/Codes", "type": "bind"}`_ (mount the decompressed GitHub repository folder to _`/WORK/Codes`_ in the container)  
    > _`{"source": "your_path\\data", "target": "/WORK/Data", "type": "bind"}`_ (mount the data folder containing required data files (possibility downloaded from Code Ocean) to _`/WORK/Data`_ in the container)
3. After installing the [Remote Development](https://code.visualstudio.com/docs/remote/remote-overview) extension, look for a small blue `><` icon in the lower-left corner of the VS Code window.
4. Click this icon and select `Reopen in Container`.
5. Wait briefly while the Docker image builds and opens. Once complete, the development environment will be ready to use.

### Running Code and Reproducing Results
For quantitative figures and key numbers, after completing the steps above, follow these instructions (**recommended to use Code Ocean instead**):

1. Locate the corresponding `.ipynb` file listed in the "Locations of quantitative results in the codes" section.
2. Click `Run All`.
3. When prompted to choose a Python kernel, select `Python Environments...`, then choose `Python 3.8.10 /usr/bin/python3`.
4. The code will execute successfully.

**Note:** Download the required data from our Code Ocean repository and place it in the appropriate folder according to your modified paths.

For the complete workflow, refer to the "Overall workflow" section and ensure all paths are correctly configured.
### Troubleshooting
1. Unable to open Docker container
    > This issue is most likely due to incorrect path settings. Please ensure all paths are correctly formatted. (For Windows, a correct path format looks like _`D:\\folder1\\folder2`_)

2. Code execution errors (e.g., `package not exist`, `cannot find file path`, etc.)
    > These errors typically occur due to incorrect path mounting in _`devcontainer.json`_. Please verify that your file structure follows this pattern: _`your_path\\code\\global_lake_area\\batch_processing\\...`_ and _`your_path\\data\\global_lake_area\\area_csvs`_. In the _`devcontainer.json`_ file, ensure that the _`your_path\\code`_ and _`your_path\\data`_ folders are properly specified. Also check all paths in the codes.

3. Kernel crashes and memory-related errors (containing keywords like "free", "mem", etc.)
    > These errors occur when your system's RAM is insufficient for running the script. Please use a high-performance computer with at least 64GB of RAM installed. For Windows 11 users, these issues may be caused by WSL2-based Docker's default memory limitations. In this case, please refer to the official documentation [here](https://learn.microsoft.com/en-us/windows/wsl/wsl-config#wslconfig) and [here](https://forums.docker.com/t/increase-container-memory-usage-limit/139437) for instructions on configuring the _`.wslconfig`_ file to allocate at least 64GB RAM to Docker.
---
---
---

## File Description and Usage

Other scripts relate to the algorithms described in this manuscript, which require several months of runtime, high-performance computing clusters, and terabytes of input data. Therefore, they are not included in the reproduction process due to time and resource limitations. For more information, brief descriptions of each file are provided below, with detailed usage instructions included within the files themselves.

1. **_`global_lake_area/`_** (folder)

    - **_`unetgee.py`_**: Provides functions for GEE authentication, U-Net sample generation, training, validation, MODIS and GSW raster export, and U-Net prediction. Many functions in this file are deprecated due to the iteration of the manuscript but are kept for reference.
    - **_`unet_train.py`_**: Serves as a command-line interface for U-Net training by calling the _`unet_train`_ function in _`unetgee.py`_.
    - **_`UNET_TRAIN_CONFIG.py`_**: Contains configuration settings for a single U-Net training session.
    - **_`update_config_unet_train_run.py`_**: Facilitates batch U-Net training by updating _`UNET_TRAIN_CONFIG.py`_ and executing _`unet_train.py`_.
    - **_`training_records.csv`_**: Stores metadata for U-Net models, including sample sizes and model metrics.
    - **_`update_training_record.py`_**: Updates the _`training_records.csv`_ file.
    - **_`unet_samples_generate_per_basin.ipynb`_**: Jupyter notebook for generating U-Net training samples for each basin.
    - **_`unet_sample_size_count.ipynb`_**: Computes sample sizes for U-Net training, evaluation, and validation; updates the _`training_records.csv`_ file.
    - **_`unet_evaluation.py`_**: Functions similarly to _`unet_train.py`_, calculating performance metrics for each U-Net model and updating the _`training_records.csv`_ file.
    - **_`UNET_EVALUATION_CONFIG.py`_**: Contains configuration settings for a single U-Net evaluation session.
    - **_`unet_evaluation_update_config_and_run.py`_**: Similar to _`update_config_unet_train_run.py`_, handles batch performance metrics calculation.
    - **_`selfee.py`_**: Implements service-account-related methods for automated authentication to resolve network issues.
    - **_`projection_wkt_generation.ipynb`_**: Creates customized Lambert Azimuth Equal Area (LAEA) projections for each basin in the BasinATLAS lev02 product.
    - **_`hydrolakes_filter_by_bas.ipynb`_**: Generates lake boundaries for U-Net sample generation (not used for final area calculation).
    - **_`gsw_export.ipynb`_**: Handles the export of GSW occurrence and recurrence data.
    - **_`gsw_occurrence_and_recurrence_mosaic.py`_**: Creates mosaics from tiled GSW occurrence and recurrence maps.
    - **_`export_modis_and_gsw_image.ipynb`_**: Exports MODIS and GSW images in LAEA projections with correct resolutions.
    - **_`draw_unet_train_history.py`_**: Generates training and validation curves for each U-Net model.
    - **_`add_final_decision_to_records.py`_**: Records the manually-selected optimal epoch in the _`training_records.csv`_ file.

---
2. **_`global_lake_area/.devcontainer`_** (folder)

Contains the _`devcontainer.json`_ file, which defines the container-based runtime environment required to reproduce the results in this manuscript.

---

3. **_`global_lake_area/my_unet_definition`_** (folder)

    - **_`__init__.py`_**: Defines this folder as a Python module.
    - **_`model.py`_**: Contains implementations of U-Net models (specifically the `attentionunet` variant was used).
    - **_`evaluation_metrics.py`_**: Contains performance metrics and loss functions, including Intersection over Union (IoU).

---

4. **_`global_lake_area/my_unet_gdal`_** (folder)

    - **_`__init__.py`_**: Initializes this folder as a Python module.
    - **_`reproject_to_target.py`_**: Deprecated.
    - **_`combined.py`_**: Deprecated.
    - **_`zonal_statistics.py`_**: Deprecated.
    - **_`reproject_to_target_tile.py`_**: Functions for clipping, reprojecting, and creating mosaics from large GeoTIFF files.
    - **_`generate_tfrecord_from_tile.py`_**: Functions for reprojecting, resampling, and converting GeoTIFF files to TFRecord format.
    - **_`align_to_target_tile.py`_**: Functions for geographically aligning and merging two raster datasets.
    - **_`unet_predictions.py`_**: Functions for processing converted TFRecords using trained U-Net models.
    - **_`reconstruct_tile_from_prediction.py`_**: Functions for converting serialized TFRecord files back to GeoTIFF tiles.
    - **_`area_calculation.py`_**: Functions for calculating areas from raster data using vector boundaries.
    - **_`quick_plotting.py`_**: Functions for generating PNG images and GIF animations from GeoTIFF files.
    - **_`quick_plotting_runner.py`_**: Command-line interface for _`quick_plotting.py`_ that accepts LAEA coordinates as input.

---

5. **_`global_lake_area/batch_processing`_** (folder)

    - **_`__init__.py`_**: Initializes this folder as a Python module.
    - **_`batch_tfrecord_generation.py`_**: Command-line interface for generating batches of MODIS-converted TFRecord files.
    - **_`batch_unet_prediction.py`_**: Command-line interface for batch predictions using U-Net.
    - **_`batch_prediction_reconstruction.py`_**: Command-line interface for reconstructing water mask maps in batches.
    - **_`batch_mosaic.py`_**: Creates mosaics from water mask tiles into a single large GeoTIFF file.
    - **_`batch_full.py`_**: Integrates multiple batch processing steps into a single command-line interface.
    - **_`asynchronous_batch.py`_**: Executes _`batch_full.py`_ asynchronously to optimize computing resource utilization.
    - **_`BATCH_CONFIG.py`_**: Configuration settings for _`batch_full.py`_ and _`asynchronous_batch.py`_.
    - **_`batch_area_calculation.py`_**: Command-line interface for calculating areas from mosaiced water mask maps in batches.
    - **_`AREA_CALCULATION_CONFIG.py`_**: Configuration settings for _`batch_area_calculation.py`_ (monthly lake surface extent results).
    - **_`MISSING_DATA_AREA_CALCULATION_CONFIG.py`_**: Configuration settings for _`batch_area_calculation.py`_ (monthly cloud contamination ratio results).
    - **_`MASKED_MY_WATER_AREA_CALCULATION_CONFIG.py`_**: Configuration settings for _`batch_area_calculation.py`_ (GSW-masked water mask map results).
    - **_`GSWR_AREA_CALCULATION_CONFIG.py`_**: Configuration settings for _`batch_area_calculation.py`_ (GSW image results for validation).
    - **_`area_calculation_update_and_run.py`_**: Automatically updates configuration files and executes _`batch_area_calculation.py`_.
    - **_`load_config_module.py`_**: Provides functionality for reading configuration files in _`.py`_ format.

---

6. **_`global_lake_area/my_plotting`_** (folder)

Contains scripts for visualizing the performance metrics of U-Net models trained in this study.

---

7. **_`global_lake_area/my_spatial_analyze`_** (folder)

    - **_`__init__.py`_**: Initializes this folder as a Python module.
    - **_`area_postprocessing.py`_**: Functions for post-processing lake surface water extracted from U-Net-generated water mask maps.
    - **_`lake_wise_area_postprocessor.py`_**: Command-line interface for lake-wise postprocessing of lake surface extent time series.
    - **_`LAKE_WISE_AREA_POSTPROCESSING_CONFIG.py`_**: Configuration settings for _`lake_wise_area_postprocessor.py`_.
    - **_`lake_wise_area_postprocess_update_and_run.py`_**: Automatically updates config files and runs _`lake_wise_area_postprocessor.py`_.
    - **_`lake_wise_lse_analyze.py`_**: Functions for lake-wise plotting.
    - **_`lake_wise_plotting.ipynb`_**: Generates plots (deprecated).
    - **_`visualization.py`_**: Functions for grid-wise plotting.
    - **_`main_grid.py`_**: Exploratory grid-wise plotting (deprecated).
    - **_`lake_concatenator.py`_**: Combines lake-wise time series of surface extent from each basin into one file for global 1.4 million lakes.
    - **_`glake_update_hydrolakes.py`_**: Functions and command-line interface for updating HydroLAKES using GLAKES.
    - **_`hylak_buffering.py`_**: Removes duplicate lakes and creates buffer zones for GLAKES-updated HydroLAKES.
    - **_`gsw_image_mosaic.py`_**: Command-line interface for mosaicking tiled GSW images into one large GeoTIFF file for validation.
    - **_`grid_concatenator.py`_**: Deprecated.
    - **_`grid_analyze.py`_**: Functions for performing grid-level analysis.
    - **_`cloud_cover_ratio_calculater.py`_**: Command-line interface for calculating cloud cover ratios based on boundary size and monthly MODIS cloud contamination area.
    - **_`basin_lse_calculation.py`_**: Deprecated.
    - **_`attach_geometry_and_generate_grid.py`_**: Functions for creating grids from global (or regional) lakes and calculating corresponding statistics.
    - **_`area_to_volume.py`_**: Deprecated.
    - **_`area_to_level.py`_**: Deprecated.
    - **_`AREA_TO_LEVEL_CONFIG.py`_**: Deprecated.
    - **_`area_to_level_batch_converter.py`_**: Deprecated.
    - **_`./data_analyze`_** (folder)
        - **_`./basin_wise_analysis`_** (folder)
            - **_`basin_wise_analysis.py`_**: Functions for basin-wise analysis and plotting.
            - **_`basin_wise_plotting.ipynb`_**: Generates basin-wise figures (including reservoir contribution).
            - **_`basinatlas_statistics_calculator.py`_**: Command-line interface for calculating BasinATLAS statistics.
            - **_`hydrobasins_merger.py`_**: Merges multiple HydroBASINS shapefile files.
            - **_`hydrobasins_statistics_calculator.py`_**: Command-line interface for calculating HydroBASINS statistics.
        - **_`./climate_analysis`_** (folder)
            - **_`attach_aridity_index.py`_**: Adds LakeATLAS aridity index to lake surface extent time series.
        - **_`./correlation_analysis`_** (folder)
            - **_`plotting.ipynb`_**: Generates plots of median relative changes in seasonality by lake size.
            - **_`correlation_plots.py`_**: Functions for plotting relationships between multiple variables.
        - **_`./extreme_analysis`_** (folder)
            - **_`area_extreme_analysis.py`_**: Functions for identifying seasonality-induced low-water extremes and other analyses.
            - **_`low_water_extreme_analysis.ipynb`_**: Adds extreme-related columns to lake surface extent time series.
            - **_`low_water_extreme_plotting.ipynb`_**: Generates plots of seasonality-induced low-water extremes and seasonality dominance.
        - **_`./grid_wise_analysis`_** (folder): Contains plots of seasonality changes.
        - **_`./permafrost_analysis`_** (folder): Adds permafrost type information to lake surface extent time series.
        - **_`./time_series_analysis`_** (folder): Generates plots of long-term trends.
    - **_`./data_validation`_** (folder): Contains validation using GSW estimates and altimetry-based water levels.

---

8. **_`global_lake_area/projection_wkt`_** (folder)

Contains Lambert Azimuthal Equal-Area (LAEA) projection definitions used in this manuscript.

9. **_`global_lake_area/revision_codes`_** (folder)
    - **_`./accuracy_assessment`_** (folder)
        - **_`sample_generation.py`_**: Generates basin-wise samples for calculating user's and producer's accuracies and F1 scores.
        - **_`sample_generation_runner.ipynb`_**: Performs batch generation of samples.
        - **_`metric_calculation.py`_**: Contains utility functions for calculating user's and producer's accuracies and F1 scores.
        - **_`metric_calculation.ipynb`_**: Calculates user's and producer's accuracies and F1 scores.
        - **_`metric_plotting.ipynb`_**: Plots user's and producer's accuracies and F1 scores.
    - **_`./relative_to_total_area`_** (folder)
        - **_`relative_to_total_area_calculation.ipynb`_**: Calculates the ratio between total variation in lake surface extent and total lake area.
        - **_`relative_to_total_area_plotting.ipynb`_**: Plots the ratio between total variation in lake surface extent and total lake area (Extended Data Fig. 4).
    - **_`./population_density_analysis`_** (folder)
        - **_`population_density_analysis_calculation.ipynb`_**: Adds population density data from BasinATLAS level-06 basins to lake surface extent time series.
        - **_`population_density_analysis_plotting.ipynb`_**: Plots relationships between seasonality dominance, changes, and population density (Extended Data Fig. 5).
    - **_`./high_water_extreme_analysis`_** (folder)
        - **_`high_water_extreme_calculate.ipynb`_**: Detects seasonality-induced high-water extremes and analyzes their magnitude and relative importance.
        - **_`high_water_extreme_plotting.ipynb`_**: Plots the relative importance of seasonality-induced high-water extremes compared to 23-year changes and regular seasonality (Extended Data Fig. 6).
    - **_`./extreme_changes`_** (folder)
        - **_`extreme_changes_calculation.ipynb`_**: Calculates frequency changes of seasonality-induced high- and low-water extremes.
        - **_`extreme_changes_plotting.ipynb`_**: Plots frequency changes of seasonality-induced high- and low-water extremes (Extended Data Fig. 7).
    - **_`./comparison_with_gsw`_** (folder)
        - **_`comparison_with_gsw_calculation.ipynb`_**: Calculates missing areas and counts in GSW monthly history product and our maps (Uses multiple iterations to avoid RAM overflow).
        - **_`comparison_with_gsw_plotting.ipynb`_**: Plots missing areas and counts in GSW monthly history product and our maps (Extended Data Fig. 10).
    - **_`./basin_seasonality_change`_** (folder)
        - **_`basin_seasonality_change_plotting.ipynb`_**: Plots median seasonality changes aggregated by BasinATLAS level-06 basins (Fig. 3c).
        
## Overall workflow
**Please note that many details in this section are omitted. For complete information, please refer to the corresponding scripts and manuscript paragraphs.**
### U-Net Training, Evaluation, and Testing
1. **Export Training, Validation, and Test Samples**   
Use `unet_samples_generate_per_basin.ipynb` to generate training, validation, and test samples for each basin (BasinATLAS level-02 basins). This step uses the Python API of Google Earth Engine (GEE) and basin-specific customized Lambert Azimuthal Equal-Area projections located in the directory _`global_lake_area/projection_wkt/Lambert_Azimuthal_Equal_Area`_. The samples are saved in TFRecord format to a Google Drive folder, which can be configured by modifying the `drive_folder` parameter. All preprocessing of MODIS images and the GSW monthly history product is handled by the sample generation function defined in `unetgee.py`.

2. **Train U-Net models**   
Use `update_config_unet_train_run.py` to update the `UNET_TRAIN_CONFIG.py` file and run `unet_train.py` sequentially for each basin. Preprocessing of training and validation samples is performed by the `unet_train` function in `unetgee.py`. The training status flag for each basin is automatically updated in the training record file `training_records.csv`.

3. **Select the optimal epoch for each U-Net model manually**  
Use `draw_unet_train_history.py` to plot the training and validation history of each U-Net model, then manually select the optimal epoch. Record the optimal epoch in the `training_records.csv` file using the `add_final_decision_to_records.py` script.  

4. **Evaluate U-Net models on test sets**
Use `unet_evaluation_update_config_and_run.py` to update the `UNET_EVALUATION_CONFIG.py` file and run `unet_evaluation.py` sequentially for each basin. The performance metrics (IoU and binary accuracy) are recorded in the `training_records.csv` file.

### Export MODIS Monthly Median Composites and GSW Products
1. **Export MODIS Monthly Median Composites**  
Use `export_modis_and_gsw_image.ipynb` to export MODIS monthly median composites in Lambert Azimuthal Equal-Area (LAEA) projections. Related functions are defined in `unetgee.py`. The exported images are saved to a Google Drive folder, which can be configured by modifying the corresponding parameter. The spatial resolution of these exports is 500 m. All preprocessing is handled automatically by the function.

2. **Export GSW Occurrence and Recurrence Maps**  
Use `gsw_export.ipynb` to export Global Surface Water (GSW) occurrence and recurrence maps. Related functions are defined in `unetgee.py`. The exported images are saved to a Google Drive folder, which can be configured by modifying the corresponding parameter. The spatial resolution of these exports is 30 m.

3. **Export GSW Monthly History Product**  
Use `export_modis_and_gsw_image.ipynb` to export the GSW monthly history product in LAEA projections. Related functions are defined in `unetgee.py`. The exported images are saved to a Google Drive folder, which can be configured by modifying the corresponding parameter. The spatial resolution of these exports is 30 m.

4. **Create Mosaics of GSW Products**  
Use `gsw_occurrence_and_recurrence_mosaic.py` to create mosaics of GSW occurrence and recurrence maps. Use `gsw_image_mosaic.py` to create mosaics of the GSW monthly history product. Basic utility functions for these operations are defined in the `my_unet_gdal` directory.

### Batch processing of MODIS images using trained U-Net models to generate raw lake surface extent maps
**Note**: This section is the most computationally intensive part of this manuscript. It requires high-performance computing clusters and multiple GPUs. This section uses all basic utility functions defined in the `my_unet_gdal` directory. A brief description of each step is provided below. Detailed usage and definitions of functions and workflows can be found in the corresponding scripts.

Two running modes are provided in this section: asynchronous batch processing (`asynchronous_batch.py`) and synchronous batch processing (`batch_full.py`). The former is recommended for high-performance computing clusters, while the latter is recommended for personal computers.

1. **Convert MODIS images to TFRecord format**  
This step involves reprojecting, resampling, merging MODIS monthly median composites with GSW occurrence and recurrence, and converting them to 128×128 kernels in TFRecord format that can be fed into U-Net models. Use `batch_tfrecord_generation.py` to generate TFRecord files for different months in one basin.

2. **Predict using U-Net models**  
This step involves using trained U-Net models to process TFRecord files generated in the previous step. Use `batch_unet_prediction.py` to generate water mask maps.

3. **Reconstruct water mask maps**  
This step involves converting serialized TFRecord files (the output of trained U-Net models with MODIS images as input) back to GeoTIFF tiles. Use `batch_prediction_reconstruction.py` to generate water mask maps.

4. **Mosaic water mask maps**  
This step combines water mask maps into one large GeoTIFF file. Use `batch_mosaic.py` to mosaic water mask maps.


### Calculate Monthly Lake Surface Extent and Post-processing
1. **Calculate Raw Monthly Lake Surface Extent for Each Lake**  
Use `area_calculation_update_and_run.py` to update the `AREA_CALCULATION_CONFIG.py` file and run `batch_area_calculation.py` sequentially for different basins. For details of this calculation, please refer to the `batch_area_calculation.py` file and the `calculate_lake_area_grid_parallel` function defined in `./my_unet_gdal/area_calculation.py`. (A raster-based calculation approach is used)

2. **Frozen Period Determination**  
Use the Simstrat model provided in [LakeEnsemblR](https://github.com/aemon-j/LakeEnsemblR) to determine the frozen period for each lake.

3. **Cloud-Contaminated Area Calculation**  
Use `area_calculation_update_and_run.py` to update the `MISSING_DATA_AREA_CALCULATION_CONFIG.py` file and run `batch_area_calculation.py` sequentially for different basins. This calculates the areas identified as clouds by the MODIS QA band.

4. **Cloud Contamination Ratio Calculation**  
Use `./my_spatial_analyze/cloud_cover_ratio_calculator.py` to calculate the cloud contamination ratio for each lake. Basic utility functions are defined in the `./my_spatial_analyze/area_postprocessing.py` file.

5. **Process and Analyze Lake Data**  
Use `./my_spatial_analyze/lake_wise_area_postprocess_update_and_run.py` to update the `LAKE_WISE_AREA_POSTPROCESSING_CONFIG.py` file and run `lake_wise_area_postprocessor.py` sequentially for different basins. This step filters out cloud-contaminated and frozen data points, and calculates lake-wise statistics including intra-annual standard deviations for further analysis. Basic utility functions are defined in the `./my_spatial_analyze/area_postprocessing.py` and `./my_spatial_analyze/attach_geometry_and_generate_grid.py` files.

### Analysis and Plotting
1. **Attach Additional Properties from LakeATLAS to the Time Series DataFrame**  
Use `./my_spatial_analyze/data_analyze/climate_analysis/attach_aridity_index.py` for aridity index, `./my_spatial_analyze/data_analyze/permafrost_analysis/attach_permafrost_type.py` for permafrost type, and `./revision_codes/population_density_analysis/population_density_analysis_calculation.ipynb` for population density.

2. **Create Grid Cells**  
Use `./my_spatial_analyze/grid_analyze.py` to create grid cells. Multiple modes and grid sizes are available (1.0 for coarse, 0.5 for medium, 0.25 for fine). All grid-wise figures are generated using this approach from lake-wise statistics.

3. **Create Fig. 1**  
This illustrative figure is created manually in Adobe Illustrator.

4. **Create Fig. 2**  
The lake-wise statistic `seasonality_dominance_percentage` is calculated in `./my_spatial_analyze/data_analyze/extreme_analysis/low_water_extreme_analysis.ipynb` following equations in the manuscript. The raw figure is generated in `./my_spatial_analyze/data_analyze/extreme_analysis/low_water_extreme_plotting.ipynb`. The final figure is polished in Adobe Illustrator by adjusting the layout, fonts, and annotations.

5. **Create Fig. 3**  
For subfigures a, b, and d, statistics are calculated in the '**Filter out cloud-contaminated and frozen data points and calculate lake-wise statistics**' step and aggregated to grid level by taking the median. For subfigure c, basin-wise statistics are calculated in `./my_spatial_analyze/data_analyze/basin_wise_analysis/basinatlas_statistics_calculator.py`. The raw figure is generated in `./my_spatial_analyze/data_analyze/grid_wise_analysis/grid_wise_plotting.ipynb`. The final figure is polished in Adobe Illustrator by adjusting the layout, fonts, and annotations.

6. **Create Fig. 4**  
For subfigures a, b, and c, time series are calculated in `./my_spatial_analyze/data_analyze/time_series_analysis/time_series_plotting.ipynb` (results are saved for quick reproduction in the Code Ocean capsule). For subfigure d, basin-wise statistics are calculated in `./my_spatial_analyze/data_analyze/basin_wise_analysis/hydrobasins_statistics_calculator.py`. For subfigure e, summed linear trends of STL trend terms are calculated and aggregated to grid level in previous steps. The raw figure is generated in `./my_spatial_analyze/data_analyze/time_series_analysis/time_series_plotting.ipynb`. The final figure is polished in Adobe Illustrator by adjusting the layout, fonts, and annotations.

7. **Create Fig. 5**  
For subfigures a, c, and d, seasonality-induced low-water extremes are detected in `./my_spatial_analyze/data_analyze/extreme_analysis/low_water_extreme_analysis.ipynb`. The relative importance of seasonality-induced low-water extremes compared to 23-year changes and regular seasonality is calculated in `./my_spatial_analyze/data_analyze/extreme_analysis/low_water_extreme_plotting.ipynb`. Raw versions of subfigures a, c, and d are generated in `./my_spatial_analyze/data_analyze/extreme_analysis/low_water_extreme_plotting.ipynb`. The final figure is polished in Adobe Illustrator, with subfigure b manually drawn.

8. **Create Extended Data Fig. 1 and 2**  
For this comparison, monthly lake surface extent from GSW monthly history product and our maps masked by GSW's mask are calculated using `./batch_processing/batch_area_calculation.py` with `GSW_AREA_CALCULATION_CONFIG.py` and `MASKED_MY_WATER_AREA_CALCULATION_CONFIG.py`. The raw figure is generated in `./my_spatial_analyze/data_validation/plot_compare_with_gsw.ipynb`. The final figure is polished and assembled in Adobe Illustrator.

9. **Create Extended Data Fig. 3**  
For subfigures a and b, basin-wise binary classification accuracy and IoU of U-Net models are calculated in the **U-Net training, evaluation, and test** section. The raw figure is generated in `./my_plotting/plotting.ipynb`. For subfigures c, d, and e, sample generation and metric calculation are performed in the `./revision_codes/accuracy_assessment` directory. The raw figure is generated in `./revision_codes/accuracy_assessment/metric_plotting.ipynb`. The final figure is polished and assembled in Adobe Illustrator.

10. **Create Extended Data Fig. 4**  
Statistics are calculated in `./revision_codes/relative_to_total_area/relatvie_to_total_area_calculation.ipynb`. The raw figure is generated in `./revision_codes/relative_to_total_area/relative_to_total_area_plotting.ipynb`. The final figure is polished in Adobe Illustrator.

11. **Create Extended Data Fig. 5**  
Statistics are attached from BasinATLAS level-06 basins in `./revision_codes/population_density_analysis/population_density_analysis_calculation.ipynb`. The raw figure is generated in `./revision_codes/population_density_analysis/population_density_analysis_plotting.ipynb`. The final figure is polished in Adobe Illustrator.

12. **Create Extended Data Fig. 6**  
Seasonality-induced high-water extremes and their magnitudes are calculated in `./revision_codes/high_water_extreme_analysis/high_water_extreme_calculate.ipynb`. The raw figure is generated in `./revision_codes/high_water_extreme_analysis/high_water_extreme_plotting.ipynb`. The final figure is polished in Adobe Illustrator.

13. **Create Extended Data Fig. 7**  
Frequency changes of seasonality-induced high- and low-water extremes are calculated in `./revision_codes/extreme_changes/extreme_changes_calculation.ipynb`. The raw figure is generated in `./revision_codes/extreme_changes/extreme_changes_plotting.ipynb`. The final figure is polished in Adobe Illustrator.

14. **Create Extended Data Fig. 8**  
Subfigure a is created using draw.io and subfigure b is created using LaTeX code. The final figure is polished in Adobe Illustrator.

15. **Create Extended Data Fig. 9**  
This figure is generated in `./my_spatial_analyze/data_validation/data_validation_nb.ipynb`. The final figure is polished in Adobe Illustrator.

16. **Create Extended Data Fig. 10**  
Missing areas in the GSW monthly history product are calculated using `./batch_processing/batch_area_calculation.py` with `GSW_MISSING_DATA_AREA_CALCULATION_CONFIG.py`. Statistics are calculated in `./revision_codes/comparison_with_gsw/comparison_with_gsw_calculation.ipynb`. The raw figure is generated in `./revision_codes/comparison_with_gsw/comparison_with_gsw_plotting.ipynb`. The final figure is polished and assembled in Adobe Illustrator.