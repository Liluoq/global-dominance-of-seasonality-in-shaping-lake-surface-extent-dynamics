import geopandas as gpd
import numpy as np
from shapely.geometry import Point, box
from shapely.vectorized import contains
from rasterio.sample import sample_gen
import rasterio
import pandas as pd
from typing import List, Optional


def grid_based_random_sampling(
    sample_boundary_gdf: gpd.GeoDataFrame,
    max_points_per_cell: int,  # expected number of points per cell with full lake coverage
    grid_size: int = 100,
    verbose: bool = False,
) -> gpd.GeoDataFrame:
    """Generate random points within a boundary using a grid-based sampling approach.

    This function divides the input boundary into a grid and generates random points within each grid cell
    that intersects with the boundary. This approach ensures a more uniform spatial distribution of points
    compared to simple random sampling.

    Args:
        sample_boundary_gdf (gpd.GeoDataFrame): GeoDataFrame containing the boundary polygons within which
            points should be generated.
        max_points_per_cell (int): Maximum number of points to generate per grid cell. The actual number
            of points will be less if the cell is not fully covered by the boundary.
        grid_size (int, optional): Number of cells along each dimension of the grid. Defaults to 100.
        verbose (bool, optional): If True, prints progress information. Defaults to False.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the generated points with their geometries.
    """

    # Get overall bounds without unary_union
    bounds = sample_boundary_gdf.total_bounds
    minx, miny, maxx, maxy = bounds

    # Create grid edges
    x_edges = np.linspace(minx, maxx, grid_size + 1)
    y_edges = np.linspace(miny, maxy, grid_size + 1)

    all_points = []
    
    # Create spatial index for faster intersection queries
    if verbose:
        print("Creating spatial index")
    spatial_index = sample_boundary_gdf.sindex

    # Process each grid cell
    total_cells = grid_size * grid_size
    for i in range(grid_size):
        for j in range(grid_size):
            if verbose:
                print(f"Processing cell {i * grid_size + j + 1}/{total_cells}")
                
            # Create cell geometry
            cell = box(x_edges[i], y_edges[j], x_edges[i + 1], y_edges[j + 1])
            
            # Query spatial index to get potentially intersecting polygons
            possible_matches_index = list(spatial_index.intersection(cell.bounds))
            if not possible_matches_index:
                continue
                
            # Get actual intersecting polygons and create local unary_union
            cell_polygons = sample_boundary_gdf.iloc[possible_matches_index]
            cell_polygons = cell_polygons[cell_polygons.intersects(cell)]
            if cell_polygons.empty:
                continue
                
            local_unified = cell_polygons.unary_union
            intersection = cell.intersection(local_unified)

            if not intersection.is_empty:
                # Generate points
                batch_size = max_points_per_cell
                x = np.random.uniform(cell.bounds[0], cell.bounds[2], size=batch_size)
                y = np.random.uniform(cell.bounds[1], cell.bounds[3], size=batch_size)

                # Vectorized containment check
                mask = contains(local_unified, x, y)
                valid_coords = np.column_stack((x[mask], y[mask]))

                # Convert valid coordinates to points
                new_points = [Point(coord) for coord in valid_coords]
                all_points.extend(new_points)

    # Create GeoDataFrame
    return gpd.GeoDataFrame(geometry=all_points, crs=sample_boundary_gdf.crs)


def extract_values_from_raster(
    raster_path: str,
    sample_points_gdf: gpd.GeoDataFrame,
) -> pd.Series:
    """
    Extract values from a raster at given sample points.

    Parameters
    ----------
    raster_path : str
        Path to the raster file to sample from
    sample_points_gdf : gpd.GeoDataFrame
        GeoDataFrame containing the sample points as geometries

    Returns
    -------
    pd.Series
        Series containing the extracted raster values, indexed by the sample points GeoDataFrame index

    Raises
    ------
    ValueError
        If the CRS of the raster and sample points do not match
    """

    """
    Extract values from raster at sample points
    """
    with rasterio.open(raster_path) as src:
        if src.crs != sample_points_gdf.crs:
            raise ValueError(
                f"Raster and sample points CRS do not match, {src.crs} != {sample_points_gdf.crs}"
            )

        coords = [(p.x, p.y) for p in sample_points_gdf.geometry]
        values = [val[0] for val in sample_gen(src, coords)]

    return pd.Series(values, index=sample_points_gdf.index)


def generate_samples_single_gdf(
    sample_boundary_gdf: gpd.GeoDataFrame,
    raster_paths: List[str],
    sample_col_names: List[str],
    max_points_per_cell: int,
    sample_grid_size: int = 100,
    save_path: Optional[str] = None,
    verbose: bool = False,
) -> gpd.GeoDataFrame:
    """
    Generate samples from rasters
    """
    if len(raster_paths) != len(sample_col_names):
        raise ValueError(
            f"Number of raster paths and sample column names must be the same, {len(raster_paths)} != {len(sample_col_names)}"
        )

    sample_points_gdf = grid_based_random_sampling(
        sample_boundary_gdf, max_points_per_cell, sample_grid_size, verbose
    )
    for raster_path, sample_col_name in zip(raster_paths, sample_col_names):
        sample_points_gdf[sample_col_name] = extract_values_from_raster(
            raster_path, sample_points_gdf
        )

    if save_path is not None:
        if save_path.endswith(".pkl"):
            sample_points_gdf.to_pickle(save_path)
        else:
            raise NotImplementedError(f"Unsupported file extension: {save_path}")

    return sample_points_gdf

def generate_samples_multiple_gdfs(
    sample_boundary_gdf: gpd.GeoDataFrame,
    raster_paths_list: List[List[str]],
    sample_col_names_list: List[List[str]],
    max_points_per_cell: int,
    sample_grid_size: int = 100,
    save_path_list: Optional[List[str]] = None,
) -> List[gpd.GeoDataFrame]:
    """
    Generate samples from multiple rasters
    """
    sample_points_gdf_list = []
    if save_path_list is not None:
        for raster_paths, sample_col_names, save_path in zip(
            raster_paths_list, sample_col_names_list, save_path_list
        ):
            sample_points_gdf = generate_samples_single_gdf(
                sample_boundary_gdf, raster_paths, sample_col_names, max_points_per_cell, sample_grid_size, save_path
            )
            sample_points_gdf_list.append(sample_points_gdf)
    else:
        for raster_paths, sample_col_names in zip(
            raster_paths_list, sample_col_names_list
        ):
            sample_points_gdf = generate_samples_single_gdf(
                sample_boundary_gdf, raster_paths, sample_col_names, max_points_per_cell, sample_grid_size
            )
            sample_points_gdf_list.append(sample_points_gdf)

    return sample_points_gdf_list

