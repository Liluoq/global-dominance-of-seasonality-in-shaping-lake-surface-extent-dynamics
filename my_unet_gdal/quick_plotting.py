from osgeo import gdal
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
import argparse
from PIL import Image
import os
import re


gdal.UseExceptions()

def get_raster_subset(raster_path, x_min, x_max, y_min, y_max):
    # Open the raster dataset
    try:
        ds = gdal.Open(raster_path)
    except RuntimeError as e:
        print('Unable to open {}'.format(raster_path))
        print(e)
        sys.exit(1)
    
    # Get geotransform and calculate inverse geotransform
    gt = ds.GetGeoTransform()
    inv_gt = gdal.InvGeoTransform(gt)
    
    # Convert projected coordinates to pixel coordinates
    px_min, py_max = gdal.ApplyGeoTransform(inv_gt, x_min, y_min) # py_max corresponds to y_min because the y-axis is flipped for projection axis and pixel axis
    px_max, py_min = gdal.ApplyGeoTransform(inv_gt, x_max, y_max)
    print(px_min, py_min, px_max, py_max)
    # Convert floats to integers (pixel indices)
    px_min, py_min, px_max, py_max = map(int, [px_min, py_min, px_max, py_max])
    
    # Compute the size of the section
    px_width = px_max - px_min
    py_height = py_max - py_min
    
    # Read the raster section
    raster_section = ds.ReadAsArray(px_min, py_min, px_width, py_height)
    
    return raster_section

def plot_raster_section(raster_section):
    # Define a binary colormap
    cmap = mcolors.ListedColormap(['yellow', 'blue'])
    bounds = [-0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    fig, ax = plt.subplots(figsize=(6, 6))  # Adjust figsize to change the plot aspect ratio
    im = ax.imshow(raster_section, cmap=cmap, norm=norm)
    
    # Create an axis on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    
    # Create a colorbar in the appended axis "cax"
    cbar = fig.colorbar(im, cax=cax, ticks=[0, 1])
    cbar.ax.set_yticklabels(['Non-Water', 'Water'], rotation=90)
    
    plt.show()
    
def create_annotated_gif(folder_path, gif_path, roi_coords):
    """
    Create a GIF from raster files in a specified folder, annotated with start times.

    Parameters:
    - folder_path: Path to the folder containing the raster files.
    - gif_path: Path where the GIF should be saved.
    - roi_coords: A tuple of the form (x_min, x_max, y_min, y_max) specifying the region of interest.
    """
    # Step 1: List and Sort Raster Files
    raster_files = [f for f in os.listdir(folder_path) if f.endswith('_water_mosaic.tif')]
    raster_files.sort(key=lambda x: re.search(r'_(\d{4}-\d{2}-\d{2})_', x).group(1))

    x_min, x_max, y_min, y_max = roi_coords

    temp_images = []

    for filename in raster_files:
        # Extract start time from the filename
        start_time = re.search(r'_(\d{4}-\d{2}-\d{2})_', filename).group(1)
        
        # Step 2: Extract and Plot Raster Data for ROI
        raster_path = os.path.join(folder_path, filename)
        raster_section = get_raster_subset(raster_path, x_min, x_max, y_min, y_max)
        
        fig, ax = plt.subplots()
        ax.imshow(raster_section, cmap='gray')
        ax.text(0.05, 0.95, start_time, transform=ax.transAxes, fontsize=14, color='yellow', ha='left', va='top', weight='bold')
        plt.axis('off')

        temp_image_path = os.path.join(os.path.dirname(gif_path), f'temp_{start_time}.png')
        plt.savefig(temp_image_path, bbox_inches='tight')
        plt.close()
        
        temp_images.append(temp_image_path)

    # Step 3: Create a GIF from the Saved Images
    images = [Image.open(img_path) for img_path in temp_images]
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=500, loop=0)

    # Cleanup temporary image files
    for img_path in temp_images:
        os.remove(img_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot a subset of a raster')
    parser.add_argument('--raster_path', type=str, default=None, help='Path to the raster file')
    parser.add_argument('--x_min', type=float, default=None, help='Minimum x coordinate')
    parser.add_argument('--x_max', type=float, default=None, help='Maximum x coordinate')
    parser.add_argument('--y_min', type=float, default=None, help='Minimum y coordinate')
    parser.add_argument('--y_max', type=float, default=None, help='Maximum y coordinate')
    parser.add_argument('--out_gif_path', type=str, default=None, help='Path to the GIF file')
    parser.add_argument('--raster_folder_path', type=str, default=None, help='Path to the folder containing raster files')
    parser.add_argument('--mode', type=str, help='Mode of operation')
    args = parser.parse_args()
    raster_path = args.raster_path
    x_min = args.x_min
    x_max = args.x_max 
    y_min = args.y_min
    y_max = args.y_max
    out_gif_path = args.out_gif_path
    raster_folder_path = args.raster_folder_path
    mode = args.mode
    
    if args.mode == 'plot_single_raster':
        if raster_path is None or x_min is None or x_max is None or y_min is None or y_max is None:
            print('Please provide the raster path and the coordinates of the region of interest')
            sys.exit(1)
        if raster_folder_path is not None:
            print('The raster path argument will be ignored because the mode is set to create_annotated_gif')
        raster_section = get_raster_subset(raster_path, x_min, x_max, y_min, y_max)
        plot_raster_section(raster_section)
    elif args.mode == 'create_annotated_gif':
        if raster_folder_path is None or out_gif_path is None or x_min is None or x_max is None or y_min is None or y_max is None:
            print('Please provide the raster folder path, GIF path, and the coordinates of the region of interest')
            sys.exit(1)
        if raster_path is not None:
            print('The raster path argument will be ignored because the mode is set to create_annotated_gif')
        create_annotated_gif(raster_folder_path, out_gif_path, (x_min, x_max, y_min, y_max))