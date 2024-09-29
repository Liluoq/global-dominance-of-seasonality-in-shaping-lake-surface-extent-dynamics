import subprocess
import os

if __name__ == '__main__':
    mode = 'create_annotated_gif'
    raster_folder_path = '/WORK/SSD_Data/global_lake_area/mosaic_tifs/6020000010'
    out_gif_path = '/WORK/SSD_Data/global_lake_area/mosaic_tifs/test_gif/6020000010.gif'
    x_min = str(-1019744)
    y_min = str(262254) 
    x_max = str(-828287) 
    y_max = str(390402)
    
    command = [
        'xvfb-run', '-a', 'python', '-u', 'quick_plotting.py',
        '--mode', mode,
        '--raster_folder_path', raster_folder_path,
        '--out_gif_path', out_gif_path,
        '--x_min', x_min,
        '--x_max', x_max,
        '--y_min', y_min,
        '--y_max', y_max
    ]
    
    subprocess.run(command)