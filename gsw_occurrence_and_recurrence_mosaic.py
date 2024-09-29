import os
from osgeo import gdal
from my_unet_gdal.reproject_to_target_tile import mosaic_tiles

# Set the working directory to the directory of the script
wkdir = '/WORK/Data/global_lake_area'
if wkdir == None:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
else:
    os.chdir(wkdir)

basin_id_list = [1020000010, 8020000010, 4020000010, 5020000010, 2020000010, 9020000010, 
                 7020000010, 6020000010, 3020000010, 1020011530, 8020008900, 4020006940, 
                 5020015660, 2020003440, 7020014250, 6020006540, 3020003790, 1020018110, 
                 8020010700, 4020015090, 5020037270, 2020018240, 7020021430, 6020008320, 
                 3020005240, 1020021940, 8020020760, 4020024190, 5020049720, 2020024230, 
                 7020024600, 6020014330, 3020008670, 1020027430, 8020022890, 4020034510, 
                 2020033490, 7020038340, 6020017370, 3020009320, 1020034170, 8020032840, 
                 2020041390, 7020046750, 6020021870, 1020035180, 8020044560, 5020082270, 
                 2020057170, 7020047840, 6020029280, 2020065840, 7020065090, 2020071190, 
                 4020050210, 3020024310, 4020050220, 1020040190, 4020050290, 4020050470]
input_folder = './gsw_occurrence_and_recurrence_laea_30m/'
file_name_base = 'gsw_occurrence_and_recurrence_30m_laea_{basin_id}'
output_folder = os.path.join(input_folder, 'mosaic')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for basin_id in basin_id_list:
    current_file_base = file_name_base.format(basin_id=basin_id)
    mosaic_tiles(
        input_folder=input_folder,
        output_tif=os.path.join(input_folder, 'mosaic', current_file_base + '_mosaic.tif'),
        input_tile_file_basename=current_file_base,
        verbose=2,
        outputdType=gdal.GDT_Int16,
        remove_tile=False
    )
