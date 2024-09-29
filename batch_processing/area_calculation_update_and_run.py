import os
import re
import subprocess

if __name__ == '__main__':
    CALCULATION_TYPE = 'gsw_area'
    # Path to area calculation configuration file
    config_folder = '/WORK/Codes/global_lake_area/batch_processing'
    if CALCULATION_TYPE == 'water_area':
        config_filename = 'AREA_CALCULATION_CONFIG.py'
    elif CALCULATION_TYPE == 'missing_data':
        config_filename = 'MISSING_DATA_AREA_CALCULATION_CONFIG.py'
    elif CALCULATION_TYPE == 'gsw_area':
        config_filename = 'GSW_AREA_CALCULATION_CONFIG.py'
    elif CALCULATION_TYPE == 'gsw_masked_my_water_area':
        config_filename = 'MASKED_MY_WATER_AREA_CALCULATION_CONFIG.py'
    
    script_folder = '/WORK/Codes/global_lake_area/batch_processing'
    script_filename = 'area_calculation_config_runner.py'
    
    with open(os.path.join(config_folder, config_filename), 'r') as file:
        config_content = file.read()
    
    calculated_hybas_id_list = [
    ]
    
    hybas_id_list = [
        1020000010, 1020011530, 1020018110, 1020021940, 1020027430, 1020034170, 1020035180, 1020040190,
        2020000010, 2020003440, 2020018240, 2020024230, 2020033490, 2020041390, 2020057170, 2020065840, 2020071190,
        3020000010, 3020003790, 3020005240, 3020008670, 3020009320, 3020024310,
        4020000010, 4020006940, 4020015090, 4020024190, 4020034510, 4020050210, 4020050220, 4020050290, 4020050470,
        5020000010, 5020015660, 5020037270, 5020049720, 5020082270, 
        6020000010, 6020006540, 6020008320, 6020014330, 6020017370, 6020021870, 6020029280,
        7020000010, 7020014250, 7020021430, 7020024600, 7020038340, 7020046750, 7020047840, 7020065090,
        8020000010, 8020008900, 8020010700, 8020020760, 8020022890, 8020032840, 8020044560,
        9020000010
    ]
    
    for hybas_id in hybas_id_list:
        if hybas_id in calculated_hybas_id_list:
            continue
        print(f'Updating configuration for basin {hybas_id}')
        config_content = re.sub(r'BASIN_ID = str\(\d+\)', f'BASIN_ID = str({hybas_id})', config_content)
        with open(os.path.join(config_folder, config_filename), 'w') as file:
            file.write(config_content)
            
        command = [
            'xvfb-run', '-a',
            'python', '-u', os.path.join(script_folder, script_filename),
            '--config', os.path.join(config_folder, config_filename)
        ]
        
        subprocess.run(command)