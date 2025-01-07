import re
import os
import subprocess

if __name__ == '__main__':
    processed_hybas_id_list = [
        
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

    config_path = '/WORK/Codes/global_lake_area/my_spatial_analyze/LAKE_WISE_AREA_POSTPROCESSING_CONFIG.py'
    runner_script_path = '/WORK/Codes/global_lake_area/my_spatial_analyze/lake_wise_area_postprocessor.py'
    with open(config_path, 'r') as file:
        config_content = file.read()
    for hybas_id in hybas_id_list:
        if hybas_id in processed_hybas_id_list:
            continue
        config_content = re.sub(r'BASIN_ID = \d+', f'BASIN_ID = {hybas_id}', config_content)
        with open(config_path, 'w') as file:
            file.write(config_content)
        command = [
            'xvfb-run', '-a',
            'python', '-u', runner_script_path,
            '--config', config_path
        ]
        subprocess.run(command)