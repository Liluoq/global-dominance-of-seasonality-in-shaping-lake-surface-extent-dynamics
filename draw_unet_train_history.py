import unetgee as ug
import json
import os

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

for basin_id in basin_id_list:
    history_path = f'/WORK/Codes/global_lake_area/trained_unet_10bands_att_laea_no_zenith/{basin_id}/history.txt'
    if not os.path.exists(history_path):
        continue
    save_path = f'/WORK/Codes/global_lake_area/trained_unet_10bands_att_laea_no_zenith/training_history_figs/{basin_id}_loss_and_accuracy_new.pdf'
    with open(history_path, 'r') as file:
        history_dict = json.load(file)
        
        ug.draw_unet_train_figure(history=history_dict, save_path=save_path)

    
    