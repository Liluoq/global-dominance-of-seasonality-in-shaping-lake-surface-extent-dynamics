import update_training_record as utr
import json
import os

os.chdir('/Codes/global_lake_area')

basin_id_list = [
    8020000010
]

final_decision_dict = {
    8020000010: 35
}

trained_dir_pattern = './trained_unet/{}'

for basin_id in basin_id_list:
    final_decision = final_decision_dict[basin_id]
    trained_dir = trained_dir_pattern.format(basin_id)
    with open(os.path.join(trained_dir, 'history.txt'), 'r') as f:
        training_records = json.load(f)

    training_acc = training_records['binary_accuracy'][final_decision]
    eval_acc = training_records['val_binary_accuracy'][final_decision]
    
    utr.update_training_record(basin_id, final_decision=final_decision, training_acc=training_acc, eval_acc=eval_acc)