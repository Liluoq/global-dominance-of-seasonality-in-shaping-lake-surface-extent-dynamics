import pandas as pd
import argparse

def update_training_record(model_name, trained_status=None, training_set_size=None, 
                           evaluation_set_size=None, test_set_size=None, final_decision=None, training_acc=None, 
                           eval_acc=None, test_acc=None, training_IoU=None, eval_IoU=None, test_IoU=None,
                           keep_ratio=None, batch_processed=None,
                           file_path='training_records.csv'):
    required_columns = ['Model Name', 'Trained', 'Training set size', 'Evaluation set size', 'Test set size', 'Keep ratio', 'Final_decision', 
                        'Training_acc', "Eval_acc", "Test_acc", "Training_IoU", "Eval_IoU", "Test_IoU", "Batch_processed"]
    try:
        # Attempt to read the existing CSV file
        df = pd.read_csv(file_path)
        # Check if the DataFrame has all the required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            # Add the missing columns to the DataFrame
            for col in missing_columns:
                df[col] = None
    except FileNotFoundError:
        # If the file does not exist, initialize a new DataFrame
        df = pd.DataFrame(columns=required_columns)

    # Check if the model already exists in the DataFrame
    if model_name in df['Model Name'].values:
        # Update only the fields that are not None
        if trained_status is not None:
            df.loc[df['Model Name'] == model_name, 'Trained'] = trained_status
        if training_set_size is not None:
            df.loc[df['Model Name'] == model_name, 'Training set size'] = training_set_size
        if evaluation_set_size is not None:
            df.loc[df['Model Name'] == model_name, 'Evaluation set size'] = evaluation_set_size
        if test_set_size is not None:
            df.loc[df['Model Name'] == model_name, 'Test set size'] = test_set_size
        if final_decision is not None:
            df.loc[df['Model Name'] == model_name, 'Final_decision'] = final_decision
        if training_acc is not None:
            df.loc[df['Model Name'] == model_name, 'Training_acc'] = training_acc
        if eval_acc is not None:
            df.loc[df['Model Name'] == model_name, 'Eval_acc'] = eval_acc
        if test_acc is not None:
            df.loc[df['Model Name'] == model_name, 'Test_acc'] = test_acc
        if training_IoU is not None:
            df.loc[df['Model Name'] == model_name, 'Training_IoU'] = training_IoU
        if eval_IoU is not None:
            df.loc[df['Model Name'] == model_name, 'Eval_IoU'] = eval_IoU
        if test_IoU is not None:
            df.loc[df['Model Name'] == model_name, 'Test_IoU'] = test_IoU
        if keep_ratio is not None:
            df.loc[df['Model Name'] == model_name, 'Keep ratio'] = keep_ratio
        if batch_processed is not None:
            df.loc[df['Model Name'] == model_name, 'Batch_processed'] = batch_processed
    else:
        # If the model does not exist, append a new record
        # Create a dictionary with all the information
        new_record = {'Model Name': model_name}
        if trained_status is not None:
            new_record['Trained'] = trained_status
        if training_set_size is not None:
            new_record['Training set size'] = training_set_size
        if evaluation_set_size is not None:
            new_record['Evaluation set size'] = evaluation_set_size
        if test_set_size is not None:
            new_record['Test set size'] = test_set_size
        if final_decision is not None:
            new_record['Final_decision'] = final_decision
        if training_acc is not None:
            new_record['Training_acc'] = training_acc
        if eval_acc is not None:
            new_record['Eval_acc'] = eval_acc
        if test_acc is not None:
            new_record['Test_acc'] = test_acc
        if training_IoU is not None:
            new_record['Training_IoU'] = training_IoU
        if eval_IoU is not None:
            new_record['Eval_IoU'] = eval_IoU
        if test_IoU is not None:
            new_record['Test_IoU'] = test_IoU
        if keep_ratio is not None:
            new_record['Keep ratio'] = keep_ratio
        if batch_processed is not None:
            new_record['Batch_processed'] = batch_processed
        # Append the new record
        df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)

    # Save the updated DataFrame back to the CSV file
    df.to_csv(file_path, index=False)

def get_training_status(model_name, file_path='training_records.csv'):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print('Training records not found')
        return None
    if model_name in df['Model Name'].values:
        return df.loc[df['Model Name'] == model_name, 'Trained'].values[0]
    else:
        print('Model not found in training records')
        return None
    
def get_batch_processed_status(model_name, file_path='training_records.csv'):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print('Training records not found')
        return None
    if model_name in df['Model Name'].values:
        return df.loc[df['Model Name'] == model_name, 'Batch_processed'].values[0]
    else:
        print('Model not found in training records')
        return None

def get_final_decision(model_name, file_path='training_records.csv'):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print('Training records not found')
        return None
    if model_name in df['Model Name'].values:
        return df.loc[df['Model Name'] == model_name, 'Final_decision'].values[0]
    else:
        print('Model not found in training records')
        return None

def get_keep_ratio(model_name, file_path='training_records.csv'):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print('Training records not found')
        return None
    if model_name in df['Model Name'].values:
        return df.loc[df['Model Name'] == model_name, 'Keep ratio'].values[0]
    else:
        print('Model not found in training records')
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Update training record')
    parser.add_argument('--model_name', type=str, required=True, help='Model name')
    parser.add_argument('--trained_status', type=str, default=None, help='Trained status')
    parser.add_argument('--training_set_size', type=int, default=None, help='Training set size')
    parser.add_argument('--evaluation_set_size', type=int, default=None, help='Evaluation set size')
    parser.add_argument('--final_decision', type=str, default=None, help='Final decision')
    parser.add_argument('--training_acc', type=float, default=None, help='Training accuracy')
    parser.add_argument('--eval_acc', type=float, default=None, help='Evaluation accuracy')
    parser.add_argument('--test_acc', type=float, default=None, help='Test accuracy')
    parser.add_argument('--training_IoU', type=float, default=None, help='Training IoU')
    parser.add_argument('--eval_IoU', type=float, default=None, help='Evaluation IoU')
    parser.add_argument('--test_IoU', type=float, default=None, help='Test IoU')
    parser.add_argument('--file_path', type=str, default='training_records.csv', help='Path to the training records CSV file')
    parser.add_argument('--keep_ratio', type=float, default=None, help='Keep ratio')
    parser.add_argument('--batch_processed', type=str, default=None, help='Batch processed status')
    
    args = parser.parse_args()
    model_name = args.model_name
    trained_status = args.trained_status
    training_set_size = args.training_set_size
    evaluation_set_size = args.evaluation_set_size
    final_decision = args.final_decision
    training_acc = args.training_acc
    eval_acc = args.eval_acc
    test_acc = args.test_acc
    training_IoU = args.training_IoU
    eval_IoU = args.eval_IoU
    test_IoU = args.test_IoU
    file_path = args.file_path
    keep_ratio = args.keep_ratio
    batch_processed = args.batch_processed
    
    update_training_record(model_name=model_name, trained_status=trained_status, training_set_size=training_set_size, 
                           evaluation_set_size=evaluation_set_size, final_decision=final_decision, training_acc=training_acc, 
                           eval_acc=eval_acc, test_acc=test_acc, training_IoU=training_IoU, eval_IoU=eval_IoU, test_IoU=test_IoU,
                           keep_ratio=keep_ratio, batch_processed=batch_processed,
                           file_path=file_path)
    