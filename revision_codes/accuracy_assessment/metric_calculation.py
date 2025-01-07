import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

def calculate_classification_metrics(gdf, pred_col='prediction', truth_col='truth'):
    """
    Calculate classification metrics for binary classification
    
    Parameters:
    -----------
    gdf : GeoDataFrame
        Input geodataframe with prediction and truth columns
    pred_col : str
        Name of prediction column
    truth_col : str
        Name of truth column
        
    Returns:
    --------
    dict
        Dictionary containing all metrics
    """
    # Get arrays
    y_true = gdf[truth_col].values
    y_pred = gdf[pred_col].values
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate metrics
    # User's accuracy (Precision) for each class
    user_acc_0 = tn / (tn + fn) if (tn + fn) > 0 else 0  # Class 0
    user_acc_1 = tp / (tp + fp) if (tp + fp) > 0 else 0  # Class 1
    
    # Producer's accuracy (Recall) for each class
    prod_acc_0 = tn / (tn + fp) if (tn + fp) > 0 else 0  # Class 0
    prod_acc_1 = tp / (tp + fn) if (tp + fn) > 0 else 0  # Class 1
    
    # Overall accuracy
    overall_acc = accuracy_score(y_true, y_pred)
    
    # F1 scores
    f1 = f1_score(y_true, y_pred, average='binary')
    f1_class0 = 2 * (user_acc_0 * prod_acc_0) / (user_acc_0 + prod_acc_0) if (user_acc_0 + prod_acc_0) > 0 else 0
    f1_class1 = 2 * (user_acc_1 * prod_acc_1) / (user_acc_1 + prod_acc_1) if (user_acc_1 + prod_acc_1) > 0 else 0
    
    # Create confusion matrix as DataFrame
    conf_matrix = np.array([[tn, fp], [fn, tp]])
    
    metrics = {
        'Confusion Matrix': conf_matrix,
        "User's Accuracy (Precision)": {
            'Class 0': user_acc_0,
            'Class 1': user_acc_1
        },
        "Producer's Accuracy (Recall)": {
            'Class 0': prod_acc_0,
            'Class 1': prod_acc_1
        },
        'Overall Accuracy': overall_acc,
        'F1 Scores': {
            'Overall': f1,
            'Class 0': f1_class0,
            'Class 1': f1_class1
        }
    }
    
    return metrics

def print_classification_report(metrics):
    """
    Print classification metrics in a formatted way
    """
    print("Classification Report")
    print("-" * 50)
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print("                Predicted")
    print("              0      1")
    print(f"Actual 0    {metrics['Confusion Matrix'][0][0]:<6} {metrics['Confusion Matrix'][0][1]:<6}")
    print(f"       1    {metrics['Confusion Matrix'][1][0]:<6} {metrics['Confusion Matrix'][1][1]:<6}")
    
    print("\nUser's Accuracy (Precision):")
    print("Class 0: {:.4f}".format(metrics['User\'s Accuracy (Precision)']['Class 0']))
    print("Class 1: {:.4f}".format(metrics['User\'s Accuracy (Precision)']['Class 1']))
    
    print("\nProducer's Accuracy (Recall):")
    print("Class 0: {:.4f}".format(metrics['Producer\'s Accuracy (Recall)']['Class 0']))
    print("Class 1: {:.4f}".format(metrics['Producer\'s Accuracy (Recall)']['Class 1']))
    
    print("\nOverall Accuracy: {:.4f}".format(metrics['Overall Accuracy']))
    
    print("\nF1 Scores:")
    print("Overall: {:.4f}".format(metrics['F1 Scores']['Overall']))
    print("Class 0: {:.4f}".format(metrics['F1 Scores']['Class 0']))
    print(f"Class 1: {metrics['F1 Scores']['Class 1']:.4f}")