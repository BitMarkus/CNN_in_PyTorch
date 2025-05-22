#############
# FUNCTIONS #
#############

import sys
from pathlib import Path
import torch
from datetime import datetime
import matplotlib.pyplot as plt
import json
import csv
import os
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
# Own modules
from settings import setting

# Function to show if CUDA is working and software versions
def show_cuda_and_versions():
    print("\n>> DEVICE:")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using Device:", device)
    print(">> VERSIONS:")
    print("Python: ", sys.version, "")
    print("Pytorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("cuDNN:", torch.backends.cudnn.version())
    return device

# Creates an input with prompt
# which is checked, if the input is an integer number
# If not, the loop will continue until a valid number is entered
def input_int(prompt):
    while(True):
        nr = input(prompt)
        if not(check_int(nr)):
            print("Input is not an integer number! Try again...")
        else:
            return int(nr)  
        
# Check variable for int
# Returns True if conversion was successful
# or False when the variable cannot be converted to an integer number
def check_int(var):
    try:
        val = int(var)
        return True
    except ValueError:
        return False
       
# Checks if int parameters are within a certain range
def check_int_range(var, min, max):
    if(var >= min and var <= max):
        return True
    else:
        return False

# Function creates all working folders in the root directory of the program
# If they do not exist yet!
def create_prg_folders():
    # Create folder for training images with classes subfolders
    train_base_pth = Path(setting["pth_train"])
    train_base_pth.mkdir(parents=True, exist_ok=True)
    for class_name in setting["classes"]:
        class_dir = train_base_pth / class_name
        class_dir.mkdir(exist_ok=True)
    # Create folder for testing images with classes subfolders
    test_base_pth = Path(setting["pth_test"])
    test_base_pth.mkdir(parents=True, exist_ok=True)
    for class_name in setting["classes"]:
        class_dir = test_base_pth / class_name
        class_dir.mkdir(exist_ok=True)
    # Other folders
    Path(setting["pth_checkpoint"]).mkdir(parents=True, exist_ok=True)
    Path(setting["pth_plots"]).mkdir(parents=True, exist_ok=True)
    Path(setting["pth_ds_gen_input"]).mkdir(parents=True, exist_ok=True)
    Path(setting["pth_ds_gen_output"]).mkdir(parents=True, exist_ok=True)
    Path(setting["pth_prediction"]).mkdir(parents=True, exist_ok=True)
    
    return True

# Function to save plots
def save_plot_to_drive(plot_path, file_name):
    # Datetime for saved files
    current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M")
    # Generate filename
    filename = f'{current_datetime}_{setting["cnn_type"]}_{file_name}.png'
    # Save plot
    plt.savefig(str(plot_path) + '/' + filename, bbox_inches='tight')

# Function to plot a confusion matrix
def plot_confusion_matrix(cm, class_list, plot_path, show_plot=True, save_plot=True):
    # # Print confusion matrix 
    # https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html#sklearn.metrics.ConfusionMatrixDisplay.from_predictions
    ConfusionMatrixDisplay.from_predictions(
        cm["y"], 
        cm["y_hat"], 
        display_labels=class_list, 
        cmap='Blues', 
        # normalize='pred', # the confusion matrix is normalized over the predicted conditions (e.g. columns)
        normalize='true', # the confusion matrix is normalized over the true conditions (e.g. rows)
    )
    plt.tight_layout()
    # Save plot
    if(save_plot):
        save_plot_to_drive(plot_path, "confusion_matrix")
    # Show and save plot
    if(show_plot):
        plt.show()   

# Save confusion matrix results with accuracies to a structured file.
# Args:
#    cm (dict): Dictionary with 'y' (true labels) and 'y_hat' (predicted labels)
#    class_list (list): List of class names in order
#    file_path (str): Path to save the file (without extension)
#    format (str): 'json' or 'csv' file format
def save_confusion_matrix_results(cm, class_list, file_path, format='json'):

    # Calculate confusion matrix
    cm_array = confusion_matrix(cm["y"], cm["y_hat"])
    
    # Calculate class-wise accuracy (normalized by true labels)
    with np.errstate(divide='ignore', invalid='ignore'):
        class_acc = cm_array.diagonal() / cm_array.sum(axis=1)
    class_acc = np.nan_to_num(class_acc)  # Replace NaN with 0
    
    # Calculate overall accuracy
    overall_acc = np.sum(cm_array.diagonal()) / np.sum(cm_array)
    
    # Prepare data structure
    results = {
        "classes": class_list,
        "confusion_matrix": cm_array.tolist(),
        "class_accuracy": dict(zip(class_list, class_acc.tolist())),
        "overall_accuracy": overall_acc,
        "true_labels": cm["y"],
        "predicted_labels": cm["y_hat"]
    }

    # Save in requested format
    if format.lower() == 'json':
        with open(f"{file_path}.json", 'w') as f:
            json.dump(results, f, indent=4)
    elif format.lower() == 'csv':
        # Save main metrics
        with open(f"{file_path}_metrics.csv", 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Class', 'Accuracy'])
            for class_name, acc in zip(class_list, class_acc):
                writer.writerow([class_name, acc])
            writer.writerow(['Overall', overall_acc])
        
        # Save full confusion matrix
        with open(f"{file_path}_matrix.csv", 'w') as f:
            writer = csv.writer(f)
            writer.writerow([''] + class_list)  # Header row
            for i, row in enumerate(cm_array):
                writer.writerow([class_list[i]] + row.tolist())
    else:
        raise ValueError("Format must be 'json' or 'csv'")

# Load confusion matrix results from file.
# Args:
#    file_path (str): Path to the file (without extension)
#    format (str): 'json' or 'csv' file format     
# Returns:
#    dict: Dictionary containing all saved results
def load_confusion_matrix_results(file_path, format='json'):

    if format.lower() == 'json':
        with open(f"{file_path}.json", 'r') as f:
            return json.load(f)
    elif format.lower() == 'csv':
        results = {}
        
        # Load metrics
        with open(f"{file_path}_metrics.csv", 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            class_acc = {}
            for row in reader:
                if row[0] == 'Overall':
                    results['overall_accuracy'] = float(row[1])
                else:
                    class_acc[row[0]] = float(row[1])
        
        # Load confusion matrix
        with open(f"{file_path}_matrix.csv", 'r') as f:
            reader = csv.reader(f)
            class_list = next(reader)[1:]  # Get class names from header
            cm_array = []
            for row in reader:
                cm_array.append([int(x) for x in row[1:]])
        
        # Reconstruct the results dictionary
        results.update({
            'classes': class_list,
            'confusion_matrix': cm_array,
            'class_accuracy': class_acc,
            # Note: CSV format doesn't store true/predicted labels
        })
        
        return results
    else:
        raise ValueError("Format must be 'json' or 'csv'")