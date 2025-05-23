#############
# FUNCTIONS #
#############

import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import json
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

# Function to plot a confusion matrix
def plot_confusion_matrix(cm, class_list, plot_path, chckpt_name=None, show_plot=False, save_plot=True):
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
        # Name of the cm file will be the checkpoint file it is based on
        # If no name was passed, it will be just called "confusion_matrix"
        if(chckpt_name is None):
            cm_file_name = "confusion_matrix"
        else:
            # Remove extension from checkpoint name if necessary
            cm_file_name = str(Path(chckpt_name).stem) + "_cm" 
        # Save plot
        plt.savefig(str(plot_path) + '/' + cm_file_name, bbox_inches='tight')
    # Show and save plot
    if(show_plot):
        plt.show()   

# Save confusion matrix results with accuracies to a json file.
# Args:
#    cm (dict): Dictionary with 'y' (true labels) and 'y_hat' (predicted labels)
#    class_list (list): List of class names in order
#    file_path (str): Path to save the file (without extension)
#    chckpt_name (str): Name of the checkpoint file the confusion matrix is based on
def save_confusion_matrix_results(cm, class_list, file_path, chckpt_name=None):

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

    # Name of the result file will be the checkpoint file it is based on
    # If no name was passed, it will be just called "results"
    if(chckpt_name is None):
        file_name = "cm_results"
    else:
        # Remove extension from checkpoint name if necessary
        file_name = str(Path(chckpt_name).stem) + "_cm"

    # Save file
    with open(f"{file_path}{file_name}.json", 'w') as f:
        json.dump(results, f, indent=4)

# Load confusion matrix results from file.
# Args:
#    file_path (str): Path to the file (without extension) 
#    file_name (str): Name of the checkpoint file the confusion matrix is based on or none
# Returns:
#    dict: Dictionary containing all saved results
def load_confusion_matrix_results(file_path, file_name=None):
        
    # Name of the result file is the checkpoint file it is based on
    # If no name was passed, the name will be "results"
    try:
        # Determine filename
        if file_name is None:
            f_name = "cm_results"
        else:
            f_name = str(Path(file_name).stem) + "_cm"
        
        # Construct full path
        full_path = Path(file_path) / f"{f_name}.json"
        
        # Check if file exists
        if not full_path.exists():
            raise FileNotFoundError(f"Results file not found at: {full_path}")
        
        # Load and return data
        with open(full_path, 'r') as f:
            return json.load(f)
            
    except Exception as e:
        print(f"Error loading confusion matrix results: {e}")
        return None
