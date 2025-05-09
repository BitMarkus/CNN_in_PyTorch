#############
# FUNCTIONS #
#############

import sys
from pathlib import Path
import torch
from datetime import datetime
import matplotlib.pyplot as plt
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



