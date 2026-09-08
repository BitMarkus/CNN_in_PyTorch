#############
# FUNCTIONS #
#############

# ===== Standard Library Imports =====
import sys
from pathlib import Path
import json
# ===== Third-Party Imports =====
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# ===== Own Modules =====
from settings import setting

# Show system information (CUDA availability and software versions)
# and select the appropriate device (GPU or CPU).
# Returns:
#   torch.device: Selected device (cuda or cpu)
def show_cuda_and_versions() -> torch.device:
    print("\n>> DEVICE:")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using Device:", device)
    print(">> VERSIONS:")
    print("Python: ", sys.version, "")
    print("Pytorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("cuDNN:", torch.backends.cudnn.version())
    # print("Captum:", captum.__version__)
    return device

# Prompt the user for input and validate that it is an integer.
# Loops until a valid integer is entered.
# Args:
#   prompt (str): The prompt to display to the user
# Returns:
#   int: The validated integer input
def input_int(prompt: str) -> int:
    while True:
        nr = input(prompt)
        if not check_int(nr):
            print("Input is not an integer number! Try again...")
        else:
            return int(nr)

# Check if a variable can be converted to an integer.
# Args:
#   var: The variable to check
# Returns:
#   bool: True if conversion is successful, False otherwise
def check_int(var) -> bool:
    try:
        val = int(var)
        return True
    except ValueError:
        return False

# Check if an integer parameter is within a specified range.
# Args:
#   var (int): The value to check
#   min (int): Minimum allowed value
#   max (int): Maximum allowed value
# Returns:
#   bool: True if within range, False otherwise
def check_int_range(var: int, min: int, max: int) -> bool:
    if var >= min and var <= max:
        return True
    else:
        return False

# Create all working folders in the root directory of the program if they do not exist.
# Creates:
#   - data/train/ with class subfolders (training images)
#   - data/test/ with class subfolders (test/validation images)
#   - checkpoints/ (for loading existing models)
#   - dataset_gen/input_synthetic/ (synthetic images for cross-validation)
#   - dataset_gen/input_real/ (real images for cross-validation)
#   - dataset_gen/input_mixed/ (mixed images for cross-validation, backward compatibility)
#   - dataset_gen/output/ (generated cross-validation datasets)
#   - input/ (all analysis inputs)
#   - output/ (all analysis outputs)
# Returns:
#   bool: True if successful
def create_prg_folders() -> bool:
    # Training and test folders
    train_base_pth = setting["pth_train"]
    train_base_pth.mkdir(parents=True, exist_ok=True)
    for class_name in setting["classes"]:
        (train_base_pth / class_name).mkdir(exist_ok=True)

    test_base_pth = setting["pth_test"]
    test_base_pth.mkdir(parents=True, exist_ok=True)
    for class_name in setting["classes"]:
        (test_base_pth / class_name).mkdir(exist_ok=True)

    # Dataset generator folders
    setting["pth_ds_gen_input_synthetic"].mkdir(parents=True, exist_ok=True)
    setting["pth_ds_gen_input_real"].mkdir(parents=True, exist_ok=True)
    setting["pth_ds_gen_input_mixed"].mkdir(parents=True, exist_ok=True)
    setting["pth_ds_gen_output"].mkdir(parents=True, exist_ok=True)

    # Checkpoints (for loading existing models)
    setting["pth_checkpoint"].mkdir(parents=True, exist_ok=True)

    # Input and Output Folders
    setting["pth_input"].mkdir(parents=True, exist_ok=True)
    setting["pth_output"].mkdir(parents=True, exist_ok=True)

    return True

# Plot a confusion matrix as a normalized heatmap.
# Args:
#   cm (dict): Dictionary with 'y' (true labels) and 'y_hat' (predicted labels)
#   class_list (list): List of class names in order
#   plot_path (Path): Directory to save the plot
#   chckpt_name (str, optional): Name of the checkpoint for the filename
#   show_plot (bool): If True, display the plot interactively
#   save_plot (bool): If True, save the plot to disk
def plot_confusion_matrix(
    cm: dict,
    class_list: list,
    plot_path: Path,
    chckpt_name: str = None,
    show_plot: bool = False,
    save_plot: bool = True
) -> None:
    # Print confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        cm["y"],
        cm["y_hat"],
        display_labels=class_list,
        cmap='Blues',
        normalize='true',
    )
    plt.tight_layout()

    # Save plot
    if save_plot:
        if chckpt_name is None:
            cm_file_name = "confusion_matrix.png"
        else:
            cm_file_name = str(chckpt_name) + "_cm.png"
        plt.savefig(plot_path / cm_file_name, bbox_inches='tight')

    # Show plot
    if show_plot:
        plt.show()


# Save confusion matrix results with accuracies to a JSON file.
# Args:
#   cm (dict or array): Dictionary with 'y' and 'y_hat' keys, or a confusion matrix array
#   class_list (list): List of class names in order
#   file_path (Path): Directory to save the file
#   chckpt_name (str, optional): Name of the checkpoint for the filename
def save_confusion_matrix_results(
    cm,
    class_list: list,
    file_path: Path,
    chckpt_name: str = None
) -> None:
    # Handle different input types
    if isinstance(cm, dict):
        # Format with 'y' and 'y_hat' keys
        y_true = cm["y"]
        y_pred = cm["y_hat"]
        cm_array = confusion_matrix(y_true, y_pred)
    elif isinstance(cm, (list, np.ndarray)):
        # Direct confusion matrix array
        cm_array = np.array(cm)
        y_true = None
        y_pred = None
    else:
        raise TypeError(f"Expected dict or array, got {type(cm)}")

    # Calculate metrics
    with np.errstate(divide='ignore', invalid='ignore'):
        class_acc = cm_array.diagonal() / cm_array.sum(axis=1)
    class_acc = np.nan_to_num(class_acc)
    overall_acc = np.sum(cm_array.diagonal()) / np.sum(cm_array)

    # Prepare results
    results = {
        "classes": class_list,
        "confusion_matrix": cm_array.tolist(),
        "class_accuracy": dict(zip(class_list, class_acc.tolist())),
        "overall_accuracy": overall_acc,
    }

    # Add labels if available
    if y_true is not None:
        results["true_labels"] = y_true.tolist() if hasattr(y_true, 'tolist') else y_true
        results["predicted_labels"] = y_pred.tolist() if hasattr(y_pred, 'tolist') else y_pred

    # Determine filename
    if chckpt_name is None:
        file_name = "cm_results"
    else:
        file_name = str(chckpt_name) + "_cm"

    # Save file
    with open(file_path / f"{file_name}.json", 'w') as f:
        json.dump(results, f, indent=4)

# Load confusion matrix results from a JSON file.
# Args:
#   file_path (Path): Directory containing the file
#   file_name (str, optional): Name of the checkpoint file
# Returns:
#   dict: Dictionary containing all saved results, or None if loading fails
def load_confusion_matrix_results(file_path: Path, file_name: str = None):
    try:
        # Determine filename
        if file_name is None:
            f_name = "cm_results"
        else:
            f_name = Path(file_name).stem + "_cm"

        # Construct full path
        full_path = file_path / f"{f_name}.json"

        # Check if file exists
        if not full_path.exists():
            raise FileNotFoundError(f"Results file not found at: {full_path}")

        # Load and return data
        with open(full_path, 'r') as f:
            return json.load(f)

    except Exception as e:
        print(f"Error loading confusion matrix results: {e}")
        return None