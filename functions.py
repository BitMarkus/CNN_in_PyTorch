#############
# FUNCTIONS #
#############

import sys
from pathlib import Path
from tqdm import tqdm
import torch
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
    return device

# Creates an input with prompt
# which is checked, if the input is an integer number
# If not, the loop will continue until a valid number is entered
def input_int(prompt):
    while(True):
        nr = input(prompt)
        if not(check_int(nr)):
            print("Input is not an integer number! Try again:")
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

# Function for Prediction
def predict(model, dataset):

    # Load dataset
    num_correct = 0
    num_samples = 0

    model.eval()
    # No need to keep track of gradients
    with torch.no_grad():
        # Loop through the data
        for i, (images, labels) in enumerate(tqdm(dataset)):
            # Send images and labels to gpu
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()
            # Forward pass
            scores = model(images)
            _, predictions = scores.max(1)
            # Check how many we got correct
            num_correct += (predictions == labels).sum()
            # Keep track of number of samples
            num_samples += predictions.size(0)

    model.train()
    return num_correct / num_samples 

# Function creates all working folders in the root directory of the program
# If they do not exist yet!
def create_prg_folders():
    # https://kodify.net/python/pathlib-path-mkdir-method/
    Path(setting["pth_data"]).mkdir(parents=True, exist_ok=True)
    Path(setting["pth_checkpoint"]).mkdir(parents=True, exist_ok=True)
    Path(setting["pth_plots"]).mkdir(parents=True, exist_ok=True)
    Path(setting["pth_prediction"]).mkdir(parents=True, exist_ok=True)



