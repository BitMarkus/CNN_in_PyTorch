#############
# FUNCTIONS #
#############

import sys
import pathlib
from tqdm import tqdm
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler

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



