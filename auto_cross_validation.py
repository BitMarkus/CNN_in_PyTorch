import os
import shutil
import torch
from tqdm import tqdm
from settings import setting
import functions as fn
from train import Train

class AutoCrossValidation:

    #############################################################################################################
    # CONSTRUCTOR:

    def __init__(self, device, cnn, dataset):

        # Passed parameters
        self.device = device
        self.cnn = cnn
        self.ds = dataset

    #############################################################################################################
    # METHODS:

