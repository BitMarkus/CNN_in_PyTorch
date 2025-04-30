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

    # Prepare training and prediction data
    def _prepare_data(self, dataset_path):
        # Clear and create directories
        for folder in ["WT", "KO"]:
            for path in [setting["pth_train"], setting["pth_test"]]:
                shutil.rmtree(os.path.join(path, folder), ignore_errors=True)
                os.makedirs(os.path.join(path, folder), exist_ok=True)        
        # Copy data
        for phase, src_dir, dest_dir in [
            ("train", os.path.join(dataset_path, "train"), setting["pth_train"]),
            ("prediction", os.path.join(dataset_path, "prediction"), setting["pth_test"])
        ]:
            for folder in ["WT", "KO"]:
                src = os.path.join(src_dir, folder)
                dst = os.path.join(dest_dir, folder)
                for img in os.listdir(src):
                    shutil.copy2(os.path.join(src, img), dst)