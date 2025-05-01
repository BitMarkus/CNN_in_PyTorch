import os
import shutil
import torch
from tqdm import tqdm
from settings import setting
import functions as fn
from train import Train
from dataset_gen import DatasetGenerator
from custom_model import Custom_CNN_Model
from dataset import Dataset
# Own modules
from settings import setting

class AutoCrossValidation:

    #############################################################################################################
    # CONSTRUCTOR:

    def __init__(self, device):

        # Passed parameters
        self.device = device

        # Settings parameters
        self.wt_lines = setting['wt_lines']
        self.ko_lines = setting['ko_lines']
        self.acv_results_dir = setting['pth_acv_results']
        self.data_dir = setting['pth_data']

        # Objects
        # Create a dataset generation object (in automatic cross validation mode)
        self.ds_gen = DatasetGenerator(mode = "acv")
        # Create a dataset object
        self.ds = Dataset()
        # Create a model object
        self.cnn = Custom_CNN_Model()

    #############################################################################################################
    # METHODS:

    def __call__(self):

        # Clean up folders train and test in the data directory (old training and test data)
        print("Cleaning up old train and test data...")
        self.ds_gen.cleanup(self.data_dir) 
        print("Cleanup finished.")  

        # Generate cross validation datasets
        configs = self.ds_gen.get_dataset_configs()
        # print(configs)
        # Iterate over each dataset
        for config in configs:

            ##################
            # Create dataset #
            ##################
            
            # Generate dataset and folders for training results
            print(f"\n>> PROCESSING DATASET {config['dataset_idx']}:")

            print(f"Cell line for testing WT group: {config['test_wt']}")
            print(f"Cell line for testing KO group: {config['test_ko']}")

            print(f"\n> Create dataset {config['dataset_idx']}...")
            dataset_dir = self.ds_gen.generate_dataset(**config) # Unpacks dict as kwargs
            # Create a subfolder for checkpoints
            ckeckpoint_dir = os.path.join(dataset_dir, "checkpoints/")
            os.makedirs(ckeckpoint_dir, exist_ok=True)
            # Create a subfolder for plots
            plot_dir = os.path.join(dataset_dir, "plots/")
            os.makedirs(plot_dir, exist_ok=True)
            print(f"Dataset {config['dataset_idx']} successfully created.")

            ######################
            # Load train dataset #
            ######################

            print(f"\n> Load dataset {config['dataset_idx']} for training...")
            self.ds.load_training_dataset()
            if(self.ds.ds_loaded):
                print(f"Dataset {config['dataset_idx']} successfully loaded.")
                print(f"Number training images/batches: {self.ds.num_train_img}/{self.ds.num_train_batches}")
                print(f"Number validation images/batches: {self.ds.num_val_img}/{self.ds.num_val_batches}") 

            ##############
            # Create CNN #
            ##############

            print(f"\n> Creating new {self.cnn.cnn_type} network...")
            if setting["cnn_type"] == "custom":
                self.cnn.model = Custom_CNN_Model().to(self.device)
            else:
                self.cnn.model = self.cnn.load_model(self.device)
            print("New network successfully created.")

            ####################
            # Train on dataset #
            ####################

            # Create Train object with loaded dataset
            self.train = Train(self.cnn, self.ds)
            print(f"\n> Start training on dataset {config['dataset_idx']}...")
            self.train.train(ckeckpoint_dir, plot_dir)
            print(f"\nTraining on dataset {config['dataset_idx']} successfully finished.")

            ##################
            # Cleanup images #
            ##################

            # Clean up folders train and test in the data directory for next dataset
            print(f"\n> Cleaning up train and test data from dataset {config['dataset_idx']}...")
            self.ds_gen.cleanup(setting["pth_data"])
            print("Cleanup finished.")    

            print(f"\n>> PROCESSING OF DATASET {config['dataset_idx']} FINISHED!")         
