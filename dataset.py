####################
# Dataset handling #
####################

from torchvision.transforms import transforms
import torchvision
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
# Own modules
from settings import setting

class Dataset():

    #############################################################################################################
    # CONSTRUCTOR:
    
    def __init__(self):
        # Path to training images
        self.pth_train = setting["pth_train"]
        # Path to test images
        self.pth_test = setting["pth_test"]
        # Path to prediction images
        self.pth_prediction = setting["pth_prediction"]
        # Settings variables
        # Shuffle training images befor validation split
        self.shuffle = setting["ds_shuffle"]
        # Shuffle seed
        self.shuffle_seed = setting["ds_shuffle_seed"]
        # Batch size for training and validation datasets (for 512x512 -> 24)
        self.batch_size = setting["ds_batch_size"]
        # Batch size for prediction dataset
        # Always needs to be 1! Or calculation of confusion matrix parameters are more complicated
        self.batch_size_pred = 1
        # Fraction of images which go into the validation dataset 
        self.val_split = setting["ds_val_split"]
        # Number of channels of training images 
        self.input_channels = setting["img_channels"]
        # Image width and height for training
        self.input_height = setting["img_height"]
        self.input_width = setting["img_width"]
        # Variable to save if the dataset was alread loaded or not
        self.ds_loaded = False   
        # Number of training and validation images in each dataset
        self.num_train_img = 0
        self.num_val_img = 0
        self.num_pred_img = 0
        # Number of training and validation batches in each dataset
        self.num_train_batches = 0
        self.num_val_batches = 0
        # Datasets
        self.ds_train = None
        self.ds_val = None
        self.ds_pred = None
        # Augentatioms for dataset
        self.train_use_augment = setting["train_use_augment"]

    #############################################################################################################
    # METHODS:

    # Transformer for testing and predicting WITHOUT AUGMENTATIONS
    def get_transformer_test(self):
        # Transformer for Grayscale images
        # https://stackoverflow.com/questions/60116208/pytorch-load-dataset-of-grayscale-images
        if(self.input_channels == 1):
            transformer = transforms.Compose([        
                # 0-255 to 0-1, numpy to tensors:
                transforms.ToTensor(), 
                # Normalization
                transforms.Grayscale(num_output_channels=1), # <- Grayscale
            ])
            return transformer

        # Transformer for RGB images
        # https://pytorch.org/hub/pytorch_vision_resnet/
        elif(self.input_channels == 3):
            transformer = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # <- RGB
            ])
            return transformer

        else:
            return False

    # Transformer for training WITH AUGMENTATIONS
    def get_transformer_train(self):
        # Transformer for Grayscale images
        # https://stackoverflow.com/questions/60116208/pytorch-load-dataset-of-grayscale-images
        if(self.input_channels == 1):
            transformer = transforms.Compose([

                # No resize needed as the images are already 512x512
                # transforms.Resize((self.input_height, self.input_width)),
                
                # Augmentations (DIC specific):
                # Random 90° rotations (0°, 90°, 180°, 270°)
                transforms.RandomChoice([
                    transforms.RandomRotation(degrees=[0, 0]),    # 0° (25% chance)
                    transforms.RandomRotation(degrees=[90, 90]),  # 90° (25% chance)
                    transforms.RandomRotation(degrees=[180, 180]), # 180° (25% chance)
                    transforms.RandomRotation(degrees=[270, 270]), # 270° (25% chance)
                ]),

                # Small-angle rotations (±10°)
                # The two rotation steps will compound (e.g., an image might get 5° + 90° rotation)
                # transforms.RandomRotation(degrees=10, expand=False, fill=0),

                # Brightness/contrast adjustments
                # Values of 0.2 (~±20% change) are conservative to avoid unrealistic images
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0, hue=0),

                # DIC images are sensitive to focus shifts. Add slight blur to simulate focal plane variability
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))], p=0.3),                

                # 0-255 to 0-1, numpy to tensors:
                transforms.ToTensor(), 

                # Normalization
                transforms.Grayscale(num_output_channels=1), # <- Grayscale
            ])
            return transformer

        # Transformer for RGB images
        # https://pytorch.org/hub/pytorch_vision_resnet/
        elif(self.input_channels == 3):
            transformer = transforms.Compose([
                # transforms.Resize((self.input_height, self.input_width)),
                # Augmentations (DIC specific):
                transforms.RandomChoice([
                    transforms.RandomRotation(degrees=[0, 0]),
                    transforms.RandomRotation(degrees=[90, 90]),
                    transforms.RandomRotation(degrees=[180, 180]),
                    transforms.RandomRotation(degrees=[270, 270]),
                ]),
                # transforms.RandomRotation(degrees=10, expand=False, fill=0),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0, hue=0),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))], p=0.3),   
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # <- RGB
            ])
            return transformer

        else:
            return False

    # Loads and splits images in a folder into 2 datasets (train, validation)
    # https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets/50544887#50544887
    def load_training_dataset(self):
        # Training dataset with or without augmentations
        if(self.train_use_augment):
            transformer = self.get_transformer_train()
        else:
            transformer = self.get_transformer_test()
        # Check if training images have either one or three channels
        if(transformer):
            # No change needed here - ImageFolder accepts Path objects
            dataset = torchvision.datasets.ImageFolder(self.pth_train, transform=transformer)
            dataset_size = len(dataset)
            indices = list(range(dataset_size))
            split = int(np.floor(self.val_split * dataset_size))
            if self.shuffle:
                np.random.seed(self.shuffle_seed)
                np.random.shuffle(indices)
            train_indices, val_indices = indices[split:], indices[:split]
            # Set number of images in each dataset
            self.num_train_img = len(train_indices)
            self.num_val_img = len(val_indices)        
            # Take a subset of indices
            # This should lead to a dataset, which contains always the same images (dependent on shuffle seed)
            # However, the order of the images within the dataset should be different
            # Maybe torch.utils.data.SequentialSampler would be better here?
            train_sampler = SubsetRandomSampler(train_indices)
            valid_sampler = SubsetRandomSampler(val_indices)       
            # Create train dataloader object
            train_loader = DataLoader(
                dataset, 
                batch_size=self.batch_size, 
                sampler=train_sampler,
            )
            # Create validation dataloader object
            validation_loader = DataLoader(
                dataset, 
                batch_size=self.batch_size,
                sampler=valid_sampler)
            # Set batch sizes of each dataset
            self.num_train_batches = len(train_loader)
            self.num_val_batches = len(validation_loader)
            # Set datasets to loaded
            self.ds_loaded = True  
            # Set datasets as member variable
            self.ds_train = train_loader
            self.ds_val = validation_loader
            return True
        else:
            print("Loading of dataset failed! Input images must have either one (grayscale) or three (RGB) channels.")
            return False

    # Loads test dataset for prediction
    def load_test_dataset(self):
        transformer = self.get_transformer_test()
        # Check if training images have either one or three channels
        if(transformer):
            dataset = torchvision.datasets.ImageFolder(self.pth_test, transform=transformer)
            # Create dataloader object
            prediction_loader = DataLoader(
                dataset, 
                batch_size=self.batch_size_pred, 
            )
            # Get number of images in test dataset
            self.num_pred_img = len(prediction_loader)
            self.ds_pred = prediction_loader  
            return True  
        else:
            print("Loading of dataset failed! Input images must have either one (grayscale) or three (RGB) channels.")
            return False

