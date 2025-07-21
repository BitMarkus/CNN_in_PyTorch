####################
# Dataset handling #
####################

import torch
from torchvision.transforms import transforms
import torchvision
import numpy as np
import random 
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.utils as vutils
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
        self.ds_test = None
        self.ds_pred = None

        ################
        # Augentations #
        ################

        # Use augmentations
        self.train_use_augment = setting["train_use_augment"]

        # FLIP AND ROTATION AUGMENTATIONS:
        # Horizontal flip probability
        self.hori_flip_prob = setting["aug_hori_flip_prob"]
        # Vertical flip probability
        self.vert_flip_prob = setting["aug_vert_flip_prob"]
        # Probability of 90° angle rotations
        self.aug_90_angle_rot_prob = setting["aug_90_angle_rot_prob"]
        # Probability of small angle rotations
        self.small_angle_rot_prob = setting["aug_small_angle_rot_prob"] 
        # Small-angle rotation
        self.small_angle_rot = setting["aug_small_angle_rot"] 
        # Fill color for gaps due to small angle rotation
        # fill=0: black background, fill=255: white background
        self.small_angle_fill_gray = setting["aug_small_angle_fill_gray"] # For grayscale images
        self.small_angle_fill_rgb = setting["aug_small_angle_fill_rgb"] # For RGB images

        # INTENSITY AUGMENTATIONS:
        self.intense_prob = setting["aug_intense_prob"]
        self.brightness = setting["aug_brightness"]
        self.contrast = setting["aug_contrast"]
        self.saturation = setting["aug_saturation"] # only for RGB images
        # Gamma correction
        # Gamma = 1: No change. The image looks "natural" (linear brightness)
        # Gamma < 1 (e.g., 0.5): Dark areas get brighter, bright areas stay mostly the same
        # Gamma > 1 (e.g., 2.0): Bright areas get darker, dark areas stay mostly the same
        self.gamma_prob = setting["aug_gamma_prob"]
        self.gamma_min = setting["aug_gamma_min"]
        self.gamma_max = setting["aug_gamma_max"]

        # OPTICAL AUGMENTATIONS:
        # Gaussian Blur Parameters
        # Probability
        self.gauss_prob = setting["aug_gauss_prob"]
        # Kernel size
        self.gauss_kernel_size = setting["aug_gauss_kernel_size"]
        # Sigma: ontrols the "spread" of the blur (how intense/smooth it is)
        self.gauss_sigma_min = setting["aug_gauss_sigma_min"]
        self.gauss_sigma_max = setting["aug_gauss_sigma_max"]
        # Poisson noise
        # Probability
        self.poiss_prob = setting["aug_poiss_prob"]
        # Controls how much the noise depends on image brightness
        # Suggested range: 0.01-0.1 (higher = more noise)
        self.poiss_scaling = setting["aug_poiss_scaling"]
        # Noise Strength: Final noise intensity multiplier
        self.poiss_noise_strength = setting["aug_poiss_noise_strength"]

    #############################################################################################################
    # METHODS:

    # Transformer for testing and predicting WITHOUT AUGMENTATIONS
    def get_transformer_test(self):
        # Common base for both grayscale and RGB
        base_transforms = [transforms.ToTensor()] 

        if self.input_channels == 1:
            # Grayscale pipeline
            base_transforms.insert(0, transforms.Grayscale(num_output_channels=1))  # Early conversion
            base_transforms.append(transforms.Normalize(mean=[0.5], std=[0.5]))  # [-1, 1]

        elif self.input_channels == 3:
            # RGB pipeline (using ImageNet stats)
            base_transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

        else:
            raise ValueError(f"Unsupported input_channels: {self.input_channels}. Use 1 (grayscale) or 3 (RGB).")

        return transforms.Compose(base_transforms)

    # Transformer for training WITH AUGMENTATIONS
    def get_transformer_train(self):
        # Transformer for Grayscale images
        # https://stackoverflow.com/questions/60116208/pytorch-load-dataset-of-grayscale-images
        if(self.input_channels == 1):
            transformer = transforms.Compose([

                # No resize needed as the images are already 512x512
                # transforms.Resize((self.input_height, self.input_width)),

                # Convert to grayscale first (1 channel)
                transforms.Grayscale(num_output_channels=1),

                # Spatial augmentations (PIL-level)
                # Flip the image horizontally and vertically with 50% probability
                transforms.RandomHorizontalFlip(p=self.hori_flip_prob),
                transforms.RandomVerticalFlip(p=self.vert_flip_prob),

                # Applies one of four fixed rotations (0°, 90°, 180°, 270°) at random
                transforms.RandomApply(
                    [
                        transforms.RandomChoice([
                            transforms.RandomRotation(degrees=[0, 0]),    # 0°
                            transforms.RandomRotation(degrees=[90, 90]),  # 90°
                            transforms.RandomRotation(degrees=[180, 180]),# 180°
                            transforms.RandomRotation(degrees=[270, 270]) # 270°
                        ])
                    ],
                    p=self.aug_90_angle_rot_prob 
                ),
                # Rotates the image by a small random angle (±10°) with a gray background
                # fill=0: black background, 
                # fill=255: white background, 
                transforms.RandomApply(
                    [transforms.RandomRotation(degrees=self.small_angle_rot , fill=self.small_angle_fill_gray)],
                    p=self.small_angle_rot_prob
                ),

                # Convert PIL image to a PyTorch tensor (shape: [1, H, W]) and scales pixel values to [0, 1]
                transforms.ToTensor(),

                # Intensity augmentations (tensor-level):
                # Randomly adjust brightness and contrast by up to ±20% to
                # simulate variations in lighting/staining intensity across samples
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=self.brightness, contrast=self.contrast)],
                    p=self.intense_prob
                ),
                # Gamma correction: Mimics nonlinear microscope/camera responses
                # Gamma = 1: No change. The image looks "natural" (linear brightness)
                # Gamma < 1 (e.g., 0.5): Dark areas get brighter, bright areas stay mostly the same
                # Gamma > 1 (e.g., 2.0): Bright areas get darker, dark areas stay mostly the same
                transforms.RandomApply(
                    [transforms.Lambda(lambda x: (x + 1e-6) ** random.uniform(self.gamma_min, self.gamma_max))],
                    p=self.gamma_prob
                ),

                # Optical augmentations (tensor-level)
                # Apply mild Gaussian blur
                # Simulates slight defocus or motion blur in microscopy
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=self.gauss_kernel_size, sigma=(self.gauss_sigma_min, self.gauss_sigma_max))], 
                    p=self.gauss_prob
                ),
                # Adds Poisson noise (a type of noise common in microscopy/imaging) scaled to 5% of pixel values
                transforms.RandomApply(
                    [transforms.Lambda(lambda x: torch.clamp(x + torch.poisson(x * self.poiss_scaling) * self.poiss_noise_strength, 0, 1))],
                    p=self.poiss_prob
                ),

                # Normalize to [-1, 1] (mean=0.5, std=0.5)
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])
            return transformer

        # Transformer for RGB images
        # https://pytorch.org/hub/pytorch_vision_resnet/
        elif(self.input_channels == 3):
            transformer = transforms.Compose([
                # transforms.Resize((self.input_height, self.input_width)),

                transforms.RandomHorizontalFlip(p=self.hori_flip_prob),
                transforms.RandomVerticalFlip(p=self.vert_flip_prob),

                # Applies one of four fixed rotations (0°, 90°, 180°, 270°) at random
                transforms.RandomApply(
                    [
                        transforms.RandomChoice([
                            transforms.RandomRotation(degrees=[0, 0]),    # 0°
                            transforms.RandomRotation(degrees=[90, 90]),  # 90°
                            transforms.RandomRotation(degrees=[180, 180]),# 180°
                            transforms.RandomRotation(degrees=[270, 270]) # 270°
                        ])
                    ],
                    p=self.aug_90_angle_rot_prob 
                ),
                # Rotates the image by a small random angle (±10°) with a gray background
                transforms.RandomApply(
                    [transforms.RandomRotation(degrees=self.small_angle_rot , fill=self.small_angle_fill_rgb)],
                    p=self.small_angle_rot_prob
                ),
                
                # Convert to tensor
                transforms.ToTensor(),
                
                # Intensity augmentations
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=self.brightness, contrast=self.contrast, saturation=self.saturation)],
                    p=self.intense_prob
                ),
                # Gamma correction
                transforms.RandomApply(
                    [transforms.Lambda(lambda x: (x + 1e-6) ** random.uniform(self.gamma_min, self.gamma_max))],
                    p=self.gamma_prob
                ),
                
                # Optical augmentations
                # Gaussian blurr
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=self.gauss_kernel_size, sigma=(self.gauss_sigma_min, self.gauss_sigma_max))], 
                    p=self.gauss_prob
                ),
                # Poisson noise 
                transforms.RandomApply(
                    [transforms.Lambda(lambda x: torch.clamp(x + torch.poisson(x * self.poiss_scaling) * self.poiss_noise_strength, 0, 1))],
                    p=self.poiss_prob
                ),
                
                # Normalization
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
            self.ds_test = prediction_loader  
            # Set datasets to loaded
            self.ds_loaded = True  
            return True  
        else:
            print("Loading of dataset failed! Input images must have either one (grayscale) or three (RGB) channels.")
            return False
        
    # Loads prediction dataset for captum
    def load_pred_dataset(self):
        transformer = self.get_transformer_test()
        # Check if training images have either one or three channels
        if(transformer):
            dataset = torchvision.datasets.ImageFolder(self.pth_prediction, transform=transformer)
            # Create dataloader object
            prediction_loader = DataLoader(
                dataset, 
                batch_size=self.batch_size_pred, 
            )
            # Get number of images in test dataset
            self.num_pred_img = len(prediction_loader)
            self.ds_pred = prediction_loader  
            # Set datasets to loaded
            self.ds_loaded = True  
            return True  
        else:
            print("Loading of dataset failed! Input images must have either one (grayscale) or three (RGB) channels.")
            return False
        
    # Show dataset examples after normalization
    def show_training_examples(self, plot_path, num_images=9, rows=3, cols=3, figsize=(10, 10), show_plot=False, save_plot=True):

        if not self.ds_loaded:
            print("Error: Dataset not loaded. Call load_training_dataset() first.")
            return
        
        # Get a batch of images from the training loader
        data_iter = iter(self.ds_train)
        images, _ = next(data_iter)  # Assumes batch_size >= num_images
        
        # If batch_size < num_images, pad with empty tensors
        if images.shape[0] < num_images:
            empty_images = torch.zeros((num_images - images.shape[0], *images.shape[1:]))
            images = torch.cat([images, empty_images], dim=0)
        else:
            images = images[:num_images]
        
        # Denormalize images back to [0,1] range for visualization
        if self.input_channels == 1:
            # Grayscale: mean=0.5, std=0.5
            images = images * 0.5 + 0.5
        elif self.input_channels == 3:
            # RGB: ImageNet stats
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            images = images * std + mean
        
        # Create grid
        grid = vutils.make_grid(images, nrow=cols, padding=2, normalize=False)
        
        # Plot
        plt.figure(figsize=figsize)
        if self.input_channels == 1:
            plt.imshow(grid[0], cmap='gray')
        else:
            plt.imshow(grid.permute(1, 2, 0))  # CHW -> HWC for matplotlib
        plt.axis('off')
        plt.title(f"Training Image Examples")

        # Adjust layout
        plt.tight_layout()
        # Save plot
        if save_plot:
            plt.savefig(str(plot_path / "training_examples"), bbox_inches='tight', dpi=300)
            plt.close()
        # Show plot
        if show_plot:
            plt.show()

"""
def load_test_dataset(self):
    transformer = self.get_transformer_test()
    if transformer:
        dataset = torchvision.datasets.ImageFolder(self.pth_test, transform=transformer)
        
        # Split test data into validation and final test portions
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.5 * dataset_size))  # 50% for validation, 50% for final test
        
        if self.shuffle:
            np.random.seed(self.shuffle_seed)
            np.random.shuffle(indices)
            
        val_indices, test_indices = indices[:split], indices[split:]
        
        val_sampler = SubsetRandomSampler(val_indices)
        test_sampler = SubsetRandomSampler(test_indices)
        
        val_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=val_sampler
        )
        
        test_loader = DataLoader(
            dataset,
            batch_size=self.batch_size_pred,
            sampler=test_sampler
        )
        
        self.num_val_img = len(val_indices)
        self.num_pred_img = len(test_indices)
        self.ds_val = val_loader  # Use this for validation during training
        self.ds_test = test_loader  # Use this for final evaluation
        self.ds_loaded = True
        return True
    else:
        print("Loading failed! Images must be grayscale or RGB.")
        return False

# Suggested workflow:
dataset = Dataset()
dataset.load_training_dataset()  # For initial exploration only
dataset.load_test_dataset()  # Splits test data into validation and final test

# Then during training:
for epoch in epochs:
    # Training loop using ds_train
    # Validation loop using ds_val (from test data)
    
# Final evaluation:
test_model(ds_test)  # On the held-out portion of test data
"""

