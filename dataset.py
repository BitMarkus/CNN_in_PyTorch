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
        # How many subprocesses are used to load data in parallel
        self.num_workers = setting["ds_num_workers"]

        # Validation split settings (False or percentage 0.0-1.0)
        # Validation split from training dataset
        self.val_from_train_split = setting["ds_val_from_train_split"]
        # Validation split from test dataset
        self.val_from_test_split = setting["ds_val_from_test_split"]
        # Add a flag to track where validation data comes from
        self.validation_from_test = (self.val_from_test_split is not False and self.val_from_train_split is False)
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

        # Number of channels of training images 
        self.input_channels = setting["img_channels"]
        # Image width and height for training
        self.input_height = setting["img_height"]
        self.input_width = setting["img_width"]
        # List of classes 
        self.classes = setting["classes"]

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

    # Validate the validation split settings and issue warnings if needed
    def validate_validation_settings(self):
        if self.val_from_train_split is False and self.val_from_test_split is False:
            print("WARNING: Both ds_val_from_train_split and ds_val_from_test_split in settings are set to False!")
            print("Defaulting to 0.1 validation split from training data.")
            self.val_from_train_split = 0.1  # Default value
            
        elif self.val_from_train_split is not False and self.val_from_test_split is not False:
            print("WARNING: Both ds_val_from_train_split and ds_val_from_test_split in settings are set!")
            print("Using only the training data split and ignoring test data split.")
            self.val_from_test_split = False 

    # Helper Methods
    def _gamma_correction(self, x):
        return (x + 1e-6) ** random.uniform(self.gamma_min, self.gamma_max)
    def _add_poisson_noise(self, x):
        return torch.clamp(x + torch.poisson(x * self.poiss_scaling) * self.poiss_noise_strength, 0, 1)

    # Prints validation strategy
    def print_dataset_info(self):
        if self.validation_from_test:
            print(f"Validation strategy: Using test set images for validation ({self.val_from_test_split*100}%)")
        else:
            print(f"Validation strategy: Using training set split for validation ({self.val_from_train_split*100}%)")

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
                    [transforms.Lambda(self._gamma_correction)],  # <- Use method reference
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
                    [transforms.Lambda(self._add_poisson_noise)],  # <- Use method reference
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
                    [transforms.Lambda(self._gamma_correction)],
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
                    [transforms.Lambda(self._add_poisson_noise)],
                    p=self.poiss_prob
                ),
                
                # Normalization
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            return transformer

        else:
            return False

    # Load dataset for training
    # If validation data comes from training images, no augmetations will be applied
    def load_training_dataset(self):
        # Training dataset with or without augmentations
        if self.train_use_augment:
            train_transformer = self.get_transformer_train()
        else:
            train_transformer = self.get_transformer_test()
            
        if not train_transformer:
            print("Loading of dataset failed! Input images must have either one (grayscale) or three (RGB) channels.")
            return False

        # Create training dataset
        train_dataset = torchvision.datasets.ImageFolder(self.pth_train, transform=train_transformer)
        dataset_size = len(train_dataset)
        indices = list(range(dataset_size))
        
        # Only create validation data if splitting from training set
        if self.val_from_train_split is not False:
            split = int(np.floor(self.val_from_train_split * dataset_size))
            if self.shuffle:
                np.random.seed(self.shuffle_seed)
                np.random.shuffle(indices)
            train_indices, val_indices = indices[split:], indices[:split]
            
            self.num_train_img = len(train_indices)
            self.num_val_img = len(val_indices)        
            
            train_sampler = SubsetRandomSampler(train_indices)
            val_sampler = SubsetRandomSampler(val_indices)
            
            # Create validation dataset with test transformer ONLY when needed
            val_dataset = torchvision.datasets.ImageFolder(
                self.pth_train, 
                transform=self.get_transformer_test()  # No augmentations for validation
            )
            validation_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                sampler=val_sampler,
                num_workers=self.num_workers,
                persistent_workers=True,
                pin_memory=True  # Faster transfer to GPU
            )
        else:
            # All training data is used for training if validation comes from test set
            self.num_train_img = dataset_size
            self.num_val_img = 0
            train_sampler = SubsetRandomSampler(indices)
            validation_loader = None

        # Create train loader
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            sampler=train_sampler,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True 
        )
        
        self.num_train_batches = len(train_loader)
        self.num_val_batches = len(validation_loader) if validation_loader else 0
        
        self.ds_loaded = True  
        self.ds_train = train_loader
        self.ds_val = validation_loader  # Will be None if val_from_train_split is False
        return True

    # Loads test dataset for prediction
    def load_test_dataset(self):
        transformer = self.get_transformer_test()
        if not transformer:
            print("Loading of dataset failed! Input images must have either one (grayscale) or three (RGB) channels.")
            return False

        dataset = torchvision.datasets.ImageFolder(self.pth_test, transform=transformer)
        dataset_size = len(dataset)
        self.num_test_img = dataset_size
        
        # If using test set for validation and training set isn't being used for validation
        if (self.val_from_test_split is not False) and (self.val_from_train_split is False):
            val_split_point = int(np.floor(self.val_from_test_split * dataset_size))
            
            indices = list(range(dataset_size))
            if self.shuffle:
                np.random.seed(self.shuffle_seed)
                np.random.shuffle(indices)
                
            val_indices, test_indices = indices[:val_split_point], indices[val_split_point:]
            
            self.num_val_from_test_img = len(val_indices)
            self.num_test_after_val_img = len(test_indices)
            
            # Create samplers and dataloaders
            val_sampler = SubsetRandomSampler(val_indices)
            test_sampler = SubsetRandomSampler(test_indices)
            
            self.ds_test_for_val = DataLoader(
                dataset,
                batch_size=self.batch_size,
                sampler=val_sampler,
                num_workers=self.num_workers,
                persistent_workers=True,
                pin_memory=True 
            )
            self.ds_test_for_test = DataLoader(
                dataset,
                batch_size=self.batch_size_pred,
                sampler=test_sampler,
                num_workers=self.num_workers,
                persistent_workers=True,
                pin_memory=True 
            )
            
            # Use the test-for-val as the validation set
            self.ds_val = self.ds_test_for_val
            self.num_val_img = self.num_val_from_test_img
            self.num_val_batches = len(self.ds_val)
        else:
            # Normal test loader when not using test set for validation
            self.ds_test_for_test = DataLoader(
                dataset,
                batch_size=self.batch_size_pred,
                num_workers=self.num_workers,
                persistent_workers=True,
                pin_memory=True  
            )
            self.num_test_after_val_img = dataset_size
        
        # For backward compatibility
        self.ds_test = self.ds_test_for_test
        self.num_pred_img = self.num_test_after_val_img
        
        return True
        
   
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
                shuffle=False, # No shuffling here!
                num_workers=self.num_workers,
                persistent_workers=True,
                pin_memory=True  
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
