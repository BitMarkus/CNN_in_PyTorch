# Diffusion-Based Phenotypic Extrapolation
# Copyright (C) 2026 Markus Reichold <markus.reichold@ur.de>
# SPDX-License-Identifier: MIT

# ===== Standard Library Imports =====
from pathlib import Path
import random
import shutil
# ===== Third-Party Imports =====
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from torchvision.transforms import transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
# ===== Own Modules =====
from settings import setting

class Dataset():

    #############################################################################################################
    # CONSTRUCTOR

    # Initialize the dataset handler with settings from the configuration file.
    # Sets up paths, batch sizes, validation strategies, and augmentation parameters.
    def __init__(self) -> None:
        
        # Paths
        self.pth_train = setting["pth_train"]
        self.pth_test = setting["pth_test"]
        self.pth_input = setting["pth_input"]

        # Settings variables
        # Shuffle training images before validation split
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
        # Variable to save if the dataset was already loaded or not
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
        # Augmentations #
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
        self.small_angle_fill_gray = setting["aug_small_angle_fill_gray"]
        self.small_angle_fill_rgb = setting["aug_small_angle_fill_rgb"]

        # INTENSITY AUGMENTATIONS:
        self.intense_prob = setting["aug_intense_prob"]
        self.brightness = setting["aug_brightness"]
        self.contrast = setting["aug_contrast"]
        self.saturation = setting["aug_saturation"]
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
        # Sigma: controls the "spread" of the blur (how intense/smooth it is)
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
    # METHODS

    # Validate the validation split settings and issue warnings if needed.
    # Ensures that only one validation source is active at a time.
    def validate_validation_settings(self) -> None:
        if self.val_from_train_split is False and self.val_from_test_split is False:
            print("WARNING: Both ds_val_from_train_split and ds_val_from_test_split in settings are set to False!")
            print("Defaulting to 0.1 validation split from training data.")
            self.val_from_train_split = 0.1

        elif self.val_from_train_split is not False and self.val_from_test_split is not False:
            print("WARNING: Both ds_val_from_train_split and ds_val_from_test_split in settings are set!")
            print("Using only the training data split and ignoring test data split.")
            self.val_from_test_split = False

    # Gamma correction augmentation.
    # Args:
    #   x (torch.Tensor): Input tensor
    # Returns:
    #   torch.Tensor: Gamma-corrected tensor
    def _gamma_correction(self, x: torch.Tensor) -> torch.Tensor:
        return (x + 1e-6) ** random.uniform(self.gamma_min, self.gamma_max)

    # Poisson noise augmentation.
    # Args:
    #   x (torch.Tensor): Input tensor
    # Returns:
    #   torch.Tensor: Tensor with Poisson noise added
    def _add_poisson_noise(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x + torch.poisson(x * self.poiss_scaling) * self.poiss_noise_strength, 0, 1)

    # Print the current validation strategy.
    def print_dataset_info(self) -> None:
        if self.validation_from_test:
            print(f"Validation strategy: Using test set images for validation ({self.val_from_test_split * 100}%)")
        else:
            print(f"Validation strategy: Using training set split for validation ({self.val_from_train_split * 100}%)")

    # Create a transformer for testing and prediction WITHOUT augmentations.
    # Returns:
    #   transforms.Compose: Transformer pipeline for test images
    def get_transformer_test(self) -> transforms.Compose:
        base_transforms = [transforms.ToTensor()]

        if self.input_channels == 1:
            base_transforms.insert(0, transforms.Grayscale(num_output_channels=1))
            base_transforms.append(transforms.Normalize(mean=[0.5], std=[0.5]))

        elif self.input_channels == 3:
            base_transforms.append(transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ))

        else:
            raise ValueError(f"Unsupported input_channels: {self.input_channels}. Use 1 (grayscale) or 3 (RGB).")

        return transforms.Compose(base_transforms)

    # Create a transformer for training WITH augmentations.
    # Returns:
    #   transforms.Compose: Transformer pipeline for training images, or False if input_channels is invalid
    def get_transformer_train(self):
        if self.input_channels == 1:
            transformer = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),

                # Spatial augmentations
                transforms.RandomHorizontalFlip(p=self.hori_flip_prob),
                transforms.RandomVerticalFlip(p=self.vert_flip_prob),

                # 90° rotations
                transforms.RandomApply([
                    transforms.RandomChoice([
                        transforms.RandomRotation(degrees=[0, 0]),
                        transforms.RandomRotation(degrees=[90, 90]),
                        transforms.RandomRotation(degrees=[180, 180]),
                        transforms.RandomRotation(degrees=[270, 270])
                    ])
                ], p=self.aug_90_angle_rot_prob),

                # Small angle rotations
                transforms.RandomApply([
                    transforms.RandomRotation(degrees=self.small_angle_rot, fill=self.small_angle_fill_gray)
                ], p=self.small_angle_rot_prob),

                transforms.ToTensor(),

                # Intensity augmentations
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=self.brightness, contrast=self.contrast)
                ], p=self.intense_prob),

                # Gamma correction
                transforms.RandomApply([
                    transforms.Lambda(self._gamma_correction)
                ], p=self.gamma_prob),

                # Optical augmentations
                transforms.RandomApply([
                    transforms.GaussianBlur(
                        kernel_size=self.gauss_kernel_size,
                        sigma=(self.gauss_sigma_min, self.gauss_sigma_max)
                    )
                ], p=self.gauss_prob),

                transforms.RandomApply([
                    transforms.Lambda(self._add_poisson_noise)
                ], p=self.poiss_prob),

                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])
            return transformer

        elif self.input_channels == 3:
            transformer = transforms.Compose([
                transforms.RandomHorizontalFlip(p=self.hori_flip_prob),
                transforms.RandomVerticalFlip(p=self.vert_flip_prob),

                # 90° rotations
                transforms.RandomApply([
                    transforms.RandomChoice([
                        transforms.RandomRotation(degrees=[0, 0]),
                        transforms.RandomRotation(degrees=[90, 90]),
                        transforms.RandomRotation(degrees=[180, 180]),
                        transforms.RandomRotation(degrees=[270, 270])
                    ])
                ], p=self.aug_90_angle_rot_prob),

                # Small angle rotations
                transforms.RandomApply([
                    transforms.RandomRotation(degrees=self.small_angle_rot, fill=self.small_angle_fill_rgb)
                ], p=self.small_angle_rot_prob),

                transforms.ToTensor(),

                # Intensity augmentations
                transforms.RandomApply([
                    transforms.ColorJitter(
                        brightness=self.brightness,
                        contrast=self.contrast,
                        saturation=self.saturation
                    )
                ], p=self.intense_prob),

                # Gamma correction
                transforms.RandomApply([
                    transforms.Lambda(self._gamma_correction)
                ], p=self.gamma_prob),

                # Optical augmentations
                transforms.RandomApply([
                    transforms.GaussianBlur(
                        kernel_size=self.gauss_kernel_size,
                        sigma=(self.gauss_sigma_min, self.gauss_sigma_max)
                    )
                ], p=self.gauss_prob),

                transforms.RandomApply([
                    transforms.Lambda(self._add_poisson_noise)
                ], p=self.poiss_prob),

                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
            return transformer

        else:
            return False

    # Load the training dataset with optional validation split from training data.
    # Returns:
    #   bool: True if successful, False otherwise
    def load_training_dataset(self) -> bool:
        if self.train_use_augment:
            train_transformer = self.get_transformer_train()
        else:
            train_transformer = self.get_transformer_test()

        if not train_transformer:
            print("Loading of dataset failed! Input images must have either one (grayscale) or three (RGB) channels.")
            return False

        train_dataset = torchvision.datasets.ImageFolder(self.pth_train, transform=train_transformer)
        dataset_size = len(train_dataset)
        indices = list(range(dataset_size))

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

            val_dataset = torchvision.datasets.ImageFolder(
                self.pth_train,
                transform=self.get_transformer_test()
            )
            validation_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                sampler=val_sampler,
                num_workers=self.num_workers,
                persistent_workers=True,
                pin_memory=True
            )
        else:
            self.num_train_img = dataset_size
            self.num_val_img = 0
            train_sampler = SubsetRandomSampler(indices)
            validation_loader = None

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
        self.ds_val = validation_loader
        return True

    # Load the test dataset, optionally splitting for validation.
    # Returns:
    #   bool: True if successful, False otherwise
    def load_test_dataset(self) -> bool:
        transformer = self.get_transformer_test()
        if not transformer:
            print("Loading of dataset failed! Input images must have either one (grayscale) or three (RGB) channels.")
            return False

        dataset = torchvision.datasets.ImageFolder(self.pth_test, transform=transformer)
        dataset_size = len(dataset)
        self.num_test_img = dataset_size

        if (self.val_from_test_split is not False) and (self.val_from_train_split is False):
            val_split_point = int(np.floor(self.val_from_test_split * dataset_size))

            indices = list(range(dataset_size))
            if self.shuffle:
                np.random.seed(self.shuffle_seed)
                np.random.shuffle(indices)

            val_indices, test_indices = indices[:val_split_point], indices[val_split_point:]

            self.num_val_from_test_img = len(val_indices)
            self.num_test_after_val_img = len(test_indices)

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

            self.ds_val = self.ds_test_for_val
            self.num_val_img = self.num_val_from_test_img
            self.num_val_batches = len(self.ds_val)
        else:
            self.ds_test_for_test = DataLoader(
                dataset,
                batch_size=self.batch_size_pred,
                num_workers=self.num_workers,
                persistent_workers=True,
                pin_memory=True
            )
            self.num_test_after_val_img = dataset_size

        self.ds_test = self.ds_test_for_test
        self.num_pred_img = self.num_test_after_val_img
        return True

    # Load the prediction dataset (uses images from pth_input).
    # Returns:
    #   bool: True if successful, False otherwise
    def load_pred_dataset(self) -> bool:
        transformer = self.get_transformer_test()
        if transformer:
            dataset = torchvision.datasets.ImageFolder(self.pth_input, transform=transformer)
            prediction_loader = DataLoader(
                dataset,
                batch_size=self.batch_size_pred,
                shuffle=False,
                num_workers=self.num_workers,
                persistent_workers=True,
                pin_memory=True
            )
            self.num_pred_img = len(prediction_loader)
            self.ds_pred = prediction_loader
            self.ds_loaded = True
            return True
        else:
            print("Loading of dataset failed! Input images must have either one (grayscale) or three (RGB) channels.")
            return False

    # Display a grid of training image examples after normalization.
    # Args:
    #   plot_path (Path): Directory to save the plot
    #   num_images (int): Number of images to display
    #   rows (int): Number of rows in the grid
    #   cols (int): Number of columns in the grid
    #   figsize (tuple): Figure size in inches
    #   show_plot (bool): If True, display the plot interactively
    #   save_plot (bool): If True, save the plot to disk
    def show_training_examples(
        self,
        plot_path: Path,
        num_images: int = 9,
        rows: int = 3,
        cols: int = 3,
        figsize: tuple = (10, 10),
        show_plot: bool = False,
        save_plot: bool = True
    ) -> None:
        if not self.ds_loaded:
            print("Error: Dataset not loaded. Call load_training_dataset() first.")
            return

        data_iter = iter(self.ds_train)
        images, _ = next(data_iter)

        if images.shape[0] < num_images:
            empty_images = torch.zeros((num_images - images.shape[0], *images.shape[1:]))
            images = torch.cat([images, empty_images], dim=0)
        else:
            images = images[:num_images]

        # Denormalize images back to [0,1] range for visualization
        if self.input_channels == 1:
            images = images * 0.5 + 0.5
        elif self.input_channels == 3:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            images = images * std + mean

        grid = vutils.make_grid(images, nrow=cols, padding=2, normalize=False)

        plt.figure(figsize=figsize)
        if self.input_channels == 1:
            plt.imshow(grid[0], cmap='gray')
        else:
            plt.imshow(grid.permute(1, 2, 0))
        plt.axis('off')
        plt.title("Training Image Examples")

        plt.tight_layout()

        if save_plot:
            plt.savefig(str(plot_path / "training_examples.png"), bbox_inches='tight', dpi=300)
            plt.close()
        if show_plot:
            plt.show()

    # Export all images from the validation dataset to a specified folder.
    # Preserves the class/subfolder structure.
    # Args:
    #   output_folder (Path): Directory to copy images to
    # Returns:
    #   bool: True if successful, False otherwise
    def export_validation_images(self, output_folder: Path) -> bool:
        if self.ds_val is None:
            print("ERROR: No validation dataset loaded.")
            return False

        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Exporting validation images to: {output_path}")

        val_dataset = self.ds_val.dataset

        if hasattr(val_dataset, 'classes'):
            class_names = val_dataset.classes
        else:
            print("ERROR: Dataset does not have 'classes' attribute.")
            return False

        for class_name in class_names:
            (output_path / class_name).mkdir(parents=True, exist_ok=True)

        if hasattr(self.ds_val, 'sampler') and hasattr(self.ds_val.sampler, 'indices'):
            indices = self.ds_val.sampler.indices
        else:
            print("ERROR: Dataloader sampler has no 'indices' attribute.")
            return False

        if hasattr(val_dataset, 'samples'):
            samples = val_dataset.samples
        elif hasattr(val_dataset, 'imgs'):
            samples = val_dataset.imgs
        else:
            print("ERROR: Dataset has no 'samples' or 'imgs' attribute.")
            return False

        copied_count = 0
        for idx in indices:
            img_path, class_idx = samples[idx]
            class_name = class_names[class_idx]
            filename = Path(img_path).name
            dest_path = output_path / class_name / filename

            try:
                shutil.copy2(img_path, dest_path)
                copied_count += 1
            except Exception as e:
                print(f"  Warning: Could not copy {img_path}: {e}")

        print(f"✓ Exported {copied_count} validation images to: {output_path}")
        return True

    #############################################################################################################
    # SYNTHETIC IMAGE HANDLING

    # Check if an image is synthetic based on filename pattern.
    # Synthetic images start with 's' followed by a number.
    # Args:
    #   path (Path or str): Path to the image file
    # Returns:
    #   bool: True if synthetic, False otherwise
    def _is_synthetic_image(self, path) -> bool:
        filename = Path(path).name
        return filename.startswith('s') and filename[1:2].isdigit()

    # Load ONLY real test images for final evaluation (excluding synthetic images).
    # Args:
    #   real_image_names (list, optional): List of real image filenames to include
    # Returns:
    #   bool: True if successful, False otherwise
    def load_real_test_dataset_only(self, real_image_names: list = None) -> bool:
        transformer = self.get_transformer_test()
        if not transformer:
            print("Loading of dataset failed!")
            return False

        full_dataset = torchvision.datasets.ImageFolder(self.pth_test, transform=transformer)

        if real_image_names is not None:
            real_indices = []
            for idx, (path, _) in enumerate(full_dataset.samples):
                filename = Path(path).name
                if filename in real_image_names:
                    real_indices.append(idx)

            if len(real_indices) == 0:
                print("ERROR: No specified real test images found!")
                return False

            print(f"Creating test dataset with {len(real_indices)} specified real images")
            test_dataset = torch.utils.data.Subset(full_dataset, real_indices)
            self.num_pred_real = len(real_indices)
        else:
            test_dataset = full_dataset
            self.num_pred_real = len(full_dataset)
            print(f"Creating test dataset with all {self.num_pred_real} images (pure real data)")

        self.ds_test_real_only = DataLoader(
            test_dataset,
            batch_size=self.batch_size_pred,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True
        )

        return True