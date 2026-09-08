# FID Score Scale Reference:
# 0-50: Excellent quality (images are very similar)
# 50-100: Good quality
# 100-200: Moderate to poor quality
# 200+: Very poor quality (images are very different)

# ===== Standard Library Imports =====
import random
from datetime import datetime
from pathlib import Path
# ===== Third-Party Imports =====
import torch
import torch.nn as nn
import numpy as np
from scipy import linalg
from torchvision import models, transforms
from PIL import Image
# ===== Own Modules =====
from settings import setting

class FIDCalculator:

    #############################################################################################################
    # CONSTRUCTOR

    # Initialize the FID (Fréchet Inception Distance) calculator.
    # Computes FID scores between a reference folder and all other folders
    # in the input directory.
    # Args:
    #   device (torch.device): Device to run feature extraction on
    #   reference_folder (str, optional): Name of the reference folder
    def __init__(self, device: torch.device, reference_folder: str = None) -> None:

        self.device = device
        self.reference_folder = reference_folder

        # Settings parameters
        self.pth_input = setting['pth_input'].resolve()
        self.pth_output = setting['pth_output'].resolve()
        self.num_channels = setting['img_channels']
        self.batch_size = setting['fid_batch_size']
        self.balance_samples = setting['fid_balance_samples']
        self.random_seed = setting['fid_random_seed']

        # Class variables
        self.folders = {}
        self.reference_folder_name = None
        self.fid_scores = {}
        self.results = {}

    #############################################################################################################
    # METHODS

    # Validate folders in the input directory and identify reference folder.
    # Returns:
    #   bool: True if validation succeeded, False otherwise
    def validate_folders(self) -> bool:
        if not self.pth_input.exists():
            print(f"Error: Input folder '{self.pth_input}' does not exist.")
            return False

        subdirs = [d for d in self.pth_input.iterdir() if d.is_dir()]

        if len(subdirs) < 2:
            print(f"Error: Expected at least 2 folders in '{self.pth_input}', found {len(subdirs)}")
            print(f"Found folders: {[d.name for d in subdirs]}")
            return False

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

        def count_images(folder: Path) -> int:
            return len([f for f in folder.iterdir()
                       if f.is_file() and f.suffix.lower() in image_extensions])

        valid_folders = {}
        for folder in subdirs:
            count = count_images(folder)
            if count > 0:
                valid_folders[folder.name] = {
                    'path': folder,
                    'count': count
                }
            else:
                print(f"Warning: No images found in folder '{folder.name}', skipping")

        if len(valid_folders) < 2:
            print(f"Error: Need at least 2 folders with images, found {len(valid_folders)}")
            return False

        self.folders = valid_folders

        if len(valid_folders) == 2:
            folder_names = list(valid_folders.keys())
            self.reference_folder_name = folder_names[0]
            print(f"Exactly 2 folders found. Using '{self.reference_folder_name}' as reference.")
        else:
            print(f"\nFound {len(valid_folders)} folders with images:")
            print("Please select the reference folder by entering the corresponding index:")
            print("-" * 50)

            folder_list = list(valid_folders.keys())
            for idx, folder_name in enumerate(folder_list, 1):
                count = valid_folders[folder_name]['count']
                print(f"  [{idx}] {folder_name} ({count} images)")

            print("-" * 50)

            while True:
                try:
                    selection = input("Enter the number of the reference folder: ").strip()
                    if not selection:
                        continue

                    selected_idx = int(selection)
                    if 1 <= selected_idx <= len(folder_list):
                        self.reference_folder_name = folder_list[selected_idx - 1]
                        break
                    else:
                        print(f"Please enter a number between 1 and {len(folder_list)}")
                except ValueError:
                    print("Please enter a valid number")
                except KeyboardInterrupt:
                    print("\nOperation cancelled by user")
                    return False

            print(f"Selected '{self.reference_folder_name}' as reference folder.")

        print(f"Folder configuration:")
        for folder_name, info in valid_folders.items():
            marker = " [REFERENCE]" if folder_name == self.reference_folder_name else ""
            print(f"  - '{folder_name}': {info['count']} images{marker}")

        self.results['all_folders'] = {name: info['count'] for name, info in valid_folders.items()}
        self.results['reference_folder'] = self.reference_folder_name
        self.results['total_folders'] = len(valid_folders)
        self.results['channels'] = self.num_channels

        return True

    # Load images from a specific folder as a PyTorch Dataset.
    # Args:
    #   folder_path (Path): Path to the folder
    #   max_images (int, optional): Maximum number of images to load
    #   random_seed (int, optional): Random seed for sampling
    # Returns:
    #   InceptionDataset: Dataset containing the images
    def load_images_from_folder(self, folder_path: Path, max_images: int = None, random_seed: int = 42):
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        all_image_paths = []
        for file_path in folder_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                all_image_paths.append(file_path)

        if max_images is not None and max_images < len(all_image_paths):
            if random_seed is not None:
                random.seed(random_seed)
                np.random.seed(random_seed)

            selected_paths = random.sample(all_image_paths, max_images)
            print(f"  - Randomly selected {max_images} out of {len(all_image_paths)} images")
        else:
            selected_paths = all_image_paths

        dataset = InceptionDataset(folder_path, num_channels=self.num_channels, image_paths=selected_paths)
        return dataset

    # Load and configure the InceptionV3 model for feature extraction.
    # Returns:
    #   torch.nn.Module: InceptionV3 model with classification layer removed
    def get_inception_model(self) -> nn.Module:
        inception_model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)

        # Remove the final classification layer
        inception_model.fc = nn.Identity()

        inception_model.eval()
        inception_model = inception_model.to(self.device)

        # Disable gradients
        for param in inception_model.parameters():
            param.requires_grad = False

        return inception_model

    # Calculate activations for all images in the dataset.
    # Args:
    #   dataset: The dataset to process
    #   model: The InceptionV3 model
    #   batch_size (int): Batch size for processing
    # Returns:
    #   np.ndarray: Activations of shape (n_samples, n_features)
    def get_activations(self, dataset, model: nn.Module, batch_size: int = 32) -> np.ndarray:
        model.eval()
        activations = []

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for batch in dataloader:
                images = batch[0].to(self.device)

                # Handle grayscale images by replicating to 3 channels for InceptionV3
                if images.shape[1] == 1:
                    images = images.repeat(1, 3, 1, 1)

                features = model(images)
                activations.append(features.cpu().numpy())

        return np.concatenate(activations, axis=0)

    # Calculate the Fréchet distance between two multivariate Gaussians.
    # Args:
    #   mu1 (np.ndarray): Mean of first distribution
    #   sigma1 (np.ndarray): Covariance of first distribution
    #   mu2 (np.ndarray): Mean of second distribution
    #   sigma2 (np.ndarray): Covariance of second distribution
    #   eps (float): Small value for numerical stability
    # Returns:
    #   float: Fréchet distance
    def calculate_frechet_distance(self, mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray, eps: float = 1e-6) -> float:
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        diff = mu1 - mu2

        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)

    # Calculate activation statistics for a dataset.
    # Args:
    #   dataset: The dataset to process
    #   model: The InceptionV3 model
    #   batch_size (int): Batch size for processing
    # Returns:
    #   tuple: (mean, covariance) of activations
    def calculate_activation_statistics(self, dataset, model: nn.Module, batch_size: int = 32) -> tuple:
        act = self.get_activations(dataset, model, batch_size)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma

    # Calculate FID score between the reference folder and all other folders.
    # Args:
    #   batch_size (int, optional): Batch size for processing
    #   balance_samples (bool, optional): Balance sample counts across folders
    #   random_seed (int, optional): Random seed for sampling
    # Returns:
    #   dict: FID scores for each folder compared to reference
    def calculate_fid(self, batch_size: int = None, balance_samples: bool = None, random_seed: int = None) -> dict:
        if batch_size is None:
            batch_size = self.batch_size
        if balance_samples is None:
            balance_samples = self.balance_samples
        if random_seed is None:
            random_seed = self.random_seed

        print("Loading InceptionV3 model...")
        model = self.get_inception_model()

        ref_folder_info = self.folders[self.reference_folder_name]
        ref_folder_path = ref_folder_info['path']
        ref_image_count = ref_folder_info['count']

        max_images = None
        if balance_samples:
            all_counts = [info['count'] for info in self.folders.values()]
            max_images = min(all_counts)
            print(f"Balancing samples: using {max_images} images from each folder")
            self.results['balanced_sample_size'] = max_images
            self.results['balancing_enabled'] = True
        else:
            self.results['balancing_enabled'] = False

        print(f"Loading reference images from '{self.reference_folder_name}'...")
        ref_dataset = self.load_images_from_folder(ref_folder_path, max_images=max_images, random_seed=random_seed)

        print("Calculating activations and statistics for reference folder...")
        ref_mu, ref_sigma = self.calculate_activation_statistics(ref_dataset, model, batch_size)

        self.fid_scores = {}

        for folder_name, folder_info in self.folders.items():
            if folder_name == self.reference_folder_name:
                continue

            print(f"\n> Processing folder '{folder_name}'...")
            folder_path = folder_info['path']

            print(f"Loading images from '{folder_name}'...")
            compare_dataset = self.load_images_from_folder(folder_path, max_images=max_images, random_seed=random_seed)

            print(f"Calculating activations and statistics for '{folder_name}'...")
            compare_mu, compare_sigma = self.calculate_activation_statistics(compare_dataset, model, batch_size)

            print(f"Calculating FID score between '{self.reference_folder_name}' and '{folder_name}'...")
            fid_value = self.calculate_frechet_distance(ref_mu, ref_sigma, compare_mu, compare_sigma)

            self.fid_scores[folder_name] = float(fid_value)

            self.results[f'fid_{folder_name}'] = float(fid_value)
            self.results[f'images_used_{folder_name}'] = len(compare_dataset)

        self.results['reference_images_used'] = len(ref_dataset)
        self.results['fid_scores'] = self.fid_scores.copy()

        return self.fid_scores

    # Save FID results to a file in the output folder.
    def save_results(self) -> None:
        self.results['timestamp'] = datetime.now().isoformat()
        self.results['device'] = str(self.device)

        # Create output directory if it doesn't exist
        self.pth_output.mkdir(parents=True, exist_ok=True)

        text_file = self.pth_output / "fid_results.txt"
        with open(text_file, 'w') as f:
            f.write("FID Score Results\n")
            f.write("=================\n\n")

            f.write(f"Reference Folder: {self.results['reference_folder']} ({self.results['reference_images_used']} images used)\n\n")

            f.write("FID Scores (compared to reference):\n")
            f.write("-" * 50 + "\n")

            sorted_scores = sorted(self.fid_scores.items(), key=lambda x: x[1])

            for folder_name, score in sorted_scores:
                f.write(f"{folder_name:<30}: {score:.4f}\n")

            f.write("\n" + "-" * 50 + "\n")
            f.write(f"Total Folders: {self.results['total_folders']}\n")
            f.write(f"Image Channels: {self.results['channels']}\n")
            if self.results.get('balancing_enabled', False):
                f.write(f"Sample Balancing: ENABLED (max: {self.results['balanced_sample_size']} images each)\n")
            else:
                f.write("Sample Balancing: DISABLED (using all available images)\n")
            f.write(f"Calculation Date: {self.results['timestamp']}\n")
            f.write(f"Device: {self.results['device']}\n")

        print(f"Results saved to: {text_file}")

    # Print FID results to console.
    def print_results(self) -> None:
        print("\n>> FID SCORE RESULTS")
        print("-" * 60)
        print(f"Reference Folder: {self.results['reference_folder']} ({self.results['reference_images_used']} images used)")
        print(f"Image Channels: {self.results['channels']}")
        print(f"Total Folders: {self.results['total_folders']}")

        print("\nFID Scores (compared to reference):")
        print("-" * 40)

        sorted_scores = sorted(self.fid_scores.items(), key=lambda x: x[1])

        for folder_name, score in sorted_scores:
            images_used = self.results.get(f'images_used_{folder_name}', 'N/A')
            print(f"  {folder_name:<25}: {score:>8.4f}  ({images_used} images)")

        print("-" * 40)

        if self.results.get('balancing_enabled', False):
            print(f"Sample Balancing: ENABLED (max: {self.results['balanced_sample_size']} images each)")
        else:
            print("Sample Balancing: DISABLED (using all available images)")
        print("-" * 60)

    #############################################################################################################
    # CALL

    # Run the complete FID calculation pipeline.
    # Returns:
    #   dict or None: FID scores if successful, None otherwise
    def __call__(self):
        try:
            if not self.validate_folders():
                print(f"\nFID calculation failed: Folder validation error!")
                return None

            fid_scores = self.calculate_fid(self.batch_size, self.balance_samples, self.random_seed)

            self.save_results()
            self.print_results()

            print(f"\nFID calculation completed successfully!")
            return fid_scores

        except Exception as e:
            print(f"\nFID calculation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

#############################################################################################################
# SIMPLE IMAGE DATASET CLASS

# Simple dataset for loading images from a folder for the Inception network architecture.
class InceptionDataset(torch.utils.data.Dataset):

    # Initialize the Inception dataset.
    # Args:
    #   folder_path (Path): Path to the folder
    #   num_channels (int): Number of channels (1 for grayscale, 3 for RGB)
    #   image_paths (list, optional): List of image paths to include
    def __init__(self, folder_path: Path, num_channels: int = 1, image_paths: list = None) -> None:
        self.folder_path = folder_path
        self.num_channels = num_channels
        self.image_paths = image_paths

        if self.num_channels == 1:
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]

        if self.num_channels == 1:
            image = Image.open(image_path).convert('L')
        else:
            image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, 0  # Return dummy label