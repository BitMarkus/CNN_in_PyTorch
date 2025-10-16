# FID Score Scale Reference:
# 0-50: Excellent quality (images are very similar)
# 50-100: Good quality
# 100-200: Moderate to poor quality
# 200+: Very poor quality (images are very different)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from pathlib import Path
import json
from datetime import datetime
from scipy import linalg
from torchvision import models, transforms
from PIL import Image
import random
# Own modules
from settings import setting


class FIDCalculator:
    """
    A class to calculate Frechet Inception Distance (FID) between two image folders.
    """
    
    def __init__(self):
        """
        Initialize the FID calculator.
        """
        self.prediction_folder = setting['pth_prediction'].resolve()
        self.folder1 = None
        self.folder2 = None
        self.fid_score = None
        self.results = {}
        self.num_channels = setting['img_channels']  # Get channels from settings
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        print(f"Image channels: {self.num_channels}")
        
        # Define image transformations based on number of channels
        if self.num_channels == 1:
            # Grayscale images
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
                transforms.Resize((299, 299)),  # InceptionV3 expects 299x299
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize for grayscale
            ])
        else:
            # RGB images
            self.transform = transforms.Compose([
                transforms.Resize((299, 299)),  # InceptionV3 expects 299x299
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
    def validate_folders(self):
        """
        Validate that exactly two folders exist in the prediction directory
        and that they contain images.
        
        Returns:
            bool: True if validation passes, False otherwise
        """
        if not self.prediction_folder.exists():
            print(f"Error: Prediction folder '{self.prediction_folder}' does not exist.")
            return False
            
        # Get all subdirectories
        subdirs = [d for d in self.prediction_folder.iterdir() if d.is_dir()]
        
        if len(subdirs) != 2:
            print(f"Error: Expected exactly 2 folders in '{self.prediction_folder}', found {len(subdirs)}")
            print(f"Found folders: {[d.name for d in subdirs]}")
            return False
            
        self.folder1, self.folder2 = subdirs[0], subdirs[1]
        
        # Check if folders contain images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        def count_images(folder):
            return len([f for f in folder.iterdir() 
                       if f.is_file() and f.suffix.lower() in image_extensions])
        
        count1 = count_images(self.folder1)
        count2 = count_images(self.folder2)
        
        if count1 == 0:
            print(f"Error: No images found in folder '{self.folder1.name}'")
            return False
            
        if count2 == 0:
            print(f"Error: No images found in folder '{self.folder2.name}'")
            return False
            
        print(f"Found folder 1: '{self.folder1.name}' with {count1} images")
        print(f"Found folder 2: '{self.folder2.name}' with {count2} images")
        
        self.results['folder1'] = self.folder1.name
        self.results['folder2'] = self.folder2.name
        self.results['folder1_count'] = count1
        self.results['folder2_count'] = count2
        self.results['channels'] = self.num_channels
        
        return True
    
    def load_images_from_folder(self, folder_path, max_images=None, random_seed=42):
        """
        Load images from a specific folder as a PyTorch Dataset.
        
        Args:
            folder_path (Path): Path to the folder containing images
            max_images (int): Maximum number of images to load (None for all)
            random_seed (int): Random seed for reproducible sampling
            
        Returns:
            torch.utils.data.Dataset: Dataset containing the images
        """
        class SimpleImageDataset(torch.utils.data.Dataset):
            def __init__(self, folder_path, transform=None, num_channels=1, image_paths=None):
                self.folder_path = folder_path
                self.transform = transform
                self.num_channels = num_channels
                self.image_paths = image_paths
            
            def __len__(self):
                return len(self.image_paths)
            
            def __getitem__(self, idx):
                image_path = self.image_paths[idx]
                
                # Load image based on number of channels
                if self.num_channels == 1:
                    image = Image.open(image_path).convert('L')  # Convert to grayscale
                else:
                    image = Image.open(image_path).convert('RGB')  # Convert to RGB
                
                if self.transform:
                    image = self.transform(image)
                
                return image, 0  # Return dummy label
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        all_image_paths = []
        for file_path in folder_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                all_image_paths.append(file_path)
        
        # Limit number of images if specified
        if max_images is not None and max_images < len(all_image_paths):
            # Set random seed for reproducible sampling
            if random_seed is not None:
                random.seed(random_seed)
                np.random.seed(random_seed)
            
            # Randomly sample images
            selected_paths = random.sample(all_image_paths, max_images)
            print(f"  - Randomly selected {max_images} out of {len(all_image_paths)} images")
        else:
            selected_paths = all_image_paths
        
        dataset = SimpleImageDataset(folder_path, transform=self.transform, 
                                   num_channels=self.num_channels, image_paths=selected_paths)
        return dataset
    
    def get_inception_model(self):
        """
        Load and configure the InceptionV3 model for feature extraction.
        
        Returns:
            torch.nn.Module: InceptionV3 model configured for feature extraction
        """
        # Load pretrained InceptionV3
        inception_model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        
        # Remove the final classification layer
        inception_model.fc = nn.Identity()
        
        # Set to evaluation mode
        inception_model.eval()
        
        # Move to device
        inception_model = inception_model.to(self.device)
        
        # Disable gradients
        for param in inception_model.parameters():
            param.requires_grad = False
            
        return inception_model
    
    def get_activations(self, dataset, model, batch_size=32):
        """
        Calculate activations for all images in the dataset.
        
        Args:
            dataset (Dataset): Dataset containing images
            model (torch.nn.Module): Inception model
            batch_size (int): Batch size for processing
            
        Returns:
            numpy.ndarray: Matrix of activations (n_images x 2048)
        """
        model.eval()
        activations = []
        
        # Create DataLoader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch[0].to(self.device)
                
                # Handle grayscale images by replicating to 3 channels for InceptionV3
                if images.shape[1] == 1:
                    images = images.repeat(1, 3, 1, 1)
                
                # Get features from InceptionV3
                features = model(images)
                
                activations.append(features.cpu().numpy())
        
        return np.concatenate(activations, axis=0)
    
    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """
        Calculate Frechet distance between two multivariate Gaussians.
        
        Args:
            mu1 (numpy.ndarray): Mean of first distribution
            sigma1 (numpy.ndarray): Covariance matrix of first distribution
            mu2 (numpy.ndarray): Mean of second distribution
            sigma2 (numpy.ndarray): Covariance matrix of second distribution
            eps (float): Epsilon for numerical stability
            
        Returns:
            float: Frechet distance
        """
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        
        diff = mu1 - mu2
        
        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real
        
        tr_covmean = np.trace(covmean)
        
        return (diff.dot(diff) + np.trace(sigma1) + 
                np.trace(sigma2) - 2 * tr_covmean)
    
    def calculate_activation_statistics(self, dataset, model, batch_size=32):
        """
        Calculate activation statistics for a dataset.
        
        Args:
            dataset (Dataset): Dataset containing images
            model (torch.nn.Module): Inception model
            batch_size (int): Batch size for processing
            
        Returns:
            tuple: (mu, sigma) mean and covariance of activations
        """
        act = self.get_activations(dataset, model, batch_size)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma
    
    def calculate_fid(self, batch_size=32, balance_samples=False, random_seed=42):
        """
        Calculate FID score between the two image folders.
        
        Args:
            batch_size (int): Batch size for processing images
            balance_samples (bool): Whether to balance sample sizes
            random_seed (int): Random seed for reproducible sampling
            
        Returns:
            float: FID score
        """
        print("Loading InceptionV3 model...")
        model = self.get_inception_model()
        
        # Determine maximum number of images to use
        max_images = None
        if balance_samples:
            # Use the smaller folder's count as the maximum for both
            max_images = min(self.results['folder1_count'], self.results['folder2_count'])
            print(f"Balancing samples: using {max_images} images from each folder")
            self.results['balanced_sample_size'] = max_images
            self.results['balancing_enabled'] = True
        else:
            self.results['balancing_enabled'] = False
        
        print("Loading images from first folder...")
        dataset1 = self.load_images_from_folder(self.folder1, max_images=max_images, random_seed=random_seed)
        
        print("Loading images from second folder...")
        dataset2 = self.load_images_from_folder(self.folder2, max_images=max_images, random_seed=random_seed)
        
        print(f"Using {len(dataset1)} images from folder 1")
        print(f"Using {len(dataset2)} images from folder 2")
        
        print("Calculating activations and statistics for folder 1...")
        m1, s1 = self.calculate_activation_statistics(dataset1, model, batch_size)
        
        print("Calculating activations and statistics for folder 2...")
        m2, s2 = self.calculate_activation_statistics(dataset2, model, batch_size)
        
        print("Calculating FID score...")
        fid_value = self.calculate_frechet_distance(m1, s1, m2, s2)
        
        self.fid_score = fid_value
        self.results['fid_score'] = float(fid_value)
        self.results['actual_images_used_1'] = len(dataset1)
        self.results['actual_images_used_2'] = len(dataset2)
        
        return fid_value
    
    def save_results(self):
        """Save FID results to a file in the prediction folder."""
        results_file = self.prediction_folder / "fid_results.json"
        
        # Add metadata
        self.results['timestamp'] = datetime.now().isoformat()
        self.results['device'] = str(self.device)
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Results saved to: {results_file}")
        
        # Also create a human-readable text file
        text_file = self.prediction_folder / "fid_results.txt"
        with open(text_file, 'w') as f:
            f.write("FID Score Results\n")
            f.write("=================\n\n")
            f.write(f"Folder 1: {self.results['folder1']} ({self.results['actual_images_used_1']} images used)\n")
            f.write(f"Folder 2: {self.results['folder2']} ({self.results['actual_images_used_2']} images used)\n")
            f.write(f"FID Score: {self.results['fid_score']:.4f}\n")
            f.write(f"Image Channels: {self.results['channels']}\n")
            if self.results.get('balancing_enabled', False):
                f.write(f"Sample Balancing: ENABLED (max: {self.results['balanced_sample_size']} images each)\n")
            else:
                f.write("Sample Balancing: DISABLED (using all available images)\n")
            f.write(f"Calculation Date: {self.results['timestamp']}\n")
            f.write(f"Device: {self.results['device']}\n")
        
        print(f"Human-readable results saved to: {text_file}")
    
    def print_results(self):
        """Print FID results to console."""
        print("\n" + "="*50)
        print("FID SCORE RESULTS")
        print("="*50)
        print(f"Folder 1: {self.results['folder1']} ({self.results['actual_images_used_1']} images used)")
        print(f"Folder 2: {self.results['folder2']} ({self.results['actual_images_used_2']} images used)")
        print(f"FID Score: {self.results['fid_score']:.4f}")
        print(f"Image Channels: {self.results['channels']}")
        if self.results.get('balancing_enabled', False):
            print(f"Sample Balancing: ENABLED (max: {self.results['balanced_sample_size']} images each)")
        else:
            print("Sample Balancing: DISABLED (using all available images)")
        print(f"Calculation Date: {self.results['timestamp']}")
        print(f"Device: {self.results['device']}")
        print("="*50)
    
    def run(self, batch_size=32, balance_samples=True, random_seed=42):
        """
        Main method to run the FID calculation pipeline.
        
        Args:
            batch_size (int): Batch size for processing images
            balance_samples (bool): Whether to balance sample sizes
            random_seed (int): Random seed for reproducible sampling
            
        Returns:
            float: FID score, or None if calculation failed
        """
        try:
            # Validate folders
            if not self.validate_folders():
                return None
            
            # Calculate FID
            fid_score = self.calculate_fid(batch_size, balance_samples, random_seed)
            
            # Save and display results
            self.save_results()
            self.print_results()
            
            return fid_score
            
        except Exception as e:
            print(f"Error during FID calculation: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
