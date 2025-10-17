#########################
# FID Score calculation #
#########################

# FID Score Scale Reference:
# 0-50: Excellent quality (images are very similar)
# 50-100: Good quality
# 100-200: Moderate to poor quality
# 200+: Very poor quality (images are very different)

import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from scipy import linalg
from torchvision import models, transforms
from PIL import Image
import random
# Own modules
from settings import setting

class FIDCalculator:

    #############################################################################################################
    # CONSTRUCTOR:
    
    def __init__(self, device, reference_folder=None):

        # Passed parameters
        self.device = device
        self.reference_folder = reference_folder  # Name of the reference folder
        
        # Settings parameters
        self.prediction_folder = setting['pth_prediction'].resolve()
        self.num_channels = setting['img_channels']  # Get channels from settings
        # Batch size for processing images for FID calculation
        self.batch_size = setting['fid_batch_size']
        # When this is set to True, the number of images is determined by the folder with the least images
        # to balance the number of images for the calculation
        self.balance_samples = setting['fid_balance_samples']
        # Random seed for randomly choosing images if balance samples is set to True
        self.random_seed = setting['fid_random_seed']
        
        # Class variables
        self.folders = {}  # Dictionary to store folder paths: {name: path}
        self.reference_folder_name = None  # Actual name of reference folder after validation
        self.fid_scores = {}  # Dictionary to store FID scores: {folder_name: score}
        self.results = {}

    #############################################################################################################
    # METHODS:

    # Validate folders in the prediction directory and identify reference folder
    def validate_folders(self):

        if not self.prediction_folder.exists():
            print(f"Error: Prediction folder '{self.prediction_folder}' does not exist.")
            return False
            
        # Get all subdirectories
        subdirs = [d for d in self.prediction_folder.iterdir() if d.is_dir()]
        
        if len(subdirs) < 2:
            print(f"Error: Expected at least 2 folders in '{self.prediction_folder}', found {len(subdirs)}")
            print(f"Found folders: {[d.name for d in subdirs]}")
            return False
            
        # Check if folders contain images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        def count_images(folder):
            return len([f for f in folder.iterdir() 
                       if f.is_file() and f.suffix.lower() in image_extensions])
        
        # Store all valid folders with their image counts
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
        
        # Determine reference folder
        if len(valid_folders) == 2:
            # If exactly 2 folders, no need for explicit reference folder
            folder_names = list(valid_folders.keys())
            self.reference_folder_name = folder_names[0]
            print(f"Exactly 2 folders found. Using '{self.reference_folder_name}' as reference.")
        else:
            # For 3+ folders, let user choose reference folder interactively
            print(f"\nFound {len(valid_folders)} folders with images:")
            print("Please select the reference folder by entering the corresponding index:")
            print("-" * 50)
            
            folder_list = list(valid_folders.keys())
            for idx, folder_name in enumerate(folder_list, 1):
                count = valid_folders[folder_name]['count']
                print(f"  [{idx}] {folder_name} ({count} images)")
            
            print("-" * 50)
            
            # Get user input for reference folder selection
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
        
        # Print folder information
        print(f"Folder configuration:")
        for folder_name, info in valid_folders.items():
            marker = " [REFERENCE]" if folder_name == self.reference_folder_name else ""
            print(f"  - '{folder_name}': {info['count']} images{marker}")
        
        self.results['all_folders'] = {name: info['count'] for name, info in valid_folders.items()}
        self.results['reference_folder'] = self.reference_folder_name
        self.results['total_folders'] = len(valid_folders)
        self.results['channels'] = self.num_channels
        
        return True

    # Load images from a specific folder as a PyTorch Dataset
    def load_images_from_folder(self, folder_path, max_images=None, random_seed=42):

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
        
        dataset = InceptionDataset(folder_path, num_channels=self.num_channels, image_paths=selected_paths)
        return dataset

    # Load and configure the InceptionV3 model for feature extraction
    def get_inception_model(self):

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

    # Calculate activations for all images in the dataset
    def get_activations(self, dataset, model, batch_size=32):

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

    # Calculate Frechet distance between two multivariate Gaussians
    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):

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

    # Calculate activation statistics for a dataset
    def calculate_activation_statistics(self, dataset, model, batch_size=32):

        act = self.get_activations(dataset, model, batch_size)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma

    # Calculate FID score between the reference folder and all other folders
    def calculate_fid(self, batch_size=None, balance_samples=None, random_seed=None):

        # Use instance variables if parameters are not provided
        if batch_size is None:
            batch_size = self.batch_size
        if balance_samples is None:
            balance_samples = self.balance_samples
        if random_seed is None:
            random_seed = self.random_seed

        print("Loading InceptionV3 model...")
        model = self.get_inception_model()
        
        # Get reference folder info
        ref_folder_info = self.folders[self.reference_folder_name]
        ref_folder_path = ref_folder_info['path']
        ref_image_count = ref_folder_info['count']
        
        # Determine maximum number of images to use for balancing
        max_images = None
        if balance_samples:
            # Find the minimum image count among all folders for balanced comparison
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
        
        # Calculate FID scores for all other folders compared to reference
        self.fid_scores = {}
        
        for folder_name, folder_info in self.folders.items():
            if folder_name == self.reference_folder_name:
                continue  # Skip reference folder
                
            print(f"\n> Processing folder '{folder_name}'...")
            folder_path = folder_info['path']
            
            print(f"Loading images from '{folder_name}'...")
            compare_dataset = self.load_images_from_folder(folder_path, max_images=max_images, random_seed=random_seed)
            
            print(f"Calculating activations and statistics for '{folder_name}'...")
            compare_mu, compare_sigma = self.calculate_activation_statistics(compare_dataset, model, batch_size)
            
            print(f"Calculating FID score between '{self.reference_folder_name}' and '{folder_name}'...")
            fid_value = self.calculate_frechet_distance(ref_mu, ref_sigma, compare_mu, compare_sigma)
            
            self.fid_scores[folder_name] = float(fid_value)
            
            # Store detailed results
            self.results[f'fid_{folder_name}'] = float(fid_value)
            self.results[f'images_used_{folder_name}'] = len(compare_dataset)
        
        # Store reference folder usage
        self.results['reference_images_used'] = len(ref_dataset)
        self.results['fid_scores'] = self.fid_scores.copy()
        
        return self.fid_scores

    # Save FID results to a file in the prediction folder
    def save_results(self):
       
        # Add metadata
        self.results['timestamp'] = datetime.now().isoformat()
        self.results['device'] = str(self.device)
        
        # Create a human-readable text file
        text_file = self.prediction_folder / "fid_results.txt"
        with open(text_file, 'w') as f:
            f.write("FID Score Results\n")
            f.write("=================\n\n")
            
            f.write(f"Reference Folder: {self.results['reference_folder']} ({self.results['reference_images_used']} images used)\n\n")
            
            f.write("FID Scores (compared to reference):\n")
            f.write("-" * 50 + "\n")
            
            # Sort folders by FID score for better readability
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

    # Print FID results to console
    def print_results(self):

        print("\n>> FID SCORE RESULTS")
        print("-" * 60)
        print(f"Reference Folder: {self.results['reference_folder']} ({self.results['reference_images_used']} images used)")
        print(f"Image Channels: {self.results['channels']}")
        print(f"Total Folders: {self.results['total_folders']}")
        
        print("\nFID Scores (compared to reference):")
        print("-" * 40)
        
        # Sort by FID score for better readability
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
    # CALL:

    # Main analysis method
    def __call__(self):

        try:
            # Validate folders
            if not self.validate_folders():
                print(f"\nFID calculation failed: Folder validation error!")
                return None
            
            # Calculate FID using parameters from constructor
            fid_scores = self.calculate_fid(self.batch_size, self.balance_samples, self.random_seed)
            
            # Save and display results
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
# SIMPLE IMAGE DATASET CLASS:

# Simple dataset for loading images from a folder for the Inception network architecture
class InceptionDataset(torch.utils.data.Dataset):
    
    def __init__(self, folder_path, num_channels=1, image_paths=None):

        self.folder_path = folder_path
        self.num_channels = num_channels
        self.image_paths = image_paths

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