import torch
import pandas as pd
from torch.utils.data import DataLoader
from PIL import Image
import torchvision
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
# Own modules
from settings import setting
from dataset import Dataset
from custom_model import Custom_CNN_Model
from model import CNN_Model

class CaptumAnalyzer:

    def __init__(self, device):
        self.device = device
        # Settings parameters
        self.pth_prediction = setting['pth_prediction'].resolve()
        self.pth_checkpoint = setting['pth_checkpoint'].resolve()
        self.classes = setting['classes']
        self.img_channels = setting['img_channels']
        self.pred_batch_size = 1
        
        # Initialize dataset to get transforms
        self.ds = Dataset()
        self.transform = self.ds.get_transformer_test()
        if not self.transform:
            raise ValueError("Failed to get image transformer")
        
        # Create model wrapper
        print(f"Creating new network...")
        if setting["cnn_type"] == "custom":
            self.cnn = Custom_CNN_Model()
            self.cnn.model = self.cnn
        else:
            self.cnn = CNN_Model()
            self.cnn.model = self.cnn.load_model(self.device)

        # Ensure everything is on the right device
        self.cnn.model = self.cnn.model.to(self.device)
        if hasattr(self.cnn, 'to'):
            self.cnn = self.cnn.to(self.device)
        print(f"Network {self.cnn.cnn_type} was successfully created.")

        # Initialize checkpoint flag
        self.checkpoint_loaded = False
        self.loaded_checkpoint_name = None

    def load_checkpoint(self):
        # First get checkpoints without printing table
        silent_checkpoints = self.cnn.print_checkpoints_table(self.pth_checkpoint, print_table=False)
        
        if not silent_checkpoints:
            print("The checkpoint folder is empty!")
            return False
        
        # If only one checkpoint exists
        if len(silent_checkpoints) == 1:
            # Extract filename from the tuple
            checkpoint_file = silent_checkpoints[0][1]  # (id, name) -> get name
            print(f"\nFound single checkpoint: {checkpoint_file}")
            print("Loading automatically...")
        else:
            # Show interactive table for multiple checkpoints
            self.cnn.print_checkpoints_table(self.pth_checkpoint)  # prints table
            checkpoint_file = self.cnn.select_checkpoint(silent_checkpoints, "Select a checkpoint: ")
            if not checkpoint_file:
                return False
        
        try:
            full_path = self.pth_checkpoint / checkpoint_file
            self.cnn.model.load_state_dict(torch.load(full_path))
            self.checkpoint_loaded = True
            self.loaded_checkpoint_name = full_path.stem
            print(f"Successfully loaded weights from {checkpoint_file}")
            return True
        except FileNotFoundError as e:
            print(f"\nError loading checkpoint: {str(e)}")
            print(f"Full path attempted: {full_path}")
            return False
        except Exception as e:
            print(f"\nError loading checkpoint: {str(e)}")
            return False
    
    def create_prediction_loader(self, folder_path):
        """Create a DataLoader that properly handles image loading and transformation"""
        class FlatImageFolder(torch.utils.data.Dataset):
            def __init__(self, folder, transform=None, img_channels=1):
                self.folder = Path(folder)
                self.transform = transform
                self.img_channels = img_channels
                self.image_files = [f for f in self.folder.iterdir() 
                                  if f.suffix.lower() in ('.png', '.jpg', '.jpeg')]
                if not self.image_files:
                    raise ValueError(f"No images found in {folder}")

            def __len__(self):
                return len(self.image_files)

            def __getitem__(self, idx):
                img_path = self.image_files[idx]
                if self.img_channels == 1:
                    img = Image.open(img_path).convert('L')
                else:
                    img = Image.open(img_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                return img, 0, str(img_path)  # Added image path as third return value

        dataset = FlatImageFolder(folder_path, transform=self.transform, img_channels=self.img_channels)
        return DataLoader(dataset, batch_size=self.pred_batch_size, shuffle=False)

    def analyze_with_captum(self, folder_path):
        """Analyze images with Captum and save visualizations (memory-optimized version)"""
        # Set matplotlib to non-interactive backend
        plt.switch_backend('Agg')
        
        # Create output folder
        output_folder = Path(folder_path).parent / (Path(folder_path).name + "_captum")
        output_folder.mkdir(exist_ok=True)
        
        # Create loader with batch_size=1
        prediction_loader = self.create_prediction_loader(folder_path)
        
        # Set model to evaluation mode
        self.cnn.model.eval()
        
        # Disable gradients for all parameters
        for param in self.cnn.model.parameters():
            param.requires_grad = False
        
        try:
            for batch in tqdm(prediction_loader, desc="Processing images with Captum"):
                images, _, img_paths = batch
                images = images.to(self.device)
                
                # Process each image individually
                for i in range(len(images)):
                    img = images[i].unsqueeze(0)  # Add batch dimension
                    img_path = img_paths[i]
                    
                    # Get prediction
                    with torch.no_grad():
                        output = self.cnn.model(img)
                        probability = torch.softmax(output, dim=1)
                        _, pred_class = torch.max(output, 1)
                    
                    # Clear any existing gradients
                    self.cnn.model.zero_grad()
                    
                    # Compute attributions
                    ig = IntegratedGradients(self.cnn.model)
                    attributions = ig.attribute(
                        img, 
                        target=pred_class.item(),
                        n_steps=25
                    )
                    
                    # Convert to numpy and move to CPU
                    img_np = img.squeeze(0).cpu().numpy()
                    attr_np = attributions.squeeze(0).cpu().numpy()
                    
                    # Handle grayscale vs RGB images differently
                    if self.img_channels == 1:
                        # For grayscale: (C, H, W) -> (H, W)
                        img_np = img_np.squeeze(0)  # Remove channel dimension
                        attr_np = attr_np.squeeze(0)
                        
                        # Add channel dimension back for visualization (H, W) -> (H, W, 1)
                        img_np = np.expand_dims(img_np, axis=2)
                        attr_np = np.expand_dims(attr_np, axis=2)
                    else:
                        # For RGB: (C, H, W) -> (H, W, C)
                        img_np = np.transpose(img_np, (1, 2, 0))
                        attr_np = np.transpose(attr_np, (1, 2, 0))
                    
                    # Normalize attribution values
                    attr_np = (attr_np - attr_np.min()) / (attr_np.max() - attr_np.min())
                    
                    # Clean up GPU memory immediately
                    del img, output, attributions
                    torch.cuda.empty_cache()
                    
                    # Create visualization
                    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
                    viz.visualize_image_attr(
                        attr_np,
                        original_image=img_np,
                        method="heat_map",
                        sign="positive",
                        cmap="inferno",
                        show_colorbar=True,
                        title=f"Class: {self.classes[pred_class.item()]} ({probability[0][pred_class].item():.2f})",
                        plt_fig_axis=(fig, ax)
                    )
                    
                    # Save and close
                    img_name = Path(img_path).stem
                    output_path = output_folder / f"{img_name}_captum.png"
                    fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
                    plt.close(fig)
                    
        finally:
            # Re-enable gradients when done
            for param in self.cnn.model.parameters():
                param.requires_grad = True
        
        print(f"Saved Captum visualizations to: {output_folder}")

    def analyze_prediction_folder(self):
        """Analyze all folders in the prediction directory with Captum"""
        if not self.load_checkpoint():
            print("WARNING: Proceeding with untrained weights!\n")
            self.loaded_checkpoint_name = "untrained"

        subfolders = [f for f in self.pth_prediction.iterdir() if f.is_dir()]
        
        if not subfolders:
            print(f"\nNo subfolders found in {self.pth_prediction}")
            return None
        
        print("\n>> Starting Captum analysis")
        print(f"Found {len(subfolders)} folders to process")
        
        for folder in subfolders:
            try:
                print(f"\n> PROCESSING FOLDER: {folder.name}")
                self.analyze_with_captum(folder)
            except Exception as e:
                print(f"Error processing folder {folder}: {str(e)}")
                continue
        
        print("\nCaptum analysis complete.")