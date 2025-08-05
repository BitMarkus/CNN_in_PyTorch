import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from scipy.ndimage import gaussian_filter
from settings import setting
from dataset import Dataset
from model import CNN_Model

class GradCAMAnalyzer:
    
    #############################################################################################################
    # CONSTRUCTOR:

    def __init__(self, device):
        self.device = device
        
        # Settings from configuration
        self.pth_prediction = setting['pth_prediction']
        self.pth_checkpoint = setting['pth_checkpoint']
        self.classes = setting['classes']
        self.img_channels = setting['img_channels']
        
        # Visualization parameters
        self.cmap_orig = setting['captum_cmap_orig']
        self.cmap_heatmap = setting['captum_cmap_heatmap']
        self.show_color_bar_orig = setting['captum_show_color_bar_orig']
        self.output_size = setting['captum_output_size']
        self.alpha_overlay = setting['captum_alpha_overlay']

        # Second iteration with blurring
        self.second_iteration = setting['gradcam_second_iteration']
        # Percentage of most prominent pixels to blur (0-1)
        self.threshold_percent = setting['gradcam_threshold_percent']
        # Gaussian blur strength
        self.sigma = setting['gradcam_blurr_sigma']

        # Initialize dataset and model
        # Initialize dataset and load prediction data
        self.ds = Dataset()
        if not self.ds.load_pred_dataset():
            raise ValueError("Failed to load prediction dataset")
        
        # Create model wrapper
        self.cnn_wrapper = CNN_Model()  
        # Load model wrapper with model information
        print(f"Creating new {self.cnn_wrapper.cnn_type} network...")
        # Get actual model (nn.Module)
        self.cnn = self.cnn_wrapper.load_model(device).to(device)
        print("New network was successfully created.")   

        # Checkpoints
        self.checkpoint_loaded = False
        self.loaded_checkpoint_name = None

        # For faster convolutions
        torch.backends.cudnn.benchmark = True  
        
    #############################################################################################################
    # METHODS:

    def load_checkpoint(self):
        # First get checkpoints without printing table
        silent_checkpoints = self.cnn_wrapper.print_checkpoints_table(self.pth_checkpoint, print_table=False)
        
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
            self.cnn_wrapper.print_checkpoints_table(self.pth_checkpoint)  # prints table
            checkpoint_file = self.cnn_wrapper.select_checkpoint(silent_checkpoints, "Select a checkpoint: ")
            if not checkpoint_file:
                return False
        
        try:
            full_path = self.pth_checkpoint / checkpoint_file
            self.cnn.load_state_dict(torch.load(full_path))
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

    def generate_heatmap(self, input_tensor, target_class):
        """Generate GradCAM heatmap for SPECIFIED target class"""
        self.cnn.zero_grad()
        
        # Forward pass - get features
        features = self.cnn.features(input_tensor.unsqueeze(0))
        features.retain_grad()  # Keep gradients for backprop
        
        # Forward through classifier
        pooled = F.adaptive_avg_pool2d(features, (1, 1))
        flattened = pooled.view(1, -1)
        output = self.cnn.classifier(flattened)
        
        # Force backward pass for TARGET CLASS
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Grad-CAM calculation
        grads = features.grad
        pooled_grads = torch.mean(grads, dim=[2, 3], keepdim=True)
        cam = (pooled_grads * features).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, input_tensor.shape[-2:], mode='bilinear', align_corners=False)
        
        # Normalize and detach
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-10)
        
        # Detach from computation graph before converting to numpy
        return cam.squeeze().detach().cpu().numpy(), target_class
    
    # Apply blur to the most prominent regions identified by the heatmap
    # Args:
    #    image: original image tensor (C, H, W)
    #    heatmap: Grad-CAM heatmap (H, W)
    # Returns:
    #    blurred_image: image with prominent regions blurred
    def apply_blur_mask(self, image, heatmap):

        # Convert to numpy
        if isinstance(image, torch.Tensor):
            image_np = image.permute(1, 2, 0).cpu().numpy() if self.img_channels != 1 else image.squeeze().cpu().numpy()
        else:
            image_np = image.copy()

        # Normalize heatmap and create mask
        heatmap_normalized = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-10)
        threshold = 1 - self.threshold_percent
        mask = (heatmap_normalized > threshold).astype(np.float32)

        # Smooth mask edges
        mask = gaussian_filter(mask, sigma=1)  # Softens edges
        mask = np.clip(mask, 0, 1)  # Ensures valid range

        # Apply blur
        if self.img_channels == 1:
            blurred = gaussian_filter(image_np, sigma=self.sigma)
            image_np = image_np * (1 - mask) + blurred * mask
        else:
            for c in range(self.img_channels):
                blurred = gaussian_filter(image_np[:, :, c], sigma=self.sigma)
                image_np[:, :, c] = image_np[:, :, c] * (1 - mask) + blurred * mask

        # Convert back to tensor
        if isinstance(image, torch.Tensor):
            if self.img_channels == 1:
                return torch.from_numpy(image_np).unsqueeze(0).to(self.device)
            else:
                return torch.from_numpy(image_np).permute(2, 0, 1).to(self.device)
        return image_np

    def visualize_gradcam(self, image, heatmap, img_path, class_idx, target_class, iteration=1):
        """Simplified visualization showing only folder-derived class"""
        # Convert image to numpy
        if self.img_channels == 1:
            img_np = image.squeeze().cpu().numpy()
        else:
            img_np = image.permute(1, 2, 0).cpu().numpy()
        
        # Create output directory
        output_folder = self.pth_prediction / f"{img_path.parent.name}_gradcam"
        output_folder.mkdir(exist_ok=True, parents=True)
        
        # Create figure
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        ax1.imshow(img_np if self.img_channels == 1 else np.mean(img_np, axis=2),
                cmap=self.cmap_orig)
        ax1.set_title(f"Original: {img_path.parent.name}")
        ax1.axis('off')
        
        # Heatmap
        ax2.imshow(heatmap, cmap='jet')
        ax2.set_title(f"Grad-CAM for {self.classes[target_class]}")
        ax2.axis('off')
        
        # Overlay
        ax3.imshow(img_np if self.img_channels == 1 else np.mean(img_np, axis=2),
                cmap='gray' if self.img_channels == 1 else None)
        ax3.imshow(heatmap, cmap='jet', alpha=self.alpha_overlay)
        ax3.set_title("Overlay")
        ax3.axis('off')
        
        # Save
        suffix = "_iter2" if iteration > 1 else ""
        output_path = output_folder / f"{img_path.stem}_gradcam{suffix}.png"
        plt.tight_layout()
        fig.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

    def predict_gradcam(self, dataset, second_iteration=False):
        self.cnn.eval()  # Ensure model is in eval mode
        
        for batch_idx, (images, _) in enumerate(tqdm(dataset, desc="Grad-CAM Analysis")):
            batch_paths = [Path(dataset.dataset.samples[i][0]) 
                        for i in range(batch_idx * dataset.batch_size,
                                    min((batch_idx + 1) * dataset.batch_size, len(dataset.dataset)))]
            
            for img_idx in range(len(images)):
                try:
                    img = images[img_idx].to(self.device)
                    img_path = batch_paths[img_idx]
                    
                    # Get target class DIRECTLY from folder name
                    folder_name = img_path.parent.name
                    target_class = self.classes.index(folder_name)  # Will raise ValueError if folder doesn't match classes
                    
                    # Generate heatmap FORCING the folder-derived class
                    heatmap, _ = self.generate_heatmap(img, target_class=target_class)
                    
                    # Visualize - pass the same target_class for both predicted and target
                    self.visualize_gradcam(img.detach(), heatmap, img_path, 
                                        target_class, target_class, iteration=1)
                    
                    if second_iteration:
                        blurred_img = self.apply_blur_mask(img, heatmap)
                        heatmap2, _ = self.generate_heatmap(blurred_img, target_class=target_class)
                        self.visualize_gradcam(blurred_img.detach(), heatmap2, img_path,
                                            target_class, target_class, iteration=2)
                    
                    torch.cuda.empty_cache()
                    
                except ValueError as e:
                    print(f"\nSkipping {img_path.name}: Folder '{folder_name}' not in classes {self.classes}")
                    continue
                except Exception as e:
                    print(f"\nError processing {img_path.name}: {str(e)}")
                    continue

    def verify_folder_structure(self):
        print("\nFolder Structure Verification:")
        for class_idx, class_name in enumerate(self.classes):
            class_path = self.pth_prediction / class_name
            if not class_path.exists():
                print(f"WARNING: Missing folder for class {class_name}")
                continue
                
            sample_files = list(class_path.glob("*.png"))[:3]  # Check first 3 samples
            print(f"\nClass {class_name} ({class_idx}): {class_path}")
            for f in sample_files:
                print(f"  {f.name}")

    def verify_model_predictions(self, num_samples=10):
        print("\nModel Prediction Verification:")
        correct = 0
        for i in range(min(num_samples, len(self.ds.ds_pred.dataset))):
            img, label = self.ds.ds_pred.dataset[i]
            img_path = Path(self.ds.ds_pred.dataset.samples[i][0])
            
            with torch.no_grad():
                output = self.cnn(img.unsqueeze(0).to(self.device))
                pred = torch.argmax(output).item()  # This is already a Python int
                prob = torch.softmax(output, dim=1)[0][pred].item()
                
                # Remove .item() from label since it's already a plain int from dataset
                status = "✅" if pred == label else "❌"  
                if status == "✅": 
                    correct += 1
                
                print(f"{status} {img_path.name}")
                print(f"  Folder: {img_path.parent.name}")
                print(f"  Pred: {self.classes[pred]} ({prob:.2%})")
                print(f"  Label: {self.classes[label]}\n")
        
        print(f"Accuracy: {correct}/{num_samples}")
        return correct == num_samples
    
    def debug_dataset_labels(self):
        print("\nDataset Label Verification:")
        for i, (img_path, label) in enumerate(self.ds.ds_pred.dataset.samples[:10]):  # Check first 10
            folder_name = Path(img_path).parent.name
            print(f"Image: {Path(img_path).name}")
            print(f"Folder: {folder_name}")
            print(f"Assigned label: {label} ({self.classes[label]})\n")

    #############################################################################################################
    # CALL:

    def __call__(self):
        print("\nInitializing GradCAM Analysis")
        
        # 1. Load dataset
        self.ds.load_pred_dataset()
        if not self.ds.ds_loaded:
            print("Failed to load dataset")
            return
        # self.debug_dataset_labels()

        # Load checkpoint
        if not self.load_checkpoint():
            print("WARNING: Using untrained weights!")
            self.loaded_checkpoint_name = "untrained"
        
        # 2. Verify model predictions first
        """
        if not self.verify_model_predictions():
            print("\nCRITICAL: Model predictions don't match dataset labels!")
            print("Cannot proceed with GradCAM until predictions are correct")
            return
        """

        # 3. Proceed with GradCAM
        print("\nStarting GradCAM processing...")
        self.predict_gradcam(self.ds.ds_pred, self.second_iteration)