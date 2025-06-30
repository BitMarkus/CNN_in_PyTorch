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
        self.ds = Dataset()
        self.transform = self.ds.get_transformer_test()
        if not self.transform:
            raise ValueError("Failed to get image transformer")
        
        print("\nCreating DenseNet-121 model...")
        self.cnn = CNN_Model()
        self.cnn.model = self.cnn.load_model(self.device)
        self.cnn.model = self.cnn.model.to(self.device)
        print(f"Network {self.cnn.cnn_type} created successfully.")

        # For faster convolutions
        torch.backends.cudnn.benchmark = True  
        
    #############################################################################################################
    # METHODS:

    def generate_heatmap(self, input_tensor, target_class=None):
        self.cnn.model.zero_grad()
        
        # Forward pass while retaining activations
        with torch.no_grad():
            features = self.cnn.model.features(input_tensor.unsqueeze(0))
            activations = features.detach().requires_grad_(True)
        
        # Get model output
        output = self.cnn.model.classifier(F.relu(F.adaptive_avg_pool2d(activations, (1, 1)).view(1, -1)))
        
        if target_class is None:
            target_class = torch.argmax(output).item()
        
        # Backward pass
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        torch.cuda.empty_cache()  # Clean up unused memory
        
        # Compute Grad-CAM
        with torch.no_grad():
            weights = torch.mean(activations.grad, dim=[2, 3], keepdim=True)
            cam = torch.sum(weights * activations, dim=1)
            cam = F.relu(cam)
            cam = F.interpolate(cam.unsqueeze(0), input_tensor.shape[1:], 
                            mode='bilinear', align_corners=False)
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-10)
        
        return cam.squeeze().cpu().numpy(), target_class
    
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

    def visualize_gradcam(self, image, heatmap, img_path, label, target_class, iteration=1):
        # Convert tensors to numpy
        if self.img_channels == 1:
            img_np = image.squeeze().cpu().numpy()
        else:
            img_np = image.permute(1, 2, 0).cpu().numpy()
        
        # Create output directory
        relative_path = img_path.relative_to(self.pth_prediction)
        output_folder = self.pth_prediction / f"{relative_path.parent.name}_gradcam"
        output_folder.mkdir(exist_ok=True, parents=True)
        
        # Create figure with adjusted layout
        plt.ioff()
        fig = plt.figure(figsize=(12, 4.5))  # Slightly adjusted height
        
        # Main grid: 2 rows (images and colorbars), 3 columns
        gs_main = gridspec.GridSpec(2, 3, height_ratios=[20, 1], width_ratios=[1, 1, 1])
        
        # Subplot 1: Original Image
        ax1 = plt.subplot(gs_main[0, 0])
        img_vis = np.mean(img_np, axis=2) if self.img_channels != 1 else img_np
        im1 = ax1.imshow(img_vis, cmap=self.cmap_orig)
        title = f"Original: Class {self.classes[label.item()]}"
        if iteration > 1:
            title += f" (Iter {iteration})"
        ax1.set_title(title)
        ax1.axis('off')
        
        # Subplot 2: Heatmap
        ax2 = plt.subplot(gs_main[0, 1])
        im2 = ax2.imshow(heatmap, cmap='jet', vmin=0, vmax=1)
        ax2.set_title(f"Grad-CAM: Target class {self.classes[target_class]}")
        ax2.axis('off')
        
        # Subplot 3: Overlay
        ax3 = plt.subplot(gs_main[0, 2])
        ax3.imshow(img_vis, cmap='gray' if self.img_channels == 1 else None)
        im3 = ax3.imshow(heatmap, cmap='jet', vmin=0, vmax=1, alpha=self.alpha_overlay)
        ax3.set_title("Overlay")
        ax3.axis('off')
        
        # Color bars - now thinner
        if self.show_color_bar_orig:
            cax1 = fig.add_subplot(gs_main[1, 0])
            cbar1 = plt.colorbar(im1, cax=cax1, orientation='horizontal')
            cbar1.ax.tick_params(labelsize=6)  # Even smaller font
        
        cax2 = fig.add_subplot(gs_main[1, 1])
        cbar2 = plt.colorbar(im2, cax=cax2, orientation='horizontal')
        cbar2.ax.tick_params(labelsize=6, pad=0.1)  # Reduced padding
        
        cax3 = fig.add_subplot(gs_main[1, 2])
        cbar3 = plt.colorbar(im3, cax=cax3, orientation='horizontal')
        cbar3.ax.tick_params(labelsize=6, pad=0.1)
        
        # Make colorbars even more compact
        for cbar in [cbar1, cbar2, cbar3] if self.show_color_bar_orig else [cbar2, cbar3]:
            cbar.ax.xaxis.set_tick_params(pad=1)  # Reduce tick label padding
            cbar.outline.set_linewidth(0.5)  # Thinner border
        
        # Adjust layout with less space for colorbars
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.05, wspace=0.1)  # Reduced vertical space
        
        # Modify output filename for second iteration
        suffix = "_iter2" if iteration > 1 else ""
        output_path = output_folder / f"{img_path.stem}_gradcam{suffix}.png"
        fig.savefig(output_path, bbox_inches='tight', dpi=100)
        plt.close(fig)

    # Args:
    #    second_iteration: if True, performs a second Grad-CAM after blurring prominent features
    def predict_gradcam(self, dataset, second_iteration=False, threshold_percent=0.2, sigma=5):
        self.cnn.model.eval()
        
        for batch_idx, (images, labels) in enumerate(tqdm(dataset, desc="Grad-CAM Analysis")):
            batch_paths = [Path(dataset.dataset.samples[i][0]) 
                        for i in range(batch_idx * dataset.batch_size,
                                    min((batch_idx + 1) * dataset.batch_size, len(dataset.dataset)))]
            
            for img_idx in range(len(images)):
                try:
                    # 1. First Grad-CAM pass
                    img = images[img_idx].to(self.device)
                    img_path = batch_paths[img_idx]
                    label = labels[img_idx]
                    
                    heatmap1, target_class1 = self.generate_heatmap(img, label.item())
                    self.visualize_gradcam(img.detach(), heatmap1, img_path, label, target_class1, iteration=1)
                    
                    # 2. Second iteration with blurring
                    if second_iteration:
                        # Apply blur to prominent regions from first heatmap
                        blurred_img = self.apply_blur_mask(img, heatmap1)
                        
                        # Generate new heatmap on blurred image
                        heatmap2, target_class2 = self.generate_heatmap(blurred_img, label.item())
                        
                        # Visualize second iteration
                        self.visualize_gradcam(blurred_img.detach(), heatmap2, img_path, label, target_class2, iteration=2)
                    
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"\nError processing {img_path.name}: {str(e)}")
                    continue

    #############################################################################################################
    # CALL:
    def __call__(self):
        print("\nLoading dataset for prediction:")
        self.ds.load_pred_dataset()
        if self.ds.ds_loaded:
            print("Dataset loaded successfully!")
        
        print("\nLoading model weights:")
        self.cnn.load_checkpoint()
        
        print("\nRunning Grad-CAM analysis:")
        self.predict_gradcam(self.ds.ds_pred, self.second_iteration)