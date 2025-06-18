import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
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

    # Generate Grad-CAM heatmap for a single input image
    # If target_class=None (default behavior):
    #   The model makes a prediction (e.g., "dog")
    #   Grad-CAM highlights regions that most influenced the model's predicted class
    # No ground truth needed! This is the most common use case
    # If target_class is specified:
    #   Grad-CAM will show regions important for that specific class (!), even if the model predicted something else
    #   Useful for debugging (e.g., "Why did the model not predict 'cat'?")
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

    # Create visualization plots for Grad-CAM results
    # Color code for heatmaps:
    # Blue: Values < 0.25 (unimportant)
    # Green: 0.25-0.75
    # Red: > 0.75 (most important)
    def visualize_gradcam(self, image, heatmap, img_path, label, target_class):
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
        # Changed height ratio to make colorbars thinner (20:1 instead of 10:1)
        gs_main = gridspec.GridSpec(2, 3, height_ratios=[20, 1], width_ratios=[1, 1, 1])
        
        # Subplot 1: Original Image
        ax1 = plt.subplot(gs_main[0, 0])
        img_vis = np.mean(img_np, axis=2) if self.img_channels != 1 else img_np
        im1 = ax1.imshow(img_vis, cmap=self.cmap_orig)
        ax1.set_title(f"Original: Class {self.classes[label.item()]}")
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
        
        output_path = output_folder / f"{img_path.stem}_gradcam.png"
        fig.savefig(output_path, bbox_inches='tight', dpi=100)
        plt.close(fig)

    # Process all images in dataset with Grad-CAM (hook-free version)
    def predict_gradcam(self, dataset):

        self.cnn.model.eval()
        
        for batch_idx, (images, labels) in enumerate(tqdm(dataset, desc="Grad-CAM Analysis")):
            batch_paths = [Path(dataset.dataset.samples[i][0]) 
                        for i in range(batch_idx * dataset.batch_size,
                                    min((batch_idx + 1) * dataset.batch_size, len(dataset.dataset)))]
            
            for img_idx in range(len(images)):
                try:
                    # 1. Prepare image
                    img = images[img_idx].to(self.device)
                    img_path = batch_paths[img_idx]
                    label = labels[img_idx]
                    
                    # 2. Generate heatmap (hook-free)
                    heatmap, target_class = self.generate_heatmap(img, label.item())
                    
                    # 3. Visualize results
                    self.visualize_gradcam(
                        image=img.detach(),
                        heatmap=heatmap,
                        img_path=img_path,
                        label=label,
                        target_class=target_class
                    )
                    
                    # 4. Cleanup
                    del img, heatmap
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
        self.predict_gradcam(self.ds.ds_pred)

"""
    def _register_hooks(self):

        target_layer = self.cnn.model.features.norm5  # Last conv layer in DenseNet-121
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def _register_hooks(self):
        target_layer = self.cnn.model.features.norm5
        
        # Use old-style hooks with gradient protection
        def backward_hook(module, grad_input, grad_output):
            with torch.no_grad():
                self.gradients = grad_output[0].clone()
        
        target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, 'activations', o.clone().detach())
        )
        target_layer.register_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor, target_class=None):

        self.cnn.model.zero_grad()
        
        # Forward pass
        output = self.cnn.model(input_tensor.unsqueeze(0))
        if target_class is None:
            target_class = torch.argmax(output).item()
        
        # Backward pass
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot)
        
        # Pool gradients and generate heatmap
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)  # Only positive influences
        cam = F.interpolate(cam, input_tensor.shape[1:], mode='bilinear', align_corners=False)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-10)
        return cam.squeeze().cpu().numpy(), target_class
"""