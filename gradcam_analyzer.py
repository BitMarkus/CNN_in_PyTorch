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
    ###########################################################################
    # CONSTRUCTOR
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
        
        # Register hooks for Grad-CAM
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks for Grad-CAM."""
        target_layer = self.cnn.model.features.norm5  # Last conv layer in DenseNet-121
        
        # Forward hook (unchanged)
        def forward_hook(module, input, output):
            self.activations = output.detach()
        target_layer.register_forward_hook(forward_hook)
        
        # Replace backward_hook with FULL backward hook
        self.backward_hook = target_layer.register_full_backward_hook(
            lambda module, grad_in, grad_out: setattr(self, 'gradients', grad_out[0].detach())
        )

    ###########################################################################
    # GRAD-CAM METHODS
    def generate_heatmap(self, input_tensor, target_class=None):
        """Generate Grad-CAM heatmap for a single input image."""
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

    def visualize_gradcam(self, image, heatmap, img_path, label, target_class):
        """Create visualization plots for Grad-CAM results."""
        # Convert tensors to numpy
        if self.img_channels == 1:
            img_np = image.squeeze().cpu().numpy()
        else:
            img_np = image.permute(1, 2, 0).cpu().numpy()
        
        # Create output directory
        relative_path = img_path.relative_to(self.pth_prediction)
        output_folder = self.pth_prediction / f"{relative_path.parent.name}_gradcam"
        output_folder.mkdir(exist_ok=True, parents=True)
        
        # Create figure
        plt.ioff()
        fig = plt.figure(figsize=(self.output_size * 3 + 0.5, self.output_size))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1.1, 1, 1])
        
        # Subplot 1: Original Image
        ax1 = plt.subplot(gs[0])
        img_vis = np.mean(img_np, axis=2) if self.img_channels != 1 else img_np
        im1 = ax1.imshow(img_vis, cmap=self.cmap_orig)
        ax1.set_title(f"Original: {self.classes[label.item()]}")
        ax1.axis('off')
        if self.show_color_bar_orig:
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # Subplot 2: Heatmap
        ax2 = plt.subplot(gs[1])
        im2 = ax2.imshow(heatmap, cmap=self.cmap_heatmap)
        ax2.set_title(f"Grad-CAM (Class {self.classes[target_class]})")
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        # Subplot 3: Overlay
        ax3 = plt.subplot(gs[2])
        ax3.imshow(img_vis, cmap='gray' if self.img_channels == 1 else None)
        im3 = ax3.imshow(heatmap, cmap=self.cmap_heatmap, alpha=self.alpha_overlay)
        ax3.set_title("Overlay")
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        
        # Save figure
        plt.tight_layout()
        output_path = output_folder / f"{img_path.stem}_gradcam.png"
        fig.savefig(output_path, bbox_inches='tight', dpi=100)
        plt.close(fig)

    ###########################################################################
    # MAIN PREDICTION LOOP
    def predict_gradcam(self, dataset):
        """Process all images in dataset with Grad-CAM."""
        self.cnn.model.eval()  # Keep model in eval mode
        
        for batch_idx, (images, labels) in enumerate(tqdm(dataset)):
            batch_paths = [Path(dataset.dataset.samples[i][0]) 
                        for i in range(batch_idx * dataset.batch_size,
                                    min((batch_idx + 1) * dataset.batch_size, len(dataset.dataset)))]
            
            for img_idx in range(len(images)):
                try:
                    torch.cuda.empty_cache()
                    
                    # 1. Prepare image with gradient tracking
                    img = images[img_idx].to(self.device)
                    img.requires_grad_()  # Enable gradients for input
                    
                    label = labels[img_idx]
                    img_path = batch_paths[img_idx]
                    
                    # 2. Forward pass with gradient context
                    with torch.set_grad_enabled(True):  # Force gradient computation
                        heatmap, target_class = self.generate_heatmap(img, label.item())
                    
                    # 3. Visualize results
                    self.visualize_gradcam(img.detach(), heatmap, img_path, label, target_class)
                    
                    del img, heatmap
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"Error processing {batch_paths[img_idx]}: {str(e)}")
                    continue

    ###########################################################################
    # CALL
    def __call__(self):
        """Main execution method."""
        print("\nLoading dataset for prediction:")
        self.ds.load_pred_dataset()
        if self.ds.ds_loaded:
            print("Dataset loaded successfully!")
        
        print("\nLoading model weights:")
        self.cnn.load_checkpoint()
        
        print("\nRunning Grad-CAM analysis:")
        self.predict_gradcam(self.ds.ds_pred)