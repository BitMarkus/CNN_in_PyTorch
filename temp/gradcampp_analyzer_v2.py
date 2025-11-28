import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tqdm import tqdm
from pathlib import Path
import warnings
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from captum.attr import LayerGradCam, LRP, IntegratedGradients 
# Own modules
from settings import setting
from dataset import Dataset
from model import CNN_Model

# Suppress specific warnings
warnings.filterwarnings("ignore", message="Input Tensor 0 did not already require gradients")

class GradCAMpp_Analyzer:

    def __init__(self, device: torch.device, debug: bool = False):
        self.device = device
        self.debug = debug
        
        # Load settings
        self.pth_prediction = Path(setting['pth_prediction'])
        self.classes = setting['classes']
        self.img_size = (setting['img_width'], setting['img_height'])
        self.img_channels = setting['img_channels']
        self.dpi = setting.get('captum_dpi', 100)
        self.alpha_overlay = setting.get('captum_alpha_overlay', 0.5)
        
        # Initialize dataset and model
        self.ds = Dataset()
        self.transform = self.ds.get_transformer_test()
        self.cnn = CNN_Model()
        
        # Load model
        with torch.no_grad():
            self.cnn.model = self.cnn.load_model(self.device).to(self.device).eval()
        
        # Freeze model
        for param in self.cnn.model.parameters():
            param.requires_grad = False
            
        # Target layer selection - using the same approach as your old script
        if hasattr(self.cnn.model, 'features'):
            self.target_layer = self.cnn.model.features[-1]  # Last convolutional layer
        else:
            # Fallback to finding last conv layer
            for name, module in reversed(list(self.cnn.model.named_modules())):
                if isinstance(module, torch.nn.Conv2d):
                    self.target_layer = module
                    break
            if not hasattr(self, 'target_layer'):
                raise ValueError("Could not find suitable convolutional layer")

    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.empty_cache()

    def create_output_structure(self, img_path: Path) -> Path:
        """Create output directory structure with _gradcampp suffix"""
        subfolder_name = f"{img_path.parent.name}_gradcampp"
        output_dir = self.pth_prediction / subfolder_name
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / f"{img_path.stem}_analysis.png"

    def _temporary_grad_enabled(self, model):
        """Context manager to temporarily enable gradients"""
        class GradContext:
            def __init__(self, model):
                self.model = model
                self.original_grad_state = {}
                
            def __enter__(self):
                for name, param in self.model.named_parameters():
                    self.original_grad_state[name] = param.requires_grad
                    param.requires_grad_(True)
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                for name, param in self.model.named_parameters():
                    param.requires_grad_(self.original_grad_state[name])
        
        return GradContext(model)

    def gradcam_pp(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        """Memory-efficient GradCAM++ implementation"""
        attributions = None
        try:
            with torch.no_grad():
                input_tensor = input_tensor.requires_grad_(True)
                
                with self._temporary_grad_enabled(self.cnn.model):
                    layer_gc = LayerGradCam(self.cnn.model, self.gradcam_target_layer)
                    attributions = layer_gc.attribute(
                        input_tensor, 
                        target=target_class,
                        relu_attributions=True
                    )
                    
                    cam = attributions.squeeze().detach().cpu().float().numpy()  # Added detach()
                    if self.img_channels == 1 and cam.ndim == 3:
                        cam = cam[0]
                        
                    cam = np.maximum(cam, 0)
                    cam = cv2.resize(cam, self.img_size[::-1])
                    return np.clip(cam / (np.percentile(cam, 99.9) + 1e-10), 0, 1)
                    
        except Exception as e:
            print(f"GradCAM++ computation failed: {str(e)}")
            return np.zeros(self.img_size[::-1])
        finally:
            if attributions is not None:
                del attributions
            torch.cuda.empty_cache()

    def _prepare_image(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Convert input tensor to numpy image"""
        img_np = input_tensor.squeeze().detach().cpu().numpy()
        
        if img_np.ndim == 3:
            if self.img_channels == 1:
                img_np = img_np[0]
            else:
                img_np = np.transpose(img_np, (1, 2, 0))
        
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-10)
        return img_np

    def ig_analysis(self, input_tensor, target_class):
        """Fixed Integrated Gradients implementation"""
        try:
            torch.cuda.empty_cache()
            
            # Ensure proper input dimensions
            input_tensor = input_tensor.unsqueeze(0) if input_tensor.dim() == 3 else input_tensor
            input_tensor = input_tensor.to(self.device).requires_grad_(True)
            
            # Initialize IG
            ig = IntegratedGradients(self.cnn.model)
            
            # Compute attributions
            attributions = ig.attribute(
                input_tensor,
                target=target_class,
                baselines=torch.zeros_like(input_tensor),
                n_steps=25,
                internal_batch_size=1
            )
            
            # Process attributions
            if attributions is None:
                return np.zeros(input_tensor.shape[2:])
                
            attr_np = attributions.squeeze().detach().cpu().float().numpy()
            
            if attr_np.ndim == 3:
                attr_np = np.mean(attr_np, axis=0)
                
            # Only positive attributions
            pos_attr = np.maximum(attr_np, 0)
            pos_attr = gaussian_filter(pos_attr, sigma=2.0)
            
            if np.max(pos_attr) > 0:
                pos_attr = (pos_attr / np.max(pos_attr)) ** 0.5
                pos_attr = pos_attr / np.max(pos_attr)
            
            return pos_attr
            
        except Exception as e:
            print(f"IG analysis failed: {str(e)}")
            return np.zeros(input_tensor.shape[2:])
        finally:
            torch.cuda.empty_cache()

    def visualize(self, input_tensor, label, img_path):
        """Fixed visualization pipeline"""
        try:
            if input_tensor is None or label is None:
                return
                
            # Prepare input - ensure no extra dimensions
            input_tensor = input_tensor.to(self.device)
            while input_tensor.dim() > 4:  # Remove any extra dimensions
                input_tensor = input_tensor.squeeze(0)
            if input_tensor.dim() == 3:
                input_tensor = input_tensor.unsqueeze(0)
            
            # Get prediction
            with torch.no_grad():
                output = self.cnn.model(input_tensor)
                pred_prob = torch.softmax(output, dim=1)[0, label.item()].item()
            
            # Generate heatmaps with proper dimension handling
            gradcam_map = self.generate_gradcam(input_tensor[0].clone(), label.item())
            ig_map = self.ig_analysis(input_tensor[0].clone(), label.item())
            
            # Prepare image
            img_np = self._prepare_image(input_tensor[0])
            self._create_visualization(img_np, gradcam_map, ig_map, label, pred_prob, img_path)
            
        except Exception as e:
            print(f"Visualization pipeline failed: {str(e)}")
        finally:
            torch.cuda.empty_cache()

    def analyze_dataset(self, dataloader):
        """Process complete dataset"""
        if not hasattr(dataloader, 'dataset'):
            raise ValueError("Invalid dataloader")
        
        self.cnn.model.eval()
        try:
            for batch_idx, (images, labels) in enumerate(tqdm(dataloader)):
                if images is None or labels is None:
                    continue
                    
                batch_paths = [Path(x[0]) for x in dataloader.dataset.samples[
                    batch_idx*dataloader.batch_size:(batch_idx+1)*dataloader.batch_size
                ]]
                
                for img_idx in range(images.size(0)):
                    try:
                        img = images[img_idx].unsqueeze(0).to(self.device)
                        self.visualize(img, labels[img_idx], batch_paths[img_idx])
                    except Exception as e:
                        print(f"Error processing image: {str(e)}")
                    finally:
                        torch.cuda.empty_cache()
        except Exception as e:
            print(f"Dataset analysis failed: {str(e)}")
        finally:
            torch.cuda.empty_cache()

    def generate_gradcam(self, input_tensor, target_class):
        """Fully debugged GradCAM matching original script behavior"""
        try:
            # 1. Force CUDA context initialization
            if 'cuda' in str(self.device):
                _ = torch.zeros(1).to(self.device)  # Ensure context exists

            # 2. Input preparation (exactly as original)
            input_tensor = input_tensor.unsqueeze(0).to(self.device)
            input_tensor.requires_grad = True  # Original uses simple assignment

            # 3. Activation capture with gradient retention
            activations = None
            gradients = None
            
            def forward_hook(module, inp, out):
                nonlocal activations
                activations = out
                # Must retain grad for non-leaf tensors
                activations.retain_grad()  # Critical difference
                
            def backward_hook(module, grad_in, grad_out):
                nonlocal gradients
                gradients = grad_out[0]  # Capture raw gradients

            # Register both hooks
            forward_handle = self.target_layer.register_forward_hook(forward_hook)
            backward_handle = self.target_layer.register_full_backward_hook(backward_hook)

            # 4. Forward pass (original method)
            if hasattr(self.cnn.model, 'features'):
                features = self.cnn.model.features(input_tensor)
                output = self.cnn.model.classifier(
                    F.relu(F.adaptive_avg_pool2d(features, (1, 1))).view(1, -1)
                )
            else:
                output = self.cnn.model(input_tensor)

            # 5. Backward pass
            one_hot = torch.zeros_like(output)
            one_hot[0][target_class] = 1
            output.backward(gradient=one_hot)

            # 6. Debug output
            # print(f"Activations shape: {activations.shape}")
            # print(f"Gradient stats - Max: {gradients.abs().max().item():.4f}")

            # 7. Compute GradCAM (original math)
            with torch.no_grad():
                weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
                cam = torch.sum(weights * activations, dim=1)
                cam = F.relu(cam)
                
                # Original normalization
                cam = cam - cam.min()
                cam = cam / (cam.max() + 1e-10)
                
                cam = F.interpolate(
                    cam.unsqueeze(0),
                    size=input_tensor.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
                
                return cam.squeeze().cpu().numpy()

        except Exception as e:
            print(f"GradCAM failed: {str(e)}")
            return np.zeros(input_tensor.shape[2:])
        finally:
            if 'forward_handle' in locals():
                forward_handle.remove()
            if 'backward_handle' in locals():
                backward_handle.remove()
            torch.cuda.empty_cache()

    def _plot_content(self, ax, title, content):
        """Plot individual panels with proper image dimension handling"""
        try:
            # Process content based on its type
            if isinstance(content, tuple):
                # Original image (always grayscale)
                img_data = content[0].squeeze()
                if img_data.ndim == 3 and img_data.shape[0] == 1:
                    img_data = img_data[0]
                img = ax.imshow(img_data, cmap='gray')
                
                # Don't show colorbar for overlay plots
                if len(content) == 1:  # Only original image
                    cbar = plt.colorbar(img, ax=ax, orientation='horizontal', 
                                      fraction=0.046, pad=0.04)
                    cbar.outline.set_visible(False)
                
                # Handle overlays with distinct colors
                for i, overlay_data in enumerate(content[1:]):
                    overlay_data = overlay_data.squeeze()
                    if "IG" in title or (i > 0 and "Combined" in title):
                        # Use coolwarm colormap for IG (blue-white-red)
                        heat = ax.imshow(overlay_data, cmap='coolwarm', alpha=self.alpha_overlay)
                    else:
                        # Use viridis colormap for GradCAM (yellow-green-blue)
                        heat = ax.imshow(overlay_data, cmap='viridis', alpha=self.alpha_overlay)
            else:
                # Single heatmap case
                heatmap_data = content.squeeze()
                cmap = 'coolwarm' if "IG" in title else 'viridis'  # Distinct colormaps
                heat = ax.imshow(heatmap_data, cmap=cmap)
                cbar = plt.colorbar(heat, ax=ax, orientation='horizontal', 
                                  fraction=0.046, pad=0.04)
                cbar.outline.set_visible(False)
            
            ax.set_title(title, pad=10)  # Added padding to title
            ax.axis('off')
        except Exception as e:
            print(f"Plotting failed: {str(e)}")

    def _create_visualization(self, img_np_plot, gradcam_map, ig_map, label, pred_prob, img_path):
        """Create visualization with proper image handling"""
        plt.ioff()
        try:
            # Create figure with adjusted layout
            fig = plt.figure(figsize=(15, 10), dpi=self.dpi)
            
            # Use constrained layout for better spacing control
            fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01)  
            
            # Grid specification with more vertical space between rows
            gs = gridspec.GridSpec(2, 3, figure=fig, 
                                  height_ratios=[1, 1],
                                  hspace=0.15,  # Increased vertical space
                                  wspace=0.01)  # Moderate horizontal space
            
            # Ensure all inputs are properly squeezed
            img_plot = img_np_plot.squeeze()
            grad_map = gradcam_map.squeeze()
            ig_heat = ig_map.squeeze()
            
            # Original image with class/probability
            ax0 = fig.add_subplot(gs[0, 0])
            self._plot_content(ax0, f"Class: {label} | Prob: {pred_prob:.2f}", (img_plot,))
            
            # GradCAM
            ax1 = fig.add_subplot(gs[0, 1])
            self._plot_content(ax1, "GradCAM", grad_map)
            
            # GradCAM Overlay
            ax2 = fig.add_subplot(gs[0, 2])
            self._plot_content(ax2, "GradCAM Overlay", (img_plot, grad_map))
            
            # IG Heatmap
            ax3 = fig.add_subplot(gs[1, 0])
            self._plot_content(ax3, "IG Heatmap", ig_heat)
            
            # IG Overlay
            ax4 = fig.add_subplot(gs[1, 1])
            self._plot_content(ax4, "IG Overlay", (img_plot, ig_heat))
            
            # Combined Overlay - now with distinct colors
            ax5 = fig.add_subplot(gs[1, 2])
            self._plot_content(ax5, "Combined", (img_plot, grad_map, ig_heat))
            
            output_path = self.create_output_structure(img_path)
            fig.savefig(output_path, bbox_inches='tight', dpi=self.dpi, pad_inches=0.2)
        except Exception as e:
            print(f"Visualization creation failed: {str(e)}")
        finally:
            plt.close(fig)
            torch.cuda.empty_cache()

    def __call__(self) -> None:
        """Main execution method"""
        print("\nLoading dataset for prediction:")
        self.ds.load_pred_dataset()
        if not self.ds.ds_loaded:
            raise RuntimeError("Failed to load prediction dataset")
        
        print("\nLoading model weights:")
        if hasattr(self.cnn, 'load_checkpoint'):
            self.cnn.load_checkpoint()
        else:
            print("Warning: Using random weights")
        
        print("\nRunning GradCAM++ analysis:")
        with self:
            self.analyze_dataset(self.ds.ds_pred)