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
from skimage.morphology import binary_closing, disk
# Own modules
from settings import setting
from dataset import Dataset
from model import CNN_Model

# Suppress specific warnings
warnings.filterwarnings("ignore", message="Input Tensor 0 did not already require gradients")

class GradCAMpp_Analyzer:

    #############################################################################################################
    # CONSTRUCTOR:

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
        
        # IG-specific settings from old script
        self.n_steps_ig = setting.get('captum_n_steps_ig', 25)
        self.threshold_percentile = setting.get('captum_threshold_percentile', 90)
        self.sigma = setting.get('captum_sigma', 1.0)
        self.outlier_perc = setting.get('captum_outlier_perc', 1)
        self.cmap_heatmap = setting.get('captum_cmap_heatmap', 'magma')
        self.cmap_overlay = setting.get('captum_cmap_overlay', 'viridis')
        self.show_color_bar_heatmap = setting.get('captum_show_color_bar_heatmap', True)
        self.show_color_bar_overlay = setting.get('captum_show_color_bar_overlay', True)
        
        # Initialize dataset and model
        self.ds = Dataset()
        if not self.ds.load_pred_dataset():
            raise ValueError("Failed to load prediction dataset")
        self.cnn = CNN_Model()
        
        # Load model
        with torch.no_grad():
            self.cnn.model = self.cnn.load_model(self.device).to(self.device).eval()
        
        # Freeze model
        for param in self.cnn.model.parameters():
            param.requires_grad = False
            
        # Target layer selection
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
            
    #############################################################################################################
    # METHODS:

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

    # GradCAM++
    def gradcam_pp(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        """Fixed GradCAM++ implementation with proper dimension handling"""
        attributions = None
        try:
            # Ensure 4D input (add batch dimension if needed)
            if input_tensor.dim() == 3:
                input_tensor = input_tensor.unsqueeze(0)  # Adds batch dimension
            input_tensor = input_tensor.requires_grad_(True)
            
            with self._temporary_grad_enabled(self.cnn.model):
                layer_gc = LayerGradCam(self.cnn.model, self.target_layer)
                attributions = layer_gc.attribute(
                    input_tensor, 
                    target=target_class,
                    relu_attributions=True
                )
                
                cam = attributions.squeeze().detach().cpu().float().numpy()
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

    # Improved GradCAM++
    def gradcam_pp_improved(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
    
        try:
            # Ensure proper input dimensions
            if input_tensor.dim() == 3:
                input_tensor = input_tensor.unsqueeze(0)
            input_tensor = input_tensor.requires_grad_(True)
            
            # Use LayerGradCam from Captum
            layer_gc = LayerGradCam(self.cnn.model, self.target_layer)
            attributions = layer_gc.attribute(
                input_tensor,
                target=target_class,
                relu_attributions=True
            )
            
            # Post-processing for smoother output
            cam = attributions.squeeze().detach().cpu().float().numpy()
            
            # Apply smoothing similar to original GradCAM
            if self.img_channels == 1 and cam.ndim == 3:
                cam = cam[0]
            
            # Enhanced normalization and smoothing
            cam = np.maximum(cam, 0)
            cam = cv2.resize(cam, self.img_size[::-1], interpolation=cv2.INTER_CUBIC)  # Higher quality interpolation
            
            # Apply Gaussian blur for smoothing (sigma=1.0 provides gentle smoothing)
            cam = gaussian_filter(cam, sigma=1.0)
            
            # Adaptive normalization
            max_val = np.percentile(cam, 99.5)  # Slightly more inclusive than 99.9
            return np.clip(cam / (max_val + 1e-10), 0, 1)
            
        except Exception as e:
            print(f"GradCAM++ computation failed: {str(e)}")
            return np.zeros(self.img_size[::-1])
        finally:
            torch.cuda.empty_cache()   
    """
    # Improved:
    def gradcam_pp_improved(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:

        try:
            if input_tensor.dim() == 3:
                input_tensor = input_tensor.unsqueeze(0)
            input_tensor = input_tensor.to(self.device).requires_grad_(True)
            
            with self._temporary_grad_enabled(self.cnn.model):
                layer_gc = LayerGradCam(self.cnn.model, self.target_layer)
                attributions = layer_gc.attribute(
                    input_tensor, 
                    target=target_class,
                    relu_attributions=True
                )
                
                cam = attributions.squeeze().detach().cpu().float().numpy()
                
                if self.img_channels == 1 and cam.ndim == 3:
                    cam = cam[0]
                
                cam = np.maximum(cam, 0)
                cam = cv2.resize(cam, self.img_size[::-1], interpolation=cv2.INTER_CUBIC)
                
                # Simplified smoothing - remove bilateral filter
                cam = gaussian_filter(cam, sigma=1.5)
                
                # Remove morphological operations completely
                threshold = np.percentile(cam, 70)
                cam[cam < threshold] = 0
                
                # Normalization only
                max_val = np.percentile(cam, 99.5) + 1e-10
                cam = cam / max_val
                
                return np.clip(cam, 0, 1)
                
        except Exception as e:
            print(f"GradCAM++ failed: {str(e)}")
            return np.zeros(self.img_size[::-1])
        finally:
            torch.cuda.empty_cache()
    """    

    # Normal GradCAM
    def generate_gradcam(self, input_tensor, target_class):
        """Use your original GradCAM implementation instead of GradCAM++"""
        self.cnn.model.zero_grad()
        
        # Forward pass - get features
        features = self.cnn.model.features(input_tensor.unsqueeze(0))
        features.retain_grad()  # Keep gradients for backprop
        
        # Forward through classifier
        pooled = F.adaptive_avg_pool2d(features, (1, 1))
        flattened = pooled.view(1, -1)
        output = self.cnn.model.classifier(flattened)
        
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
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-10)
        
        return cam.squeeze().detach().cpu().numpy()

    def visualize(self, input_tensor, img_path, target_class):
        """Fixed visualization pipeline with dimension handling"""
        try:
            if input_tensor is None:
                return
                
            # Ensure proper input dimensions
            input_tensor = input_tensor.to(self.device)
            if input_tensor.dim() == 3:
                input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension if missing
            elif input_tensor.dim() > 4:
                input_tensor = input_tensor.squeeze(0)  # Remove extra dimensions
            
            # Get prediction probability (for display only)
            with torch.no_grad():
                output = self.cnn.model(input_tensor)
                pred_prob = torch.softmax(output, dim=1)[0, target_class].item()
            
            # Generate heatmaps - ensure we pass the first image from batch
            gradcam_map = self.gradcam_pp_improved(input_tensor[0].clone(), target_class)
            ig_map = self.ig_analysis(input_tensor[0].clone(), target_class)
            
            # Prepare image
            img_np = self._prepare_image(input_tensor[0])
            self._create_visualization(img_np, gradcam_map, ig_map, target_class, pred_prob, img_path)
            
        except Exception as e:
            print(f"Visualization pipeline failed: {str(e)}")
        finally:
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

        try:
            torch.cuda.empty_cache()
            
            # Ensure proper input dimensions
            input_tensor = input_tensor.unsqueeze(0) if input_tensor.dim() == 3 else input_tensor
            input_tensor = input_tensor.to(self.device).requires_grad_(True)
            
            # Initialize IG with baseline as channel-wise mean
            ig = IntegratedGradients(self.cnn.model)
            attributions = ig.attribute(
                input_tensor,
                target=target_class,
                n_steps=self.n_steps_ig,
                internal_batch_size=2,
                baselines=input_tensor.mean(dim=(2, 3)),  # Channel-wise mean
                return_convergence_delta=False
            )
            
            if attributions is None:
                return np.zeros(input_tensor.shape[2:])
                
            # Process attributions like in old script
            if self.img_channels == 1:
                attr_np = attributions.squeeze().detach().cpu().numpy()  # Added detach()
            else:
                attr_np = attributions.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()  # Added detach()
                attr_np = np.max(attr_np, axis=2)  # Max projection

            # Normalization and thresholding like in old script
            attr_np = np.abs(attr_np)  # Focus on magnitude
            p99 = np.percentile(attr_np, 99)
            attr_np = np.clip(attr_np, 0, p99)
            
            # Apply Gaussian blur with configured sigma
            attr_np = gaussian_filter(attr_np, sigma=self.sigma)
            
            # Dynamic thresholding using configured percentile
            threshold = np.percentile(attr_np, self.threshold_percentile)
            attr_np[attr_np < threshold] = 0

            # Normalize for visualization
            attr_np = (attr_np - attr_np.min()) / (attr_np.max() - attr_np.min() + 1e-10)
            
            return attr_np
            
        except Exception as e:
            print(f"IG analysis failed: {str(e)}")
            return np.zeros(input_tensor.shape[2:])
        finally:
            torch.cuda.empty_cache()
    """
    # Improved:
    def ig_analysis(self, input_tensor, target_class):

        try:
            torch.cuda.empty_cache()
            
            input_tensor = input_tensor.unsqueeze(0) if input_tensor.dim() == 3 else input_tensor
            input_tensor = input_tensor.to(self.device).requires_grad_(True)
            
            # Simplified baseline
            baseline = input_tensor.mean(dim=(2,3), keepdim=True)
            
            ig = IntegratedGradients(self.cnn.model)
            attributions = ig.attribute(
                input_tensor,
                target=target_class,
                baselines=baseline,
                n_steps=self.n_steps_ig
            )
            
            if attributions is None:
                return np.zeros(input_tensor.shape[2:])
                
            # Convert to proper format
            if self.img_channels == 1:
                attr_np = attributions.squeeze().abs().detach().cpu().numpy().astype(np.float32)
            else:
                attr_np = attributions.squeeze(0).permute(1, 2, 0).abs().detach().cpu().numpy()
                attr_np = np.max(attr_np, axis=2).astype(np.float32)

            # Only use Gaussian blur
            attr_np = gaussian_filter(attr_np, sigma=self.sigma)
            
            # Simple thresholding
            threshold = np.percentile(attr_np, self.threshold_percentile)
            attr_np[attr_np < threshold] = 0
            
            # Normalize
            attr_np = (attr_np - attr_np.min()) / (attr_np.max() - attr_np.min() + 1e-10)
            return attr_np
            
        except Exception as e:
            print(f"IG analysis failed: {str(e)}")
            return np.zeros(input_tensor.shape[2:])
        finally:
            torch.cuda.empty_cache()
    """

    def _create_visualization(self, img_np_plot, gradcam_map, ig_map, target_class, pred_prob, img_path):
        """Create visualization matching original GradCAM style but with old script's IG style"""
        plt.ioff()
        try:
            # Create figure with adjusted layout
            fig = plt.figure(figsize=(15, 10), dpi=self.dpi)
            
            # Get class name
            class_name = self.classes[target_class]
            
            # Prepare visualization elements
            img_plot = img_np_plot.squeeze()
            grad_map = gradcam_map.squeeze()
            ig_heat = ig_map.squeeze()
            
            # Use gridspec for layout control
            gs = gridspec.GridSpec(2, 3, figure=fig, 
                                height_ratios=[1, 1],
                                hspace=0.15,
                                wspace=0.1)
            
            # Original image
            ax0 = fig.add_subplot(gs[0, 0])
            ax0.imshow(img_plot if self.img_channels == 1 else np.mean(img_plot, axis=2),
                    cmap='gray' if self.img_channels == 1 else None)
            ax0.set_title(f"Original: {class_name} | Prob: {pred_prob:.2f}")
            ax0.axis('off')
            
            # GradCAM (using jet colormap like original)
            ax1 = fig.add_subplot(gs[0, 1])
            im1 = ax1.imshow(grad_map, cmap='jet', vmin=0, vmax=1)
            ax1.set_title(f"GradCAM (target: {class_name})")
            ax1.axis('off')
            plt.colorbar(im1, ax=ax1, orientation='horizontal', fraction=0.046, pad=0.04)
            
            # GradCAM Overlay (matching original alpha)
            ax2 = fig.add_subplot(gs[0, 2])
            ax2.imshow(img_plot if self.img_channels == 1 else np.mean(img_plot, axis=2),
                    cmap='gray' if self.img_channels == 1 else None)
            ax2.imshow(grad_map, cmap='jet', alpha=self.alpha_overlay)
            ax2.set_title(f"GradCAM Overlay")
            ax2.axis('off')
            
            # IG Heatmap (using style from old script)
            ax3 = fig.add_subplot(gs[1, 0])
            im3 = ax3.imshow(ig_heat, cmap=self.cmap_heatmap)
            ax3.set_title(f"IG Heatmap (target: {class_name})")
            ax3.axis('off')
            if self.show_color_bar_heatmap:
                plt.colorbar(im3, ax=ax3, orientation='horizontal', fraction=0.046, pad=0.04)
            
            # IG Overlay (using style from old script)
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.imshow(img_plot if self.img_channels == 1 else np.mean(img_plot, axis=2),
                    cmap='gray' if self.img_channels == 1 else None)
            ax4.imshow(ig_heat, cmap=self.cmap_overlay, alpha=self.alpha_overlay)
            ax4.set_title(f"IG Overlay")
            ax4.axis('off')
            if self.show_color_bar_overlay:
                plt.colorbar(ax4.images[-1], ax=ax4, orientation='horizontal', fraction=0.046, pad=0.04)
            
            # Combined Overlay
            ax5 = fig.add_subplot(gs[1, 2])
            ax5.imshow(img_plot if self.img_channels == 1 else np.mean(img_plot, axis=2),
                    cmap='gray' if self.img_channels == 1 else None)
            ax5.imshow(grad_map, cmap='jet', alpha=self.alpha_overlay/2)
            ax5.imshow(ig_heat, cmap=self.cmap_overlay, alpha=self.alpha_overlay/2)
            ax5.set_title(f"Combined")
            ax5.axis('off')
            
            output_path = self.create_output_structure(img_path)
            fig.savefig(output_path, bbox_inches='tight', dpi=self.dpi, pad_inches=0.2)
        except Exception as e:
            print(f"Visualization creation failed: {str(e)}")
        finally:
            plt.close(fig)
            torch.cuda.empty_cache()

    def analyze_dataset(self, dataloader):
        """Process complete dataset using folder-based target classes"""
        if not hasattr(dataloader, 'dataset'):
            raise ValueError("Invalid dataloader")
        
        self.cnn.model.eval()
        try:
            for batch_idx, (images, _) in enumerate(tqdm(dataloader)):
                if images is None:
                    continue
                    
                batch_paths = [Path(x[0]) for x in dataloader.dataset.samples[
                    batch_idx*dataloader.batch_size:(batch_idx+1)*dataloader.batch_size
                ]]
                
                for img_idx in range(images.size(0)):
                    try:
                        img = images[img_idx].unsqueeze(0).to(self.device)
                        img_path = batch_paths[img_idx]
                        
                        # Get target class from folder name
                        folder_name = img_path.parent.name
                        try:
                            target_class = self.classes.index(folder_name)
                        except ValueError:
                            print(f"\nSkipping {img_path.name}: Folder '{folder_name}' not in classes {self.classes}")
                            continue
                        
                        self.visualize(img, img_path, target_class)
                    except Exception as e:
                        print(f"Error processing image: {str(e)}")
                    finally:
                        torch.cuda.empty_cache()
        except Exception as e:
            print(f"Dataset analysis failed: {str(e)}")
        finally:
            torch.cuda.empty_cache()

    #############################################################################################################
    # CALL:

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