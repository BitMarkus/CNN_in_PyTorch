import torch
from tqdm import tqdm
from pathlib import Path
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
# Set non-interactive backend at the start
import matplotlib
matplotlib.use('Agg')  # Set before other matplotlib imports
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from scipy.ndimage import gaussian_filter
import gc  # For garbage collection
# Own modules
from settings import setting
from dataset import Dataset
from model import CNN_Model

class CaptumAnalyzer:

    #############################################################################################################
    # CONSTRUCTOR:

    def __init__(self, device):
        self.device = device
        # Settings parameters
        self.pth_prediction = setting['pth_prediction']
        self.pth_checkpoint = setting['pth_checkpoint']
        self.classes = setting['classes']
        self.img_channels = setting['img_channels']
        self.classes = setting['classes']
        # Captum specific
        self.show_overlay = setting['captum_show_overlay']
        self.sign = setting['captum_sign']
        self.cmap_orig = setting['captum_cmap_orig']
        self.cmap_heatmap = setting['captum_cmap_heatmap']
        self.cmap_overlay = setting['captum_cmap_overlay']
        self.show_color_bar_orig = setting['captum_show_color_bar_orig']
        self.show_color_bar_heatmap = setting['captum_show_color_bar_heatmap']
        self.show_color_bar_overlay = setting['captum_show_color_bar_overlay']
        self.n_steps_ig = setting['captum_n_steps_ig']
        self.output_size = setting['captum_output_size']
        self.alpha_overlay = setting['captum_alpha_overlay']
        self.threshold_percentile = setting['captum_threshold_percentile']
        self.dpi = setting['captum_dpi']
        self.sigma = setting['captum_sigma']
        self.outlier_perc = setting['captum_outlier_perc']

        # Aggregate across channels (recommended for heatmaps): Use aggregate_channels=True (default)
        # For channel-specific analysis (process per-channel independently): Use aggregate_channels=False
        self.aggregate_channels = True
        
        # Initialize dataset to get transforms
        self.ds = Dataset()
        self.transform = self.ds.get_transformer_test()
        if not self.transform:
            raise ValueError("Failed to get image transformer")
        
        # Create model wrapper
        self.cnn_wrapper = CNN_Model()  
        # Load model wrapper with model information
        print(f"Creating new {self.cnn_wrapper.cnn_type} network...")
        # Get actual model (nn.Module)
        self.cnn = self.cnn_wrapper.load_model(device).to(device)
        print("New network was successfully created.")   

    #############################################################################################################
    # METHODS

    # Compute and visualize Integrated Gradients attributions with optional overlay display.
    # Args:
    #    dataset: Input dataset
    #    device: Torch device (cpu/cuda)
    #    show_overlay: Whether to show the blended heatmap overlay (default: True)
    def predict_captum(self, dataset, device, show_overlay=None):
        self.cnn.eval()
        if show_overlay is None:
            show_overlay = self.show_overlay  # Use setting if not overridden

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(dataset)):
                batch_paths = [Path(dataset.dataset.samples[i][0]) 
                            for i in range(batch_idx * dataset.batch_size,
                                        min((batch_idx + 1) * dataset.batch_size, len(dataset.dataset)))]
                
                for img_idx in range(len(images)):
                    try:
                        # --- Preparation ---
                        img = images[img_idx].unsqueeze(0).to(device)
                        label = labels[img_idx].to(device)
                        img_path = batch_paths[img_idx]
                        
                        # Create output folder
                        relative_path = img_path.relative_to(self.pth_prediction)
                        output_folder = self.pth_prediction / f"{relative_path.parent.name}_captum"
                        output_folder.mkdir(exist_ok=True, parents=True)
                        
                        # Convert image to numpy
                        img_np = img.squeeze().cpu().numpy() if self.img_channels == 1 else img.squeeze(0).permute(1, 2, 0).cpu().numpy()

                        # --- Attribution Calculation ---
                        ig = IntegratedGradients(self.cnn)
                        attributions = ig.attribute(
                            img,
                            target=label.item(),
                            n_steps=self.n_steps_ig,  # From settings
                            internal_batch_size=2,
                            baselines=img.mean(dim=(2,3), keepdim=True),  # Channel-wise mean
                            return_convergence_delta=False
                        )
                        
                        # --- Attribution Processing ---
                        if self.img_channels == 1:
                            attr_np = attributions.squeeze().cpu().numpy()
                        else:
                            attr_np = attributions.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
                            attr_np = np.max(attr_np, axis=2)  # Max projection

                        # Normalization and thresholding
                        attr_np = np.abs(attr_np)  # Focus on magnitude
                        p99 = np.percentile(attr_np, 99)
                        attr_np = np.clip(attr_np, 0, p99)
                        
                        # Apply Gaussian blur with configured sigma
                        attr_np = gaussian_filter(attr_np, sigma=self.sigma)
                        
                        # Dynamic thresholding using configured percentile
                        threshold = np.percentile(attr_np, self.threshold_percentile)
                        attr_np[attr_np < threshold] = 0

                        # Prepare for visualization
                        if attr_np.ndim == 2:
                            attr_np = np.expand_dims(attr_np, axis=-1)
                        
                        # --- Visualization ---
                        plt.ioff()
                        num_cols = 3 if show_overlay else 2
                        fig = plt.figure(figsize=(self.output_size * num_cols + 0.5, self.output_size))
                        gs = gridspec.GridSpec(1, num_cols, width_ratios=[1.1 if i == 0 else 1 for i in range(num_cols)])
                        
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
                        importance = (attr_np - attr_np.min()) / (attr_np.max() - attr_np.min() + 1e-10)
                        viz.visualize_image_attr(
                            importance,
                            method="heat_map",
                            sign=self.sign,
                            show_colorbar=self.show_color_bar_heatmap,
                            cmap=self.cmap_heatmap,
                            plt_fig_axis=(fig, ax2),
                            outlier_perc=self.outlier_perc,
                            title="Feature Importance"
                        )
                        
                        # Subplot 3: Overlay
                        if show_overlay:
                            ax3 = plt.subplot(gs[2])
                            viz.visualize_image_attr(
                                np.expand_dims(attr_np, -1) if attr_np.ndim == 2 else attr_np,
                                original_image=np.expand_dims(img_np, axis=-1) if self.img_channels == 1 else img_np,
                                method="blended_heat_map",
                                sign=self.sign,
                                show_colorbar=self.show_color_bar_overlay,
                                cmap=self.cmap_overlay,
                                alpha_overlay=self.alpha_overlay,
                                plt_fig_axis=(fig, ax3),
                                outlier_perc=self.outlier_perc,
                                title="Important Cell Regions"
                            )
                        
                        # Save with configured DPI
                        plt.tight_layout()
                        output_path = output_folder / f"{img_path.stem}_captum.png"
                        fig.savefig(output_path, bbox_inches='tight', dpi=self.dpi)
                        plt.close(fig)
                        
                    except Exception as e:
                        print(f"Error processing {img_path.name}: {str(e)}")
                    finally:
                        torch.cuda.empty_cache()
                        gc.collect()

    #############################################################################################################
    # CALL

    def __call__(self):
        # Load dataset for prediction
        print("\nLoading dataset for prediction:") 
        self.ds.load_pred_dataset()
        if(self.ds.ds_loaded):
            print("Dataset successfully loaded!")

        # Load weights
        print("\nLoading weights:")
        self.cnn_wrapper.load_checkpoint()

        # Iterate over images and predict
        self.predict_captum(self.ds.ds_pred, self.device, self.show_overlay)

"""
    def predict_captum(self, dataset, device, show_overlay=True):
        self.cnn.eval()
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(dataset)):
                batch_paths = [Path(dataset.dataset.samples[i][0]) 
                            for i in range(batch_idx * dataset.batch_size,
                                        min((batch_idx + 1) * dataset.batch_size, len(dataset.dataset)))]
                
                for img_idx in range(len(images)):
                    try:
                        # 1. Prepare image and paths
                        img = images[img_idx].unsqueeze(0).to(device)
                        label = labels[img_idx].to(device)
                        img_path = batch_paths[img_idx]
                        
                        # Create output folder
                        relative_path = img_path.relative_to(self.pth_prediction)
                        output_folder = self.pth_prediction / f"{relative_path.parent.name}_captum"
                        output_folder.mkdir(exist_ok=True, parents=True)
                        
                        # 2. Convert image to numpy for visualization
                        img_np = img.squeeze().cpu().numpy() if self.img_channels == 1 else img.squeeze(0).permute(1, 2, 0).cpu().numpy()
                        
                        # 3. Compute attributions with fibroblast-optimized parameters
                        ig = IntegratedGradients(self.cnn)
                        attributions = ig.attribute(
                            img,
                            target=label.item(),
                            n_steps=self.n_steps_ig,
                            internal_batch_size=2,
                            baselines=img.mean(dim=(2,3), keepdim=True),  # Channel-wise mean baseline
                            return_convergence_delta=False
                        )
                        
                        # 4. Process attributions for cell images
                        if self.img_channels == 1:
                            attr_np = attributions.squeeze().cpu().numpy()
                        else:
                            attr_np = attributions.squeeze(0).permute(1, 2, 0).cpu().numpy()
                            attr_np = np.max(attr_np, axis=2)  # Max projection
                        
                        # 5. Cell-specific normalization
                        attr_np = np.abs(attr_np)  # Focus on magnitude
                        p99 = np.percentile(attr_np, 99)
                        attr_np = np.clip(attr_np, 0, p99)  # Clip extreme values
                        
                        # 6. Specialized postprocessing
                        sigma = 0.7 if self.img_channels == 1 else 1.2
                        attr_np = gaussian_filter(attr_np, sigma=sigma)
                        dyn_threshold = np.mean(attr_np) + 2*np.std(attr_np)
                        attr_np[attr_np < dyn_threshold] = 0
                        
                        # 7. Prepare for visualization
                        if attr_np.ndim == 2:
                            attr_np = np.expand_dims(attr_np, axis=-1)
                        
                        # 8. Visualization
                        plt.ioff()
                        num_cols = 3 if show_overlay else 2
                        fig = plt.figure(figsize=(self.output_size * num_cols + 0.5, self.output_size))
                        gs = gridspec.GridSpec(1, num_cols, width_ratios=[1.1 if i == 0 else 1 for i in range(num_cols)])

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
                        # First convert attributions to importance scores (0-1)
                        importance = (attr_np - attr_np.min()) / (attr_np.max() - attr_np.min() + 1e-10)
                        # Apply threshold to focus only on most important regions
                        importance[importance < 0.3] = 0  # Only show top 70% important pixels
                        viz.visualize_image_attr(
                            importance,
                            method="heat_map",
                            sign=self.sign,  # Only show positive importance
                            show_colorbar=self.show_color_bar_heatmap,
                            cmap=self.cmap_heatmap,  # More scientific colormap
                            plt_fig_axis=(fig, ax2),
                            title="Feature Importance",
                            use_pyplot=False
                        )

                        # Subplot 3: Overlay with cell-focused visualization
                        if show_overlay:
                            ax3 = plt.subplot(gs[2])
                            # Create mask of important regions
                            mask = importance > 0
                            # Visualize only important cell regions
                            viz.visualize_image_attr(
                                np.expand_dims(mask.astype(np.float32), -1) if mask.ndim == 2 else mask,
                                original_image=np.expand_dims(img_np, axis=-1) if self.img_channels == 1 else img_np,
                                method="blended_heat_map",
                                sign=self.sign,
                                show_colorbar=self.show_color_bar_overlay,
                                cmap=self.cmap_overlay,
                                alpha_overlay=self.alpha_overlay,  
                                plt_fig_axis=(fig, ax3),
                                title="Important Cell Regions",
                                use_pyplot=False
                            )

                        plt.tight_layout()
                        output_path = output_folder / f"{img_path.stem}_captum.png"
                        fig.savefig(output_path, bbox_inches='tight', dpi=150)
                        plt.close(fig)
                        
                    except Exception as e:
                        print(f"Error processing {img_path.name}: {str(e)}")
                    finally:
                        torch.cuda.empty_cache()
                        gc.collect()
"""