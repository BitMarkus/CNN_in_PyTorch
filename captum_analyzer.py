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
# Own modules
from settings import setting
from dataset import Dataset
from custom_model import Custom_CNN_Model
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

    #############################################################################################################
    # METHODS

    # Compute and visualize Integrated Gradients attributions with optional overlay display.
    # Args:
    #    dataset: Input dataset
    #    device: Torch device (cpu/cuda)
    #    show_overlay: Whether to show the blended heatmap overlay (default: True)
    def predict_captum(self, dataset, device, show_overlay=True):

        self.cnn.model.eval()
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(dataset)):
                batch_paths = [Path(dataset.dataset.samples[i][0])  # Convert to Path objects
                            for i in range(batch_idx * dataset.batch_size,
                                        min((batch_idx + 1) * dataset.batch_size, len(dataset.dataset)))]
                
                for img_idx in range(len(images)):
                    try:
                        torch.cuda.empty_cache()
                        
                        img = images[img_idx].unsqueeze(0).to(device)
                        label = labels[img_idx].to(device)
                        img_path = batch_paths[img_idx]
                        
                        # Create output folder matching original subfolder structure
                        relative_path = img_path.relative_to(self.pth_prediction)
                        output_folder = self.pth_prediction / f"{relative_path.parent.name}_captum"
                        output_folder.mkdir(exist_ok=True, parents=True)
                        
                        # Compute attributions
                        ig = IntegratedGradients(self.cnn.model)
                        attributions = ig.attribute(
                            img,
                            target=label.item(),
                            n_steps=10,
                            internal_batch_size=1
                        )
                        
                        # Convert to numpy based on number of channels
                        if self.img_channels == 1:
                            # Grayscale image
                            img_np = img.squeeze(0).squeeze(0).cpu().numpy()  # Remove channel dim
                            attr_np = attributions.squeeze(0).squeeze(0).cpu().detach().numpy()
                        else:
                            # RGB image
                            img_np = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
                            attr_np = attributions.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
                        
                        # Create figure with subplots
                        plt.ioff()  # Turn off interactive mode
                        
                        # Determine number of columns based on overlay option
                        num_cols = 3 if show_overlay else 2
                        # Adjust figure width to account for colorbar
                        fig = plt.figure(figsize=(6 * num_cols + 0.5, 6))  # Added 0.5 for colorbar
                        gs = gridspec.GridSpec(1, num_cols, width_ratios=[1.1 if i == 0 else 1 for i in range(num_cols)])
                        
                        # Original image in inferno colormap
                        ax1 = plt.subplot(gs[0])
                        if self.img_channels == 1:
                            # Grayscale - already single channel
                            img_vis = img_np
                        else:
                            # RGB - convert to grayscale
                            img_vis = np.mean(img_np, axis=2)
                        
                        im1 = ax1.imshow(img_vis, cmap='inferno')
                        ax1.set_title(f"Original Image: Class {self.classes[label.item()]}")
                        ax1.axis('off')
                        # Add colorbar with adjusted positioning
                        cbar = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
                        cbar.ax.tick_params(labelsize=8)  # Make colorbar ticks smaller if needed
                        
                        # Heatmap only
                        ax2 = plt.subplot(gs[1])
                        if self.img_channels == 1:
                            viz.visualize_image_attr(
                                np.expand_dims(attr_np, axis=-1),
                                method="heat_map",
                                sign="absolute_value",
                                show_colorbar=False,
                                cmap="inferno",
                                plt_fig_axis=(fig, ax2),
                                title="Attribution Heatmap"
                            )
                        else:
                            viz.visualize_image_attr(
                                attr_np,
                                method="heat_map",
                                sign="absolute_value",
                                show_colorbar=False,
                                cmap="inferno",
                                plt_fig_axis=(fig, ax2),
                                title="Attribution Heatmap"
                            )
                        
                        # Overlay (only if requested)
                        if show_overlay:
                            ax3 = plt.subplot(gs[2])
                            if self.img_channels == 1:
                                # For grayscale images
                                viz.visualize_image_attr(
                                    np.expand_dims(attr_np, axis=-1),  # Add channel dim
                                    original_image=np.expand_dims(img_np, axis=-1),
                                    method="blended_heat_map",
                                    sign="positive",
                                    show_colorbar=False,
                                    cmap="inferno",
                                    alpha_overlay=0.7,
                                    plt_fig_axis=(fig, ax3),
                                    title="Blended Heatmap Overlay"
                                )
                            else:
                                # For RGB images
                                viz.visualize_image_attr(
                                    attr_np,
                                    original_image=img_np,
                                    method="blended_heat_map",
                                    sign="positive",
                                    show_colorbar=False,
                                    cmap="inferno",
                                    alpha_overlay=0.7,
                                    plt_fig_axis=(fig, ax3),
                                    title="Blended Heatmap Overlay"
                                )
                        
                        plt.tight_layout()
                        
                        # Save to correct location
                        output_path = output_folder / f"{img_path.stem}_captum.png"
                        fig.savefig(output_path, bbox_inches='tight', dpi=100)
                        plt.close(fig)
                        
                        # Clean up
                        del img, attributions, attr_np, fig
                        torch.cuda.empty_cache()
                        
                    except Exception as e:
                        print(f"Error processing image {batch_paths[img_idx]}: {str(e)}")
                        continue

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
        self.cnn.load_checkpoint()

        # Iterate over images and predict
        self.predict_captum(self.ds.ds_pred, self.device, show_overlay=True)

    
