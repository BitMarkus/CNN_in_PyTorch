# ===== Standard Library Imports =====
from pathlib import Path
# ===== Third-Party Imports =====
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from prettytable import PrettyTable
# ===== Own Modules =====
from settings import setting
from dataset import Dataset
from model import CNN_Model

class GradCAMAnalyzer:

    #############################################################################################################
    # CONSTRUCTOR

    # Initialize the GradCAM analyzer for visualizing class activation maps.
    # Supports DenseNet-121 architecture with optional second iteration blurring.
    # Args:
    #   device (torch.device): Device to run analysis on
    def __init__(self, device: torch.device) -> None:

        self.device = device

        # Settings from configuration
        self.pth_input = setting['pth_input']
        self.pth_output = setting['pth_output']
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
        self.threshold_percent = setting['gradcam_threshold_percent']
        self.sigma = setting['gradcam_blurr_sigma']

        # Export mode
        self.export_only_overlay = setting.get('gradcam_export_only_overlay', False)

        # Initialize dataset and model
        self.ds = Dataset()
        if not self.ds.load_pred_dataset():
            raise ValueError("Failed to load prediction dataset")

        self.cnn_wrapper = CNN_Model()
        print(f"Creating new {self.cnn_wrapper.cnn_type} network...")
        self.cnn = self.cnn_wrapper.load_model(device).to(device)
        print("New network was successfully created.")

        # Checkpoints
        self.checkpoint_loaded = False
        self.loaded_checkpoint_name = None

        # Class selection for GradCAM
        self.selected_class = None
        self.selected_class_name = None

        # For faster convolutions
        torch.backends.cudnn.benchmark = True

    #############################################################################################################
    # METHODS

    # Prompt the user to select a class for GradCAM analysis.
    # Returns:
    #   bool: True if selection succeeded, False otherwise
    def select_gradcam_class(self) -> bool:
        table = PrettyTable()
        table.field_names = ["ID", "Class Name"]
        table.align["ID"] = "c"
        table.align["Class Name"] = "l"

        for idx, class_name in enumerate(self.classes, 1):
            table.add_row([idx, class_name])

        print()
        print(table)

        while True:
            try:
                selection = input(f"Select class for GradCAM analysis (1-{len(self.classes)}): ").strip()
                class_idx = int(selection)

                if 1 <= class_idx <= len(self.classes):
                    self.selected_class = class_idx - 1
                    self.selected_class_name = self.classes[self.selected_class]
                    print(f"✓ Selected class: {self.selected_class_name}")
                    return True
                else:
                    print(f"Invalid selection. Please enter a number between 1 and {len(self.classes)}.")

            except ValueError:
                print("Invalid input. Please enter a valid number.")
            except KeyboardInterrupt:
                print("\nClass selection cancelled.")
                return False

    # Load model weights from a selected checkpoint.
    # Returns:
    #   bool: True if loading succeeded, False otherwise
    def load_checkpoint(self) -> bool:
        silent_checkpoints = self.cnn_wrapper.print_checkpoints_table(self.pth_checkpoint, print_table=False)

        if not silent_checkpoints:
            print("The checkpoint folder is empty!")
            return False

        if len(silent_checkpoints) == 1:
            checkpoint_file = silent_checkpoints[0][1]
            print(f"\nFound single checkpoint: {checkpoint_file}")
            print("Loading automatically...")
        else:
            self.cnn_wrapper.print_checkpoints_table(self.pth_checkpoint)
            checkpoint_file = self.cnn_wrapper.select_checkpoint(silent_checkpoints, "Select a checkpoint: ")
            if not checkpoint_file:
                return False

        try:
            full_path = self.pth_checkpoint / checkpoint_file

            checkpoint = torch.load(full_path, map_location=self.device)

            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.cnn.load_state_dict(checkpoint['model_state_dict'])
                epoch_info = checkpoint.get('epoch', 'unknown')
                if epoch_info != 'unknown':
                    epoch_info = epoch_info + 1
                accuracy = checkpoint.get('accuracy', 'unknown')
                balanced_acc = checkpoint.get('balanced_accuracy', 'unknown')
                composite_score = checkpoint.get('composite_score', 'unknown')

                print(f"  ✓ Loaded checkpoint from epoch {epoch_info}")
                if accuracy != 'unknown':
                    print(f"    Accuracy: {accuracy:.2%}")
                if balanced_acc != 'unknown':
                    print(f"    Balanced Acc: {balanced_acc:.2%}")
                if composite_score != 'unknown':
                    print(f"    Composite Score: {composite_score:.4f}")

            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                self.cnn.load_state_dict(checkpoint['state_dict'])
                print(f"  ✓ Loaded checkpoint (alternative format)")

            else:
                self.cnn.load_state_dict(checkpoint)
                print(f"  ✓ Loaded checkpoint (direct state dict)")

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
            print("WARNING: Using untrained weights!")
            self.loaded_checkpoint_name = "untrained"
            return False

    # Generate GradCAM heatmap for a specific target class.
    # Args:
    #   input_tensor (torch.Tensor): Input image tensor
    #   target_class (int): Target class index
    # Returns:
    #   tuple: (heatmap_numpy, target_class)
    def generate_heatmap(self, input_tensor: torch.Tensor, target_class: int) -> tuple:
        self.cnn.zero_grad()

        features = self.cnn.features(input_tensor.unsqueeze(0))
        features.retain_grad()

        pooled = F.adaptive_avg_pool2d(features, (1, 1))
        flattened = pooled.view(1, -1)
        output = self.cnn.classifier(flattened)

        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        grads = features.grad
        pooled_grads = torch.mean(grads, dim=[2, 3], keepdim=True)
        cam = (pooled_grads * features).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, input_tensor.shape[-2:], mode='bilinear', align_corners=False)

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-10)

        return cam.squeeze().detach().cpu().numpy(), target_class

    # Apply blur to the most prominent regions identified by the heatmap.
    # Args:
    #   image (torch.Tensor): Original image tensor
    #   heatmap (np.ndarray): Grad-CAM heatmap
    # Returns:
    #   torch.Tensor: Image with prominent regions blurred
    def apply_blur_mask(self, image: torch.Tensor, heatmap: np.ndarray) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            if self.img_channels == 1:
                image_np = image.squeeze().cpu().numpy()
            else:
                image_np = image.permute(1, 2, 0).cpu().numpy()
        else:
            image_np = image.copy()

        heatmap_normalized = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-10)
        threshold = 1 - self.threshold_percent
        mask = (heatmap_normalized > threshold).astype(np.float32)

        mask = gaussian_filter(mask, sigma=1)
        mask = np.clip(mask, 0, 1)

        if self.img_channels == 1:
            blurred = gaussian_filter(image_np, sigma=self.sigma)
            image_np = image_np * (1 - mask) + blurred * mask
        else:
            for c in range(self.img_channels):
                blurred = gaussian_filter(image_np[:, :, c], sigma=self.sigma)
                image_np[:, :, c] = image_np[:, :, c] * (1 - mask) + blurred * mask

        if isinstance(image, torch.Tensor):
            if self.img_channels == 1:
                return torch.from_numpy(image_np).unsqueeze(0).to(self.device)
            else:
                return torch.from_numpy(image_np).permute(2, 0, 1).to(self.device)
        return image_np

    # Visualize GradCAM results with export options.
    # Args:
    #   image (torch.Tensor): Input image
    #   heatmap (np.ndarray): Grad-CAM heatmap
    #   img_path (Path): Path to the image
    #   class_idx (int): Class index
    #   target_class (int): Target class for GradCAM
    #   iteration (int): Iteration number (1 or 2)
    def visualize_gradcam(self, image: torch.Tensor, heatmap: np.ndarray, img_path: Path, class_idx: int, target_class: int, iteration: int = 1) -> None:
        if self.img_channels == 1:
            img_np = image.squeeze().cpu().numpy()
        else:
            img_np = image.permute(1, 2, 0).cpu().numpy()

        if self.selected_class_name:
            output_folder = self.pth_output / f"{img_path.parent.name}_gradcam_{self.selected_class_name}"
        else:
            output_folder = self.pth_output / f"{img_path.parent.name}_gradcam"
        output_folder.mkdir(exist_ok=True, parents=True)

        if self.export_only_overlay:
            self._export_overlay_only(img_np, heatmap, img_path, iteration, output_folder)
        else:
            self._export_all_images(img_np, heatmap, img_path, target_class, iteration, output_folder)

    # Export all three images (original, gradcam, overlay).
    def _export_all_images(self, img_np: np.ndarray, heatmap: np.ndarray, img_path: Path, target_class: int, iteration: int, output_folder: Path) -> None:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        ax1.imshow(img_np if self.img_channels == 1 else np.mean(img_np, axis=2),
                   cmap=self.cmap_orig)
        ax1.set_title(f"Original: {img_path.parent.name}")
        ax1.axis('off')

        class_name = self.selected_class_name if self.selected_class_name else self.classes[target_class]
        ax2.imshow(heatmap, cmap='jet')
        ax2.set_title(f"Grad-CAM for {class_name}")
        ax2.axis('off')

        ax3.imshow(img_np if self.img_channels == 1 else np.mean(img_np, axis=2),
                   cmap='gray' if self.img_channels == 1 else None)
        ax3.imshow(heatmap, cmap='jet', alpha=self.alpha_overlay)
        ax3.set_title("Overlay")
        ax3.axis('off')

        suffix = "_iter2" if iteration > 1 else ""
        gradcam_suffix = f"_gradcam-{self.selected_class_name}" if self.selected_class_name else "_gradcam"
        output_path = output_folder / f"{img_path.stem}{gradcam_suffix}{suffix}.png"
        plt.tight_layout()
        fig.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

    # Export only the overlay image without axes or whitespace.
    def _export_overlay_only(self, img_np: np.ndarray, heatmap: np.ndarray, img_path: Path, iteration: int, output_folder: Path) -> None:
        fig = plt.figure(figsize=(5.12, 5.12), dpi=100)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        ax.imshow(img_np if self.img_channels == 1 else np.mean(img_np, axis=2),
                  cmap='gray' if self.img_channels == 1 else None)
        ax.imshow(heatmap, cmap='jet', alpha=self.alpha_overlay)

        suffix = "_iter2" if iteration > 1 else ""
        gradcam_suffix = f"_gradcam-{self.selected_class_name}" if self.selected_class_name else "_gradcam"
        output_path = output_folder / f"{img_path.stem}{gradcam_suffix}{suffix}.png"
        fig.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close(fig)

    # Run GradCAM prediction on the dataset.
    # Args:
    #   dataset: The dataset to process
    #   second_iteration (bool): Whether to run second iteration with blurring
    def predict_gradcam(self, dataset, second_iteration: bool = False) -> None:
        self.cnn.eval()

        target_class = self.selected_class
        print(f"\nApplying GradCAM for class: {self.selected_class_name}")
        print("Processing images...")

        for batch_idx, (images, _) in enumerate(tqdm(dataset, desc="Grad-CAM Analysis")):
            batch_paths = [Path(dataset.dataset.samples[i][0])
                          for i in range(batch_idx * dataset.batch_size,
                                        min((batch_idx + 1) * dataset.batch_size, len(dataset.dataset)))]

            for img_idx in range(len(images)):
                try:
                    img = images[img_idx].to(self.device)
                    img_path = batch_paths[img_idx]

                    heatmap, _ = self.generate_heatmap(img, target_class=target_class)

                    self.visualize_gradcam(img.detach(), heatmap, img_path,
                                          target_class, target_class, iteration=1)

                    if second_iteration:
                        blurred_img = self.apply_blur_mask(img, heatmap)
                        heatmap2, _ = self.generate_heatmap(blurred_img, target_class=target_class)
                        self.visualize_gradcam(blurred_img.detach(), heatmap2, img_path,
                                              target_class, target_class, iteration=2)

                    torch.cuda.empty_cache()

                except Exception as e:
                    print(f"\nError processing {img_path.name}: {str(e)}")
                    continue

    # Verify folder structure in the input directory.
    def verify_folder_structure(self) -> None:
        print("\nFolder Structure Verification:")
        for class_idx, class_name in enumerate(self.classes):
            class_path = self.pth_input / class_name
            if not class_path.exists():
                print(f"WARNING: Missing folder for class {class_name}")
                continue

            sample_files = list(class_path.glob("*.png"))[:3]
            print(f"\nClass {class_name} ({class_idx}): {class_path}")
            for f in sample_files:
                print(f"  {f.name}")

    # Verify model predictions against dataset labels.
    # Args:
    #   num_samples (int): Number of samples to check
    # Returns:
    #   bool: True if all predictions match labels
    def verify_model_predictions(self, num_samples: int = 10) -> bool:
        print("\nModel Prediction Verification:")
        correct = 0
        for i in range(min(num_samples, len(self.ds.ds_pred.dataset))):
            img, label = self.ds.ds_pred.dataset[i]
            img_path = Path(self.ds.ds_pred.dataset.samples[i][0])

            with torch.no_grad():
                output = self.cnn(img.unsqueeze(0).to(self.device))
                pred = torch.argmax(output).item()
                prob = torch.softmax(output, dim=1)[0][pred].item()

                status = "✅" if pred == label else "❌"
                if status == "✅":
                    correct += 1

                print(f"{status} {img_path.name}")
                print(f"  Folder: {img_path.parent.name}")
                print(f"  Pred: {self.classes[pred]} ({prob:.2%})")
                print(f"  Label: {self.classes[label]}\n")

        print(f"Accuracy: {correct}/{num_samples}")
        return correct == num_samples

    # Debug dataset labels (for troubleshooting).
    def debug_dataset_labels(self) -> None:
        print("\nDataset Label Verification:")
        for i, (img_path, label) in enumerate(self.ds.ds_pred.dataset.samples[:10]):
            folder_name = Path(img_path).parent.name
            print(f"Image: {Path(img_path).name}")
            print(f"Folder: {folder_name}")
            print(f"Assigned label: {label} ({self.classes[label]})\n")

    #############################################################################################################
    # CALL

    # Run the complete GradCAM analysis pipeline.
    def __call__(self) -> None:
        print("\nInitializing GradCAM Analysis")

        self.ds.load_pred_dataset()
        if not self.ds.ds_loaded:
            print("Failed to load dataset")
            return

        if not self.load_checkpoint():
            print("WARNING: Using untrained weights!")
            self.loaded_checkpoint_name = "untrained"

        if not self.select_gradcam_class():
            print("GradCAM analysis cancelled.")
            return

        print("\nStarting GradCAM processing...")
        self.predict_gradcam(self.ds.ds_pred, self.second_iteration)