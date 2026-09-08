# TENSORBOARD USAGE INSTRUCTIONS:
# Navigate to project folder in explorer, type cmd in the address bar to open terminal
# Activate virtual environment: conda activate {env_name}
# Open logfile from path: python -m tensorboard.main --logdir "{path to logdir}\logs"
# Start tensorboard: python -m tensorboard.main --logdir ./logs

# ===== Standard Library Imports =====
from pathlib import Path
from datetime import datetime
import json
import shutil
# ===== Third-Party Imports =====
import torch
from torch import nn
from torch.optim import Adam, SGD, AdamW
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (
    f1_score, confusion_matrix, roc_curve, auc,
    roc_auc_score, precision_recall_curve, average_precision_score,
    balanced_accuracy_score
)
# ===== Own Modules =====
from settings import setting
import functions as fn

class Train():

    #############################################################################################################
    # CONSTRUCTOR

    # Initialize the training handler with model, dataset, and configuration.
    # Sets up optimizer, loss function, learning rate scheduler, and TensorBoard logging.
    # Args:
    #   cnn_wrapper (CNN_Model): Wrapper containing the model
    #   dataset (Dataset): Dataset handler with train/val loaders
    #   device (torch.device): Device to run training on
    #   dataset_idx (int, optional): Index for cross-validation dataset naming
    def __init__(self, cnn_wrapper, dataset, device: torch.device, dataset_idx: int = None) -> None:
        # Input validation
        assert len(dataset.ds_train) > 0, "Training dataset is empty"
        assert len(dataset.ds_val) > 0, "Validation dataset is empty"

        self.device = device
        self.cnn_wrapper = cnn_wrapper
        self.cnn = cnn_wrapper.model
        self.dataset_idx = dataset_idx

        # Generate timestamp for this training run
        if self.dataset_idx is not None:
            # Cross-validation: use dataset index as folder name
            self.timestamp = f"ds{self.dataset_idx:02d}"
            # For cross-validation, keep the old behavior (checkpoints stay in dataset folder)
            self.train_output_dir = None
            self.checkpoint_dir = None
            self.plot_dir = None
            self.log_dir = Path("logs") / self.timestamp
        else:
            # Single training: create timestamped folder in output/train/
            self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.train_output_dir = setting['pth_output'] / "train" / self.timestamp
            self.train_output_dir.mkdir(parents=True, exist_ok=True)

            # Create subdirectories
            self.checkpoint_dir = self.train_output_dir / "checkpoints"
            self.checkpoint_dir.mkdir(exist_ok=True)
            self.plot_dir = self.train_output_dir / "plots"
            self.plot_dir.mkdir(exist_ok=True)
            self.log_dir = self.train_output_dir / "logs"
            self.log_dir.mkdir(exist_ok=True)

            # Copy settings file for reproducibility
            self._copy_settings_file(self.train_output_dir)

            print(f"\n📁 Training results will be saved to: {self.train_output_dir}")

        # Initialize TensorBoard writer
        self.writer = SummaryWriter(str(self.log_dir))

        # Create directory for saving per-epoch probability data
        self.prob_dir = self.log_dir / "probabilities"
        self.prob_dir.mkdir(parents=True, exist_ok=True)

        # Class names
        self.classes = setting["classes"]

        # Enhanced TensorBoard layout
        custom_layout = {
            'Accuracy': {
                'Training Accuracy': ['Scalar', 'Accuracy/Train'],
                'Validation Accuracy': ['Scalar', 'Accuracy/Val'],
                'Standard Accuracy': ['Scalar', 'Accuracy/val_standard'],
                'Balanced Accuracy': ['Scalar', 'Metrics/Balanced_Accuracy'],
                'Per-Class Accuracy': ['Multiline', [f'Accuracy/class/{cls}' for cls in self.classes]],
            },
            'Loss': {
                'Training Loss': ['Scalar', 'Loss/Train'],
                'Validation Loss': ['Scalar', 'Loss/Val'],
            },
            'Metrics': {
                'F1 Scores': ['Multiline', ['Metrics/F1/Macro', 'Metrics/F1/Weighted']],
                'AUC Scores': ['Multiline', ['Metrics/AUC'] + [f'Metrics/AUC/{cls}' for cls in self.classes]],
                'AP Scores': ['Multiline', ['Metrics/AP'] + [f'Metrics/AP/{cls}' for cls in self.classes]],
                'Learning Rate': ['Scalar', 'Metrics/LR'],
                'Composite Score': ['Scalar', 'Metrics/Composite_Score'],
                'Balanced Accuracy': ['Scalar', 'Metrics/Balanced_Accuracy'],
            },
            'ROC Curves': {
                'ROC - All Classes': ['Image', 'ROC/All_Classes'],
            },
            'PR Curves': {
                'PR - All Classes': ['Image', 'PR/All_Classes'],
            },
            'Data': {
                'Class Distribution': ['Multiline', [f'Data/class_count/{cls}' for cls in self.classes]],
                'Class Weights': ['Multiline', [f'Data/class_weight/{cls}' for cls in self.classes]],
            },
            'System': {
                'GPU Memory': ['Scalar', 'System/GPU_Memory']
            }
        }
        self.writer.add_custom_scalars(custom_layout)

        # Datasets
        self.ds_train = dataset.ds_train
        self.ds_val = dataset.ds_val
        self.num_train_img = dataset.num_train_img
        self.num_val_img = dataset.num_val_img
        self.num_train_batches = dataset.num_train_batches
        self.num_val_batches = dataset.num_val_batches

        # Training hyperparameters
        self.num_epochs = setting["train_num_epochs"]
        self.init_lr = setting["train_init_lr"]
        self.warmup_epochs = setting["train_lr_warmup_epochs"]
        self.lr_eta_min = setting["train_lr_eta_min"]
        self.weight_decay = setting["train_weight_decay"]
        self.label_smoothing = setting["train_label_smoothing"]

        # Optimizer specific
        self.optimizer_type = setting["train_optimizer_type"]
        self.sgd_momentum = setting["train_sgd_momentum"]
        self.sgd_use_nesterov = setting["train_sgd_use_nesterov"]
        self.adam_beta1 = setting["train_adam_beta1"]
        self.adam_beta2 = setting["train_adam_beta2"]

        # Loss function settings
        self.use_weighted_loss = setting["train_use_weighted_loss"]

        # Save checkpoints
        self.chckpt_save = setting["chckpt_save"]
        # Checkpoint selection method
        self.chckpt_selection_method = setting.get("chckpt_selection_method", "both")
        # Composite score
        self.min_class_acc_threshold = setting["chckpt_min_class_acc_threshold"]
        self.penalty_weight = setting["chckpt_penalty_weight"]
        # Balanced accuracy
        self.min_balanced_acc_threshold = setting.get("chckpt_min_balanced_acc_threshold", 0.60)
        self.min_per_class_acc_balanced = setting.get("chckpt_min_per_class_acc_balanced", 0.0)

        # Validation split settings
        self.val_from_train_split = setting["ds_val_from_train_split"]
        self.val_from_test_split = setting["ds_val_from_test_split"]

        # Calculate class weights for metrics
        self.class_weights = self._get_class_weights_for_metrics(dataset)

        # Optimizer
        if self.optimizer_type == "ADAM":
            self.optimizer = Adam(
                self.cnn.parameters(),
                lr=self.init_lr,
                weight_decay=self.weight_decay,
                betas=(self.adam_beta1, self.adam_beta2),
                amsgrad=False
            )
        elif self.optimizer_type == "ADAMW":
            self.optimizer = AdamW(
                self.cnn.parameters(),
                lr=self.init_lr,
                weight_decay=self.weight_decay,
                betas=(self.adam_beta1, self.adam_beta2),
                amsgrad=False
            )
        elif self.optimizer_type == "SGD":
            self.optimizer = SGD(
                self.cnn.parameters(),
                lr=self.init_lr,
                momentum=self.sgd_momentum,
                weight_decay=self.weight_decay,
                nesterov=self.sgd_use_nesterov
            )

        # Loss function setup
        if self.use_weighted_loss:
            print("\n> Using WEIGHTED loss function. Calculating class weights...")
            if self.val_from_train_split is not False:
                class_weights, class_counts = self._calculate_weights_from_dataloader()
            else:
                class_weights, class_counts = self._calculate_weights_from_folder(dataset.pth_train)

            self._print_class_analysis(class_counts, class_weights)
            self.class_counts = class_counts

            self.loss_function = nn.CrossEntropyLoss(
                weight=class_weights.to(device),
                label_smoothing=self.label_smoothing
            )
        else:
            print("\n> Using NON-WEIGHTED loss function.")
            if self.val_from_train_split is not False:
                _, class_counts = self._calculate_weights_from_dataloader()
            else:
                _, class_counts = self._calculate_weights_from_folder(dataset.pth_train)
            self._print_class_distribution(class_counts)
            self.loss_function = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

        # Learning rate scheduler
        self.scheduler_CA = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.num_epochs - self.warmup_epochs,
            eta_min=self.lr_eta_min,
        )
        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.01,
            total_iters=self.warmup_epochs,
        )
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[self.warmup_scheduler, self.scheduler_CA],
            milestones=[self.warmup_epochs],
        )

        # Mixed precision training
        self.scaler = GradScaler()

        # GPU memory monitoring
        self.total_gpu_memory = torch.cuda.get_device_properties(device).total_memory if torch.cuda.is_available() else 0

        # Print checkpoint saving configuration
        print(f"\n> Checkpoint saving: {self.chckpt_selection_method.upper()}")
        if self.chckpt_selection_method in ["balanced_accuracy", "both"]:
            print(f"  - Balanced Accuracy: min={self.min_balanced_acc_threshold:.0%}, per-class={self.min_per_class_acc_balanced:.0%}")
        if self.chckpt_selection_method in ["composite_score", "both"]:
            print(f"  - Composite Score: min_class={self.min_class_acc_threshold:.0%}, penalty={self.penalty_weight}")

    #############################################################################################################
    # METHODS

    # Save training examples to the training output directory.
    # This provides a quick visual sanity check of the training data.
    # Args:
    #   output_dir (Path): Directory to save the examples
    #   num_images (int): Number of images to display
    #   rows (int): Number of rows in the grid
    #   cols (int): Number of columns in the grid
    #   figsize (tuple): Figure size in inches
    def _save_training_examples(
        self,
        output_dir: Path,
        num_images: int = 25,
        rows: int = 5,
        cols: int = 5,
        figsize: tuple = (12, 12)
    ) -> None:
        try:
            # Get a batch of images from the training loader
            data_iter = iter(self.ds_train)
            images, _ = next(data_iter)

            # If batch_size < num_images, pad with empty tensors
            if images.shape[0] < num_images:
                empty_images = torch.zeros((num_images - images.shape[0], *images.shape[1:]))
                images = torch.cat([images, empty_images], dim=0)
            else:
                images = images[:num_images]

            # Denormalize images back to [0,1] range for visualization
            if self.device.type == 'cuda':
                images = images.cpu()

            if self.cnn_wrapper.input_channels == 1:
                # Grayscale: mean=0.5, std=0.5
                images = images * 0.5 + 0.5
            else:
                # RGB: ImageNet stats
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                images = images * std + mean

            # Create grid
            import torchvision.utils as vutils
            grid = vutils.make_grid(images, nrow=cols, padding=2, normalize=False)

            # Plot
            plt.figure(figsize=figsize)
            if self.cnn_wrapper.input_channels == 1:
                plt.imshow(grid[0], cmap='gray')
            else:
                plt.imshow(grid.permute(1, 2, 0))
            plt.axis('off')
            plt.title("Training Image Examples")

            plt.tight_layout()
            plt.savefig(str(output_dir / "training_examples.png"), bbox_inches='tight', dpi=300)
            plt.close()

            print(f"✓ Training examples saved to: {output_dir / 'training_examples.png'}")
        except Exception as e:
            print(f"⚠️  Could not save training examples: {e}")

    # Copy the settings file to the training output directory for reproducibility.
    # Args:
    #   output_dir (Path): The training output directory
    def _copy_settings_file(self, output_dir: Path) -> None:
        settings_src = Path(__file__).parent / "settings.py"
        settings_dst = output_dir / "settings_copy.py"
        if settings_src.exists():
            shutil.copy2(settings_src, settings_dst)
            print(f"✓ Settings file copied to: {settings_dst}")

    # Calculate the composite score for checkpoint selection.
    # Formula: Composite = Overall_Accuracy - penalty_weight * (Standard_Deviation_of_Class_Accuracies)
    # Args:
    #   class_accuracies (dict): Per-class accuracies
    #   overall_accuracy (float): Mean accuracy across all classes
    #   penalty_weight (float): Penalty for class imbalance (higher = stricter)
    # Returns:
    #   tuple: (composite_score, class_std, min_class_acc)
    def _calculate_composite_score(self, class_accuracies: dict, overall_accuracy: float, penalty_weight: float = 2.0) -> tuple:
        acc_values = list(class_accuracies.values())
        class_std = np.std(acc_values)
        min_class_acc = min(acc_values)
        composite_score = overall_accuracy - (penalty_weight * class_std)
        return composite_score, class_std, min_class_acc

    # Plot ROC curves and save to TensorBoard.
    # Args:
    #   fpr (dict): False positive rates per class
    #   tpr (dict): True positive rates per class
    #   roc_auc (dict): AUC values per class
    #   epoch (int): Current epoch
    #   title (str): Plot title
    def _plot_roc_curve_to_tensorboard(self, fpr: dict, tpr: dict, roc_auc: dict, epoch: int, title: str = "ROC Curves") -> None:
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.classes)))

        for i, (class_name, color) in enumerate(zip(self.classes, colors)):
            if i in fpr and i in tpr and i in roc_auc:
                ax.plot(fpr[i], tpr[i], color=color, lw=2,
                       label=f'{class_name} (AUC = {roc_auc[i]:.3f})')

        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC=0.5)')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'{title} (Epoch {epoch+1})', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=10, frameon=False)
        ax.grid(True, alpha=0.3)

        self.writer.add_figure('ROC/All_Classes', fig, epoch)
        plt.close(fig)

    # Plot Precision-Recall curves and save to TensorBoard.
    # Args:
    #   precision (dict): Precision values per class
    #   recall (dict): Recall values per class
    #   average_precision (dict): AP values per class
    #   epoch (int): Current epoch
    #   title (str): Plot title
    def _plot_pr_curve_to_tensorboard(self, precision: dict, recall: dict, average_precision: dict, epoch: int, title: str = "Precision-Recall Curves") -> None:
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.classes)))

        for i, (class_name, color) in enumerate(zip(self.classes, colors)):
            if i in precision and i in recall and i in average_precision:
                ax.plot(recall[i], precision[i], color=color, lw=2,
                       label=f'{class_name} (AP = {average_precision[i]:.3f})')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(f'{title} (Epoch {epoch+1})', fontsize=14, fontweight='bold')
        ax.legend(loc="lower left", fontsize=10, frameon=False)
        ax.grid(True, alpha=0.3)

        self.writer.add_figure('PR/All_Classes', fig, epoch)
        plt.close(fig)

    # Save probability data for retrospective analysis.
    # Saves as .npz (full data) and .json (summary).
    # Args:
    #   all_probs (np.ndarray): Probabilities (n_samples, n_classes)
    #   all_labels (np.ndarray): True labels
    #   all_preds (np.ndarray): Predictions
    #   epoch (int): Current epoch
    def _save_probability_data(self, all_probs: np.ndarray, all_labels: np.ndarray, all_preds: np.ndarray, epoch: int) -> None:
        np.savez_compressed(
            self.prob_dir / f'probabilities_epoch_{epoch:03d}.npz',
            probabilities=all_probs,
            labels=all_labels,
            predictions=all_preds,
            classes=np.array(self.classes)
        )

        summary = {
            'epoch': epoch,
            'num_samples': len(all_labels),
            'num_classes': len(self.classes),
            'classes': self.classes,
            'class_distribution': {self.classes[i]: int(np.sum(all_labels == i)) for i in range(len(self.classes))}
        }

        with open(self.prob_dir / f'probabilities_epoch_{epoch:03d}_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

    # Get class weights for metrics calculation.
    # Args:
    #   dataset (Dataset): Dataset handler
    # Returns:
    #   torch.Tensor: Class weights
    def _get_class_weights_for_metrics(self, dataset) -> torch.Tensor:
        if self.use_weighted_loss:
            if self.val_from_train_split is not False:
                weights, _ = self._calculate_weights_from_dataloader()
            else:
                weights, _ = self._calculate_weights_from_folder(dataset.pth_train)
        else:
            weights = torch.ones(len(self.classes))
        return weights

    # Calculate class weights from a DataLoader.
    # Returns:
    #   tuple: (normalized_weights, class_counts)
    def _calculate_weights_from_dataloader(self) -> tuple:
        train_dataset = self.ds_train.dataset
        class_counts = torch.zeros(len(self.classes))
        for _, label in train_dataset:
            class_counts[label] += 1

        weights = 1.0 / (class_counts + 1e-6)
        weights_normalized = weights / weights.sum()
        return weights_normalized, class_counts

    # Calculate class weights from folder structure.
    # Args:
    #   folder_path (Path): Path to folder with class subdirectories
    # Returns:
    #   tuple: (normalized_weights, class_counts)
    def _calculate_weights_from_folder(self, folder_path: Path) -> tuple:
        class_counts = []
        for class_dir in sorted(Path(folder_path).iterdir()):
            if class_dir.is_dir():
                class_counts.append(len(list(class_dir.glob("*.*"))))

        counts = torch.tensor(class_counts, dtype=torch.float32)
        weights = 1.0 / (counts + 1e-6)
        weights_normalized = weights / weights.sum()
        return weights_normalized, counts

    # Print comprehensive class analysis including distribution and weights.
    # Args:
    #   class_counts (torch.Tensor): Number of samples per class
    #   class_weights (torch.Tensor): Loss weights per class
    def _print_class_analysis(self, class_counts: torch.Tensor, class_weights: torch.Tensor) -> None:
        total = class_counts.sum().item()

        print("\n" + "=" * 60)
        print("CLASS DISTRIBUTION ANALYSIS")
        print("=" * 60)

        print("\n📊 ACTUAL CLASS DISTRIBUTION:")
        max_len = max(len(cls) for cls in self.classes)
        for i, cls in enumerate(self.classes):
            count = int(class_counts[i].item())
            percentage = (class_counts[i] / total) * 100
            print(f"  {cls.ljust(max_len)} : {count:6d} images ({percentage:.2f}%)")
        print(f"  {'Total'.ljust(max_len)} : {int(total):6d} images (100.00%)")

        print("\n⚖️  LOSS FUNCTION WEIGHTS (for CrossEntropyLoss):")
        print("  Note: Higher weight = more importance in loss calculation")
        for i, cls in enumerate(self.classes):
            weight_pct = class_weights[i].item() * 100
            print(f"  {cls.ljust(max_len)} : {weight_pct:.2f}% weight")

        print("\n📝 INTERPRETATION:")
        majority_idx = torch.argmax(class_counts).item()
        minority_idx = torch.argmin(class_counts).item()
        majority_class = self.classes[majority_idx]
        minority_class = self.classes[minority_idx]

        majority_weight = class_weights[majority_idx].item() * 100
        minority_weight = class_weights[minority_idx].item() * 100

        print(f"  • {majority_class} is MAJORITY class ({class_counts[majority_idx].item():.0f} images)")
        print(f"  • {minority_class} is MINORITY class ({class_counts[minority_idx].item():.0f} images)")
        print(f"  • Loss weight ratio: {minority_class}:{majority_class} = {minority_weight/majority_weight:.2f}:1")
        print(f"  • Each {minority_class} sample gets {minority_weight/majority_weight:.2f}x more importance")
        print("=" * 60 + "\n")

        for i, cls in enumerate(self.classes):
            self.writer.add_scalar(f'Data/class_count/{cls}', class_counts[i].item(), 0)
            self.writer.add_scalar(f'Data/class_weight/{cls}', class_weights[i].item(), 0)

    # Print class distribution only (for non-weighted loss).
    # Args:
    #   class_counts (torch.Tensor): Number of samples per class
    def _print_class_distribution(self, class_counts: torch.Tensor) -> None:
        total = class_counts.sum().item()

        print("\n📊 CLASS DISTRIBUTION:")
        max_len = max(len(cls) for cls in self.classes)
        for i, cls in enumerate(self.classes):
            count = int(class_counts[i].item())
            percentage = (class_counts[i] / total) * 100
            print(f"  {cls.ljust(max_len)} : {count:6d} images ({percentage:.2f}%)")
        print(f"  {'Total'.ljust(max_len)} : {int(total):6d} images")

        for i, cls in enumerate(self.classes):
            self.writer.add_scalar(f'Data/class_count/{cls}', class_counts[i].item(), 0)

    # Calculate weighted accuracy if using weighted loss, otherwise standard accuracy.
    # Args:
    #   preds (torch.Tensor): Predictions
    #   labels (torch.Tensor): True labels
    # Returns:
    #   tuple: (accuracy, total_weight_or_samples)
    def _calculate_weighted_accuracy(self, preds: torch.Tensor, labels: torch.Tensor) -> tuple:
        if self.use_weighted_loss:
            if self.class_weights.device != labels.device:
                self.class_weights = self.class_weights.to(labels.device)

            correct = (preds == labels).float()
            batch_weights = self.class_weights[labels]

            weighted_accuracy = (correct * batch_weights).sum().item()
            total_weight = batch_weights.sum().item()
            return weighted_accuracy, total_weight
        else:
            accuracy = (preds == labels).sum().item()
            return accuracy, labels.size(0)

    # Generate metrics plot at the end of training.
    # Args:
    #   history (dict): Training history
    #   show_plot (bool): If True, display the plot
    #   save_plot (bool): If True, save the plot to disk
    def _plot_metrics(self, history: dict, show_plot: bool = True, save_plot: bool = True) -> None:
        plt.figure(figsize=(24, 12))

        # 1. Accuracy Plot
        plt.subplot(2, 3, 1)
        epochs_range = range(1, len(history["train_acc"]) + 1)
        plt.plot(epochs_range, history["train_acc"], 'g-', label='Training')
        plt.plot(epochs_range, history["val_acc"], 'r-', label='Validation')
        plt.title('Accuracy' + (' (Weighted)' if self.use_weighted_loss else ''))
        plt.legend()

        # 2. Loss Plot
        plt.subplot(2, 3, 2)
        plt.plot(epochs_range, history["train_loss"], 'g-', label='Training')
        plt.plot(epochs_range, history["val_loss"], 'r-', label='Validation')
        plt.title('Loss' + (' (Weighted)' if self.use_weighted_loss else ''))
        plt.legend()
        plt.ylim(0, min(5, max(max(history["val_loss"]), max(history["train_loss"])) * 1.1))

        # 3. Learning Rate
        plt.subplot(2, 3, 3)
        plt.plot(epochs_range, history["lr"], 'b-')
        plt.title('Learning Rate')

        # 4. F1 Scores
        plt.subplot(2, 3, 4)
        plt.plot(epochs_range, history["f1_macro"], 'b-', label='Macro')
        plt.plot(epochs_range, history["f1_weighted"], 'orange', label='Weighted')
        plt.title('F1 Scores')
        plt.legend()
        plt.ylim(0, 1)

        # 5. ROC Curves (last epoch only)
        plt.subplot(2, 3, 5)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.classes)))
        last_epoch = -1
        for i, color in zip(range(len(self.classes)), colors):
            if i in history["fpr"][last_epoch]:
                plt.plot(history["fpr"][last_epoch][i],
                        history["tpr"][last_epoch][i],
                        color=color,
                        label=f'{self.classes[i]} (AUC={history["roc_auc"][last_epoch][i]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title('ROC Curves')
        plt.legend()

        # 6. PR Curves (last epoch only)
        plt.subplot(2, 3, 6)
        for i, color in zip(range(len(self.classes)), colors):
            if i in history["precision"][last_epoch]:
                plt.plot(history["recall"][last_epoch][i],
                        history["precision"][last_epoch][i],
                        color=color,
                        label=f'{self.classes[i]} (AP={history["average_precision"][last_epoch][i]:.2f})')
        plt.title('Precision-Recall')
        plt.legend()

        plt.tight_layout()
        if save_plot:
            weight_suffix = "_weighted" if self.use_weighted_loss else "_standard"
            plt.savefig(str(self.plot_dir / f"train_metrics{weight_suffix}"), bbox_inches='tight', dpi=300)
            plt.close()
        if show_plot:
            plt.show()

    # Create a styled confusion matrix plot.
    # Args:
    #   cm (np.ndarray): Confusion matrix
    #   class_names (list): List of class names
    #   epoch (int, optional): Current epoch
    # Returns:
    #   matplotlib.figure.Figure: The figure object
    def _plot_confusion_matrix(self, cm: np.ndarray, class_names: list = None, epoch: int = None):
        fig, ax = plt.subplots(figsize=(8, 8))

        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)

        title_font = {'size': 14, 'weight': 'bold'}
        label_font = {'size': 12}
        tick_font = {'size': 12}
        text_font = {'size': 12}

        ax.set_xlabel('Predicted Label', fontdict=label_font)
        ax.set_ylabel('True Label', fontdict=label_font)
        title = f'Confusion Matrix (Epoch {epoch+1})' if epoch else 'Confusion Matrix'
        if self.use_weighted_loss:
            title += ' - Weighted Loss'
        ax.set_title(title, fontdict=title_font)

        if class_names:
            ax.set_xticks(np.arange(len(class_names)))
            ax.set_yticks(np.arange(len(class_names)))
            ax.set_xticklabels(class_names, rotation=45, ha="right", fontdict=tick_font)
            ax.set_yticklabels(class_names, fontdict=tick_font)

        thresh = cm_normalized.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i,
                    f"{cm_normalized[i,j]:.1%}\n({cm[i,j]})",
                    ha="center", va="center",
                    color="white" if cm_normalized[i,j] > thresh else "black",
                    fontsize=text_font['size'])

        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=12)
        plt.tight_layout()

        return fig

    #############################################################################################################
    # TRAIN FUNCTION

    # Main training loop.
    # Args:
    #   chckpt_pth (Path, optional): Directory to save checkpoints. Required for cross-validation.
    #   plot_pth (Path, optional): Directory to save plots. Required for cross-validation.
    # Returns:
    #   dict: Training history
    def train(self, chckpt_pth: Path = None, plot_pth: Path = None) -> dict:

        # For single training: use pre-created directories
        if chckpt_pth is None and self.dataset_idx is None:
            chckpt_pth = self.checkpoint_dir
            plot_pth = self.plot_dir

        # For cross-validation: chckpt_pth and plot_pth should be provided
        if chckpt_pth is None or plot_pth is None:
            raise ValueError("checkpoint and plot paths must be provided for cross-validation")

        # Set the paths for this training run
        self.checkpoint_dir = chckpt_pth
        self.plot_dir = plot_pth

        # For single training: save training examples
        if self.dataset_idx is None:
            self._save_training_examples(self.plot_dir)       

        # Initialize metrics storage
        history = {
            "train_acc": [], "train_loss": [],
            "val_acc": [], "val_loss": [],
            "lr": [],
            "f1_macro": [], "f1_weighted": [],
            "roc_auc": [], "fpr": [], "tpr": [],
            "precision": [], "recall": [], "average_precision": [],
            "confusion_matrices": []
        }

        # Track best scores for both metrics
        best_balanced_acc = -float('inf')
        best_composite_score = -float('inf')
        best_epoch_bal = -1
        best_epoch_comp = -1

        # Build dynamic base name from settings
        pretrained_str = "pretr" if setting["cnn_is_pretrained"] else "scratch"
        model_name = setting["cnn_type"]
        dataset_suffix = f"_ds{self.dataset_idx}" if self.dataset_idx is not None else ""

        for epoch in range(self.num_epochs):
            print(f"\n>> Epoch [{epoch+1}/{self.num_epochs}]:")

            #################
            # Training loop #
            #################

            self.cnn.train()
            weighted_train_accuracy = 0.0
            standard_train_accuracy = 0.0
            train_loss = 0.0
            total_train_weight = 0.0
            actual_train_samples = 0

            if self.use_weighted_loss and self.class_weights.device != self.device:
                self.class_weights = self.class_weights.to(self.device)

            with tqdm(self.ds_train, unit="batch") as tepoch:
                for batch_idx, (images, labels) in enumerate(tepoch):
                    tepoch.set_description("Train")

                    images, labels = images.to(self.device), labels.to(self.device)

                    self.optimizer.zero_grad()
                    with autocast(device_type=self.device.type, enabled=self.device.type == 'cuda'):
                        outputs = self.cnn(images)
                        loss = self.loss_function(outputs, labels)

                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.cnn.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    if torch.isnan(loss).any():
                        raise ValueError("NaN loss detected during training")

                    train_loss += loss.item() * images.size(0)
                    _, preds = torch.max(outputs, 1)

                    if self.use_weighted_loss:
                        batch_weighted_accuracy, batch_weight = self._calculate_weighted_accuracy(preds, labels)
                        weighted_train_accuracy += batch_weighted_accuracy
                        total_train_weight += batch_weight
                        batch_standard_accuracy = (preds == labels).sum().item()
                        standard_train_accuracy += batch_standard_accuracy
                    else:
                        batch_standard_accuracy, _ = self._calculate_weighted_accuracy(preds, labels)
                        standard_train_accuracy += batch_standard_accuracy
                        weighted_train_accuracy = standard_train_accuracy

                    actual_train_samples += labels.size(0)

                    if batch_idx % 10 == 0:
                        torch.cuda.empty_cache()

            self.scheduler.step()

            if self.use_weighted_loss:
                if total_train_weight > 0:
                    final_weighted_train_accuracy = weighted_train_accuracy / total_train_weight
                else:
                    final_weighted_train_accuracy = 0.0

                if actual_train_samples > 0:
                    final_standard_train_accuracy = standard_train_accuracy / actual_train_samples
                else:
                    final_standard_train_accuracy = 0.0
            else:
                if actual_train_samples > 0:
                    final_standard_train_accuracy = standard_train_accuracy / actual_train_samples
                    final_weighted_train_accuracy = final_standard_train_accuracy
                else:
                    final_standard_train_accuracy = 0.0
                    final_weighted_train_accuracy = 0.0

            train_loss = train_loss / len(self.ds_train.dataset)
            current_lr = self.scheduler.get_last_lr()[0]

            if self.use_weighted_loss:
                print(f"> Train Loss: {train_loss:.5f} | Weighted Train Acc: {final_weighted_train_accuracy:.2f} | Standard Train Acc: {final_standard_train_accuracy:.2f} | Learning Rate: {current_lr:.6f}")
            else:
                print(f"> Train Loss: {train_loss:.5f} | Train Acc: {final_standard_train_accuracy:.2f} | Learning Rate: {current_lr:.6f}")

            history["train_acc"].append(final_weighted_train_accuracy if self.use_weighted_loss else final_standard_train_accuracy)
            history["train_loss"].append(train_loss)
            history["lr"].append(current_lr)

            ###################
            # Validation loop #
            ###################

            self.cnn.eval()
            val_accuracy = 0.0
            val_loss = 0.0
            total_val_weight = 0.0
            all_preds = []
            all_labels = []
            all_probs = []

            if self.use_weighted_loss and self.class_weights.device != self.device:
                self.class_weights = self.class_weights.to(self.device)

            class_correct = {i: 0 for i in range(len(self.classes))}
            class_total = {i: 0 for i in range(len(self.classes))}

            with torch.inference_mode():
                with tqdm(self.ds_val, unit="batch") as tepoch:
                    tepoch.set_description("Valid")
                    for images, labels in tepoch:
                        images, labels = images.to(self.device), labels.to(self.device)

                        with autocast(device_type=self.device.type, enabled=self.device.type == 'cuda'):
                            outputs = self.cnn(images)
                            loss = self.loss_function(outputs, labels)
                            probs = torch.softmax(outputs.float(), dim=1)

                        val_loss += loss.item() * images.size(0)
                        _, preds = torch.max(outputs, 1)

                        batch_accuracy, batch_weight = self._calculate_weighted_accuracy(preds, labels)
                        val_accuracy += batch_accuracy
                        total_val_weight += batch_weight

                        all_probs.append(probs.cpu().detach())
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())

                        for i in range(len(self.classes)):
                            class_mask = (labels.cpu() == i)
                            class_correct[i] += (preds[class_mask] == labels[class_mask]).sum().item()
                            class_total[i] += class_mask.sum().item()

            if self.use_weighted_loss:
                if total_val_weight > 0:
                    weighted_val_accuracy = val_accuracy / total_val_weight
                else:
                    weighted_val_accuracy = 0.0
            else:
                weighted_val_accuracy = val_accuracy / len(self.ds_val.dataset)

            val_loss = val_loss / len(self.ds_val.dataset)

            class_accuracies = {}
            for i in range(len(self.classes)):
                if class_total[i] > 0:
                    class_accuracies[self.classes[i]] = class_correct[i] / class_total[i]
                else:
                    class_accuracies[self.classes[i]] = 0.0

            standard_val_accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
            balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)

            class_acc_str = " | ".join([f"{acc:.2f} {cls}" for cls, acc in class_accuracies.items()])

            composite_score, class_std, min_class_acc = self._calculate_composite_score(
                class_accuracies, standard_val_accuracy, penalty_weight=self.penalty_weight
            )

            with torch.no_grad():
                all_probs = torch.cat(all_probs).numpy()
            all_labels_np = np.array(all_labels)
            all_preds_np = np.array(all_preds)

            if all_labels_np.ndim > 1 and all_labels_np.shape[1] > 1:
                all_labels_np = np.argmax(all_labels_np, axis=1)

            f1_macro = f1_score(all_labels_np, all_preds_np, average='macro')
            f1_weighted = f1_score(all_labels_np, all_preds_np, average='weighted')

            fpr, tpr, roc_auc = {}, {}, {}
            precision, recall, average_precision = {}, {}, {}

            for i in range(len(self.classes)):
                class_mask = (all_labels_np == i)
                if np.sum(class_mask) > 0:
                    fpr[i], tpr[i], _ = roc_curve(class_mask, all_probs[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                    precision[i], recall[i], _ = precision_recall_curve(class_mask, all_probs[:, i])
                    average_precision[i] = average_precision_score(class_mask, all_probs[:, i])

            if len(self.classes) == 2:
                roc_auc_weighted = roc_auc_score(all_labels_np, all_probs[:, 1])
            else:
                roc_auc_weighted = roc_auc_score(
                    all_labels_np,
                    all_probs,
                    multi_class='ovr',
                    average='weighted'
                )
            ap_weighted = np.mean(list(average_precision.values())) if average_precision else 0

            cm = confusion_matrix(all_labels_np, all_preds_np)

            self._plot_roc_curve_to_tensorboard(fpr, tpr, roc_auc, epoch)
            self._plot_pr_curve_to_tensorboard(precision, recall, average_precision, epoch)

            cm_fig = self._plot_confusion_matrix(cm, class_names=self.classes, epoch=epoch)
            self.writer.add_figure('Confusion Matrix', cm_fig, epoch, close=True)
            plt.close(cm_fig)

            # Log metrics to TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Accuracy/train', final_weighted_train_accuracy if self.use_weighted_loss else final_standard_train_accuracy, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/val', weighted_val_accuracy if self.use_weighted_loss else standard_val_accuracy, epoch)
            self.writer.add_scalar('Accuracy/val_standard', standard_val_accuracy, epoch)
            self.writer.add_scalar('Metrics/Balanced_Accuracy', balanced_accuracy, epoch)
            self.writer.add_scalar('Metrics/Composite_Score', composite_score, epoch)
            self.writer.add_scalar('Metrics/Class_Accuracy_StdDev', class_std, epoch)
            self.writer.add_scalar('Metrics/Min_Class_Accuracy', min_class_acc, epoch)

            if self.use_weighted_loss:
                self.writer.add_scalar('Accuracy/train_standard', final_standard_train_accuracy, epoch)

            self.writer.add_scalar('Metrics/LR', current_lr, epoch)
            self.writer.add_scalars('Metrics/F1', {'Macro': f1_macro, 'Weighted': f1_weighted}, epoch)
            self.writer.add_scalar('Metrics/AUC', roc_auc_weighted, epoch)
            self.writer.add_scalar('Metrics/AP', ap_weighted, epoch)

            for i, class_name in enumerate(self.classes):
                if i in roc_auc:
                    self.writer.add_scalar(f'Metrics/AUC/{class_name}', roc_auc[i], epoch)
                if i in average_precision:
                    self.writer.add_scalar(f'Metrics/AP/{class_name}', average_precision[i], epoch)
                if class_name in class_accuracies:
                    self.writer.add_scalar(f'Accuracy/class/{class_name}', class_accuracies[class_name], epoch)

            for i, class_name in enumerate(self.classes):
                if i in class_total:
                    self.writer.add_scalar(f'Data/class_count/{class_name}', class_total[i], epoch)

            if torch.cuda.is_available():
                mem_usage = torch.cuda.memory_reserved() / self.total_gpu_memory
                self.writer.add_scalar('System/GPU_Memory', mem_usage, epoch)

            # Console output
            if self.use_weighted_loss:
                print(f"> Val Loss: {val_loss:.5f} | Weighted Val Acc: {weighted_val_accuracy:.2f} | Standard Val Acc: {standard_val_accuracy:.2f} | Per-class: ({class_acc_str})")
                print(f"> Balanced Accuracy: {balanced_accuracy:.4f} | Composite Score: {composite_score:.4f}")
                print(f"> F1 (Macro): {f1_macro:.4f} | F1 (Weighted): {f1_weighted:.4f} | AUC: {roc_auc_weighted:.4f} | AP: {ap_weighted:.4f}")
            else:
                print(f"> Val Loss: {val_loss:.5f} | Val Acc: {standard_val_accuracy:.2f} | Per-class: ({class_acc_str})")
                print(f"> Balanced Accuracy: {balanced_accuracy:.4f} | Composite Score: {composite_score:.4f}")
                print(f"> F1 (Macro): {f1_macro:.4f} | AUC: {roc_auc_weighted:.4f} | AP: {ap_weighted:.4f}")

            print(f"> Class STD: {class_std:.4f} | Min Class Acc: {min_class_acc:.2%}")

            #####################
            # CHECKPOINT SAVING #
            #####################

            save_checkpoint = False
            trigger_reason = []

            # METHOD 1: Balanced Accuracy (if enabled)
            if self.chckpt_selection_method in ["balanced_accuracy", "both"]:
                bal_threshold_met = balanced_accuracy >= self.min_balanced_acc_threshold
                per_class_ok = True
                if self.min_per_class_acc_balanced > 0:
                    if min_class_acc < self.min_per_class_acc_balanced:
                        per_class_ok = False

                if bal_threshold_met and per_class_ok:
                    if balanced_accuracy > best_balanced_acc:
                        save_checkpoint = True
                        best_balanced_acc = balanced_accuracy
                        best_epoch_bal = epoch
                        trigger_reason.append(f"BalancedAcc improved to {balanced_accuracy:.4f}")
                        print(f"> Balanced accuracy improved to {balanced_accuracy:.4f}!")
                    else:
                        print(f"> Balanced accuracy {balanced_accuracy:.4f} did not improve over best {best_balanced_acc:.4f}")
                else:
                    if not bal_threshold_met:
                        print(f"> Balanced accuracy ({balanced_accuracy:.4f}) below threshold ({self.min_balanced_acc_threshold:.0%})")
                    elif not per_class_ok:
                        print(f"> Min class accuracy ({min_class_acc:.2%}) below per-class threshold ({self.min_per_class_acc_balanced:.0%})")
            else:
                # Not tracking balanced accuracy
                print(f"> Balanced accuracy: {balanced_accuracy:.4f} (not used for checkpoint selection)")

            # METHOD 2: Composite Score (if enabled)
            if self.chckpt_selection_method in ["composite_score", "both"]:
                comp_threshold_met = min_class_acc >= self.min_class_acc_threshold

                if comp_threshold_met:
                    if composite_score > best_composite_score:
                        save_checkpoint = True
                        best_composite_score = composite_score
                        best_epoch_comp = epoch
                        trigger_reason.append(f"CompositeScore improved to {composite_score:.4f}")
                        print(f"> Composite score improved to {composite_score:.4f}!")
                    else:
                        print(f"> Composite score {composite_score:.4f} did not improve over best {best_composite_score:.4f}")
                else:
                    print(f"> Min class accuracy ({min_class_acc:.2%}) below composite threshold ({self.min_class_acc_threshold:.0%})")
            else:
                # Not tracking composite score
                print(f"> Composite score: {composite_score:.4f} (not used for checkpoint selection)")

            # Save checkpoint if either method triggered (or only one, depending on selection)
            if save_checkpoint:

                checkpoint_name = f"ckpt_{pretrained_str}_{model_name}_e{epoch+1:02d}_bal{balanced_accuracy:.3f}_comp{composite_score:.3f}{dataset_suffix}"

                checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.pt"

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.cnn.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'accuracy': standard_val_accuracy,
                    'balanced_accuracy': balanced_accuracy,
                    'composite_score': composite_score,
                    'per_class_accuracy': class_accuracies,
                    'loss': val_loss,
                }, checkpoint_path)

                # Save validation confusion matrix
                val_cm_filename = checkpoint_name.replace('.pt', '') + "_val"
                fn.plot_confusion_matrix(
                    {"y": all_labels_np, "y_hat": all_preds_np},
                    self.classes,
                    self.plot_dir,
                    chckpt_name=val_cm_filename,
                    show_plot=False,
                    save_plot=True
                )

                fn.save_confusion_matrix_results(
                    {"y": all_labels_np, "y_hat": all_preds_np},
                    self.classes,
                    self.plot_dir,
                    chckpt_name=val_cm_filename
                )

                roc_pr_data = {
                    'epoch': epoch,
                    'classes': self.classes,
                    'composite_score': composite_score,
                    'balanced_accuracy': balanced_accuracy,
                    'class_std': float(class_std),
                    'min_class_acc': float(min_class_acc),
                    'roc_auc': {self.classes[i]: float(roc_auc[i]) for i in roc_auc},
                    'average_precision': {self.classes[i]: float(average_precision[i]) for i in average_precision},
                    'per_class_accuracy': {cls: float(acc) for cls, acc in class_accuracies.items()},
                }

                with open(self.plot_dir / f"{val_cm_filename}_roc_pr.json", 'w') as f:
                    json.dump(roc_pr_data, f, indent=2)

                print(f"✓ Model saved! Epoch {epoch+1}")
                print(f"  - Triggers: {', '.join(trigger_reason)}")
                print(f"  - Balanced Accuracy: {balanced_accuracy:.4f}")
                print(f"  - Composite Score: {composite_score:.4f}")
                print(f"  - Standard Accuracy: {standard_val_accuracy:.2%}")
                print(f"  - Min Class Accuracy: {min_class_acc:.2%}")
                print(f"  - Class STD: {class_std:.4f}")
                print(f"✓ Validation confusion matrix saved: {val_cm_filename}.png/.json")

            # ALWAYS save probability data for EVERY epoch
            self._save_probability_data(all_probs, all_labels_np, all_preds_np, epoch)
            print(f"✓ Probability data saved for epoch {epoch+1}")

            # Store validation metrics in history
            history["val_acc"].append(standard_val_accuracy)
            history["val_loss"].append(val_loss)
            history["f1_macro"].append(f1_macro)
            history["f1_weighted"].append(f1_weighted)
            history["roc_auc"].append(roc_auc)
            history["fpr"].append(fpr)
            history["tpr"].append(tpr)
            history["precision"].append(precision)
            history["recall"].append(recall)
            history["average_precision"].append(average_precision)
            history["confusion_matrices"].append(cm)

        # Final summary
        print(f"\n>> Training completed!")
        print(f"   Best Balanced Accuracy: {best_balanced_acc:.4f} at epoch {best_epoch_bal+1}")
        print(f"   Best Composite Score: {best_composite_score:.4f} at epoch {best_epoch_comp+1}")

        # Final cleanup and plotting
        self.writer.close()
        self._plot_metrics(history, show_plot=False, save_plot=True)

        return history