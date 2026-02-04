######################
# Class for training #
######################

from torch.optim import Adam, SGD, AdamW
from torch import nn
from tqdm import tqdm
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from torch.amp import GradScaler, autocast 
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import (f1_score, confusion_matrix, roc_curve, auc, 
                            roc_auc_score, precision_recall_curve, average_precision_score)
from datetime import datetime
import numpy as np
# Own modules
from settings import setting
import functions as fn

class Train():

    #############################################################################################################
    # CONSTRUCTOR:
    
    def __init__(self, cnn_wrapper, dataset, device):
        # Input validation
        assert len(dataset.ds_train) > 0, "Training dataset is empty"
        assert len(dataset.ds_val) > 0, "Validation dataset is empty"
        
        self.device = device
        self.cnn_wrapper = cnn_wrapper
        self.cnn = cnn_wrapper.model

        # Initialize TensorBoard writer
        self.writer = SummaryWriter(f"logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        
        # Class names
        self.classes = setting["classes"] 
        
        # Enhanced TensorBoard layout
        custom_layout = {
            'Accuracy': {
                'Training Accuracy': ['Scalar', 'Accuracy/Train'],
                'Validation Accuracy': ['Scalar', 'Accuracy/Val'],
                'Standard Accuracy': ['Scalar', 'Accuracy/val_standard'],
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
        
        # Pretrained
        self.is_pretrained = setting["cnn_is_pretrained"] 
        # Datasets
        self.ds_train = dataset.ds_train
        self.ds_val = dataset.ds_val

        # Number of training and validation images in each dataset
        self.num_train_img = dataset.num_train_img
        self.num_val_img = dataset.num_val_img
        # Number of training and validation batches in each dataset
        self.num_train_batches = dataset.num_train_batches
        self.num_val_batches = dataset.num_val_batches
        # Number of epochs
        self.num_epochs = setting["train_num_epochs"] 
        # Initial learning rate and scheduler
        self.init_lr = setting["train_init_lr"]
        self.warmup_epochs = setting["train_lr_warmup_epochs"] 
        self.lr_eta_min = setting["train_lr_eta_min"] 
        # Weight decay
        self.weight_decay = setting["train_weight_decay"] 
        # Label smoothing
        self.label_smoothing = setting["train_label_smoothing"] 
        # Optimizer specific
        self.optimizer_type = setting["train_optimizer_type"]
        self.sgd_momentum = setting["train_sgd_momentum"]
        self.sgd_use_nesterov = setting["train_sgd_use_nesterov"]
        self.adam_beta1 = setting["train_adam_beta1"]
        self.adam_beta2 = setting["train_adam_beta2"]

        # Use weighted loss function and training metrics
        # Useful for imbalanced datasets
        self.use_weighted_loss = setting["train_use_weighted_loss"]

        # Validation split settings
        # Validation split from training dataset in the folder data/train/
        self.val_from_train_split = setting["ds_val_from_train_split"]
        # Validation split from test dataset in the folder data/test/
        self.val_from_test_split = setting["ds_val_from_test_split"]

        # Calculate class weights for metrics
        self.class_weights = self._get_class_weights_for_metrics(dataset)

        # Optmizer, learning rate scheduler and loss function
        if(self.optimizer_type == "ADAM"):
            self.optimizer = Adam(
                self.cnn.parameters(),
                lr=self.init_lr, 
                weight_decay=self.weight_decay, 
                betas=(self.adam_beta1, self.adam_beta2),
                amsgrad=False
            )
        elif(self.optimizer_type == "ADAMW"):
            self.optimizer = AdamW(
                self.cnn.parameters(),
                lr=self.init_lr, 
                weight_decay=self.weight_decay, 
                betas=(self.adam_beta1, self.adam_beta2),
                amsgrad=False
            )
        elif(self.optimizer_type == "SGD"):
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
            
            # Print BOTH class distribution AND weights
            self._print_class_analysis(class_counts, class_weights)
            
            # Store class counts for TensorBoard logging
            self.class_counts = class_counts

            self.loss_function = nn.CrossEntropyLoss(
                weight=class_weights.to(device),
                label_smoothing=self.label_smoothing
            )
        else:
            print("\n> Using NON-WEIGHTED loss function.")
            # Still calculate and show distribution for reference
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

        # Add gradient scaler for mixed precision training
        self.scaler = GradScaler()
        
        # Add memory monitoring
        self.total_gpu_memory = torch.cuda.get_device_properties(device).total_memory if torch.cuda.is_available() else 0

    #############################################################################################################
    # METHODS:

    # Get class weights for metrics calculation
    def _get_class_weights_for_metrics(self, dataset):
        if self.use_weighted_loss:
            if self.val_from_train_split is not False:
                weights, _ = self._calculate_weights_from_dataloader()
            else:
                weights, _ = self._calculate_weights_from_folder(dataset.pth_train)
        else:
            # Return equal weights for all classes (on CPU initially)
            weights = torch.ones(len(self.classes))
        
        return weights  # Keep on CPU initially, will move to device when needed

    # Calculate class weights from a DataLoader
    def _calculate_weights_from_dataloader(self):
        train_dataset = self.ds_train.dataset
        class_counts = torch.zeros(len(self.classes))
        for _, label in train_dataset:
            class_counts[label] += 1
        
        # Calculate inverse frequency weights
        weights = 1.0 / (class_counts + 1e-6)
        weights_normalized = weights / weights.sum()
        
        return weights_normalized, class_counts

    # For standalone train/val folders
    def _calculate_weights_from_folder(self, folder_path):
        class_counts = []
        for class_dir in sorted(Path(folder_path).iterdir()):
            if class_dir.is_dir():
                class_counts.append(len(list(class_dir.glob("*.*"))))
        
        counts = torch.tensor(class_counts, dtype=torch.float32)
        
        # Calculate inverse frequency weights (higher weight for minority class)
        weights = 1.0 / (counts + 1e-6)
        weights_normalized = weights / weights.sum()
        
        return weights_normalized, counts

    # Print both class distribution and weights
    def _print_class_analysis(self, class_counts, class_weights):
        """Print comprehensive class analysis including distribution and weights"""
        total = class_counts.sum().item()
        
        print("\n" + "="*60)
        print("CLASS DISTRIBUTION ANALYSIS")
        print("="*60)
        
        # Print actual class distribution
        print("\n📊 ACTUAL CLASS DISTRIBUTION:")
        max_len = max(len(cls) for cls in self.classes)
        for i, cls in enumerate(self.classes):
            count = int(class_counts[i].item())
            percentage = (class_counts[i] / total) * 100
            print(f"  {cls.ljust(max_len)} : {count:6d} images ({percentage:.2f}%)")
        print(f"  {'Total'.ljust(max_len)} : {int(total):6d} images (100.00%)")
        
        # Print loss weights
        print("\n⚖️  LOSS FUNCTION WEIGHTS (for CrossEntropyLoss):")
        print("  Note: Higher weight = more importance in loss calculation")
        for i, cls in enumerate(self.classes):
            weight_pct = class_weights[i].item() * 100
            print(f"  {cls.ljust(max_len)} : {weight_pct:.2f}% weight")
        
        # Print explanation
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
        print("="*60 + "\n")
        
        # Log to TensorBoard
        for i, cls in enumerate(self.classes):
            self.writer.add_scalar(f'Data/class_count/{cls}', class_counts[i].item(), 0)
            self.writer.add_scalar(f'Data/class_weight/{cls}', class_weights[i].item(), 0)

    # Print only class distribution (for non-weighted loss)
    def _print_class_distribution(self, class_counts):
        """Print class distribution when not using weighted loss"""
        total = class_counts.sum().item()
        
        print("\n📊 CLASS DISTRIBUTION:")
        max_len = max(len(cls) for cls in self.classes)
        for i, cls in enumerate(self.classes):
            count = int(class_counts[i].item())
            percentage = (class_counts[i] / total) * 100
            print(f"  {cls.ljust(max_len)} : {count:6d} images ({percentage:.2f}%)")
        print(f"  {'Total'.ljust(max_len)} : {int(total):6d} images")
        
        # Log to TensorBoard
        for i, cls in enumerate(self.classes):
            self.writer.add_scalar(f'Data/class_count/{cls}', class_counts[i].item(), 0)

    # Calculate weighted accuracy if using weighted loss, otherwise standard accuracy
    def _calculate_weighted_accuracy(self, preds, labels):
        if self.use_weighted_loss:
            # Ensure class weights are on the same device as labels
            if self.class_weights.device != labels.device:
                self.class_weights = self.class_weights.to(labels.device)
            
            # Calculate accuracy directly on GPU
            correct = (preds == labels).float()
            batch_weights = self.class_weights[labels]
            
            # Calculate weighted accuracy
            weighted_accuracy = (correct * batch_weights).sum().item()
            total_weight = batch_weights.sum().item()
            
            return weighted_accuracy, total_weight
        else:
            # Standard accuracy calculation
            accuracy = (preds == labels).sum().item()
            return accuracy, labels.size(0)

    # Generate metrics plot at the end of a training
    def _plot_metrics(self, history, plot_path, show_plot=True, save_plot=True):
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
            plt.savefig(str(plot_path / f"train_metrics{weight_suffix}"), bbox_inches='tight', dpi=300)
            plt.close()
        if show_plot:
            plt.show()

    # Creates a styled confusion matrix plot
    def _plot_confusion_matrix(self, cm, class_names=None, epoch=None):
        fig, ax = plt.subplots(figsize=(8, 8))

        # Normalize and plot
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)

        # Customize fonts
        title_font = {'size': 14, 'weight': 'bold'}
        label_font = {'size': 12}
        tick_font = {'size': 12}
        text_font = {'size': 12}

        # Add labels/title
        ax.set_xlabel('Predicted Label', fontdict=label_font)
        ax.set_ylabel('True Label', fontdict=label_font)
        title = f'Confusion Matrix (Epoch {epoch+1})' if epoch else 'Confusion Matrix'
        if self.use_weighted_loss:
            title += ' - Weighted Loss'
        ax.set_title(title, fontdict=title_font)
        
        # Class names
        if class_names:
            ax.set_xticks(np.arange(len(class_names)))
            ax.set_yticks(np.arange(len(class_names)))
            ax.set_xticklabels(class_names, rotation=45, ha="right", fontdict=tick_font)
            ax.set_yticklabels(class_names, fontdict=tick_font)

        # Annotations
        thresh = cm_normalized.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, 
                    f"{cm_normalized[i,j]:.1%}\n({cm[i,j]})", 
                    ha="center", va="center",
                    color="white" if cm_normalized[i,j] > thresh else "black",
                    fontsize=text_font['size'])
                
        # Colorbar
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=12)
        plt.tight_layout()

        return fig
    
    def debug_real_validation(self, val_loader):
        """Debug what's happening during real image validation"""
        print("\n=== DEBUG REAL IMAGE VALIDATION ===")
        
        # Get one batch
        images, labels = next(iter(val_loader))
        
        print(f"Batch size: {images.shape[0]}")
        print(f"Image shape: {images.shape}")
        
        # Check label distribution
        unique, counts = torch.unique(labels, return_counts=True)
        print(f"Labels in batch: KO={counts[0].item() if 0 in unique else 0}, "
            f"WT={counts[1].item() if 1 in unique else 0}")
        
        # Make predictions
        with torch.no_grad():
            images = images.to(self.device)
            outputs = self.model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            print(f"\nFirst 5 predictions:")
            for i in range(min(5, len(predictions))):
                true_label = "KO" if labels[i] == 0 else "WT"
                pred_label = "KO" if predictions[i] == 0 else "WT"
                ko_prob = probabilities[i][0].item()
                wt_prob = probabilities[i][1].item()
                print(f"  Image {i}: True={true_label}, Pred={pred_label}, "
                    f"P(KO)={ko_prob:.3f}, P(WT)={wt_prob:.3f}")
        
        # Check if model outputs are extremely confident
        print(f"\nPrediction confidence analysis:")
        print(f"  Mean WT probability: {probabilities[:, 1].mean().item():.3f}")
        print(f"  Mean KO probability: {probabilities[:, 0].mean().item():.3f}")
        print(f"  % predicting WT: {(predictions == 1).float().mean().item():.1%}")
        
        return predictions, labels

    ##################
    # TRAIN FUNCTION #
    ##################

    def train(self, chckpt_pth, plot_pth):
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

        # Track best accuracy for checkpoint saving
        best_accuracy = 0.0

        # Iterate over epochs
        for epoch in range(self.num_epochs):

            # Console output:
            print(f"\n>> Epoch [{epoch+1}/{self.num_epochs}]:")

            #################
            # Training loop #
            #################

            self.cnn.train()
            weighted_train_accuracy = 0.0  # Separate accumulator for weighted
            standard_train_accuracy = 0.0  # Separate accumulator for standard  
            train_loss = 0.0
            total_train_weight = 0.0
            actual_train_samples = 0  # Track actual samples for standard accuracy

            # Ensure class weights are on the correct device before training loop
            if self.use_weighted_loss and self.class_weights.device != self.device:
                self.class_weights = self.class_weights.to(self.device)

            with tqdm(self.ds_train, unit="batch") as tepoch:
                for batch_idx, (images, labels) in enumerate(tepoch):
                    tepoch.set_description("Train")

                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    self.optimizer.zero_grad()
                    with autocast(device_type='cuda', enabled=self.device.type == 'cuda'):
                        outputs = self.cnn(images)
                        loss = self.loss_function(outputs, labels)
                    
                    # Scale loss and backpropagate
                    self.scaler.scale(loss).backward()
                    
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.cnn.parameters(), max_norm=1.0)
                    
                    # Update weights
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    
                    # Check for NaN/inf values
                    if torch.isnan(loss).any():
                        raise ValueError("NaN loss detected during training")
                    
                    # Calculate train loss and accuracy
                    train_loss += loss.item() * images.size(0)
                    _, preds = torch.max(outputs, 1)
                    
                    # Calculate BOTH weighted and standard accuracy
                    if self.use_weighted_loss:
                        # Weighted accuracy calculation
                        batch_weighted_accuracy, batch_weight = self._calculate_weighted_accuracy(preds, labels)
                        weighted_train_accuracy += batch_weighted_accuracy
                        total_train_weight += batch_weight
                        
                        # Also calculate standard accuracy for this batch
                        batch_standard_accuracy = (preds == labels).sum().item()
                        standard_train_accuracy += batch_standard_accuracy
                    else:
                        # Standard accuracy calculation
                        batch_standard_accuracy, _ = self._calculate_weighted_accuracy(preds, labels)
                        standard_train_accuracy += batch_standard_accuracy
                        weighted_train_accuracy = standard_train_accuracy  # Same for non-weighted
                    
                    # Track actual samples for standard accuracy calculation
                    actual_train_samples += labels.size(0)

                    # Clear memory every 10 batches
                    if batch_idx % 10 == 0:
                        torch.cuda.empty_cache()

            # Update learning rate and store training metrics
            self.scheduler.step()

            # Calculate final training accuracies with PROPER sample counting
            if self.use_weighted_loss:
                # Weighted accuracy uses total_weight
                if total_train_weight > 0:
                    final_weighted_train_accuracy = weighted_train_accuracy / total_train_weight
                else:
                    final_weighted_train_accuracy = 0.0
                
                # Standard accuracy uses actual sample count
                if actual_train_samples > 0:
                    final_standard_train_accuracy = standard_train_accuracy / actual_train_samples
                else:
                    final_standard_train_accuracy = 0.0
            else:
                # For standard loss, use actual samples processed
                if actual_train_samples > 0:
                    final_standard_train_accuracy = standard_train_accuracy / actual_train_samples
                    final_weighted_train_accuracy = final_standard_train_accuracy  # Same for non-weighted
                else:
                    final_standard_train_accuracy = 0.0
                    final_weighted_train_accuracy = 0.0

            train_loss = train_loss / len(self.ds_train.dataset)  # Loss calculation is fine as-is
            current_lr = self.scheduler.get_last_lr()[0]

            # Console output: Print training summary - CONDITIONAL DISPLAY
            if self.use_weighted_loss:
                print(f"> Train Loss: {train_loss:.5f} | Weighted Train Acc: {final_weighted_train_accuracy:.2f} | Standard Train Acc: {final_standard_train_accuracy:.2f} | Learning Rate: {current_lr:.6f}")
            else:
                print(f"> Train Loss: {train_loss:.5f} | Train Acc: {final_standard_train_accuracy:.2f} | Learning Rate: {current_lr:.6f}")

            # Add training metrics to history
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

            # Ensure class weights are on the correct device before validation loop
            if self.use_weighted_loss and self.class_weights.device != self.device:
                self.class_weights = self.class_weights.to(self.device)

            # Initialize per-class counters
            class_correct = {i: 0 for i in range(len(self.classes))}
            class_total = {i: 0 for i in range(len(self.classes))}

            with torch.inference_mode():
                with tqdm(self.ds_val, unit="batch") as tepoch:
                    tepoch.set_description("Valid")
                    for images, labels in tepoch:

                        images, labels = images.to(self.device), labels.to(self.device)
                        
                        with autocast(device_type='cuda', enabled=self.device.type == 'cuda'):
                            outputs = self.cnn(images)
                            loss = self.loss_function(outputs, labels)
                            probs = torch.softmax(outputs.float(), dim=1)
                        
                        val_loss += loss.item() * images.size(0)
                        _, preds = torch.max(outputs, 1)
                        
                        # Use weighted or standard accuracy calculation
                        batch_accuracy, batch_weight = self._calculate_weighted_accuracy(preds, labels)
                        val_accuracy += batch_accuracy
                        total_val_weight += batch_weight
                        
                        all_probs.append(probs.cpu().detach())
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())

                        # Calculate per-class accuracy
                        for i in range(len(self.classes)):
                            class_mask = (labels.cpu() == i)
                            class_correct[i] += (preds[class_mask] == labels[class_mask]).sum().item()
                            class_total[i] += class_mask.sum().item()

            # Calculate final validation metrics with safety check
            if self.use_weighted_loss:
                # Avoid division by zero
                if total_val_weight > 0:
                    weighted_val_accuracy = val_accuracy / total_val_weight
                else:
                    weighted_val_accuracy = 0.0
            else:
                weighted_val_accuracy = val_accuracy / len(self.ds_val.dataset)
                
            val_loss = val_loss / len(self.ds_val.dataset)

            # Calculate per-class accuracies
            class_accuracies = {}
            for i in range(len(self.classes)):
                if class_total[i] > 0:
                    # Calculate ACTUAL per-class accuracy (not weighted)
                    class_accuracies[self.classes[i]] = class_correct[i] / class_total[i]
                else:
                    class_accuracies[self.classes[i]] = 0.0

            # Calculate standard (unweighted) accuracy directly from predictions
            standard_val_accuracy = (np.array(all_preds) == np.array(all_labels)).mean()

            # Format per-class accuracies for display
            class_acc_str = " | ".join([f"{acc:.2f} {cls}" for cls, acc in class_accuracies.items()])
            
            # Create and log confusion matrix
            cm = confusion_matrix(all_labels, all_preds)
            cm_fig = self._plot_confusion_matrix(cm, class_names=self.classes, epoch=epoch)
            self.writer.add_figure('Confusion Matrix', cm_fig, epoch, close=True)
            plt.close(cm_fig)
            
            # Convert to numpy arrays safely
            with torch.no_grad():
                all_probs = torch.cat(all_probs).numpy()
            all_labels = np.array(all_labels)
            all_preds = np.array(all_preds)

            # Convert to class indices if one-hot encoded
            if all_labels.ndim > 1 and all_labels.shape[1] > 1:
                all_labels = np.argmax(all_labels, axis=1)        

            # Calculate classification metrics
            f1_macro = f1_score(all_labels, all_preds, average='macro')
            f1_weighted = f1_score(all_labels, all_preds, average='weighted')

            # Initialize metric dictionaries
            fpr, tpr, roc_auc = {}, {}, {}
            precision, recall, average_precision = {}, {}, {}

            # Calculate per-class metrics with checks
            for i in range(len(self.classes)):
                class_mask = (all_labels == i)
                if np.sum(class_mask) > 0:
                    fpr[i], tpr[i], _ = roc_curve(class_mask, all_probs[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                    precision[i], recall[i], _ = precision_recall_curve(class_mask, all_probs[:, i])
                    average_precision[i] = average_precision_score(class_mask, all_probs[:, i])

            # Calculate weighted metrics
            if len(self.classes) == 2:
                roc_auc_weighted = roc_auc_score(all_labels, all_probs[:, 1])
            else:
                roc_auc_weighted = roc_auc_score(
                    all_labels,
                    all_probs,
                    multi_class='ovr',
                    average='weighted'
                )
            ap_weighted = np.mean(list(average_precision.values())) if average_precision else 0

            # Log to TensorBoard - ALL METRICS (regardless of weighted setting)
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Accuracy/train', final_weighted_train_accuracy if self.use_weighted_loss else final_standard_train_accuracy, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/val', weighted_val_accuracy if self.use_weighted_loss else standard_val_accuracy, epoch)

            # Log both weighted and standard accuracy
            self.writer.add_scalar('Accuracy/val_standard', standard_val_accuracy, epoch)
            if self.use_weighted_loss:
                self.writer.add_scalar('Accuracy/train_standard', standard_train_accuracy, epoch)

            self.writer.add_scalar('Metrics/LR', current_lr, epoch)
            self.writer.add_scalars('Metrics/F1', {'Macro': f1_macro, 'Weighted': f1_weighted}, epoch)
            self.writer.add_scalar('Metrics/AUC', roc_auc_weighted, epoch)
            self.writer.add_scalar('Metrics/AP', ap_weighted, epoch)

            # Log per-class metrics
            for i, class_name in enumerate(self.classes):
                if i in roc_auc:  # Only if class exists in this batch
                    self.writer.add_scalar(f'Metrics/AUC/{class_name}', roc_auc[i], epoch)
                if i in average_precision:
                    self.writer.add_scalar(f'Metrics/AP/{class_name}', average_precision[i], epoch)
                if class_name in class_accuracies:
                    self.writer.add_scalar(f'Accuracy/class/{class_name}', class_accuracies[class_name], epoch)

            # Log class distribution (useful for understanding weighted metrics)
            for i, class_name in enumerate(self.classes):
                if i in class_total:
                    self.writer.add_scalar(f'Data/class_count/{class_name}', class_total[i], epoch)

            # Log GPU memory usage
            if torch.cuda.is_available():
                mem_usage = torch.cuda.memory_reserved() / self.total_gpu_memory
                self.writer.add_scalar('System/GPU_Memory', mem_usage, epoch)

            # Print validation summary - CONDITIONAL DISPLAY
            if self.use_weighted_loss:
                print(f"> Val Loss: {val_loss:.5f} | Weighted Val Acc: {weighted_val_accuracy:.2f} | Standard Val Acc: {standard_val_accuracy:.2f} | Per-class: ({class_acc_str})")
                print(f"> F1 (Macro): {f1_macro:.4f} | F1 (Weighted): {f1_weighted:.4f} | AUC: {roc_auc_weighted:.4f} | AP: {ap_weighted:.4f}")
            else:
                print(f"> Val Loss: {val_loss:.5f} | Val Acc: {standard_val_accuracy:.2f} | Per-class: ({class_acc_str})")
                print(f"> F1 (Macro): {f1_macro:.4f} | AUC: {roc_auc_weighted:.4f} | AP: {ap_weighted:.4f}")

            # Store validation metrics
            current_val_accuracy = weighted_val_accuracy if self.use_weighted_loss else standard_val_accuracy
            history["val_acc"].append(current_val_accuracy)
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

            # Save best mode
            accuracy_for_checkpoint = weighted_val_accuracy if self.use_weighted_loss else standard_val_accuracy
            accuracy_type = "weighted" if self.use_weighted_loss else "standard"

            # Store the old best accuracy to check if it improved
            old_best_accuracy = best_accuracy

            # Call save_weights - it will handle the threshold check internally
            best_accuracy = self.cnn_wrapper.save_weights(
                accuracy_for_checkpoint, 
                best_accuracy, 
                epoch, 
                chckpt_pth
            )

            # Only print message if a checkpoint was actually saved
            if best_accuracy > old_best_accuracy:
                print(f"Model with {accuracy_type} validation accuracy {best_accuracy:.2f} saved!")

        # Final cleanup and plotting
        self.writer.close()
        self._plot_metrics(history, plot_pth, show_plot=False, save_plot=True)
        
        return history