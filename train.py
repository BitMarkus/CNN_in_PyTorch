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
        # To use tensorboard for training, navigate to the root of the project folder
        # Type in CMD in the the adress line
        # Run the following command: tensorboard --logdir=logs/ --host=localhost (Optional: --reload_interval 30)
        # Open http://localhost:6006/ in web browser
        self.writer = SummaryWriter(f"logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        
        # Enhanced TensorBoard layout
        custom_layout = {
            'Accuracy': {
                'Training Accuracy': ['Scalar', 'Accuracy/Train'],
                'Validation Accuracy': ['Scalar', 'Accuracy/Val'],
            },
            'Loss': {
                'Training Loss': ['Scalar', 'Loss/Train'],
                'Validation Loss': ['Scalar', 'Loss/Val'],
            },
            'Metrics': {
                'F1 Scores': ['Multiline', ['Metrics/F1/Macro', 'Metrics/F1/Weighted']],
                'AUC': ['Scalar', 'Metrics/AUC'],
                'AP': ['Scalar', 'Metrics/AP'],
                'Learning Rate': ['Scalar', 'Metrics/LR'],
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

        # Class names
        self.classes = setting["classes"] 
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
        # Options: "SGD" and "ADAM/ADAMW"
        self.optimizer_type = setting["train_optimizer_type"]
        self.sgd_momentum = setting["train_sgd_momentum"] # SGD
        self.sgd_use_nesterov = setting["train_sgd_use_nesterov"] # SGD
        self.adam_beta1 = setting["train_adam_beta1"]   # ADAM/ADAMW
        self.adam_beta2 = setting["train_adam_beta2"]   # ADAM/ADAMW 
        # Loss function 
        self.use_weighted_loss = setting["train_use_weighted_loss"]
        # Validation split settings (False or percentage 0.0-1.0)
        # Validation split from training dataset
        self.val_from_train_split = setting["ds_val_from_train_split"]
        # Validation split from test dataset
        self.val_from_test_split = setting["ds_val_from_test_split"]

        # Optmizer, learning rate scheduler and loss function
        if(self.optimizer_type == "ADAM"):
            # ADAM:
            # amsgrad=True: Avoids LR instability
            self.optimizer = Adam(
                self.cnn.parameters(),
                lr=self.init_lr, 
                weight_decay=self.weight_decay, 
                betas=(self.adam_beta1, self.adam_beta2),
                amsgrad=False
            )
        elif(self.optimizer_type == "ADAMW"):
            # ADAMW:
            self.optimizer = AdamW(
                self.cnn.parameters(),
                lr=self.init_lr, 
                weight_decay=self.weight_decay, 
                betas=(self.adam_beta1, self.adam_beta2),
                amsgrad=False
            )
        elif(self.optimizer_type == "SGD"):
            # SGD:
            self.optimizer = SGD(
                self.cnn.parameters(), 
                lr=self.init_lr, 
                momentum=self.sgd_momentum,
                weight_decay=self.weight_decay,
                nesterov=self.sgd_use_nesterov
            )

        # Loss function setup
        if self.use_weighted_loss:
            print("\nLoss function: Using weighted loss function. Calculating class weights...")
            if self.val_from_train_split is not False:
                # Case 1: Validation split from training data
                # Access only training samples (excludes validation split)
                class_weights = self._calculate_weights_from_dataloader()
            else:
                # Case 2: Validation is separate (e.g., from test or own folder)
                # Counts files per class without loading images (fast)
                class_weights = self._calculate_weights_from_folder(dataset.pth_train)
            print(f"Class weights: {dict(zip(dataset.classes, class_weights.tolist()))}")

            self.loss_function = nn.CrossEntropyLoss(
                weight=class_weights.to(device),
                label_smoothing=self.label_smoothing
        )
        else:
            # Case 3: Using non-weighted loss function
            print("\nLoss function: Using non-weighted loss function.")
            self.loss_function = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

        # Learning rate scheduler:
        # CosineAnnealingLR:
        self.scheduler_CA = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.num_epochs - self.warmup_epochs, # Adjust for warmup
            eta_min=self.lr_eta_min,
        )
        # Warmup scheduler
        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, 
            start_factor=0.01,  # Start at 1% of target LR
            total_iters=self.warmup_epochs ,      # Ramp over 5 epochs
        )
        # Then chain with CosineAnnealingLR
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

    # Generate metrics plot at the end of a training
    def _plot_metrics(self, history, plot_path, show_plot=True, save_plot=True):
        plt.figure(figsize=(24, 12))
        
        # 1. Accuracy Plot
        plt.subplot(2, 3, 1)
        epochs_range = range(1, len(history["train_acc"]) + 1)
        plt.plot(epochs_range, history["train_acc"], 'g-', label='Training')
        plt.plot(epochs_range, history["val_acc"], 'r-', label='Validation')
        plt.title('Accuracy')
        plt.legend()
        
        # 2. Loss Plot
        plt.subplot(2, 3, 2)
        plt.plot(epochs_range, history["train_loss"], 'g-', label='Training')
        plt.plot(epochs_range, history["val_loss"], 'r-', label='Validation')
        plt.title('Loss')
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
            plt.savefig(str(plot_path / "train_metrics"), bbox_inches='tight', dpi=300)
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
        ax.set_title(f'Confusion Matrix (Epoch {epoch+1})' if epoch else 'Confusion Matrix', 
                    fontdict=title_font)
        
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
    
    # Calculate class weights from a DataLoader
    # For when validation is split from training data
    def _calculate_weights_from_dataloader(self):
        train_dataset = self.ds_train.dataset
        class_counts = torch.zeros(len(self.classes))
        for _, label in train_dataset:
            class_counts[label] += 1
        weights = 1.0 / (class_counts + 1e-6)
        return weights / weights.sum()
    # For standalone train/val folders
    def _calculate_weights_from_folder(self, folder_path):
        class_counts = []
        for class_dir in Path(folder_path).iterdir():
            if class_dir.is_dir():
                class_counts.append(len(list(class_dir.glob("*.*"))))
        weights = 1.0 / (torch.tensor(class_counts, dtype=torch.float32) + 1e-6)
        return weights / weights.sum()

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
            train_accuracy = 0.0
            train_loss = 0.0
            
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
                    train_accuracy += torch.sum(preds == labels.data).item()

                    # Clear memory every 10 batches
                    if batch_idx % 10 == 0:
                        torch.cuda.empty_cache()

            # Update learning rate and store training metrics
            self.scheduler.step()
            train_accuracy /= len(self.ds_train.dataset)
            train_loss /= len(self.ds_train.dataset)
            current_lr = self.scheduler.get_last_lr()[0]

            # Console output: Print training summary
            print(f"> Train Loss: {train_loss:.5f} | Train Acc: {train_accuracy:.2f} | Learning Rate: {current_lr:.6f}")
            
            # Add training metrics to history
            history["train_acc"].append(train_accuracy)
            history["train_loss"].append(train_loss)
            history["lr"].append(current_lr)

            ###################
            # Validation loop #
            ###################

            self.cnn.eval()
            val_accuracy = 0.0
            val_loss = 0.0
            total_samples = 0
            all_preds = []
            all_labels = []
            all_probs = []

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
                        val_accuracy += (preds == labels).sum().item()
                        total_samples += labels.size(0)
                        
                        all_probs.append(probs.cpu().detach())
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())

            # Calculate metrics
            val_accuracy /= total_samples
            val_loss /= total_samples
            
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
                if np.sum(class_mask) > 0:  # Only if class exists in batch
                    fpr[i], tpr[i], _ = roc_curve(class_mask, all_probs[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                    precision[i], recall[i], _ = precision_recall_curve(class_mask, all_probs[:, i])
                    average_precision[i] = average_precision_score(class_mask, all_probs[:, i])

            # Calculate weighted metrics: HANDLE BINARY VS MULTICLASS DIFFERENTLY!
            if len(self.classes) == 2:
                # Binary classification: use probabilities for positive class only
                roc_auc_weighted = roc_auc_score(all_labels, all_probs[:, 1])
            else:
                # Multiclass classification
                roc_auc_weighted = roc_auc_score(
                    all_labels,
                    all_probs,
                    multi_class='ovr',
                    average='weighted'
                )
            ap_weighted = np.mean(list(average_precision.values())) if average_precision else 0

            # Log to TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_accuracy, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/val', val_accuracy, epoch)
            self.writer.add_scalar('Metrics/LR', current_lr, epoch)
            self.writer.add_scalars('Metrics/F1', {'Macro': f1_macro, 'Weighted': f1_weighted}, epoch)
            self.writer.add_scalar('Metrics/AUC', roc_auc_weighted, epoch)
            self.writer.add_scalar('Metrics/AP', ap_weighted, epoch)
            
            # Log GPU memory usage
            if torch.cuda.is_available():
                mem_usage = torch.cuda.memory_reserved() / self.total_gpu_memory
                self.writer.add_scalar('System/GPU_Memory', mem_usage, epoch)

            # Print validation summary
            print(f"> Val Loss: {val_loss:.5f} | Val Acc: {val_accuracy:.2f}")
            print(f"> F1 (Macro): {f1_macro:.4f} | F1 (Weighted): {f1_weighted:.4f} | AUC: {roc_auc_weighted:.4f} | AP: {ap_weighted:.4f}")

            # Store validation metrics
            history["val_acc"].append(val_accuracy)
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

            # Save best model
            best_accuracy = self.cnn_wrapper.save_weights(
                val_accuracy, 
                best_accuracy, 
                epoch, 
                chckpt_pth
            )

        # Final cleanup and plotting
        self.writer.close()
        self._plot_metrics(history, plot_pth, show_plot=False, save_plot=True)
        
        return history