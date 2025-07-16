######################
# Class for training #
######################

from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
from torch import nn
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast 
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, confusion_matrix
from datetime import datetime
import numpy as np
# Own modules
from settings import setting
import functions as fn

class Train():

    #############################################################################################################
    # CONSTRUCTOR:
    
    def __init__(self, cnn_wrapper, dataset):
        # CNN object for methods
        self.cnn_wrapper = cnn_wrapper
        # Model
        self.cnn = cnn_wrapper.model
        # Initialize TensorBoard writer
        # To use tensorboard for training, navigate to the root of the project folder
        # Type in CMD in the the adress line
        # Run the following command: tensorboard --logdir=logs/ --host=localhost (Optional: --reload_interval 30)
        # Open http://localhost:6006/ in web browser
        self.writer = SummaryWriter(f"logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        # Create a custom layout for TensorBoard
        custom_layout = {
            'Loss': {
                'TrainingLoss': ['Scalar', 'Loss/train'],
                'ValidationLoss': ['Scalar', 'Loss/val'],
            },
            'Accuracy': {
                'TrainingAccuracy': ['Scalar', 'Accuracy/train'],
                'ValidationAccuracy': ['Scalar', 'Accuracy/val'],
            },
            'Metrics': {
                'LearningRate': ['Scalar', 'Learning Rate'],
                'F1Score': ['Scalar', 'Metrics/F1'],
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
        # Options: "SGD" and "ADAM"
        self.optimizer_type = setting["train_optimizer_type"]
        self.sgd_momentum = setting["train_sgd_momentum"] # SGD
        self.sgd_use_nesterov = setting["train_sgd_use_nesterov"] # SGD
        self.adam_beta1 = setting["train_adam_beta1"]   # ADAM
        self.adam_beta2 = setting["train_adam_beta2"]   # ADAM 

        # Optmizer, learning rate scheduler and loss function
        if(self.optimizer_type == "ADAM"):
            # ADAM:
            # amsgrad=True: Avoids LR instability
            self.optimizer = Adam(
                self.cnn.parameters(),
                lr=self.init_lr, 
                weight_decay=self.weight_decay, 
                betas=(0.9, 0.999),
                amsgrad=True
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
        # This loss function combines nn.LogSoftmax() and nn.NLLLoss() in one single class
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
        # Then chain with your CosineAnnealingLR
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[self.warmup_scheduler, self.scheduler_CA],
            milestones=[self.warmup_epochs],
        )

        # Add gradient scaler for mixed precision training
        # Initialize once before training
        self.scaler = GradScaler()

    #############################################################################################################
    # METHODS:

    # Plots accuracy, loss, f1 score, and learning rate after training
    def _plot_metrics(self, history, plot_path, show_plot=True, save_plot=True):
        # Number of epochs
        epochs_range = range(1, len(history["train_loss"]) + 1)
        # Create 2x2 grid layout
        plt.figure(figsize=(15, 12))  # Increased height for 2x2 layout
        # Accuracy plot (top-left)
        plt.subplot(2, 2, 1)
        plt.plot(epochs_range, history["train_acc"], label='Training Accuracy', color='green')
        plt.plot(epochs_range, history["val_acc"], label='Validation Accuracy', color='red')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')
        # Loss plot (top-right)
        plt.subplot(2, 2, 2)
        plt.ylim(0, 5)  # Set the range of y-axis
        plt.plot(epochs_range, history["train_loss"], label='Training Loss', color='green')
        plt.plot(epochs_range, history["val_loss"], label='Validation Loss', color='red')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        # Learning rate plot (bottom-left)
        plt.subplot(2, 2, 3)
        plt.plot(epochs_range, history["lr"], label='Learning Rate', color='blue')
        plt.legend(loc='upper right')
        plt.title('Learning Rate')
        # F1 score plot (bottom-right)
        plt.subplot(2, 2, 4)
        plt.plot(epochs_range, history["f1"], label='F1 Score', color='red')
        plt.ylim(0, 1)  # F1 score ranges between 0 and 1
        plt.legend(loc='lower right')
        plt.title('F1 Score (Macro)')
        
        # Adjust layout
        plt.tight_layout()
        # Save plot
        if save_plot:
            plt.savefig(str(plot_path / "train_metrics"), bbox_inches='tight', dpi=300)
        # Show plot
        if show_plot:
            plt.show()

    # Creates a styled confusion matrix plot
    def _plot_confusion_matrix(self, cm, class_names=None, epoch=None):
        fig, ax = plt.subplots(figsize=(8, 8))
        # Normalize and plot
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
        # Customize fonts
        title_font = {'size': 20, 'weight': 'bold'}
        label_font = {'size': 20}
        tick_font = {'size': 20}
        text_font = {'size': 20}
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
                    fontdict=text_font)
        # Colorbar
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=20)
        
        plt.tight_layout()
        return fig

    def train(self, chckpt_pth, plot_pth):

        # Save best accuracy for model saving
        best_accuracy = 0.0
        # Track train and validation accuracy, train and accuracy loss and learning rate every epoch
        history = {"train_acc": [], "train_loss": [], "val_acc": [], "val_loss": [], "lr": [], "f1": []}

        # Iterate over epochs
        for epoch in range(self.num_epochs):

            print(f"\n>> Epoch [{epoch+1}/{self.num_epochs}]:")

            #################
            # Training loop #
            #################        

            # Switch model to train mode
            self.cnn.train()
            # Set variables for train accuracy and loss
            train_accuracy = 0.0
            train_loss = 0.0
            
            # Iterate over batches
            # https://towardsdatascience.com/training-models-with-a-progress-a-bar-2b664de3e13e
            with tqdm(self.ds_train, unit="batch") as tepoch:
                for batch_idx, (images, labels) in enumerate(tepoch):
                    tepoch.set_description("Train")

                    # Send images and labels to gpu or cpu
                    if torch.cuda.is_available():
                        images, labels = images.cuda(), labels.cuda()
                    
                    # Clear gradients
                    self.optimizer.zero_grad() 
                    # Enable mixed precision
                    with autocast():
                        # Forward          
                        outputs = self.cnn(images)
                        loss = self.loss_function(outputs, labels)
                    # Scales loss and backprops scaled gradients
                    self.scaler.scale(loss).backward()
                    # Unscales gradients + optimizer step (skips if gradients are inf/NaN)
                    self.scaler.step(self.optimizer)
                    # Updates scale for next iteration
                    self.scaler.update()
                    
                    train_loss += loss.cpu().data * images.size(0)
                    _, prediction = torch.max(outputs.data, 1)           
                    train_accuracy += int(torch.sum(prediction==labels.data))

                    # Tensorboard: Log batch-level metrics
                    if batch_idx % 50 == 0:  # Log every 50 batches
                        self.writer.add_scalar('Loss/train_batch', loss.item(), epoch * len(self.ds_train) + batch_idx)

                # Set learning rate scheduler
                self.scheduler.step()

                # Calculate train loss and accuracy   
                train_accuracy = train_accuracy / self.num_train_img
                train_loss = train_loss / self.num_train_img
                lr = self.scheduler.get_last_lr()[0]

                # Tensorboard: Log epoch-level training metrics
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Accuracy/train', train_accuracy, epoch)
                self.writer.add_scalar('Learning Rate', lr, epoch)              

                print(f"> train_loss: {train_loss:.5f}, train_acc: {train_accuracy:.2f}, lr: {lr:.6f}")

                # Save train loss, accuracy and learning rate
                history["train_loss"].append(train_loss)
                history["train_acc"].append(train_accuracy)
                history["lr"].append(lr)

            ###################
            # Evaluation loop #
            ###################

            # Switch model to evaluation mode
            self.cnn.eval()
            # Set variables for validation accuracy and loss
            validation_accuracy = 0.0
            validation_loss = 0.0
            all_preds = []  # Store all predictions for advanced metrics
            all_labels = []  # Store all ground truth labels
            
            # Clear cache before validation to save V-RAM
            torch.cuda.empty_cache()

            # Iterate over batches
            # torch.inference_mode(): More efficient than no_grad()
            with torch.inference_mode():
                with tqdm(self.ds_val, unit="batch") as tepoch:
                    for images, labels in tepoch:
                        tepoch.set_description("Valid")

                        # Send images and labels to gpu or cpu
                        if torch.cuda.is_available():
                            images, labels = images.cuda(), labels.cuda()
                        
                        # Explicit FP32 validation (no autocast)
                        outputs = self.cnn(images) # Let PyTorch handle dtype automatically
                        loss = self.loss_function(outputs, labels)
                        
                        validation_loss += loss.cpu().data * images.size(0)
                        _, prediction = torch.max(outputs.data, 1)
                        validation_accuracy += int(torch.sum(prediction==labels.data))
                        
                        # Store predictions and labels for advanced metrics
                        all_preds.extend(prediction.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
            
            # Calculate standard metrics
            validation_accuracy = validation_accuracy / self.num_val_img
            validation_loss = validation_loss / self.num_val_img
            # Log validation metrics to TensorBoard
            self.writer.add_scalar('Loss/val', validation_loss, epoch)
            self.writer.add_scalar('Accuracy/val', validation_accuracy, epoch)
            
            # Calculate F1 score and confusion matrix
            # Options for parameter average:
            # 'binary': 
            # Only reports F1 for the positive class in binary classification. Uses pos_label to specify which class is "positive".
            # Binary classification only (not for multiclass).
            # 'micro': 
            # Calculates global F1 by summing all TPs, FPs, FNs across classes, then computes F1. Class-agnostic. 
            # Imbalanced data where you care about overall performance (favors majority classes).
            # 'weighted': 
            # Like 'macro', but weights each class’s F1 by its support (number of true instances). 
            # Imbalanced data where you want to reflect class frequencies.
            # None: Returns F1 per class as an array. Debugging per-class performance.
            # 'macro': 
            # Computes F1 for each class independently, then takes the unweighted mean. Treats all classes equally.
            # Multiclass data where you want equal importance for all classes (ignores imbalance).
            f1 = f1_score(all_labels, all_preds, average='macro')
            cm = confusion_matrix(all_labels, all_preds)

            # Log advanced metrics to TensorBoard
            self.writer.add_scalar('Metrics/F1', f1, epoch)

            # Create and log confusion matrix figure
            cm_fig = self._plot_confusion_matrix(cm, class_names=self.classes, epoch=epoch)
            self.writer.add_figure('Confusion Matrix', cm_fig, epoch, close=True)  # Added close=True
            plt.close(cm_fig)  # Extra safety to prevent memory leaks

            print(f"> val_loss: {validation_loss:.5f}, val_acc: {validation_accuracy:.2f}, F1: {f1:.4f}")

            # Save validation loss and accuracy
            history["val_loss"].append(validation_loss)
            history["val_acc"].append(validation_accuracy)
            history["f1"].append(f1)  # Store F1 score in history

            ###################
            # Save best model #
            ###################

            best_accuracy = self.cnn_wrapper.save_weights(validation_accuracy, best_accuracy, epoch, chckpt_pth)

        ################
        # Metircs plot #
        ################

        # Tensorboard: Log hyperparameters and close writer
        self.writer.add_hparams(
            {"lr": self.init_lr, "batch_size": setting["ds_batch_size"], "momentum": self.sgd_momentum},
            {"hparam/val_accuracy": max(history["val_acc"]), "hparam/val_loss": min(history["val_loss"])},
        )
        self.writer.close()

        self._plot_metrics(history, plot_pth, show_plot=False, save_plot=True)

        return history