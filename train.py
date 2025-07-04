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
        # Optimizer momentum
        self.train_momentum = setting["train_momentum"] 

        # Optmizer, learning rate scheduler and loss function
        # ADAM:
        # amsgrad=True: Avoids LR instability
        # self.optimizer = Adam(model.parameters(), lr=self.init_lr, weight_decay=self.weight_decay, amsgrad=True)
        
        # SGD:
        self.optimizer = SGD(
            self.cnn.parameters(), 
            lr=self.init_lr, 
            momentum=self.train_momentum,
            weight_decay=self.weight_decay
        )
        # This loss function combines nn.LogSoftmax() and nn.NLLLoss() in one single class
        self.loss_function = nn.CrossEntropyLoss()
        # With label smoothing:
        # self.loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Learning rate scheduler:
        # CosineAnnealingLR:
        self.scheduler_CA = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.num_epochs - self.warmup_epochs, # Adjust for warmup
            eta_min=self.lr_eta_min
        )
        # Warmup scheduler
        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, 
            start_factor=0.01,  # Start at 1% of target LR
            total_iters=self.warmup_epochs       # Ramp over 5 epochs
        )
        # Then chain with your CosineAnnealingLR
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[self.warmup_scheduler, self.scheduler_CA],
            milestones=[self.warmup_epochs]
        )

        # Add gradient scaler for mixed precision training
        # Initialize once before training
        self.scaler = GradScaler()

    #############################################################################################################
    # METHODS:

    # Plots accuracy, loss, and learning rate after training
    def plot_metrics(self, history, plot_path, show_plot=True, save_plot=True):
        # Number of epochs
        epochs_range = range(1, len(history["train_loss"]) + 1)
        # Draw plots
        plt.figure(figsize=(15, 5))
        # Accuracy plot:
        plt.subplot(1, 3, 1)
        plt.plot(epochs_range, history["train_acc"], label='Training Accuracy', color='green')
        plt.plot(epochs_range, history["val_acc"], label='Validation Accuracy', color='red')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')
        # Loss plot:
        plt.subplot(1, 3, 2)
        # Set the range of y-axis
        plt.ylim(0, 5)
        plt.plot(epochs_range, history["train_loss"], label='Training Loss', color='green')
        plt.plot(epochs_range, history["val_loss"], label='Validation Loss', color='red')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        # Learning rate plot:
        plt.subplot(1, 3, 3)
        # convert y-axis to Logarithmic scale
        # plt.yscale("log")
        plt.plot(epochs_range, history["lr"], label='Learning Rate', color='blue')
        plt.legend(loc='upper right')
        plt.title('Learning Rate')
        # Reduce unnecessary whitespaces around the plots
        # https://stackoverflow.com/questions/4042192/reduce-left-and-right-margins-in-matplotlib-plot
        plt.tight_layout()
        # Save plot
        if(save_plot):
            plt.savefig(str(plot_path / "train_metrics"), bbox_inches='tight')  # Modified path handling
        # Show and save plot
        if(show_plot):
            plt.show()

    def train(self, chckpt_pth, plot_pth):

        # Save best accuracy for model saving
        best_accuracy = 0.0
        # Track train and validation accuracy, train and accuracy loss and learning rate every epoch
        history = {"train_acc": [], "train_loss": [], "val_acc": [], "val_loss": [], "lr": []}

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
                for images, labels in tepoch:
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

                # Set learning rate scheduler
                self.scheduler.step()

                # Calculate train loss and accuracy   
                train_accuracy = train_accuracy / self.num_train_img
                train_loss = train_loss / self.num_train_img
                lr = self.scheduler.get_last_lr()[0]

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
            
            validation_accuracy = validation_accuracy / self.num_val_img
            validation_loss = validation_loss / self.num_val_img

            print(f"> val_loss: {validation_loss:.5f}, val_acc: {validation_accuracy:.2f}")

            # Save validation loss and accuracy
            history["val_loss"].append(validation_loss)
            history["val_acc"].append(validation_accuracy)

            ###################
            # Save best model #
            ###################

            best_accuracy = self.cnn_wrapper.save_weights(validation_accuracy, best_accuracy, epoch, chckpt_pth)

        ################
        # Metircs plot #
        ################

        self.plot_metrics(history, plot_pth, show_plot=False, save_plot=True)

        return history