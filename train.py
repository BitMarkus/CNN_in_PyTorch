######################
# Class for training #
######################

from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
from torch import nn
from tqdm import tqdm
import torch
from datetime import datetime
import matplotlib.pyplot as plt
# Own modules
from settings import setting
import functions as fn

class Train():

    #############################################################################################################
    # CONSTRUCTOR:
    
    def __init__(self, model, dataset):
        # Model
        self.model = model
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
        self.lr_step_size = setting["train_lr_step_size"] 
        self.lr_multiplier = setting["train_lr_multiplier"]  
        # Weight decay
        self.weight_decay = setting["train_weight_decay"] 
        # Checkpoint saving options
        self.save_checkpoint = setting["chckpt_save"] 
        self.checkpoint_min_acc = setting["chckpt_min_acc"] 
        self.pth_checkpoint = setting["pth_checkpoint"]

        # Optmizer, learning rate scheduler and loss function
        # self.optimizer = Adam(model.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
        self.optimizer = SGD(model.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
        self.loss_function = nn.CrossEntropyLoss()
        # This lr scheduler takes the initial lr for the optimizer
        # and multiplies it with gamma (default = 0.1) every step_size
        # When last_epoch=-1, sets initial lr as lr
        # https://pytorch.org/docs/stable/optim.html
        # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html#torch.optim.lr_scheduler.StepLR
        self.scheduler = StepLR(self.optimizer, step_size=self.lr_step_size, gamma=self.lr_multiplier) 

    #############################################################################################################
    # METHODS:

    def train(self):

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
            self.model.train()
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
                    # Forward          
                    outputs = self.model(images)
                    loss = self.loss_function(outputs, labels)
                    # Backward
                    loss.backward()
                    self.optimizer.step()            
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
            self.model.eval()
            # Set variables for validation accuracy and loss
            validation_accuracy = 0.0
            validation_loss = 0.0

            # Iterate over batches
            # for i, (images,labels) in enumerate(tqdm(self.ds_val)):
            with tqdm(self.ds_val, unit="batch") as tepoch:
                for images, labels in tepoch:
                    tepoch.set_description("Valid")

                    # Send images and labels to gpu or cpu
                    if torch.cuda.is_available():
                        images, labels = images.cuda(), labels.cuda()
                        
                    outputs = self.model(images)
                    loss = self.loss_function(outputs, labels)
                    validation_loss += loss.cpu().data * images.size(0)
                    _,prediction = torch.max(outputs.data, 1)
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

            # Save checkpoint if the accuracy has improved AND
            # if it is higher than a predefined percentage (min_acc_for_saving) AND
            # if models should be saved at all
            if(validation_accuracy > best_accuracy and 
                validation_accuracy > self.checkpoint_min_acc and
                self.save_checkpoint):
                # Datetime for saved files
                current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M")
                print(f"Model with test accuracy {validation_accuracy:.2f} saved!")
                # Add datetime, epoch and validation accuracy to the filename and save model
                filename = f'{self.pth_checkpoint}{current_datetime}_checkpoint_e{epoch+1}_vacc{validation_accuracy*100:.0f}.model'
                torch.save(self.model.state_dict(), filename)

                # Update best accuracy
                best_accuracy = validation_accuracy    

        return history
    
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
        plt.yscale("log")
        plt.plot(epochs_range, history["lr"], label='Learning Rate', color='blue')
        plt.legend(loc='upper right')
        plt.title('Learning Rate')
        # Reduce unnecessary whitespaces around the plots
        # https://stackoverflow.com/questions/4042192/reduce-left-and-right-margins-in-matplotlib-plot
        plt.tight_layout()
        # Save plot
        if(save_plot):
            fn.save_plot_to_drive(plot_path, "train_metrics")
        # Show and save plot
        if(show_plot):
            plt.show()
