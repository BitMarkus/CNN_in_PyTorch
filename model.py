##################################
# Predefined model architectures #
##################################

import torchvision.models as models
import pathlib
from torch import nn
import torch.nn.init as init
import torch
from datetime import datetime
from tqdm import tqdm
from prettytable import PrettyTable
from PIL import Image
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
import os
from pathlib import Path
# Own modules
from custom_cnn import CustomCNN
import functions as fn
from settings import setting

class CNN_Model():

    #############################################################################################################
    # CONSTRUCTOR:
    
    def __init__(self):
        # Path to training images
        self.pth_train = setting["pth_train"]
        # Path to prediction images
        self.pth_pred = setting["pth_prediction"]
        # Path to checkpoints
        self.pth_checkpoint = setting["pth_checkpoint"]
        # Input shape data
        self.input_channels = setting["img_channels"]
        self.batch_size = setting["ds_batch_size"]
        self.img_width = setting["img_width"]
        self.img_height = setting["img_height"]
        self.input_size = (self.batch_size, self.input_channels, self.img_height, self.img_width)
        # Variable for model architecture name
        self.cnn_type = setting["cnn_type"] 
        # Variable to save if a model was alread loaded or not
        self.model_loaded = False   
        # Variable to save if a checkpoint for this model was already loaded
        self.checkpoint_loaded = False
        # Model
        self.model = None
        # Pretrained
        self.is_pretrained = setting["cnn_is_pretrained"] 
        # Initialization type for non-pretrained cnns
        self.initialization = setting["cnn_initialization"] 
        # Checkpoint saving options
        self.chckpt_min_acc = setting["chckpt_min_acc"] 
        self.chckpt_save = setting["chckpt_save"]
        self.chckpt_pth = setting["pth_checkpoint"]
        # heckpoint loading options
        # self.chckpt_weights_file = setting["chckpt_weights_file"] 
        # Get number of classes = number of output nodes
        # self.class_list = self.get_class_list()
        self.class_list = setting["classes"]   
        self.num_classes = len(self.class_list)

    #############################################################################################################
    # METHODS:

    def load_model(self, device):
        """
        Universal CNN loader for:
        - ResNet (18/34/50/101/152)
        - ResNeXt-101 (32x8d/64x4d)
        - AlexNet
        - VGG (11/13/16/19, with/without BN)
        - DenseNet (121/169/201/264)
        - EfficientNet (B0-B7)
        """
        # Supported models and their constructors
        model_dict = {
            # ResNet variants
            "resnet18": models.resnet18,
            "resnet34": models.resnet34,
            "resnet50": models.resnet50,
            "resnet101": models.resnet101,
            "resnet152": models.resnet152,
            # ResNeXt variants
            "resnext101_32x8d": models.resnext101_32x8d,
            "resnext101_64x4d": models.resnext101_64x4d,
            # AlexNet
            "alexnet": models.alexnet,
            # VGG variants
            "vgg11": models.vgg11,
            "vgg13": models.vgg13,
            "vgg16": models.vgg16,
            "vgg19": models.vgg19,
            "vgg11_bn": models.vgg11_bn,
            "vgg13_bn": models.vgg13_bn,
            "vgg16_bn": models.vgg16_bn,
            "vgg19_bn": models.vgg19_bn,
            # DenseNet variants
            "densenet121": models.densenet121,
            "densenet169": models.densenet169,
            "densenet201": models.densenet201,
            # EfficientNet variants
            "efficientnet_b0": models.efficientnet_b0,
            "efficientnet_b7": models.efficientnet_b7,
            # Add custom model constructor
            "custom": self._build_custom_model
        }
        
        # Custom Model Case
        if self.cnn_type == "custom":
            self.is_pretrained = False  # Custom models cannot be pretrained
            self.model = CustomCNN(
                input_channels=self.input_channels,
                num_classes=self.num_classes,
                batch_size = self.batch_size,
                img_size=(self.img_height, self.img_width),
                dropout=setting.get("cnn_dropout", 0.5)
            )
        
        # Predefined Model Case
        else:
            # Load with/without pretrained weights
            constructor = model_dict[self.cnn_type]
            self.model = constructor(weights="DEFAULT" if self.is_pretrained else None)
            
            # Adapt for grayscale input (only for predefined models)
            if self.input_channels == 1:
                original_conv = self._adapt_first_conv(self.model, self.input_channels)
                if self.is_pretrained and original_conv:
                    # Transfer pretrained weights by averaging RGB channels
                    if hasattr(self.model, 'conv1'):
                        self.model.conv1.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
                    elif hasattr(self.model.features, '0'):
                        self.model.features[0].weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
            
            elif self.input_channels not in [1, 3]:
                raise ValueError("Input must be 1 (grayscale) or 3 (RGB) channels")

        # Common Modifications
        # Modify final layer for all models
        if hasattr(self.model, 'fc'):  # ResNet-style
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, self.num_classes)
        elif hasattr(self.model, 'classifier'):  # DenseNet/VGG-style
            if isinstance(self.model.classifier, nn.Sequential):
                in_features = self.model.classifier[-1].in_features
                self.model.classifier[-1] = nn.Linear(in_features, self.num_classes)
            else:
                in_features = self.model.classifier.in_features
                self.model.classifier = nn.Linear(in_features, self.num_classes)
        
        # Initialize non-pretrained weights
        if not self.is_pretrained:
            self.model.apply(self._init_weights)

        # Final setup
        self.model.to(device)
        self.model_loaded = True
        return self.model

    # Constructs your custom CNN architecture
    def _build_custom_model(self):
        return CustomCNN(
            input_channels=self.input_channels,
            num_classes=self.num_classes,
            batch_size=self.batch_size,
            img_size=(self.img_height, self.img_width),
            dropout=setting.get("cnn_dropout", 0.5)
        ) 
    
    # Weight initialization for non-pretrained cnns
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if self.initialization == "kaiming":
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif self.initialization == "xavier":
                init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, 0, 0.01)
            init.constant_(m.bias, 0)

    # Modify first layer to match image channels
    def _adapt_first_conv(self, model, input_channels):
        # ResNet/ResNeXt/AlexNet:
        if hasattr(model, 'conv1'):  
            original_conv = model.conv1
            model.conv1 = nn.Conv2d(
                input_channels, 
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False
            )
            return original_conv
        # VGG:
        elif hasattr(model, 'features') and isinstance(model.features[0], nn.Conv2d):
            original_conv = model.features[0]
            model.features[0] = nn.Conv2d(
                input_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False
            )
            return original_conv
        # DenseNet:
        elif hasattr(model, 'features') and hasattr(model.features, 'conv0'):
            original_conv = model.features.conv0
            model.features.conv0 = nn.Conv2d(
                input_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False
            )
            return original_conv
        # EfficientNet:
        elif hasattr(model, 'features') and hasattr(model.features[0], '0'):
            original_conv_block = model.features[0]
            original_conv = original_conv_block[0]  # Access Conv2d inside the block
            new_conv = nn.Conv2d(
                input_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False
            )
            # Replace the entire block while preserving BN/activation
            model.features[0][0] = new_conv
            return original_conv

        return None

    # Read classes from training data directory (subfolder names) and returns a list
    def get_class_list(self):
        class_list = sorted([j.name for j in self.pth_train.iterdir() if j.is_dir()])
        return class_list
    
    # Prints all classes
    def print_class_list(self):
        print("Number of classes:", self.num_classes)
        print("Classes: ", end="")
        print(', '.join(self.class_list))

    # Saves a checkpoint/weights
    def save_weights(self, val_acc, best_acc, epoch, chckpt_pth):
        # Save checkpoint if the accuracy has improved AND
        # if it is higher than a predefined percentage (min_acc_for_saving) AND
        # if models should be saved at all
        if(val_acc > best_acc and 
            val_acc > self.chckpt_min_acc and
            self.chckpt_save):
            # Datetime for saved files
            # current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M")
            print(f"Model with test accuracy {val_acc:.2f} saved!")
            # Add datetime, epoch and validation accuracy to the filename and save model
            if(self.is_pretrained):
                pretr = "_pretr"
            else:
                pretr = ""
            filename = chckpt_pth / f'ckpt{pretr}_{self.cnn_type}_e{epoch+1}_vacc{val_acc*100:.0f}.model'
            torch.save(self.model.state_dict(), filename)
            # Update best accuracy
            best_acc = val_acc 

        return best_acc
    
    def get_checkpoints_list(self, pth_checkpoint):
        # List of all model files in the checkpoint directory with the .model extension
        checkpoints_list = [file.name for file in pth_checkpoint.glob('*.model')]
        # Return list of tuples with (display_id, filename) where display_id starts at 1
        return [(i+1, name) for i, name in enumerate(checkpoints_list)]
    
    def print_checkpoints_table(self, pth_checkpoint, print_table=True):
        # Get a list of all checkpoints in the checkpoint folder with ID
        checkpoints = self.get_checkpoints_list(pth_checkpoint)
        # print(checkpoints)
        if print_table and checkpoints:
            table = PrettyTable(["ID", "Checkpoint"])
            for display_id, name in checkpoints:
                table.add_row([display_id, name])
            print()
            print(table)
        # print(f"Number of checkpoints: {len(checkpoints)}")
        return checkpoints
    
    def select_checkpoint(self, checkpoints, prompt):
        # Check input
        max_id = len(checkpoints)
        while(True):
            nr = input(prompt)
            if not(fn.check_int(nr)):
                print("Input is not an integer number! Try again...")
            else:
                nr = int(nr)
                if not(fn.check_int_range(nr, 1, max_id)):
                    print("Index out of range! Try again...")
                else:
                        return checkpoints[nr-1][1] 
    
    # Load a checkpoint/weights
    def load_weights(self, chckpt_pth, chckpt_file):
        self.model.load_state_dict(torch.load(chckpt_pth / chckpt_file))
        print(f'Weights from checkpoint {chckpt_file} successfully loaded.')
        return chckpt_file
    
    def load_checkpoint(self):
        # First get checkpoints without printing table
        silent_checkpoints = self.print_checkpoints_table(self.pth_checkpoint, print_table=False)
        # In case the folder is empty
        if not silent_checkpoints:
            print("The checkpoint folder is empty!")
            return False
        # If only one checkpoint exists
        if len(silent_checkpoints) == 1:
            # Extract filename from the tuple: (id, name)
            checkpoint_file = silent_checkpoints[0][1]
            print(f"\nFound single checkpoint: {checkpoint_file}")
            print("Loading automatically...")
        else:
            # Show interactive table for multiple checkpoints
            self.print_checkpoints_table(self.pth_checkpoint)
            checkpoint_file = self.select_checkpoint(silent_checkpoints, "Select a checkpoint: ")
            if not checkpoint_file:
                return False
        # Load weights from checkpoint
        try:
            full_path = self.pth_checkpoint / checkpoint_file
            self.load_weights(self.pth_checkpoint, checkpoint_file)
            self.checkpoint_loaded = True
            self.loaded_checkpoint_name = full_path.stem
            return True
        except FileNotFoundError as e:
            print(f"\nError loading checkpoint: {str(e)}")
            print(f"Full path attempted: {full_path}")
            return False
        except Exception as e:
            print(f"\nError loading checkpoint: {str(e)}")
            return False

    # Function for Prediction
    def predict(self, dataset):

        # Load dataset
        num_correct = 0
        num_samples = 0
        # Parameters for confusion matrix
        cm = {"y": [], "y_hat": []}
        # Set model to evaluation mode
        self.model.eval()

        # No need to keep track of gradients
        with torch.no_grad():
            # Loop through the data
            for i, (images, labels) in enumerate(tqdm(dataset)):
                # Send images and labels to gpu
                if torch.cuda.is_available():
                    images, labels = images.cuda(), labels.cuda()
                # Forward pass
                scores = self.model(images)
                # print(scores)         
                _, predictions = scores.max(1)
                # Check how many we got correct
                num_correct += (predictions == labels).sum()
                # Keep track of number of samples
                num_samples += predictions.size(0)
                # Confusion matrix data
                cm["y"].append(labels.item())
                # print(labels.item())
                cm["y_hat"].append(predictions.item())
                # print(predictions.item())

        acc = num_correct / num_samples 
        # Set model back to training mode
        self.model.train()
        return acc, cm

    # Prints number of model parameters
    def print_model_size(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {total_params:,}")

    # Captures output shapes of all layers through forward hooks.
    # Returns: 
    #    - layer_info: Dictionary with unique layer data
    #    - hooks: List of hook references for cleanup
    def get_layer_info(self, device):
        layer_info = {}
        hooks = []

        def hook_fn(module, input, output):
            # Only process if not already seen
            if id(module) not in layer_info:
                params = 0
                # Only count parameters for leaf modules (no children)
                if not list(module.children()):
                    params = sum(p.numel() for p in module.parameters())
                
                layer_info[id(module)] = {
                    'name': module.__class__.__name__,
                    'output': list(output.shape),
                    'params': params,
                    'trainable': any(p.requires_grad for p in module.parameters())
                }

        # Register hooks only for leaf modules
        for name, module in self.model.named_modules():
            if not list(module.children()):  # Only leaf nodes
                hooks.append(module.register_forward_hook(hook_fn))

        # Forward pass
        dummy_input = torch.rand(*self.input_size).to(device)
        self.model.to(device).eval()
        with torch.no_grad():
            self.model(dummy_input)

        return layer_info, hooks
    
    # Generates accurate model summary without double-counting parameters
    # Skips layers without trainable parameters
    def model_summary(self, device, show_non_trainable=False):

        # Get layer information
        layer_info, hooks = self.get_layer_info(device)

        # Initialize table
        table = PrettyTable()
        table.field_names = ["Layer (type)", "Output Shape", "Param #", "Trainable"]
        table.align = "l"

        # Calculate totals
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # Populate table (skip zero-param layers)
        for module_id, info in layer_info.items():
            if not show_non_trainable:
                if info['params'] == 0 and not info['trainable']:
                    continue
                
            table.add_row([
                info['name'],
                str(info['output']),
                f"{info['params']:,}",
                str(info['trainable'])
            ])

        # Clean up hooks
        for hook in hooks:
            hook.remove()

        # Print results
        print(table)
        print(f"\nInput: {list(self.input_size)} (Device: {device})")
        print(f"Total params: {total_params:,} (ground truth)")
        print(f"Trainable params: {trainable_params:,}")
        print(f"Non-trainable params: {total_params - trainable_params:,}")
