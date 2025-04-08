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

from PIL import Image
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
from captum.attr import visualization as viz
# Own modules
from settings import setting

class CNN_Model():

    #############################################################################################################
    # CONSTRUCTOR:
    
    def __init__(self):
        # Path to training images
        self.pth_data = setting["pth_data"]
        # Number of channels of training images 
        self.input_channels = setting["img_channels"]
        # Variable for model architecture name
        self.cnn_type = setting["cnn_type"] 
        # Variable to save if a model was alread loaded or not
        self.model_loaded = False   
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
        self.chckpt_weights_file = setting["chckpt_weights_file"] 
        # Get number of classes = number of output nodes
        self.class_list = self.get_class_list()   
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
        }
        
        # Load model
        if self.is_pretrained:
            self.model = model_dict[self.cnn_type](weights="DEFAULT")
        else:
            self.model = model_dict[self.cnn_type](weights=None)
        
        # Adapt input for one channel (grayscale)
        if self.input_channels == 1:
            original_conv = self._adapt_first_conv(self.model, self.input_channels)
            if self.is_pretrained and original_conv:
                # Handle all architecture cases:
                if hasattr(self.model, 'conv1'):  # ResNet/ResNeXt/AlexNet
                    self.model.conv1.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
                elif hasattr(self.model.features, '0') and isinstance(self.model.features[0], nn.Conv2d):  # VGG
                    self.model.features[0].weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
                elif hasattr(self.model.features, 'conv0'):  # DenseNet
                    self.model.features.conv0.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
                elif hasattr(self.model.features[0], '0'):  # EfficientNet
                    self.model.features[0][0].weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
        elif(self.input_channels == 2 or self.input_channels > 3):
            print("The input images must either have one (grayscale) or three (RGB) channels!")
            return False
        
        # Modify last layer to match number of classes
        # ResNet/ResNeXt/AlexNet:
        if hasattr(self.model, 'fc'):  
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, self.num_classes)
        # VGG/DenseNet:
        elif hasattr(self.model, 'classifier') and isinstance(self.model.classifier, nn.Sequential):  
            in_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(in_features, self.num_classes)
        # EfficientNet:
        elif hasattr(self.model, 'classifier'):  
            if isinstance(self.model.classifier, nn.Sequential):
                in_features = self.model.classifier[-1].in_features
                self.model.classifier[-1] = nn.Linear(in_features, self.num_classes)
            else:
                in_features = self.model.classifier.in_features
                self.model.classifier = nn.Linear(in_features, self.num_classes)
        
        # Initialize non-pretrained models
        if not self.is_pretrained:         
            self.model.apply(self._init_weights)
        
        # Send model to gpu or cpu
        self.model.to(device) 
        # Set model to loaded
        self.model_loaded = True  

        return self.model   
    
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
        root = pathlib.Path(self.pth_data)
        class_list = sorted([j.name.split('/')[-1] for j in root.iterdir()])
        return class_list
    
    # Prints all classes
    def print_class_list(self):
        print("Number of classes:", self.num_classes)
        print("Classes: ", end="")
        print(', '.join(self.class_list))

    # Saves a checkpoint/weights
    def save_weights(self, val_acc, best_acc, epoch):
        # Save checkpoint if the accuracy has improved AND
        # if it is higher than a predefined percentage (min_acc_for_saving) AND
        # if models should be saved at all
        if(val_acc > best_acc and 
            val_acc > self.chckpt_min_acc and
            self.chckpt_save):
            # Datetime for saved files
            current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M")
            print(f"Model with test accuracy {val_acc:.2f} saved!")
            # Add datetime, epoch and validation accuracy to the filename and save model
            if(self.is_pretrained):
                pretr = "_pretrained"
            else:
                pretr = ""
            filename = f'{self.chckpt_pth}{current_datetime}_checkpoint{pretr}_{self.cnn_type}_e{epoch+1}_vacc{val_acc*100:.0f}.model'
            torch.save(self.model.state_dict(), filename)
            # Update best accuracy
            best_acc = val_acc 

        return best_acc
    
    # Load a checkpoint/weights
    def load_weights(self, device):
        # https://stackoverflow.com/questions/49941426/attributeerror-collections-ordereddict-object-has-no-attribute-eval
        self.model.load_state_dict(torch.load(self.chckpt_pth + self.chckpt_weights_file))
        # self.model.to(device)
        print(f'Weights from checkpoint {self.chckpt_weights_file} successfully loaded.')

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
    
    def predict_single(self, dataset):
        # Set model to evaluation mode
        self.model.eval()
        # No need to keep track of gradients
        with torch.no_grad():
            # Loop through the data
            for i, (images, labels) in enumerate(dataset):
                # Send images and labels to gpu
                if torch.cuda.is_available():
                    images, labels = images.cuda(), labels.cuda()

                for i, img in enumerate(images):
                    img = img.unsqueeze(0)  # Add batch dimension
                    # print(img.shape)
                    score = self.model(img)
                    probability = torch.softmax(score, dim=1)

                    # Initialize Integrated Gradients
                    ig = IntegratedGradients(self.model) 
                    # Read target class (e.g., class 1)
                    # ko = 0, wt = 1  
                    tar_class = labels[i].item()
                    # Compute attributions
                    attributions = ig.attribute(img, target=tar_class) 
                    print("Target class:", tar_class)
                    print("Probability:", probability)

                    # Convert the input image and attributions to a format suitable for visualization
                    image_np = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    # print(image_np.shape)
                    attributions_np = attributions.squeeze(0).permute(1, 2, 0).cpu().numpy()     

                    # Visualize the attributions
                    fig, ax = viz.visualize_image_attr(
                        attributions_np,  # Attributions
                        original_image=image_np,  # Original image
                        method="heat_map",  # Visualization method # blended_heat_map, heat_map or masked_image
                        sign="positive",  # Show positive and negative attributions # all
                        cmap="inferno",
                        show_colorbar=True,  # Show color bar
                        title=f"Feature Attribution for Class {tar_class}",
                        fig_size=(12,12)
                    )                       
                    plt.show() 

