##################################
# Predefined model architectures #
##################################

import torchvision.models as models
import pathlib
from torch import nn
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

    # Load model: Pretrained or custom trained
    # https://pytorch.org/vision/0.9/models.html  
    def load_model(self, device):

        # Use weights="DEFAULT" for pretrained weights

        ##########
        # ResNet #
        ##########      

        if(self.cnn_type == "ResNet-18"):
            self.model = models.resnet18(weights=None)
            # Set number of input channels
            self.model.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            # Set number of output nodes
            # https://discuss.pytorch.org/t/how-to-modify-the-final-fc-layer-based-on-the-torch-model/766/23
            self.model.fc = nn.Linear(512, self.num_classes)

        elif(self.cnn_type == "ResNet-34"):
            self.model = models.resnet34(weights=None)
            # Set number of input channels
            self.model.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            # Set number of output nodes
            self.model.fc = nn.Linear(512, self.num_classes)

        elif(self.cnn_type == "ResNet-50"):
            self.model = models.resnet50(weights=None)
            # Set number of input channels
            self.model.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            # Set number of output nodes
            self.model.fc = nn.Linear(2048, self.num_classes) 

        elif(self.cnn_type == "ResNet-101"):
            self.model = models.resnet101(weights=None)
            # Set number of input channels
            self.model.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            # Set number of output nodes
            self.model.fc = nn.Linear(2048, self.num_classes) 

        elif(self.cnn_type == "ResNet-152"):
            self.model = models.resnet152(weights=None)
            # Set number of input channels
            self.model.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            # Set number of output nodes
            self.model.fc = nn.Linear(2048, self.num_classes) 

        elif(self.cnn_type == "ResNeXt-101"):
            self.model = models.resnext101_32x8d(weights=None)
            # Set number of input channels
            self.model.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            # Set number of output nodes
            self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)

        ###########
        # Alexnet #
        ###########

        elif(self.cnn_type == "AlexNet"):
            self.model = models.alexnet(weights=None)
            # https://discuss.pytorch.org/t/how-to-modify-the-final-fc-layer-based-on-the-torch-model/766/11
            # Set number of input channels
            self.model.features._modules['0'] = nn.Conv2d(self.input_channels, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
            # Set number of output nodes
            self.model.classifier._modules['6'] = nn.Linear(4096, self.num_classes)   

        #######
        # VGG #
        #######  

        elif(self.cnn_type == "VGG-11"):
            self.model = models.vgg11_bn(weights=None)
            # https://discuss.pytorch.org/t/how-to-modify-the-final-fc-layer-based-on-the-torch-model/766/11
            # Set number of input channels
            self.model.features._modules['0'] = nn.Conv2d(self.input_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            # Set number of output nodes
            self.model.classifier._modules['6'] = nn.Linear(4096, self.num_classes, bias=True)  

        elif(self.cnn_type == "VGG-13"):
            self.model = models.vgg13_bn(weights=None)
            # Set number of input channels
            self.model.features._modules['0'] = nn.Conv2d(self.input_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            # Set number of output nodes
            self.model.classifier._modules['6'] = nn.Linear(4096, self.num_classes, bias=True)  

        elif(self.cnn_type == "VGG-16"):
            self.model = models.vgg16_bn(weights=None)
            # Set number of input channels
            self.model.features._modules['0'] = nn.Conv2d(self.input_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            # Set number of output nodes
            self.model.classifier._modules['6'] = nn.Linear(4096, self.num_classes, bias=True)    

        elif(self.cnn_type == "VGG-19"):
            self.model = models.vgg19_bn(weights=None)
            # Set number of input channels
            self.model.features._modules['0'] = nn.Conv2d(self.input_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            # Set number of output nodes
            self.model.classifier._modules['6'] = nn.Linear(4096, self.num_classes, bias=True)  

        ############
        # DenseNet #  
        ############

        elif(self.cnn_type == "DenseNet-121"):
            self.model = models.densenet121(weights=None)
            # Set number of input channels
            self.model.features._modules['conv0'] = nn.Conv2d(self.input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            # Set number of output nodes
            self.model.classifier = nn.Linear(1024, self.num_classes, bias=True) 

        elif(self.cnn_type == "DenseNet-161"):
            self.model = models.densenet161(weights=None)
            # Set number of input channels
            self.model.features._modules['conv0'] = nn.Conv2d(self.input_channels, 96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            # Set number of output nodes
            self.model.classifier = nn.Linear(1024, self.num_classes, bias=True) 

        elif(self.cnn_type == "DenseNet-169"):
            self.model = models.densenet169(weights=None)
            # Set number of input channels
            self.model.features._modules['conv0'] = nn.Conv2d(self.input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            # Set number of output nodes
            self.model.classifier = nn.Linear(1664, self.num_classes, bias=True) 

        elif(self.cnn_type == "DenseNet-201"):
            self.model = models.densenet201(weights=None)
            # Set number of input channels
            self.model.features._modules['conv0'] = nn.Conv2d(self.input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            # Set number of output nodes
            self.model.classifier = nn.Linear(1920, self.num_classes, bias=True) 
            # print(self.model.classifier)

        ################
        # EfficientNet #
        ################

        elif(self.cnn_type == "EfficientNet-B7"):
            # Load EfficientNet-B7
            self.model = models.efficientnet_b7(weights=None)
            # Set number of input channels
            first_conv = self.model.features[0][0]
            self.model.features[0][0] = nn.Conv2d(
                self.input_channels,
                first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=False
            )
            # Set number of output nodes
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, self.num_classes)

        elif(self.cnn_type == "EfficientNet-B0"):
            # Load EfficientNet-B0
            self.model = models.efficientnet_b0(weights=None)
            # Set number of input channels
            first_conv = self.model.features[0][0]
            self.model.features[0][0] = nn.Conv2d(
                self.input_channels,
                first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=False
            )
            # Set number of output nodes
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, self.num_classes)
       
        # Send model to gpu or cpu
        self.model.to(device) 
        # Set model to loaded
        self.model_loaded = True  
        
        return self.model       

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
            filename = f'{self.chckpt_pth}{current_datetime}_checkpoint_{self.cnn_type}_e{epoch+1}_vacc{val_acc*100:.0f}.model'
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

