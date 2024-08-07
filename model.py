######################
# Model architecture #
######################

import torchvision.models as models
import pathlib
from torch import nn
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

    #############################################################################################################
    # METHODS:

    # Load model: Pretrained or custom trained
    def load_model(self, device):
        # Get number of classes = number of output nodes
        class_list = self.get_class_list()   
        num_classes = len(class_list)

        if(self.cnn_type == "ResNet-18"):
            self.model = models.resnet18(weights=None)
            # Set number of input channels
            self.model.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            # Set number of output nodes
            # https://discuss.pytorch.org/t/how-to-modify-the-final-fc-layer-based-on-the-torch-model/766/23
            self.model.fc = nn.Linear(512, num_classes)

        ##########
        # ResNet #
        ##########

        elif(self.cnn_type == "ResNet-34"):
            self.model = models.resnet34(weights=None)
            # Set number of input channels
            self.model.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            # Set number of output nodes
            self.model.fc = nn.Linear(512, num_classes)

        elif(self.cnn_type == "ResNet-50"):
            self.model = models.resnet50(weights=None)
            # Set number of input channels
            self.model.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            # Set number of output nodes
            self.model.fc = nn.Linear(2048, num_classes) 

        elif(self.cnn_type == "ResNet-101"):
            self.model = models.resnet101(weights=None)
            # Set number of input channels
            self.model.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            # Set number of output nodes
            self.model.fc = nn.Linear(2048, num_classes) 

        elif(self.cnn_type == "ResNet-152"):
            self.model = models.resnet152(weights=None)
            # Set number of input channels
            self.model.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            # Set number of output nodes
            self.model.fc = nn.Linear(2048, num_classes) 
        
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
        class_list = self.get_class_list()
        print("Number of classes:", len(class_list))
        print("Classes: ", end="")
        print(', '.join(class_list))
