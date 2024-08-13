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

        ##########
        # ResNet #
        ##########        

        if(self.cnn_type == "ResNet-18"):
            self.model = models.resnet18(weights=None)
            # Set number of input channels
            self.model.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            # Set number of output nodes
            # https://discuss.pytorch.org/t/how-to-modify-the-final-fc-layer-based-on-the-torch-model/766/23
            self.model.fc = nn.Linear(512, num_classes)

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

        ###########
        # Alexnet #
        ###########

        elif(self.cnn_type == "AlexNet"):
            self.model = models.alexnet(weights=None)
            # https://discuss.pytorch.org/t/how-to-modify-the-final-fc-layer-based-on-the-torch-model/766/11
            # Set number of input channels
            self.model.features._modules['0'] = nn.Conv2d(self.input_channels, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
            # Set number of output nodes
            self.model.classifier._modules['6'] = nn.Linear(4096, num_classes)   

        #######
        # VGG #
        #######  

        elif(self.cnn_type == "VGG-11"):
            self.model = models.vgg11_bn(weights=None)
            # https://discuss.pytorch.org/t/how-to-modify-the-final-fc-layer-based-on-the-torch-model/766/11
            # Set number of input channels
            self.model.features._modules['0'] = nn.Conv2d(self.input_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            # Set number of output nodes
            self.model.classifier._modules['6'] = nn.Linear(4096, num_classes, bias=True)  

        elif(self.cnn_type == "VGG-13"):
            self.model = models.vgg13_bn(weights=None)
            # Set number of input channels
            self.model.features._modules['0'] = nn.Conv2d(self.input_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            # Set number of output nodes
            self.model.classifier._modules['6'] = nn.Linear(4096, num_classes, bias=True)  

        elif(self.cnn_type == "VGG-16"):
            self.model = models.vgg16_bn(weights=None)
            # Set number of input channels
            self.model.features._modules['0'] = nn.Conv2d(self.input_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            # Set number of output nodes
            self.model.classifier._modules['6'] = nn.Linear(4096, num_classes, bias=True)    

        elif(self.cnn_type == "VGG-19"):
            self.model = models.vgg19_bn(weights=None)
            # Set number of input channels
            self.model.features._modules['0'] = nn.Conv2d(self.input_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            # Set number of output nodes
            self.model.classifier._modules['6'] = nn.Linear(4096, num_classes, bias=True)  

        ############
        # DenseNet #  
        ############

        elif(self.cnn_type == "DenseNet-121"):
            self.model = models.densenet121(weights=None)
            # Set number of input channels
            self.model.features._modules['conv0'] = nn.Conv2d(self.input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            # Set number of output nodes
            self.model.classifier = nn.Linear(1024, num_classes, bias=True) 

        elif(self.cnn_type == "DenseNet-161"):
            self.model = models.densenet161(weights=None)
            # Set number of input channels
            self.model.features._modules['conv0'] = nn.Conv2d(self.input_channels, 96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            # Set number of output nodes
            self.model.classifier = nn.Linear(1024, num_classes, bias=True) 

        elif(self.cnn_type == "DenseNet-169"):
            self.model = models.densenet169(weights=None)
            # Set number of input channels
            self.model.features._modules['conv0'] = nn.Conv2d(self.input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            # Set number of output nodes
            self.model.classifier = nn.Linear(1664, num_classes, bias=True) 

        elif(self.cnn_type == "DenseNet-201"):
            self.model = models.densenet201(weights=None)
            # Set number of input channels
            self.model.features._modules['conv0'] = nn.Conv2d(self.input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            # Set number of output nodes
            self.model.classifier = nn.Linear(1920, num_classes, bias=True) 
            # print(self.model.classifier)

        
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
