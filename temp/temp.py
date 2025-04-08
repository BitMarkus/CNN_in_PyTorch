import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models
from torchvision.models.efficientnet import EfficientNet

def load_cnn(
    model_name="resnet18",
    pretrained=True,
    input_channels=3,
    num_classes=2,
    initialization="kaiming"
):
    """
    Universal CNN loader for:
    - ResNet (18/34/50/101/152)
    - ResNeXt-101 (32x8d/64x4d)
    - AlexNet
    - VGG (11/13/16/19, with/without BN)
    - DenseNet (121/169/201/264)
    - EfficientNet (B0-B7)
    
    Args:
        model_name: Any supported model name (e.g., "resnet50", "densenet121")
        pretrained: True for pretrained weights, False for random initialization
        input_channels: 1 (grayscale), 3 (RGB), or N (custom)
        num_classes: Output classes
        initialization: "kaiming" (ReLU) or "xavier" (sigmoid/tanh)
    """
    # Supported models and their constructors
    """
    # Compressed version:
    model_dict = {
        # ResNets
        **{f"resnet{d}": getattr(models, f"resnet{d}") for d in [18, 34, 50, 101, 152]},
        # ResNeXt
        **{f"resnext{cfg}": getattr(models, f"resnext{cfg.replace('_', '')}") for cfg in ["101_32x8d", "101_64x4d"]},
        # AlexNet
        "alexnet": models.alexnet,
        # VGGs
        **{f"vgg{v}": getattr(models, f"vgg{v}") for v in [11, 13, 16, 19, "11_bn", "13_bn", "16_bn", "19_bn"]},
        # DenseNets
        **{f"densenet{d}": getattr(models, f"densenet{d}") for d in [121, 169, 201, 264]},
        # EfficientNets (fixed parenthesis issue here)
        **{f"efficientnet_b{b}": (lambda **kwargs: models.efficientnet_b0(**kwargs)) if b == 0 
           else (lambda **kwargs: models.efficientnet_b7(**kwargs)) 
           for b in [0, 7]},
    }
    """
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
        # "densenet264": models.densenet264,
        
        # EfficientNet variants
        "efficientnet_b0": models.efficientnet_b0,
        "efficientnet_b7": models.efficientnet_b7
    }
    
    # Load model
    if pretrained:
        model = model_dict[model_name](weights="DEFAULT")
    else:
        model = model_dict[model_name](weights=None)
    
    # --- Handle input channel adaptation ---
    def _adapt_first_conv(model, input_channels):
        if hasattr(model, 'conv1'):  # ResNet/ResNeXt/AlexNet
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
        elif hasattr(model.features, '0'):  # VGG/DenseNet
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
        elif hasattr(model.features, '0.0'):  # EfficientNet
            original_conv = model.features[0][0]
            model.features[0][0] = nn.Conv2d(
                input_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False
            )
            return original_conv
        return None
    
    if input_channels != 3:
        original_conv = _adapt_first_conv(model, input_channels)
        if pretrained and original_conv:
            if input_channels == 1:
                # Grayscale: Average RGB weights
                if hasattr(model, 'conv1'):
                    model.conv1.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
                elif hasattr(model.features, '0'):
                    model.features[0].weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
                elif hasattr(model.features, '0.0'):
                    model.features[0][0].weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
            else:
                # Multi-channel: Repeat pretrained weights
                repeat_factor = input_channels // 3
                new_weights = original_conv.weight.data.repeat(1, repeat_factor, 1, 1)
                if input_channels % 3 != 0:
                    remaining = input_channels - (repeat_factor * 3)
                    new_weights = torch.cat([
                        new_weights, 
                        torch.randn(new_weights.shape[0], remaining, *new_weights.shape[2:]) 
                    ], dim=1)
                if hasattr(model, 'conv1'):
                    model.conv1.weight.data = new_weights
                elif hasattr(model.features, '0'):
                    model.features[0].weight.data = new_weights
                elif hasattr(model.features, '0.0'):
                    model.features[0][0].weight.data = new_weights
    
    # --- Replace final layer ---
    if hasattr(model, 'fc'):  # ResNet/ResNeXt/AlexNet
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):  # VGG/DenseNet
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif hasattr(model, 'classifier'):  # EfficientNet
        if isinstance(model.classifier, nn.Sequential):
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
        else:
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
    
    # --- Initialize non-pretrained models ---
    if not pretrained:
        def _init_weights(m):
            if isinstance(m, nn.Conv2d):
                if initialization == "kaiming":
                    init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif initialization == "xavier":
                    init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.01)
                init.constant_(m.bias, 0)
        
        model.apply(_init_weights)
    
    return model

# Pretrained ResNeXt-101 (32x8d) for 1-channel images
model = load_cnn("resnext101_32x8d", pretrained=True, input_channels=1)

# Non-pretrained DenseNet-169 for 4-channel data
model = load_cnn("densenet169", pretrained=False, input_channels=4)

# Pretrained EfficientNet-B7 for RGB
model = load_cnn("efficientnet_b7", pretrained=True)

# Non-pretrained VGG16-BN with Xavier init
model = load_cnn("vgg16_bn", pretrained=False, initialization="xavier")

"""
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
        
    elif(self.cnn_type == "ResNet-50"):
        # Load pretrained ResNet-50 (trained on ImageNet)
        self.model = models.resnet50(weights="DEFAULT")  # "DEFAULT" loads the best available weights (e.g., IMAGENET1K_V2)      
        # Adjust the first convolutional layer for your input channels
        if self.input_channels != 3:
            # Replace conv1 while preserving pretrained weights by averaging/summing channels
            original_conv1 = self.model.conv1
            new_conv1 = nn.Conv2d(
                self.input_channels, 
                64, 
                kernel_size=(7, 7), 
                stride=(2, 2), 
                padding=(3, 3), 
                bias=False
            )              
            # Initialize new conv1 layer weights (optional strategies below)
            if self.input_channels == 1:
                # Grayscale to RGB: Copy mean of pretrained weights across all input channels
                new_conv1.weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True)
            else:
                # For >3 channels: Repeat pretrained weights or use custom initialization
                new_conv1.weight.data = original_conv1.weight.data.repeat(1, self.input_channels // 3, 1, 1)
                # Add random initialization for remaining channels (if not divisible by 3)
                if self.input_channels % 3 != 0:
                    new_conv1.weight.data[:, -(self.input_channels % 3):, :, :].normal_(0, 0.02)             
            self.model.conv1 = new_conv1  
        # Replace the final fully connected layer for your task
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
    
    # Weight initialization for non-pretrained Networks
    # self.initialize_weights(self.model) 
    # Send model to gpu or cpu
    self.model.to(device) 
    # Set model to loaded
    self.model_loaded = True  

    return self.model   
"""