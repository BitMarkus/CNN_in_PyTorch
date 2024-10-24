#############################
# Custom model architecture #
#############################

import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm
# Own modules
from model import CNN_Model
from settings import setting

class Custom_CNN_Model(nn.Module, CNN_Model):

    #############################################################################################################
    # CONSTRUCTOR:
    
    def __init__(self):

        # Initialize base classes
        nn.Module.__init__(self)
        CNN_Model.__init__(self)

        # Parameters from settings
        self.dropout = setting["cnn_dropout"]
        self.batch_size = setting["ds_batch_size"]
        self.img_width = setting["img_width"]
        self.img_height = setting["img_height"]

        # Define layers
        self.act = nn.ReLU()
        self.drop = nn.Dropout(self.dropout)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten = nn.Flatten()

        # Conv layers
        # Conv 1:
        self.conv1 = nn.Conv2d(
             self.input_channels, 
             64, 
             kernel_size=(3,3), 
             stride=1, 
             padding=1)
        # Conv 2:
        self.conv2 = nn.Conv2d(
             64, 
             128, 
             kernel_size=(3,3), 
             stride=1, 
             padding=1)
        # Conv 3:
        self.conv3 = nn.Conv2d(
             128, 
             256, 
             kernel_size=(3,3), 
             stride=1, 
             padding=1)
        # Conv 4:
        self.conv4 = nn.Conv2d(
             256, 
             512, 
             kernel_size=(3,3), 
             stride=1, 
             padding=1)      
        # Conv 5:
        self.conv5 = nn.Conv2d(
             512, 
             512, 
             kernel_size=(3,3), 
             stride=1, 
             padding=1)
        
        # Fc layers
        # Fc 1:
        self.fc1 = nn.Linear(
             in_features=32768, 
             out_features=4096)
        # Fc 2:
        self.fc2 = nn.Linear(
             in_features=4096, 
             out_features=512)
        
        # Classifier
        self.classifier = nn.Linear(
             in_features=512, 
             out_features=self.num_classes)        

    #############################################################################################################
    # METHODS:        

    def forward(self, x):
        assert x.shape == (self.batch_size, 
                           self.input_channels, 
                           self.img_height, 
                           self.img_width)

        # Conv block 1:
        # 512x512 -> 256x256
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.pool(x)
        assert x.shape == (self.batch_size, 64, 256, 256)

        # Conv block 2:
        # 256x256 -> 128x128
        x = self.conv2(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.pool(x)
        assert x.shape == (self.batch_size, 128, 128, 128)

        # Conv block 3:
        # 128x128 -> 64x64
        x = self.conv3(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.pool(x)
        assert x.shape == (self.batch_size, 256, 64, 64)

        # Conv block 4:
        # 64x64 -> 32x32
        x = self.conv4(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.pool(x)
        assert x.shape == (self.batch_size, 512, 32, 32)

        # Conv block 5:
        # 32x32 -> 16x16
        x = self.conv5(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.pool(x)
        assert x.shape == (self.batch_size, 512, 16, 16)

        # Conv block 6:
        # 16x16 -> 8x8
        x = self.conv5(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.pool(x)
        assert x.shape == (self.batch_size, 512, 8, 8)

        # Flatten
        # 8x8x512 -> 32768
        x = self.flatten(x)
        assert x.shape == (self.batch_size, 32768)

        # Fc layer 1: 
        # 32768 -> 4096
        x = self.fc1(x)
        assert x.shape == (self.batch_size, 4096)

        # Fc layer 2: 
        # 4096 -> 512
        x = self.fc2(x)
        assert x.shape == (self.batch_size, 512)

        # Classifier
        x = self.classifier(x)
        assert x.shape == (self.batch_size, self.num_classes)

        return x
    
    """
    # Layers
    self.model = nn.Sequential(
        # Conv block 1:
        # 512x512 -> 256x256
        self._cnn_block(self.input_channels, 64, first_cnn_block=True),
        # Conv block 2:
        # 256x256 -> 128x128
        self._cnn_block(64, 128, first_cnn_block=False),
        # Conv block 3:
        # 128x128 -> 64x64
        self._cnn_block(128, 256, first_cnn_block=False),
        # Conv block 4:
        # 64x64 -> 32x32
        self._cnn_block(256, 512, first_cnn_block=False),
        # Conv block 5:
        # 32x32 -> 16x16
        self._cnn_block(512, 512, first_cnn_block=False),
        # Conv block 6:
        # 16x16 -> 8x8
        self._cnn_block(512, 512, first_cnn_block=False),
        self._classifier(32768)
    )
    """
    """
    def _cnn_block(        
            self,
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            first_cnn_block = False):
        if(first_cnn_block):
            return nn.Sequential(
                nn.Conv2d(self.input_channels, out_channels, kernel_size, stride, padding),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.MaxPool2d(kernel_size=(2, 2))
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.MaxPool2d(kernel_size=(2, 2))
            )  

    def _classifier(self, in_features):  
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=in_features, out_features=4096),
            nn.Linear(in_features=4096, out_features=512),
            nn.Linear(in_features=512, out_features=self.num_classes),
        ) 

    def forward(self, x):
        out = self.model(x)

        return out
    """

