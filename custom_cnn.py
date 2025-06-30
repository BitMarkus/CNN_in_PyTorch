import torch.nn as nn

class CustomCNN(nn.Module):

    def __init__(self, input_channels, num_classes, batch_size, img_size, dropout=0.5):
        super().__init__()
        self.input_channels = input_channels
        self.batch_size = batch_size
        self.img_height, self.img_width = img_size
        self.num_classes = num_classes
        self.dropout = dropout
        
        # Define convolutional blocks = encoder
        # Conv block 1:
        # 512x512 -> 256x256
        self.conv_block_1 = self._cnn_block(self.input_channels, 64, first_cnn_block=True)
        # Conv block 2:
        # 256x256 -> 128x128
        self.conv_block_2 = self._cnn_block(64, 128, first_cnn_block=False)
        # Conv block 3:
        # 128x128 -> 64x64
        self.conv_block_3 = self._cnn_block(128, 256, first_cnn_block=False)
        # Conv block 4:
        # 64x64 -> 32x32
        self.conv_block_4 = self._cnn_block(256, 512, first_cnn_block=False)
        # Conv block 5:
        # 32x32 -> 16x16
        self.conv_block_5 = self._cnn_block(512, 512, first_cnn_block=False)
        # Conv block 6:
        # 16x16 -> 8x8
        self.conv_block_6 = self._cnn_block(512, 512, first_cnn_block=False)
        
        # Define decoder
        # self.decoder = self._decoder_1(in_features=32768, out_features=self.num_classes)
        self.decoder = self._decoder_2(in_features=512, out_features=self.num_classes)
   

    #############################################################################################################
    # METHODS:
    
    def _cnn_block(        
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            first_cnn_block = False):

        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

    def _decoder_1(self, in_features, out_features):  
        
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=in_features, out_features=4096),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(in_features=4096, out_features=512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            # Classifier
            nn.Linear(in_features=512, out_features=out_features),
        )  
    
    def _decoder_2(self, in_features, out_features):  
        
        return nn.Sequential(
            # 1x1 convolution to reduce feature maps to number of classes
            nn.Conv2d(in_features, out_features, 1, 1, 0, bias=False),
            # https://discuss.pytorch.org/t/global-average-pooling-in-pytorch/6721/4
            # https://blog.paperspace.com/global-pooling-in-convolutional-neural-networks/
            # Global average pooling: 8 as the size of the last feature maps is 8x8 
            nn.AvgPool2d(8),
        )  

    #############################################################################################################
    # FORWARD:       

    def forward(self, x):
        # print(x.shape)
        assert (x.shape[1] == self.input_channels and 
                x.shape[2] == self.img_height and 
                x.shape[3] == self.img_width)
        
        # ENCODER
        # Conv block 1:
        # 512x512 -> 256x256
        x = self.conv_block_1(x)
        assert (x.shape[1] == 64 and 
                x.shape[2] == 256 and 
                x.shape[3] == 256)
        # Conv block 2:
        # 256x256 -> 128x128
        x = self.conv_block_2(x)
        assert (x.shape[1] == 128 and 
                x.shape[2] == 128 and 
                x.shape[3] == 128)
        # Conv block 3:
        # 128x128 -> 64x64
        x = self.conv_block_3(x)
        assert (x.shape[1] == 256 and 
                x.shape[2] == 64 and 
                x.shape[3] == 64)
        # Conv block 4:
        # 64x64 -> 32x32
        x = self.conv_block_4(x)
        assert (x.shape[1] == 512 and 
                x.shape[2] == 32 and 
                x.shape[3] == 32)
        # Conv block 5:
        # 32x32 -> 16x16
        x = self.conv_block_5(x)
        assert (x.shape[1] == 512 and 
                x.shape[2] == 16 and 
                x.shape[3] == 16)
        # Conv block 6:
        # 16x16 -> 8x8
        x = self.conv_block_6(x) 
        assert (x.shape[1] == 512 and 
                x.shape[2] == 8 and 
                x.shape[3] == 8)
        
        # DECODER
        x = self.decoder(x) 
        # print(x.shape)
        #######################################################################################################
        # Line specific for decoder_2
        # Reshapes the tensor from the global average pooling layer [batch_size, 2, 1, 1]
        # to the desired output tensor [batch size, 2]
        # x = x.view(-1, 2 * 1 * 1) -> better:
        x = x.view(x.size(0), -1)  # Flatten all except batch dim
        #######################################################################################################
        # print(x.shape)

        assert (x.shape[1] == self.num_classes)

        return x