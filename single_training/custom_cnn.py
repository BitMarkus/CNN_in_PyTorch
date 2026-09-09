# ===== Third-Party Imports =====
import torch.nn as nn

class CustomCNN(nn.Module):

    #############################################################################################################
    # CONSTRUCTOR

    # Custom CNN architecture for fibroblast image classification.
    # Encoder-decoder style network with 6 convolutional blocks and configurable decoder.
    # Args:
    #   input_channels (int): Number of input channels (1 for grayscale, 3 for RGB)
    #   num_classes (int): Number of output classes
    #   batch_size (int): Batch size (used for assertion checks)
    #   img_size (tuple): (height, width) of input images
    #   dropout (float): Dropout rate for decoder (default: 0.5)
    def __init__(self, input_channels: int, num_classes: int, batch_size: int, img_size: tuple, dropout: float = 0.5) -> None:

        super().__init__()

        self.input_channels = input_channels
        self.batch_size = batch_size
        self.img_height, self.img_width = img_size
        self.num_classes = num_classes
        self.dropout = dropout

        # Define convolutional blocks (encoder)
        # Conv block 1: 512x512 -> 256x256
        self.conv_block_1 = self._cnn_block(self.input_channels, 64, first_cnn_block=True)
        # Conv block 2: 256x256 -> 128x128
        self.conv_block_2 = self._cnn_block(64, 128, first_cnn_block=False)
        # Conv block 3: 128x128 -> 64x64
        self.conv_block_3 = self._cnn_block(128, 256, first_cnn_block=False)
        # Conv block 4: 64x64 -> 32x32
        self.conv_block_4 = self._cnn_block(256, 512, first_cnn_block=False)
        # Conv block 5: 32x32 -> 16x16
        self.conv_block_5 = self._cnn_block(512, 512, first_cnn_block=False)
        # Conv block 6: 16x16 -> 8x8
        self.conv_block_6 = self._cnn_block(512, 512, first_cnn_block=False)

        # Define decoder (using decoder_2 by default)
        self.decoder = self._decoder_2(in_features=512, out_features=self.num_classes)


    #############################################################################################################
    # METHODS

    # Create a convolutional block with Conv2d, BatchNorm2d, ReLU, and MaxPool2d.
    # Args:
    #   in_channels (int): Number of input channels
    #   out_channels (int): Number of output channels
    #   kernel_size (int): Kernel size (default: 3)
    #   stride (int): Stride (default: 1)
    #   padding (int): Padding (default: 1)
    #   first_cnn_block (bool): Unused, kept for compatibility
    # Returns:
    #   nn.Sequential: The convolutional block
    def _cnn_block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        first_cnn_block: bool = False
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

    # Decoder with fully connected layers (original implementation).
    # Args:
    #   in_features (int): Number of input features
    #   out_features (int): Number of output features (classes)
    # Returns:
    #   nn.Sequential: The decoder network
    def _decoder_1(self, in_features: int, out_features: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=in_features, out_features=4096),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(in_features=4096, out_features=512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(in_features=512, out_features=out_features),
        )

    # Decoder with 1x1 convolution and global average pooling.
    # More parameter-efficient than decoder_1.
    # Args:
    #   in_features (int): Number of input channels (feature maps)
    #   out_features (int): Number of output features (classes)
    # Returns:
    #   nn.Sequential: The decoder network
    def _decoder_2(self, in_features: int, out_features: int) -> nn.Sequential:
        return nn.Sequential(
            # 1x1 convolution to reduce feature maps to number of classes
            nn.Conv2d(in_features, out_features, 1, 1, 0, bias=False),
            # Global average pooling (kernel size = 8 for 8x8 feature maps)
            nn.AvgPool2d(8),
        )

    #############################################################################################################
    # FORWARD

    # Forward pass through the custom CNN.
    # Args:
    #   x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
    # Returns:
    #   torch.Tensor: Output logits of shape (batch_size, num_classes)
    def forward(self, x):
        # Input validation
        assert x.shape[1] == self.input_channels, f"Expected {self.input_channels} channels, got {x.shape[1]}"
        assert x.shape[2] == self.img_height, f"Expected height {self.img_height}, got {x.shape[2]}"
        assert x.shape[3] == self.img_width, f"Expected width {self.img_width}, got {x.shape[3]}"

        # ENCODER
        # Conv block 1: 512x512 -> 256x256
        x = self.conv_block_1(x)
        assert x.shape[1] == 64 and x.shape[2] == 256 and x.shape[3] == 256

        # Conv block 2: 256x256 -> 128x128
        x = self.conv_block_2(x)
        assert x.shape[1] == 128 and x.shape[2] == 128 and x.shape[3] == 128

        # Conv block 3: 128x128 -> 64x64
        x = self.conv_block_3(x)
        assert x.shape[1] == 256 and x.shape[2] == 64 and x.shape[3] == 64

        # Conv block 4: 64x64 -> 32x32
        x = self.conv_block_4(x)
        assert x.shape[1] == 512 and x.shape[2] == 32 and x.shape[3] == 32

        # Conv block 5: 32x32 -> 16x16
        x = self.conv_block_5(x)
        assert x.shape[1] == 512 and x.shape[2] == 16 and x.shape[3] == 16

        # Conv block 6: 16x16 -> 8x8
        x = self.conv_block_6(x)
        assert x.shape[1] == 512 and x.shape[2] == 8 and x.shape[3] == 8

        # DECODER
        x = self.decoder(x)

        # Reshape from [batch_size, num_classes, 1, 1] to [batch_size, num_classes]
        x = x.view(x.size(0), -1)

        assert x.shape[1] == self.num_classes

        return x