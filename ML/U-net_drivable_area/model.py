import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    """A double convolution block for the U-Net architecture.

    Applies two consecutive convolutional layers, each followed by batch normalization, ReLU activation,
    and a dropout layer to prevent overfitting.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        dropout_rate (float, optional): Dropout probability to prevent overfitting. Defaults to 0.1.

    Attributes:
        conv (nn.Sequential): Sequential container of the double convolution layers, including
            Conv2d, BatchNorm2d, ReLU, and Dropout.
    """
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        """Forward pass for the double convolution block.

        Args:
            x (torch.Tensor): Input tensor with shape [batch_size, in_channels, height, width].

        Returns:
            torch.Tensor: Output tensor with shape [batch_size, out_channels, height, width].
        """
        return self.conv(x)

class UNET(nn.Module):
    """U-Net architecture for semantic segmentation.

    Implements a U-Net with an encoder-decoder structure, including skip connections, for multiclass
    segmentation tasks. The model consists of a downsampling path (encoder), a bottleneck, and an
    upsampling path (decoder), with a final convolution to produce the output segmentation map for
    three classes (background, driveable, lanes).

    Args:
        in_channels (int, optional): Number of input channels (e.g., 3 for RGB images). Defaults to 3.
        out_channels (int, optional): Number of output channels/classes (e.g., 3 for multiclass segmentation).
            Defaults to 3.
        features (list, optional): List of feature sizes for each level in the encoder and decoder.
            Defaults to [32, 64, 128, 256].

    Attributes:
        ups (nn.ModuleList): List of upsampling layers (ConvTranspose2d and DoubleConv).
        downs (nn.ModuleList): List of downsampling layers (DoubleConv).
        pool (nn.MaxPool2d): Max pooling layer for downsampling.
        bottleneck (DoubleConv): Bottleneck layer connecting encoder and decoder.
        final_conv (nn.Conv2d): Final convolution layer to produce the output segmentation map.
    """
    def __init__(self, in_channels=3, out_channels=3, features=[32, 64, 128, 256]):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Downsampling path (encoder)
        for feature in features:
            dropout_rate = 0.1 if feature <= 128 else 0.2
            self.downs.append(DoubleConv(in_channels, feature, dropout_rate))
            in_channels = feature

        # Upsampling path (decoder)
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            dropout_rate = 0.1 if feature <= 128 else 0.2
            self.ups.append(DoubleConv(feature*2, feature, dropout_rate))

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1]*2, dropout_rate=0.3)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        self.initialize_weights()

    def forward(self, x):
        """Forward pass for the U-Net model.

        Processes the input through the downsampling path, bottleneck, and upsampling path with skip
        connections, producing a segmentation map. Resizes tensors if necessary to match skip connection
        dimensions.

        Args:
            x (torch.Tensor): Input tensor with shape [batch_size, in_channels, height, width].

        Returns:
            torch.Tensor: Output segmentation map with shape [batch_size, out_channels, height, width].
        """
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

    def initialize_weights(self):
        """Initializes weights for Conv2d, ConvTranspose2d, and BatchNorm2d layers.

        Uses Kaiming uniform initialization for convolutional layers and constant initialization
        for batch normalization layers to improve training stability.

        Notes:
            - Conv2d and ConvTranspose2d weights are initialized with Kaiming uniform (ReLU nonlinearity).
            - BatchNorm2d weights are set to 1, and biases are set to 0.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)