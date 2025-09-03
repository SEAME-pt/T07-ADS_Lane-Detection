import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """Double convolution block for U-Net.

    Applies two 3x3 convolutions with ReLU activation and batch normalization.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.

    Attributes:
        conv (nn.Sequential): Sequential container of convolution, batch norm, and ReLU layers.
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
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
    """U-Net model for binary segmentation.

    Implements a U-Net architecture with encoder-decoder structure, skip connections,
    and double convolution blocks for lane detection.

    Args:
        in_channels (int, optional): Number of input channels (e.g., 3 for RGB images). Defaults to 3.
        out_channels (int, optional): Number of output channels (e.g., 1 for binary segmentation). Defaults to 1.
        features (list, optional): Number of features in each encoder/decoder level. Defaults to [64, 128, 256, 512].

    Attributes:
        downs (nn.ModuleList): List of encoder blocks (double convolution + max pooling).
        ups (nn.ModuleList): List of decoder blocks (upsampling + double convolution).
        bottleneck (DoubleConv): Bottleneck block at the bottom of the U-Net.
        final_conv (nn.Conv2d): Final 1x1 convolution to produce output channels.
        pool (nn.MaxPool2d): Max pooling layer for downsampling.
    """
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder (downsampling path)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder (upsampling path)
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        # Final convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        """Forward pass for the U-Net model.

        Processes the input through the encoder, bottleneck, and decoder, using skip connections
        to combine features from the encoder with the decoder.

        Args:
            x (torch.Tensor): Input tensor with shape [batch_size, in_channels, height, width].

        Returns:
            torch.Tensor: Output tensor with shape [batch_size, out_channels, height, width].
        """
        skip_connections = []

        # Encoder path
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            if x.shape != skip_connection.shape:
                x = torch.nn.functional.interpolate(x, size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        # Final convolution
        return self.final_conv(x)