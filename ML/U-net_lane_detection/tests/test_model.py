import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import UNET, DoubleConv

def test_unet_forward_output_shape():
    """Test that UNET.forward produces the correct output shape.

    Verifies that the output shape matches [batch_size, out_channels, height, width]
    for a given input.
    """
    model = UNET(in_channels=3, out_channels=1)
    x = torch.randn(2, 3, 144, 256)
    y = model(x)
    assert y.shape == (2, 1, 144, 256), f"Expected shape (2, 1, 144, 256), got {y.shape}"

def test_unet_no_nan_inference():
    """Test that UNET.forward produces finite outputs (no NaN or Inf).

    Ensures the model output is numerically stable for a random input.
    """
    model = UNET(in_channels=3, out_channels=1)
    x = torch.randn(1, 3, 144, 256)
    with torch.no_grad():
        y = model(x)
    assert torch.isfinite(y).all(), "Output contains NaN or Inf values"

def test_unet_different_input_size():
    """Test UNET.forward with non-standard input size to trigger interpolation.

    Verifies that the model handles inputs requiring interpolation in the decoder path.
    """
    model = UNET(in_channels=3, out_channels=1)
    x = torch.randn(1, 3, 160, 320)  # Different size to trigger interpolation
    y = model(x)
    assert y.shape == (1, 1, 160, 320), f"Expected shape (1, 1, 160, 320), got {y.shape}"

def test_double_conv_forward():
    """Test DoubleConv.forward for correct output shape and functionality.

    Verifies that the double convolution block processes inputs correctly.
    """
    conv_block = DoubleConv(in_channels=3, out_channels=64)
    x = torch.randn(1, 3, 144, 256)
    y = conv_block(x)
    assert y.shape == (1, 64, 144, 256), f"Expected shape (1, 64, 144, 256), got {y.shape}"
    assert torch.isfinite(y).all(), "DoubleConv output contains NaN or Inf values"

def test_unet_initialization():
    """Test UNET initialization with custom parameters.

    Verifies that the model initializes with custom in_channels, out_channels, and features.
    """
    model = UNET(in_channels=1, out_channels=2, features=[32, 64, 128])
    assert len(model.downs) == 3, f"Expected 3 down blocks, got {len(model.downs)}"
    assert len(model.ups) == 6, f"Expected 6 up blocks (3 ConvTranspose2d + 3 DoubleConv), got {len(model.ups)}"
    assert isinstance(model.bottleneck, DoubleConv), "Bottleneck is not a DoubleConv instance"
    assert model.final_conv.out_channels == 2, f"Expected 2 output channels, got {model.final_conv.out_channels}"