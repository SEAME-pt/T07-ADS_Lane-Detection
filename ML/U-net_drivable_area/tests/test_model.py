import torch
import torch.nn as nn
import pytest
import sys
from pathlib import Path

# Adiciona o diretório raiz ao sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import DoubleConv, UNET

def test_double_conv_init():
    in_channels = 3
    out_channels = 64
    dropout_rate = 0.1
    double_conv = DoubleConv(in_channels, out_channels, dropout_rate)
    
    assert isinstance(double_conv.conv, nn.Sequential)
    assert len(double_conv.conv) == 7
    assert isinstance(double_conv.conv[0], nn.Conv2d)
    assert double_conv.conv[0].in_channels == in_channels
    assert double_conv.conv[0].out_channels == out_channels
    assert isinstance(double_conv.conv[6], nn.Dropout)
    assert double_conv.conv[6].p == dropout_rate

def test_double_conv_forward():
    double_conv = DoubleConv(in_channels=3, out_channels=64)
    x = torch.randn(2, 3, 144, 256)
    output = double_conv(x)
    assert output.shape == (2, 64, 144, 256)
    assert output.dtype == torch.float32

def test_unet_init():
    in_channels = 3
    out_channels = 3
    features = [32, 64, 128, 256]
    unet = UNET(in_channels, out_channels, features)
    
    assert len(unet.downs) == len(features)
    assert len(unet.ups) == 2 * len(features)
    assert isinstance(unet.bottleneck, DoubleConv)
    assert isinstance(unet.final_conv, nn.Conv2d)
    assert unet.final_conv.out_channels == out_channels
    assert unet.bottleneck.conv[0].in_channels == features[-1]
    assert unet.bottleneck.conv[0].out_channels == features[-1] * 2

def test_unet_forward():
    unet = UNET(in_channels=3, out_channels=3, features=[32, 64, 128, 256])
    x = torch.randn(2, 3, 144, 256)
    output = unet(x)
    assert output.shape == (2, 3, 144, 256)
    assert output.dtype == torch.float32

def test_unet_weight_initialization():
    unet = UNET(in_channels=3, out_channels=3)
    for m in unet.modules():
        if isinstance(m, nn.Conv2d):
            assert torch.all(m.weight >= -1) and torch.all(m.weight <= 1)
            if m.bias is not None:
                assert torch.all(m.bias == 0)
        elif isinstance(m, nn.BatchNorm2d):
            assert torch.all(m.weight == 1)
            assert torch.all(m.bias == 0)
        elif isinstance(m, nn.ConvTranspose2d):
            assert torch.all(m.weight >= -1) and torch.all(m.weight <= 1)
            if m.bias is not None:
                assert torch.all(m.bias == 0)

def test_unet_forward_different_size():
    unet = UNET(in_channels=3, out_channels=3, features=[32, 64, 128, 256])
    x = torch.randn(2, 3, 160, 240)
    output = unet(x)
    assert output.shape == (2, 3, 160, 240)
    assert output.dtype == torch.float32