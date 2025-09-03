# Lane Segmentation with U-Net

![Architecture Diagram](https://miro.medium.com/max/1400/1*f7YOaE4TWubwaFF7Z1fzNw.png)
*U-Net Architecture Diagram (Reference)*

Implementation of a U-Net based deep learning model for lane segmentation in autonomous driving scenarios. Combines advanced techniques for handling class imbalance and data augmentation.

## Features 

-  Custom U-Net with dropout and batch normalization
-  Albumentations for data augmentation
-  Combined Focal Loss + Dice Loss for class imbalance
-  Mixed-precision training support
-  Comprehensive metrics (Pixel Accuracy, Dice Score)
-  Model checkpointing and prediction visualization

## Project Structure 

```plaintext
.
├── data/
│   ├── train/         		  # Training images (RGB)
│   ├── train_masks/   		  # Training masks (Binary)
│   ├── val/           		  # Validation images
│   └── val_masks/     		  # Validation masks
├── model.py                  # U-Net implementation
├── dataset.py                # Dataset & transforms
├── train.py                  # Training script
├── utils.py                  # Helper functions
├── saved_images/             # Prediction samples
├── model.pth.tar             # Trained model weights
└── README.md
```

## Requirements 

- Python 3.8+
- PyTorch 1.12+
- Torchvision 0.13+
- Albumentations 1.3+
- CUDA 11.6+ (recommended)
- NVIDIA GPU (recommended)
- Netron (optional, for model visualization)

```sh
pip install torch torchvision albumentations tqdm numpy pillow
```

## Dataset Preparation 

- Download dataset from Kaggle.com - Lane Detection for Carla Driving Simulator
- Organize files:
```plaintext
data/
├── train/
│   ├── 0001.png
│   ├── 0002.png
│   └── ...
├── train_masks/
│   ├── 0001_label.png
│   ├── 0002_label.png
│   └── ...
└── ... (similar for validation)
```

## Training

Configure parameters in train.py:

```sh
# Hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 16
NUM_EPOCHS = 10
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 258 
```

Start training:
```sh
python train.py
```

## Evaluation

Metrics are automatically calculated during training (example):
```sh
#Exmple
Epoch 89 - Mean training loss: 0.0615
Dice Score: 0.9305
```

## Model Architecture

```sh
class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        # Custom U-Net with:
        - DoubleConv blocks with dropout
        - Learnable upsampling
        - Kaiming weight initialization
        - Skip connections
```

## Model Visualization

[![Model Visualization](https://img.shields.io/badge/Model_Visualization-Netron-blue?style=plastic)](https://github.com/SEAME-pt/T07-ADS_Lane-Detection/blob/24---lane-detection-ML/U-net/assets/model.onnx.svg)


The U-Net model architecture has been visualized using Netron, a tool for inspecting neural network graphs. The model was exported to ONNX format (model_graph.onnx) and loaded into Netron to generate a detailed graph of the network's layers, connections, and parameters. This visualization aids in understanding the flow of data through the encoder-decoder structure and skip connections.

To view the graph:

- Install Netron: pip install netron or use the web version at netron.app.
- Run netron model_graph.onnx or upload model_graph.onnx to the Netron interface.


## Documentation

[![Docs](https://img.shields.io/badge/Doxygen-Documentation-blue?style=plastic)](https://github.com/SEAME-pt/T07-ADS_Lane-Detection/blob/24---lane-detection-ML/U-net/docs/refman.pdf)

Comprehensive documentation for the codebase has been generated using Doxygen. The documentation includes detailed descriptions of classes, functions, and modules, making it easier to understand the implementation details of the U-Net model, dataset handling, and training pipeline.

To view the documentation:

- Ensure Doxygen is installed (see Requirements).
- Navigate to the docs/ directory and open docs/html/index.html in a web browser.
- Alternatively, regenerate the documentation by running:

```sh
doxygen Doxyfile
```

from the docs/ directory.

## Loss Functions

Combined loss function:
```sh
loss = 0.7 * FocalLoss() + 1.3 * DiceLoss()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.95, gamma=2.0):
        # Handles class imbalance

class DiceLoss(nn.Module):
    def forward(self, inputs, targets):
        # Optimizes for segmentation overlap
```

## Performance
```plaintext
| Metric          | Validation |
|-----------------|:----------:|
| Accuracy        | 99.66%     |
| Dice Score      | 0.9305     |
| loss      	  | 0.0615     |
```
*Tested on NVIDIA GeForce GTX 1050Ti GPU*


![Results Preview](./saved_images/combined_epoch9_batch20.png)  
*Input | Binary Prediction | Raw Prediction*


---


Developed by: Team07 - SEA:ME Portugal

[![Team07](https://img.shields.io/badge/SEAME-Team07-blue?style=plastic)](https://github.com/orgs/SEAME-pt/teams/team07)

