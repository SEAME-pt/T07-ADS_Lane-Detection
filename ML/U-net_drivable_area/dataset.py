import os
import torch
import cv2
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
# import matplotlib.pyplot as plt

class MultiClassLaneDataset(Dataset):
    """A custom dataset for multiclass lane detection.

    Loads images and their corresponding multiclass masks (0: background, 1: driveable, 2: lanes),
    applies optional data augmentations, and returns them as PyTorch tensors for training or validation.

    Args:
        image_dir (str): Directory containing the input images (expected to be .jpg files).
        mask_dir (str): Directory containing the ground truth multiclass masks (expected to be .png files).
        transform (callable, optional): Albumentations pipeline for data augmentation. Defaults to None.
        num_augmentations (int, optional): Number of augmented versions per image. Defaults to 2.

    Attributes:
        image_dir (str): Directory with input images.
        mask_dir (str): Directory with ground truth masks.
        transform (callable): Data augmentation pipeline.
        num_augmentations (int): Number of augmentations per image.
        images (list): List of image filenames.
        total_samples (int): Total dataset size, including augmentations.
        base_transform (A.Compose): Base transformation pipeline to ensure fixed size and normalization.
    """
    def __init__(self, image_dir, mask_dir, transform=None, num_augmentations=2):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.num_augmentations = num_augmentations
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        self.total_samples = len(self.images) * (1 + self.num_augmentations if transform else 1)

        # Base transformation to ensure fixed size without augmentations
        self.base_transform = A.Compose([
            A.Resize(height=144, width=256, interpolation=cv2.INTER_NEAREST),
            A.Normalize(mean=[0.41, 0.39, 0.42], std=[0.15, 0.14, 0.15], max_pixel_value=255.0),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})

    def __len__(self):
        """Returns the total number of samples in the dataset.

        The total includes the original images and their augmented versions, if applicable.

        Returns:
            int: Total number of samples.
        """
        return self.total_samples

    def __getitem__(self, index):
        """Retrieves an image and its corresponding multiclass mask by index.

        Loads an image and mask, applies augmentations if specified, and returns them as tensors.
        Validates that the mask contains only valid class values (0, 1, 2).

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - image (torch.Tensor): Image tensor with shape [C, H, W].
                - mask (torch.Tensor): Mask tensor with shape [H, W], containing values 0 (background),
                  1 (driveable), or 2 (lanes).

        Raises:
            AssertionError: If the mask contains invalid class values (not in [0, 1, 2]).
        """
        samples_per_image = (1 + self.num_augmentations) if self.transform else 1
        img_idx = index // samples_per_image
        sample_idx = index % samples_per_image
        
        img_path = os.path.join(self.image_dir, self.images[img_idx])
        mask_path = os.path.join(self.mask_dir, self.images[img_idx].replace('.jpg', '_mask.png'))
        
        # Load as NumPy arrays
        image = np.asarray(Image.open(img_path).convert("RGB"))  # [H, W, 3]
        mask = np.asarray(Image.open(mask_path), dtype=np.int64)  # [H, W], values 0, 1, 2
        
        # Apply transformations
        if self.transform is not None and sample_idx > 0:
            # Apply augmentations for transformed versions
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]  # [C, H, W]
            mask = augmentations["mask"]    # [H, W]
        else:
            # Apply base transformation for original image
            augmentations = self.base_transform(image=image, mask=mask)
            image = augmentations["image"]  # [C, H, W]
            mask = augmentations["mask"]    # [H, W]
        
        # Convert mask to torch.long for CrossEntropyLoss
        mask = mask.clone().detach().to(dtype=torch.long)
        assert torch.all(torch.isin(mask, torch.tensor([0, 1, 2], dtype=torch.long))), \
            f"Mask {mask_path} contains invalid values: {torch.unique(mask).tolist()}"
        
        return image, mask

train_transforms = A.Compose([
    A.Resize(height=144, width=256, interpolation=cv2.INTER_NEAREST),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.RandomGamma(p=0.5),
    A.Normalize(mean=[0.41, 0.39, 0.42], std=[0.15, 0.14, 0.15], max_pixel_value=255.0),
    ToTensorV2(),
], additional_targets={'mask': 'mask'})
"""Training data augmentation pipeline.

Applies a series of transformations to augment images and masks for training:
- Resizes images and masks to 144x256 pixels.
- Randomly flips images horizontally with 50% probability.
- Adjusts brightness and contrast randomly with 50% probability.
- Adjusts gamma to enhance lane lines with 50% probability.
- Normalizes pixel values using specified mean and standard deviation.
- Converts images and masks to PyTorch tensors.

Args:
    image (numpy.ndarray): Input image with shape [H, W, 3].
    mask (numpy.ndarray): Input mask with shape [H, W].

Returns:
    dict: Dictionary containing augmented image and mask tensors.
"""

val_transforms = A.Compose([
    A.Resize(height=144, width=256, interpolation=cv2.INTER_NEAREST),
    A.Normalize(mean=[0.41, 0.39, 0.42], std=[0.15, 0.14, 0.15], max_pixel_value=255.0),
    ToTensorV2(),
], additional_targets={'mask': 'mask'})
"""Validation data transformation pipeline.

Applies minimal transformations to prepare images and masks for validation:
- Resizes images and masks to 144x256 pixels.
- Normalizes pixel values using specified mean and standard deviation.
- Converts images and masks to PyTorch tensors.

Args:
    image (numpy.ndarray): Input image with shape [H, W, 3].
    mask (numpy.ndarray): Input mask with shape [H, W].

Returns:
    dict: Dictionary containing transformed image and mask tensors.
"""