import os
import torch
import cv2
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

class MultiClassLaneDataset(Dataset):
    """!
    @brief A custom dataset for multiclass lane detection.

    This class loads images and their corresponding multiclass masks (0: background, 1: driveable, 2: lanes),
    applies optional data augmentations, and returns them as PyTorch tensors for training or validation.

    @param image_dir (str): Directory containing the input images.
    @param mask_dir (str): Directory containing the ground truth multiclass masks.
    @param transform (albumentations.Compose, optional): Data augmentation pipeline (default: None).
    @param num_augmentations (int, optional): Number of augmented versions per image (default: 2).
    """
    def __init__(self, image_dir, mask_dir, transform=None, num_augmentations=2):
        self.image_dir = image_dir  #!< Directory with input images.
        self.mask_dir = mask_dir  #!< Directory with ground truth masks.
        self.transform = transform  #!< Optional augmentation pipeline.
        self.num_augmentations = num_augmentations  #!< Number of augmentations per image.
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]  #!< List of image filenames.
        self.total_samples = len(self.images) * (1 + self.num_augmentations if transform else 1)  #!< Total dataset size.

        # Base transformation to ensure fixed size without augmentations
        self.base_transform = A.Compose([
            A.Resize(height=144, width=256, interpolation=cv2.INTER_NEAREST),
            A.Normalize(mean=[0.41, 0.39, 0.42], std=[0.15, 0.14, 0.15], max_pixel_value=255.0),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})

    def __len__(self):
        """!
        @brief Returns the total number of samples in the dataset.

        @return int: Total number of samples, including augmentations.
        """
        return self.total_samples

    def __getitem__(self, index):
        """!
        @brief Retrieves an image and its corresponding multiclass mask by index.

        Loads an image and mask, applies augmentations if specified, and returns them as tensors.

        @param index (int): Index of the sample to retrieve.
        @return tuple: A tuple containing the image tensor ([C, H, W]) and mask tensor ([H, W]).
        """
        samples_per_image = (1 + self.num_augmentations) if self.transform else 1
        img_idx = index // samples_per_image
        sample_idx = index % samples_per_image
        
        img_path = os.path.join(self.image_dir, self.images[img_idx])
        mask_path = os.path.join(self.mask_dir, self.images[img_idx].replace('.jpg', '_mask.png'))
        
        # Load as NumPy arrays
        image = np.array(Image.open(img_path).convert("RGB"))  # [H, W, 3]
        mask = np.array(Image.open(mask_path), dtype=np.int64)  # [H, W], values 0, 1, 2
        
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
            f"Máscara {mask_path} contém valores inválidos: {torch.unique(mask).tolist()}"
        
        return image, mask

# Transformations for training
train_transforms = A.Compose([
    A.Resize(height=144, width=256, interpolation=cv2.INTER_NEAREST),
    A.HorizontalFlip(p=0.5),  #!< Randomly flip images horizontally.
    A.RandomBrightnessContrast(p=0.5),  #!< Adjust brightness and contrast randomly.
    A.RandomGamma(p=0.5),  #!< Adjust gamma to enhance lane lines.
    A.Normalize(mean=[0.41, 0.39, 0.42], std=[0.15, 0.14, 0.15], max_pixel_value=255.0),
    ToTensorV2(),
], additional_targets={'mask': 'mask'})

# Transformations for validation
val_transforms = A.Compose([
    A.Resize(height=144, width=256, interpolation=cv2.INTER_NEAREST),
    A.Normalize(mean=[0.41, 0.39, 0.42], std=[0.15, 0.14, 0.15], max_pixel_value=255.0),
    ToTensorV2(),
], additional_targets={'mask': 'mask'})