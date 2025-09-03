import os
import sys
from pathlib import Path
import pytest
import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import DataLoader
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset import MultiClassLaneDataset, train_transforms, val_transforms

def test_dataset_len_without_aug(tmp_image_mask_dirs):
    image_dir, mask_dir = tmp_image_mask_dirs
    ds = MultiClassLaneDataset(image_dir=image_dir, mask_dir=mask_dir, transform=None)
    assert len(ds.images) == len(ds)

def test_dataset_len_with_aug(tmp_image_mask_dirs):
    image_dir, mask_dir = tmp_image_mask_dirs
    
    transform = A.Compose([
        A.Resize(height=144, width=256, interpolation=cv2.INTER_CUBIC),
        A.HorizontalFlip(p=1.0),
        ToTensorV2(),
    ], additional_targets={'mask': 'mask'})
    
    ds = MultiClassLaneDataset(image_dir=image_dir, mask_dir=mask_dir, transform=transform, num_augmentations=2)
    assert len(ds) == len(ds.images) * (1 + 2)

def test_dataset_getitem_shapes_and_types(tmp_image_mask_dirs):
    image_dir, mask_dir = tmp_image_mask_dirs
    ds = MultiClassLaneDataset(image_dir=image_dir, mask_dir=mask_dir, transform=None)
    img, mask = ds[0]
    assert img.dim() == 3 and img.shape[0] == 3
    assert mask.dim() == 2
    assert img.shape[1:] == mask.shape == (144, 256)
    assert img.dtype == torch.float32
    assert mask.dtype == torch.long
    assert mask.min().item() >= 0 and mask.max().item() <= 2

def test_dataset_getitem_with_aug(tmp_image_mask_dirs):
    image_dir, mask_dir = tmp_image_mask_dirs
    
    # transform = A.Compose([
    #     A.Resize(height=144, width=256, interpolation=cv2.INTER_CUBIC),
    #     A.HorizontalFlip(p=1.0),
    #     ToTensorV2(),
    # ], additional_targets={'mask': 'mask'})
    
    transform = A.Compose([
            A.Resize(height=144, width=256, interpolation=cv2.INTER_NEAREST),
            # A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
            A.Normalize(mean=[0.41, 0.39, 0.42], std=[0.15, 0.14, 0.15], max_pixel_value=255.0),
            ToTensorV2(),
        ], additional_targets={'mask': 'mask'})
    
    ds = MultiClassLaneDataset(image_dir=image_dir, mask_dir=mask_dir, transform=transform, num_augmentations=2)
    img, mask = ds[1]
    assert img.dim() == 3 and img.shape[0] == 3
    assert mask.dim() == 2
    assert img.shape[1:] == mask.shape == (144, 256)
    assert img.dtype == torch.float32
    assert mask.dtype == torch.long
    assert mask.min().item() >= 0 and mask.max().item() <= 2

def test_train_transforms(tmp_image_mask_dirs):
    image_dir, mask_dir = tmp_image_mask_dirs
    ds = MultiClassLaneDataset(image_dir=image_dir, mask_dir=mask_dir, transform=train_transforms, num_augmentations=2)
    img, mask = ds[1]
    assert img.dim() == 3 and img.shape == (3, 144, 256)
    assert mask.dim() == 2 and mask.shape == (144, 256)
    assert img.dtype == torch.float32
    assert mask.dtype == torch.long
    assert mask.min().item() >= 0 and mask.max().item() <= 2

def test_val_transforms(tmp_image_mask_dirs):
    image_dir, mask_dir = tmp_image_mask_dirs
    ds = MultiClassLaneDataset(image_dir=image_dir, mask_dir=mask_dir, transform=val_transforms, num_augmentations=0)
    img, mask = ds[0]
    assert img.dim() == 3 and img.shape == (3, 144, 256)
    assert mask.dim() == 2 and mask.shape == (144, 256)
    assert img.dtype == torch.float32
    assert mask.dtype == torch.long
    assert mask.min().item() >= 0 and mask.max().item() <= 2

def test_dataset_empty_dir(tmp_path: Path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    ds = MultiClassLaneDataset(image_dir=str(empty_dir), mask_dir=str(empty_dir), transform=None)
    assert len(ds.images) == 0
    assert len(ds) == 0

def test_dataset_invalid_image(tmp_path: Path):
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()
    with open(image_dir / "invalid_image.jpg", "w") as f:
        f.write("not an image")
    with open(mask_dir / "invalid_image_mask.png", "w") as f:
        f.write("not a mask")
    ds = MultiClassLaneDataset(image_dir=str(image_dir), mask_dir=str(mask_dir), transform=None)
    with pytest.raises(Exception):  # PIL.UnidentifiedImageError
        ds[0]

def test_dataset_transform_synthetic():
    img_tensor = torch.randint(0, 256, (572, 572, 3), dtype=torch.uint8)
    mask_tensor = torch.randint(0, 3, (572, 572), dtype=torch.int64)  # Multiclass mask as int64
    img = img_tensor.numpy()
    mask = mask_tensor.numpy()
    transformed = train_transforms(image=img, mask=mask)
    image = transformed["image"]
    mask = torch.as_tensor(transformed["mask"], dtype=torch.long)  # Convert to long
    dataset = [(image, mask) for _ in range(4)]
    loader = DataLoader(dataset, batch_size=4)
    data, targets = next(iter(loader))
    assert data.shape == (4, 3, 144, 256)
    assert targets.shape == (4, 144, 256)
    assert data.dtype == torch.float32
    assert targets.dtype == torch.long
    assert targets.min().item() >= 0 and targets.max().item() <= 2

def test_dataset_dataloader(tmp_image_mask_dirs):
    image_dir, mask_dir = tmp_image_mask_dirs
    transform = A.Compose([
        A.Resize(height=144, width=256, interpolation=cv2.INTER_CUBIC),
        A.Rotate(limit=10, p=1.0),
        ToTensorV2(),
    ], additional_targets={'mask': 'mask'})
    dataset = MultiClassLaneDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        transform=transform,
        num_augmentations=2,
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    data, targets = next(iter(loader))
    assert data.shape[0] == 2
    assert data.shape[1:] == (3, 144, 256)
    assert targets.shape[1:] == (144, 256)
    assert data.dtype == torch.float32
    assert targets.dtype == torch.long
    assert targets.min().item() >= 0 and targets.max().item() <= 2

def test_train_transforms_full(tmp_image_mask_dirs):
    image_dir, mask_dir = tmp_image_mask_dirs
    # transform = A.Compose([
    #     A.Resize(height=144, width=256, interpolation=cv2.INTER_CUBIC),
    #     A.HorizontalFlip(p=1.0),
    #     A.RandomBrightnessContrast(p=1.0),
    #     A.RandomGamma(p=1.0),
    #     ToTensorV2(),
    # ], additional_targets={'mask': 'mask'})
    
    transform = A.Compose([
    	A.Resize(height=144, width=256, interpolation=cv2.INTER_NEAREST),
    	A.HorizontalFlip(p=0.5),  #!< Randomly flip images horizontally.
    	A.RandomBrightnessContrast(p=0.5),  #!< Adjust brightness and contrast randomly.
    	A.RandomGamma(p=0.5),  #!< Adjust gamma to enhance lane lines.
    	A.Normalize(mean=[0.41, 0.39, 0.42], std=[0.15, 0.14, 0.15], max_pixel_value=255.0),
    	ToTensorV2(),
	], additional_targets={'mask': 'mask'})
    ds = MultiClassLaneDataset(image_dir=image_dir, mask_dir=mask_dir, transform=transform, num_augmentations=2)
    img, mask = ds[1]
    assert img.dim() == 3 and img.shape == (3, 144, 256)
    assert mask.dim() == 2 and mask.shape == (144, 256)
    assert img.dtype == torch.float32
    assert mask.dtype == torch.long
    assert mask.min().item() >= 0 and mask.max().item() <= 2