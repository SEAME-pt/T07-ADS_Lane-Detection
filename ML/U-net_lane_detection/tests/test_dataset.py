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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset import LaneDataset, train_transforms, val_transforms

def test_dataset_len_without_aug(tmp_image_mask_dirs):
    image_dir, mask_dir = tmp_image_mask_dirs
    ds = LaneDataset(image_dir=image_dir, mask_dir=mask_dir, transform=None)
    assert len(ds.images) == len(ds)

def test_dataset_len_with_aug(tmp_image_mask_dirs):
    image_dir, mask_dir = tmp_image_mask_dirs
    transform = A.Compose([
        A.Resize(height=144, width=256, interpolation=cv2.INTER_CUBIC),
        A.HorizontalFlip(p=1.0),
        ToTensorV2(),
    ])
    ds = LaneDataset(image_dir=image_dir, mask_dir=mask_dir, transform=transform, num_augmentations=2)
    assert len(ds) == len(ds.images) * (1 + 2)

def test_dataset_getitem_shapes_and_types(tmp_image_mask_dirs):
    image_dir, mask_dir = tmp_image_mask_dirs
    ds = LaneDataset(image_dir=image_dir, mask_dir=mask_dir, transform=None)
    img, mask = ds[0]
    assert img.dim() == 3 and img.shape[0] == 3
    assert mask.dim() == 3 and mask.shape[0] == 1
    assert img.shape[1:] == mask.shape[1:] == (144, 256)
    assert mask.min().item() >= 0 and mask.max().item() <= 1

def test_dataset_getitem_with_aug(tmp_image_mask_dirs):
    image_dir, mask_dir = tmp_image_mask_dirs
    transform = A.Compose([
        A.Resize(height=144, width=256, interpolation=cv2.INTER_CUBIC),
        A.HorizontalFlip(p=1.0),
        ToTensorV2(),
    ])
    ds = LaneDataset(image_dir=image_dir, mask_dir=mask_dir, transform=transform, num_augmentations=2)
    img, mask = ds[1]
    assert img.dim() == 3 and img.shape[0] == 3
    assert mask.dim() == 3 and mask.shape[0] == 1
    assert img.shape[1:] == mask.shape[1:] == (144, 256)
    assert mask.min().item() >= 0 and mask.max().item() <= 1

def test_train_transforms(tmp_image_mask_dirs):
    image_dir, mask_dir = tmp_image_mask_dirs
    ds = LaneDataset(image_dir=image_dir, mask_dir=mask_dir, transform=train_transforms, num_augmentations=2)
    img, mask = ds[1]
    assert img.dim() == 3 and img.shape == (3, 144, 256)
    assert mask.dim() == 3 and mask.shape == (1, 144, 256)
    assert img.dtype == torch.float32
    assert mask.dtype == torch.float32
    assert mask.min().item() >= 0 and mask.max().item() <= 1

def test_val_transforms(tmp_image_mask_dirs):
    image_dir, mask_dir = tmp_image_mask_dirs
    ds = LaneDataset(image_dir=image_dir, mask_dir=mask_dir, transform=val_transforms, num_augmentations=0)
    img, mask = ds[0]
    assert img.dim() == 3 and img.shape == (3, 144, 256)
    assert mask.dim() == 3 and mask.shape == (1, 144, 256)
    assert img.dtype == torch.float32
    assert mask.dtype == torch.float32
    assert mask.min().item() >= 0 and mask.max().item() <= 1

def test_dataset_empty_dir(tmp_path: Path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    ds = LaneDataset(image_dir=str(empty_dir), mask_dir=str(empty_dir), transform=None)
    assert len(ds.images) == 0
    assert len(ds) == 0

def test_dataset_invalid_image(tmp_path: Path):
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()
    with open(image_dir / "invalid_image.png", "w") as f:
        f.write("not an image")
    with open(mask_dir / "invalid_image_label.png", "w") as f:
        f.write("not a mask")
    ds = LaneDataset(image_dir=str(image_dir), mask_dir=str(mask_dir), transform=None)
    with pytest.raises(Exception):  # PIL.UnidentifiedImageError para imagens inválidas
        ds[0]

def test_dataset_transform_synthetic():
    img_tensor = torch.randint(0, 256, (572, 572, 3), dtype=torch.uint8)
    mask_tensor = torch.randint(0, 2, (572, 572), dtype=torch.uint8)
    img = img_tensor.numpy()
    # mask = mask_tensor.numpy()
    mask = mask_tensor.numpy().astype(np.float32)  # Converte para float32
    transformed = train_transforms(image=img, mask=mask)
    dataset = [(transformed["image"], transformed["mask"].unsqueeze(0)) for _ in range(4)]  # Adiciona dimensão do canal
    loader = DataLoader(dataset, batch_size=4)
    data, targets = next(iter(loader))
    assert data.shape == (4, 3, 144, 256)
    assert targets.shape == (4, 1, 144, 256)
    assert data.dtype == torch.float32
    assert targets.dtype == torch.float32

def test_dataset_dataloader(tmp_image_mask_dirs):
    image_dir, mask_dir = tmp_image_mask_dirs
    transform = A.Compose([
        A.Resize(height=144, width=256, interpolation=cv2.INTER_CUBIC),  # Consistente com base_transform
        A.Rotate(limit=10),
        ToTensorV2()
    ])
    dataset = LaneDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        transform=transform,
        num_augmentations=2,
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    data, targets = next(iter(loader))
    assert data.shape[0] == 2
    assert data.shape[1:] == (3, 144, 256)
    assert targets.shape[1:] == (1, 144, 256)