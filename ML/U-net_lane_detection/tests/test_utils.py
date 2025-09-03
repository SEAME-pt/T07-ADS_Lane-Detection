import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import check_accuracy, save_predictions_as_imgs, save_predictions_as_imgs_1, calculate_alpha
from model import UNET


def _make_small_loader(batch_size: int = 2, num_batches: int = 3):
    H, W = 144, 256
    x = torch.rand(num_batches * batch_size, 3, H, W)
    # Binary masks 0/1 with channel dim
    y = (torch.rand(num_batches * batch_size, 1, H, W) > 0.5).float()
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def test_check_accuracy_runs_cpu(tmp_path: Path):
    device = "cpu"
    loader = _make_small_loader()
    model = UNET(in_channels=3, out_channels=1).to(device)
    dice = check_accuracy(loader, model, device=device)
    assert 0.0 <= float(dice) <= 1.0


def test_save_predictions_as_imgs(tmp_path: Path):
    device = "cpu"
    out_dir = tmp_path / "preds"
    out_dir.mkdir(parents=True, exist_ok=True)
    loader = _make_small_loader(num_batches=1)
    model = UNET(in_channels=3, out_channels=1).to(device)
    save_predictions_as_imgs(loader, model, folder=str(out_dir), device=device)
    # Expect at least the prediction file inside the folder
    files = list(out_dir.glob("pred_*.png"))
    assert len(files) >= 1


def test_save_predictions_as_imgs_1(tmp_path: Path):
    device = "cpu"
    out_dir = tmp_path / "combined"
    out_dir.mkdir(parents=True, exist_ok=True)
    loader = _make_small_loader(num_batches=10)
    model = UNET(in_channels=3, out_channels=1).to(device)
    save_predictions_as_imgs_1(loader, model, epoch=0, folder=str(out_dir), device=device)
    files = list(out_dir.glob("combined_epoch0_batch*.png"))
    # Function saves every 10th batch starting at 0 → expect at least 1 file
    assert len(files) >= 1


def test_calculate_alpha(tmp_path: Path):
    loader = _make_small_loader(num_batches=2)
    alpha = calculate_alpha(loader)
    assert 0.0 <= float(alpha) <= 1.

