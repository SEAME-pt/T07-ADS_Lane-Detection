import torch
from torch.utils.data import DataLoader, TensorDataset
import pytest
from unittest.mock import patch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import train
from train import FocalLoss, DiceLoss, main
from model import UNET

def _mini_loader(batch_size: int = 2, num_batches: int = 2):
    H, W = 144, 256
    x = torch.randn(num_batches * batch_size, 3, H, W)
    y = (torch.rand(num_batches * batch_size, 1, H, W) > 0.5).float()
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)

def test_losses_forward():
    with patch('builtins.print'):  # Mockar print
        logits = torch.randn(2, 1, 144, 256)
        targets = (torch.rand(2, 1, 144, 256) > 0.5).float()
        focal = FocalLoss(alpha=0.9, gamma=2.0, reduction='mean')
        dice = DiceLoss()
        lf = focal(logits, targets)
        ld = dice(logits, targets)
        assert torch.isfinite(lf).all()
        assert torch.isfinite(ld).all()
        assert lf.dim() == 0
        assert ld.dim() == 0

def test_focal_loss_reductions():
    with patch('builtins.print'):  # Mockar print
        logits = torch.randn(2, 1, 144, 256)
        targets = (torch.rand(2, 1, 144, 256) > 0.5).float()
        focal_sum = FocalLoss(alpha=0.9, gamma=2.0, reduction='sum')
        lf_sum = focal_sum(logits, targets)
        assert torch.isfinite(lf_sum).all()
        assert lf_sum.dim() == 0
        focal_none = FocalLoss(alpha=0.9, gamma=2.0, reduction='none')
        lf_none = focal_none(logits, targets)
        assert torch.isfinite(lf_none).all()
        assert lf_none.shape == logits.shape

def test_focal_loss_nan_inf():
    with patch('builtins.print'):  # Mockar print
        focal = FocalLoss(alpha=0.9, gamma=2.0)
        logits = torch.tensor([float('nan'), float('inf')], dtype=torch.float32).view(1, 1, 1, 2)
        targets = torch.tensor([0, 1], dtype=torch.float32).view(1, 1, 1, 2)
        with patch('train.logging.warning') as mock_logging:
            loss = focal(logits, targets)
            mock_logging.assert_called_with("Inputs contain NaN or Inf.")
        assert torch.isfinite(loss).any()

def test_train_one_epoch_cpu():
    device = "cpu"
    train.DEVICE = device
    model = UNET(in_channels=3, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    with patch('builtins.print'):  # Mockar print
        loss_fn_focal = FocalLoss(alpha=0.95, gamma=2.0)
        loss_fn_dice = DiceLoss()
        loader = _mini_loader(num_batches=2)
        mean_loss = train.train_fn(loader, model, optimizer, loss_fn_focal, loss_fn_dice, scaler=None, epoch=0)
        assert mean_loss >= 0

def test_train_one_epoch_with_scaler():
    device = "cpu"
    train.DEVICE = device
    model = UNET(in_channels=3, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    with patch('builtins.print'):  # Mockar print
        loss_fn_focal = FocalLoss(alpha=0.95, gamma=2.0)
        loss_fn_dice = DiceLoss()
        scaler = torch.amp.GradScaler('cpu')
        loader = _mini_loader(num_batches=2)
        mean_loss = train.train_fn(loader, model, optimizer, loss_fn_focal, loss_fn_dice, scaler=scaler, epoch=0)
        assert mean_loss >= 0

def test_train_fn_logging():
    device = "cpu"
    train.DEVICE = device
    model = UNET(in_channels=3, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    with patch('builtins.print'):  # Mockar print
        loss_fn_focal = FocalLoss(alpha=0.95, gamma=2.0)
        loss_fn_dice = DiceLoss()
        loader = _mini_loader(num_batches=2)
        with patch('train.logging.info') as mock_logging:
            mean_loss = train.train_fn(loader, model, optimizer, loss_fn_focal, loss_fn_dice, scaler=None, epoch=1)
            mock_logging.assert_called_with(f"Epoch 1 - Mean training loss: {mean_loss:.4f}")
        assert mean_loss >= 0

@patch('train.get_loaders')
@patch('train.load_checkpoint')
@patch('train.check_accuracy')
@patch('train.save_checkpoint')
@patch('train.save_predictions_as_imgs_1')
def test_main(mock_save_preds, mock_save_checkpoint, mock_check_accuracy, mock_load_checkpoint, mock_get_loaders):
    mock_get_loaders.return_value = (_mini_loader(), _mini_loader())
    mock_check_accuracy.side_effect = [0.9, 0.95]
    mock_load_checkpoint.return_value = None
    train.LOAD_MODEL = False
    train.NUM_EPOCHS = 2
    with patch('builtins.print') as mocked_print:
        train.main()
        mocked_print.assert_any_call(f'Dice score: 0.0')
        mocked_print.assert_any_call("Model saved (best validation)!")
    assert mock_get_loaders.called
    assert mock_check_accuracy.called
    assert mock_save_checkpoint.called
    assert mock_save_preds.called
    

def test_check_accuracy_cpu():
    device = "cpu"
    train.DEVICE = device
    model = UNET(in_channels=3, out_channels=1).to(device)
    loader = _mini_loader(num_batches=2)
    with patch('builtins.print'):  # evitar prints se houver
        score = train.check_accuracy(loader, model, device=device)
    # Verificações
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


# @patch('train.get_loaders')
# @patch('train.load_checkpoint')
# @patch('train.check_accuracy')
# @patch('train.save_checkpoint')
# @patch('train.save_predictions_as_imgs_1')
# @patch('torch.load')
# def test_main_with_load_model(mock_torch_load, mock_save_preds, mock_save_checkpoint, mock_check_accuracy, mock_load_checkpoint, mock_get_loaders):
#     mock_get_loaders.return_value = (_mini_loader(), _mini_loader())
#     mock_check_accuracy.side_effect = [0.8, 0.9]
#     mock_load_checkpoint.return_value = None
#     mock_torch_load.return_value = {"state_dict": {}, "optimizer": {}}
#     train.LOAD_MODEL = True
#     train.NUM_EPOCHS = 2
#     with patch('builtins.print') as mocked_print:
#         train.main()
#         mocked_print.assert_any_call("Modelo carregado com sucesso!")
#         mocked_print.assert_any_call("Modelo salvo (melhor validação)!")
#     assert mock_torch_load.called
#     assert mock_load_checkpoint.called
#     assert mock_get_loaders.called
#     assert mock_check_accuracy.called
#     assert mock_save_checkpoint.called
#     assert mock_save_preds.called

