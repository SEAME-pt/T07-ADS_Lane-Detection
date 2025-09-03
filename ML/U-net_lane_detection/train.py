import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
import torch.nn.functional as F
from dataset import train_transforms, val_transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler
from torch.amp import autocast
import logging
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    save_predictions_as_imgs_1,
    calculate_alpha,
)

# Configurations
"""Global configurations for training the lane detection model.

Attributes:
    LEARNING_RATE (float): Learning rate for the optimizer. Defaults to 0.001.
    DEVICE (str): Device for computation ('cuda' if available, else 'cpu').
    BATCH_SIZE (int): Number of samples per batch. Defaults to 16.
    NUM_EPOCHS (int): Number of training epochs. Defaults to 3.
    NUM_WORKERS (int): Number of subprocesses for data loading. Defaults to 2.
    IMAGE_HEIGHT (int): Height of input images (originally 1280). Defaults to 144.
    IMAGE_WIDTH (int): Width of input images (originally 1918). Defaults to 256.
    PIN_MEMORY (bool): Whether to pin memory for faster data transfer to GPU. Defaults to True.
    LOAD_MODEL (bool): Whether to load a pre-trained model. Defaults to False.
    TRAIN_IMG_DIR (str): Directory for training images. Defaults to 'data/train_little/'.
    TRAIN_MASK_DIR (str): Directory for training masks. Defaults to 'data/train_masks_little/'.
    VAL_IMG_DIR (str): Directory for validation images. Defaults to 'data/val_little/'.
    VAL_MASK_DIR (str): Directory for validation masks. Defaults to 'data/val_masks_little/'.
"""
LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 144
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/train_little/"
TRAIN_MASK_DIR = "data/train_masks_little/"
VAL_IMG_DIR = "data/val_little/"
VAL_MASK_DIR = "data/val_masks_little/"

# Logging configuration
logging.basicConfig(filename='training_log.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s')
logging.getLogger().addHandler(logging.StreamHandler())  # Also display logs in terminal

def train_fn(loader, model, optimizer, loss_fn_focal, loss_fn_dice, scaler=None, epoch=0):
    """Trains the model for one epoch.

    Performs a single training epoch, computing combined Focal and Dice losses, and updating model parameters.
    Supports mixed precision training when running on CUDA.

    Args:
        loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        model (torch.nn.Module): The U-Net model to train.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        loss_fn_focal (nn.Module): Focal loss function for imbalanced classes.
        loss_fn_dice (nn.Module): Dice loss function for segmentation.
        scaler (torch.cuda.amp.GradScaler, optional): Gradient scaler for mixed precision training. Defaults to None.
        epoch (int, optional): Current epoch number for logging. Defaults to 0.

    Returns:
        float: Mean training loss for the epoch.
    """
    model.train()  # Activate training mode
    loop = tqdm(loader)
    total_loss = 0
    total_batches = len(loader)
    
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(DEVICE)
        targets = targets.to(DEVICE).float()
        
        # Forward pass with mixed precision (optional)
        if scaler is not None and torch.cuda.is_available():
            with autocast('cuda'):
                predictions = model(data)
                loss_focal = loss_fn_focal(predictions, targets)
                loss_dice = loss_fn_dice(predictions, targets)
                loss = 0.7 * loss_focal + 1.3 * loss_dice
        else:
            predictions = model(data)
            loss_focal = loss_fn_focal(predictions, targets)
            loss_dice = loss_fn_dice(predictions, targets)
            loss = 0.7 * loss_focal + 1.3 * loss_dice
        
        # Backward pass
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        loop.set_postfix(loss=total_loss / (batch_idx + 1))  # Running average loss per batch
    
    mean_loss = total_loss / total_batches
    logging.info(f"Epoch {epoch} - Mean training loss: {mean_loss:.4f}")
    return mean_loss

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in binary segmentation tasks.

    Implements Focal Loss, which focuses training on hard examples by reducing the weight
    of easily classified samples.

    Args:
        alpha (float, optional): Weight for the positive class. Defaults to 0.95.
        gamma (float, optional): Focusing parameter to emphasize hard examples. Defaults to 2.0.
        reduction (str, optional): Reduction method for the loss ('mean', 'sum', or 'none'). Defaults to 'mean'.

    Attributes:
        alpha (float): Weight for the positive class.
        gamma (float): Focusing parameter.
        reduction (str): Reduction method.
    """
    def __init__(self, alpha=0.95, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Weight for positive class
        self.gamma = gamma  # Focusing factor
        self.reduction = reduction
        print(f"FocalLoss initialized with alpha={self.alpha}, gamma={self.gamma}. "
              f"Ensure alpha reflects dataset imbalance.")

    def forward(self, inputs, targets):
        """Computes the Focal Loss between predictions and targets.

        Args:
            inputs (torch.Tensor): Raw model outputs (logits) with shape [batch_size, 1, height, width].
            targets (torch.Tensor): Ground truth binary masks with shape [batch_size, 1, height, width], values in {0, 1}.

        Returns:
            torch.Tensor: Computed Focal Loss.
        """
        # Check for invalid values
        if torch.isnan(inputs).any() or torch.isinf(inputs).any():
            logging.warning("Inputs contain NaN or Inf.")
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)
        if torch.isnan(targets).any() or torch.isinf(targets).any():
            logging.warning("Targets contain NaN or Inf.")
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)
        
        # Clamp logits to avoid overflow
        inputs = torch.clamp(inputs, min=-100, max=100)
        
        # Compute BCE loss with logits
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Adjusted probabilities
        p_t = torch.sigmoid(inputs)
        p_t = targets * p_t + (1 - targets) * (1 - p_t)  # Corrected probability
        
        # Focusing factor: reduce weight of easy examples
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha-based weighting
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        
        # Final loss
        focal_loss = alpha_weight * focal_weight * bce_loss
        
        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation tasks.

    Measures the overlap between predicted and ground truth binary masks using the Dice coefficient.
    Adds a small smoothing factor to avoid division by zero.

    Args:
        smooth (float, optional): Smoothing factor to avoid division by zero. Defaults to 1e-8.

    Attributes:
        None
    """
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-8):
        """Computes the Dice Loss between predictions and targets.

        Args:
            inputs (torch.Tensor): Raw model outputs (logits) with shape [batch_size, 1, height, width].
            targets (torch.Tensor): Ground truth binary masks with shape [batch_size, 1, height, width], values in {0, 1}.
            smooth (float, optional): Smoothing factor to avoid division by zero. Defaults to 1e-8.

        Returns:
            torch.Tensor: Computed Dice Loss (1 - Dice coefficient).
        """
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice

def save_checkpoint(state, filename="model.pth.tar"):
    """Saves the model state to a file.

    Args:
        state (dict): Dictionary containing model and optimizer state dictionaries.
        filename (str, optional): Path to save the checkpoint. Defaults to 'model.pth.tar'.
    """
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    """Loads a saved model checkpoint.

    Args:
        checkpoint (dict): Checkpoint dictionary containing model state.
        model (torch.nn.Module): The U-Net model to load the state into.
    """
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def check_accuracy(loader, model, device="cuda"):
    """Evaluates the model on the validation dataset.

    Computes the Dice score for the validation dataset to assess model performance.

    Args:
        loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        model (torch.nn.Module): The U-Net model to evaluate.
        device (str, optional): Device to run the model. Defaults to 'cuda'.

    Returns:
        float: Mean Dice score for the validation dataset.
    """
    model.eval()
    total_dice = 0.0
    total_batches = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device).float()
            predictions = model(images)
            predictions = torch.sigmoid(predictions)
            predictions = (predictions > 0.5).float()
            intersection = (predictions * masks).sum()
            dice = (2. * intersection) / (predictions.sum() + masks.sum() + 1e-8)
            total_dice += dice.item()
            total_batches += 1

    mean_dice = total_dice / total_batches
    return mean_dice

def main():
    """Main training function.

    Orchestrates the training process, including model initialization, data loading, training,
    validation, and checkpoint saving for the U-Net model.
    """
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn_focal = FocalLoss(alpha=0.95, gamma=2.0)
    loss_fn_dice = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Load data
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transforms,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    best_dice_score = 0.0
    
    if LOAD_MODEL:
        load_checkpoint(torch.load("model.pth.tar"), model)
        print("Model loaded successfully!")
        dice_score_saved = check_accuracy(val_loader, model, device=DEVICE)
        best_dice_score = dice_score_saved

    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None  # Mixed precision
    
    print(f'Dice score: {best_dice_score}')
    
    for epoch in range(NUM_EPOCHS):
        # Training
        train_fn(train_loader, model, optimizer, loss_fn_focal, loss_fn_dice, scaler, epoch)
        
        # Validation
        dice_score = check_accuracy(val_loader, model, device=DEVICE)
        
        print(f"Epoch {epoch}: Dice Score = {dice_score:.4f}")
        
        if dice_score > best_dice_score:
            best_dice_score = dice_score
            checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
            save_checkpoint(checkpoint, filename="model.pth.tar")
            print("Model saved (best validation)!")
        
        # Save predictions as images
        save_predictions_as_imgs_1(val_loader, model, epoch, folder="saved_images/", device=DEVICE)

if __name__ == "__main__":
    main()