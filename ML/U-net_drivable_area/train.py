import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchmetrics import JaccardIndex
from tqdm import tqdm
import os
import logging
import numpy as np
from PIL import Image
from model import UNET
from dataset import MultiClassLaneDataset, train_transforms, val_transforms

# Configurations
"""Global configurations for training the lane detection model.

Attributes:
    device (torch.device): Device to run the model ('cuda' if available, else 'cpu').
    batch_size (int): Number of samples per batch. Defaults to 8.
    num_epochs (int): Number of training epochs. Defaults to 1.
    learning_rate (float): Learning rate for the optimizer. Defaults to 1e-3.
    save_dir (str): Directory to save model checkpoints. Defaults to 'checkpoints'.
    saved_images_dir (str): Directory to save prediction images. Defaults to 'saved_images'.
    LOAD_MODEL (bool): Whether to load a pre-trained model. Defaults to False.
    patience (int): Number of epochs to wait for improvement before early stopping. Defaults to 5.
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 8
num_epochs = 1
learning_rate = 1e-3
save_dir = 'checkpoints'
saved_images_dir = 'saved_images'
LOAD_MODEL = False
patience = 5  # For early stopping
os.makedirs(save_dir, exist_ok=True)
os.makedirs(saved_images_dir, exist_ok=True)

# Logging configuration
logging.basicConfig(filename='training_log.log', level=logging.INFO, 
                    format='%(asctime)s - %(message)s')
logging.getLogger().addHandler(logging.StreamHandler())

def check_masks(image_dir, mask_dir, dataset_type="train"):
    """Checks for missing image-mask pairs in the dataset.

    Verifies that each image has a corresponding mask and vice versa, reporting any mismatches.

    Args:
        image_dir (str): Directory containing input images (expected to be .jpg files).
        mask_dir (str): Directory containing ground truth masks (expected to be _mask.png files).
        dataset_type (str, optional): Type of dataset ('train' or 'val'). Defaults to 'train'.

    Returns:
        list: List of error messages for mismatched image-mask pairs.
    """
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]
    mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith('_mask.png')]
    image_basenames = {os.path.splitext(f)[0]: f for f in image_files}
    mask_basenames = {os.path.splitext(f.replace('_mask', ''))[0]: f for f in mask_files}
    errors = []
    for base_name in image_basenames:
        if base_name not in mask_basenames:
            errors.append(f"Image without mask in {dataset_type}: {image_basenames[base_name]}")
    for base_name in mask_basenames:
        if base_name not in image_basenames:
            errors.append(f"Mask without image in {dataset_type}: {mask_basenames[base_name]}")
    return errors

class FocalLoss_w(nn.Module):
    """Focal Loss with class weights for multiclass segmentation.

    Adjusts the loss to focus on hard-to-classify examples and balance class importance using weights.

    Args:
        gamma (float, optional): Focusing parameter to emphasize hard examples. Defaults to 2.0.
        alpha (torch.Tensor, optional): Class weights for balancing (shape [num_classes]). Defaults to None.
        reduction (str, optional): Reduction method ('mean' or 'sum'). Defaults to 'mean'.

    Attributes:
        gamma (float): Focusing parameter.
        alpha (torch.Tensor): Class weights.
        reduction (str): Reduction method.
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss_w, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        logging.info(f"FocalLoss_w initialized with gamma={self.gamma}, alpha={self.alpha}")

    def forward(self, inputs, targets):
        """Forward pass for Focal Loss with weights.

        Args:
            inputs (torch.Tensor): Model predictions with shape [batch_size, num_classes, height, width].
            targets (torch.Tensor): Ground truth labels with shape [batch_size, height, width], values in {0, 1, 2}.

        Returns:
            torch.Tensor: Computed focal loss.
        """
        CE_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        focal_loss = (1 - pt) ** self.gamma * CE_loss
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()

class FocalLoss(nn.Module):
    """Focal Loss for multiclass segmentation.

    Focuses on hard-to-classify examples with a fixed alpha weight for foreground classes.

    Args:
        alpha (float, optional): Weight for foreground classes (values > 0). Defaults to 0.95.
        gamma (float, optional): Focusing parameter to emphasize hard examples. Defaults to 2.0.
        reduction (str, optional): Reduction method ('mean' or 'sum'). Defaults to 'mean'.

    Attributes:
        alpha (float): Weight for foreground classes.
        gamma (float): Focusing parameter.
        reduction (str): Reduction method.
    """
    def __init__(self, alpha=0.95, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """Forward pass for Focal Loss.

        Args:
            inputs (torch.Tensor): Model predictions with shape [batch_size, num_classes, height, width].
            targets (torch.Tensor): Ground truth labels with shape [batch_size, height, width], values in {0, 1, 2}.

        Returns:
            torch.Tensor: Computed focal loss.
        """
        inputs = torch.clamp(inputs, min=-100, max=100)
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        p_t = torch.softmax(inputs, dim=1)
        p_t = torch.gather(p_t, dim=1, index=targets.unsqueeze(1).long()).squeeze(1)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_weight = self.alpha * (targets > 0).float() + (1 - self.alpha) * (targets == 0).float()
        focal_loss = alpha_weight * focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DiceLoss(nn.Module):
    """Dice Loss for multiclass segmentation.

    Measures the overlap between predicted and ground truth segmentation maps using the Dice coefficient.

    Args:
        smooth (float, optional): Smoothing factor to avoid division by zero. Defaults to 1e-8.

    Attributes:
        smooth (float): Smoothing factor.
    """
    def __init__(self, smooth=1e-8):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        """Forward pass for Dice Loss.

        Args:
            inputs (torch.Tensor): Model predictions with shape [batch_size, num_classes, height, width].
            targets (torch.Tensor): Ground truth labels with shape [batch_size, height, width], values in {0, 1, 2}.

        Returns:
            torch.Tensor: Computed Dice loss (1 - Dice coefficient).
        """
        inputs = torch.softmax(inputs, dim=1)
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=3).permute(0, 3, 1, 2).float()
        inputs = inputs.reshape(-1)
        targets_one_hot = targets_one_hot.reshape(-1)
        intersection = (inputs * targets_one_hot).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets_one_hot.sum() + self.smooth)
        return 1 - dice

def save_predictions_as_imgs(loader, model, epoch, folder="saved_images/", device="cuda"):
    """Saves model predictions as images alongside input images and ground truth masks.

    Generates side-by-side images of the input, ground truth mask, and predicted mask for visualization.
    Images are saved in the specified folder with filenames indicating the epoch and batch index.

    Args:
        loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        model (torch.nn.Module): The U-Net model to evaluate.
        epoch (int): Current epoch number.
        folder (str, optional): Directory to save images. Defaults to 'saved_images/'.
        device (str, optional): Device to run the model. Defaults to 'cuda'.
    """
    model.eval()
    colors = np.array([
        [0, 0, 0],      # Class 0: background (black)
        [255, 0, 0],    # Class 1: driveable (red)
        [0, 255, 0],    # Class 2: lanes (green)
    ], dtype=np.uint8)

    with torch.no_grad():
        for idx, (images, masks) in enumerate(loader):
            images = images.to(device)
            preds = model(images)
            preds = torch.argmax(preds, dim=1).cpu().numpy()
            masks = masks.cpu().numpy()
            images = images.cpu().numpy()

            for i in range(images.shape[0]):
                img = images[i].transpose(1, 2, 0)
                img = (img * np.array([0.15, 0.14, 0.15]) + np.array([0.41, 0.39, 0.42])) * 255
                img = img.astype(np.uint8)

                mask_true = masks[i]
                mask_true_colored = colors[mask_true]
                mask_pred = preds[i]
                mask_pred_colored = colors[mask_pred]

                combined_width = img.shape[1] * 3
                combined_img = np.zeros((img.shape[0], combined_width, 3), dtype=np.uint8)
                combined_img[:, 0:img.shape[1], :] = img
                combined_img[:, img.shape[1]:2*img.shape[1], :] = mask_true_colored
                combined_img[:, 2*img.shape[1]:, :] = mask_pred_colored

                combined_img_pil = Image.fromarray(combined_img)
                combined_img_pil.save(os.path.join(folder, f"epoch_{epoch}_combined_{idx}_{i}.png"))

            break

def train_fn(loader, model, optimizer, loss_fn_focal, loss_fn_dice, scaler, epoch):
    """Trains the model for one epoch.

    Performs a single training epoch, computing combined Focal and Dice losses, and updating model parameters.
    Supports mixed precision training when running on CUDA.

    Args:
        loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        model (torch.nn.Module): The U-Net model to train.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        loss_fn_focal (nn.Module): Focal loss function.
        loss_fn_dice (nn.Module): Dice loss function.
        scaler (torch.cuda.amp.GradScaler): Gradient scaler for mixed precision training.
        epoch (int): Current epoch number.

    Returns:
        tuple: A tuple containing:
            - mean_loss (float): Mean training loss for the epoch.
            - mean_iou (float): Mean IoU (Jaccard Index) for the epoch.
    """
    model.train()
    loop = tqdm(loader, desc=f'Epoch {epoch+1}/{num_epochs}')
    total_loss = 0.0
    total_iou = 0.0
    jaccard = JaccardIndex(task='multiclass', num_classes=3).to(device)

    for batch_idx, (images, masks) in enumerate(loop):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        if device.type == 'cuda' and scaler is not None:
            with autocast():
                outputs = model(images)
                loss_focal = loss_fn_focal(outputs, masks)
                loss_dice = loss_fn_dice(outputs, masks)
                loss = 0.7 * loss_focal + 1.3 * loss_dice
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss_focal = loss_fn_focal(outputs, masks)
            loss_dice = loss_fn_dice(outputs, masks)
            loss = 0.7 * loss_focal + 1.3 * loss_dice
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_iou += jaccard(outputs, masks).item()
        loop.set_postfix(loss=total_loss / (batch_idx + 1), mIoU=total_iou / (batch_idx + 1))

    mean_loss = total_loss / len(loader)
    mean_iou = total_iou / len(loader)
    logging.info(f"Epoch {epoch+1} - Train Loss: {mean_loss:.4f}, Train mIoU: {mean_iou:.4f}")
    return mean_loss, mean_iou

def check_accuracy(loader, model, loss_fn_focal, loss_fn_dice, device):
    """Evaluates the model on the validation dataset.

    Computes the validation loss (combined Focal and Dice) and mean IoU for the given dataset.

    Args:
        loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        model (torch.nn.Module): The U-Net model to evaluate.
        loss_fn_focal (nn.Module): Focal loss function.
        loss_fn_dice (nn.Module): Dice loss function.
        device (str): Device to run the model.

    Returns:
        tuple: A tuple containing:
            - mean_loss (float): Mean validation loss.
            - mean_iou (float): Mean IoU (Jaccard Index).
    """
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    jaccard = JaccardIndex(task='multiclass', num_classes=3).to(device)

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss_focal = loss_fn_focal(outputs, masks)
            loss_dice = loss_fn_dice(outputs, masks)
            loss = 0.7 * loss_focal + 1.3 * loss_dice

            total_loss += loss.item()
            total_iou += jaccard(outputs, masks).item()

    mean_loss = total_loss / len(loader)
    mean_iou = total_iou / len(loader)
    return mean_loss, mean_iou

def load_checkpoint(checkpoint, model):
    """Loads a saved model checkpoint.

    Args:
        checkpoint (dict): Checkpoint dictionary containing model state.
        model (torch.nn.Module): The U-Net model to load the state into.
    """
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def main():
    """Main training function.

    Orchestrates the training process, including dataset loading, model training, validation, early stopping,
    and checkpoint saving.
    """
    # Verify dataset integrity
    train_errors = check_masks('dataset/train', 'dataset/train_masks', 'train')
    val_errors = check_masks('dataset/val', 'dataset/val_masks', 'val')
    if train_errors or val_errors:
        print("Dataset errors:", train_errors + val_errors)
        exit()

    # Dataset and DataLoader
    train_dataset = MultiClassLaneDataset(
        image_dir='dataset/train',
        mask_dir='dataset/train_masks',
        transform=train_transforms
    )
    
    val_dataset = MultiClassLaneDataset(
        image_dir='dataset/val',
        mask_dir='dataset/val_masks',
        transform=val_transforms
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    # Model, loss functions, and optimizer
    model = UNET(in_channels=3, out_channels=3).to(device)
    weights = torch.tensor([0.2118, 0.2732, 2.5150]).to(device)
    loss_fn_focal_w = FocalLoss_w(gamma=2.0, alpha=weights, reduction='mean')
    loss_fn_focal = FocalLoss(alpha=0.95, gamma=2.0)
    loss_fn_dice = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Early stopping variables
    best_iou = 0.0
    patience_counter = 0

    if LOAD_MODEL:
        load_checkpoint(torch.load("model.pth"), model)
        print("Model loaded successfully!")
        _, best_iou = check_accuracy(val_loader, model, loss_fn_focal_w, loss_fn_dice, device)
    
    scaler = GradScaler() if device.type == 'cuda' else None

    for epoch in range(num_epochs):
        train_loss, train_iou = train_fn(train_loader, model, optimizer, loss_fn_focal_w, loss_fn_dice, scaler, epoch)
        val_loss, val_iou = check_accuracy(val_loader, model, loss_fn_focal_w, loss_fn_dice, device)
        logging.info(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f}, Val mIoU: {val_iou:.4f}")
        print(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f}, Val mIoU: {val_iou:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            patience_counter = 0
            torch.save({
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(save_dir, 'best_model.pth'))
            logging.info(f"Model saved (best mIoU: {best_iou:.4f})")
        
        save_predictions_as_imgs(val_loader, model, epoch, folder=saved_images_dir, device=device)

if __name__ == "__main__":
    main()