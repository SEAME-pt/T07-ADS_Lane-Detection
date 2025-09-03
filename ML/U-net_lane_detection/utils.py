import torch
import torchvision
from dataset import LaneDataset
from torch.utils.data import DataLoader
import logging
import os

# Logging configuration
logging.basicConfig(filename='training_log.log', level=logging.INFO, 
                    format='%(asctime)s - %(message)s')
"""Configures logging to save metrics to a file and display them in the terminal.

Logs are saved to 'training_log.log' with timestamped messages at the INFO level.
"""

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """Saves the model state to a checkpoint file.

    Args:
        state (dict): Dictionary containing the model and optimizer state dictionaries.
        filename (str, optional): Path to save the checkpoint file. Defaults to 'my_checkpoint.pth.tar'.

    Returns:
        None
    """
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    """Loads the model state from a checkpoint file.

    Args:
        checkpoint (dict): Dictionary containing the saved state dictionary.
        model (torch.nn.Module): The model instance to load the state into.

    Returns:
        None
    """
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=2,
    pin_memory=True,
):
    """Creates data loaders for training and validation datasets.

    Initializes `LaneDataset` instances for training and validation data, and returns PyTorch DataLoader objects.

    Args:
        train_dir (str): Directory containing training images.
        train_maskdir (str): Directory containing training masks.
        val_dir (str): Directory containing validation images.
        val_maskdir (str): Directory containing validation masks.
        batch_size (int): Number of samples per batch.
        train_transform (albumentations.Compose): Transformations for training data.
        val_transform (albumentations.Compose): Transformations for validation data.
        num_workers (int, optional): Number of subprocesses for data loading. Defaults to 2.
        pin_memory (bool, optional): Whether to pin memory for faster data transfer to GPU. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
            - val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
    """
    train_ds = LaneDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = LaneDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    """Evaluates the model's accuracy and Dice score on a dataset.

    Computes pixel-wise accuracy and Dice score for binary segmentation predictions.

    Args:
        loader (torch.utils.data.DataLoader): DataLoader containing the dataset to evaluate.
        model (torch.nn.Module): The trained model to evaluate.
        device (str, optional): Device to perform computations on. Defaults to 'cuda'.

    Returns:
        float: Mean Dice score across the dataset.
    """
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )
            mean_dice = dice_score / len(loader)

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
    print(f"Dice score: {mean_dice:.4f}")
    logging.info(f"Dice Score: {mean_dice:.4f}")
    
    model.train()
    return mean_dice

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    """Saves model predictions as images.

    Generates and saves binary prediction masks alongside ground truth masks for each batch in the dataset.

    Args:
        loader (torch.utils.data.DataLoader): DataLoader containing the dataset.
        model (torch.nn.Module): The trained model to generate predictions.
        folder (str, optional): Directory to save the prediction images. Defaults to 'saved_images/'.
        device (str, optional): Device to perform computations on. Defaults to 'cuda'.

    Returns:
        None
    """
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y, f"{folder}{idx}.png")

    model.train()

def save_predictions_as_imgs_1(loader, model, epoch, folder="saved_images/", device="cuda"):
    """Saves combined images of input, ground truth, and predictions.

    Creates composite images with the original input, ground truth mask, and predicted mask, separated by spaces,
    and saves them for visualization.

    Args:
        loader (torch.utils.data.DataLoader): DataLoader containing the dataset.
        model (torch.nn.Module): The trained model to generate predictions.
        epoch (int): Current training epoch for naming the output files.
        folder (str, optional): Directory to save the combined images. Defaults to 'saved_images/'.
        device (str, optional): Device to perform computations on. Defaults to 'cuda'.

    Returns:
        None
    """
    model.eval()
    
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    for idx, (x, y) in enumerate(loader):
        print("")
        if idx % 10 == 0:
            x = x.to(device=device)
            y = y.to(device=device).float()
            
            with torch.no_grad():
                preds = torch.sigmoid(model(x))
                preds = (preds > 0.5).float()
            
            img_original = x[0]  # [3, H, W]
            
            mask = y[0]  # [1, H, W] or [H, W]
            if mask.dim() == 2:  # If [H, W], add channel dimension
                mask = mask.unsqueeze(0)  # [1, H, W]
            mask_rgb = mask.repeat(3, 1, 1)  # [3, H, W]
            
            pred = preds[0]  # [1, H, W]
            if pred.dim() == 2:  # If [H, W], add channel dimension
                pred = pred.unsqueeze(0)  # [1, H, W]
            pred_rgb = pred.repeat(3, 1, 1)  # [3, H, W]
            
            space_width = 10
            space = torch.ones(3, img_original.size(1), space_width, device=device)
            
            combined = torch.cat(
                (img_original, space, mask_rgb, space, pred_rgb), dim=2
            )  # [3, H, 3*W + 2*space_width]
            
            torchvision.utils.save_image(
                combined, f"{folder}/combined_epoch{epoch}_batch{idx}.png"
            )
    
    model.train()

def calculate_alpha(dataset_loader):
    """Calculates the alpha weight for Focal Loss based on class imbalance.

    Computes the proportion of positive pixels in the dataset to suggest an alpha value for Focal Loss.

    Args:
        dataset_loader (torch.utils.data.DataLoader): DataLoader containing the dataset with masks.

    Returns:
        float: Suggested alpha value (proportion of negative pixels).
    """
    total_pixels = 0
    positive_pixels = 0
    
    for _, targets in dataset_loader:
        total_pixels += targets.numel()  # Total number of pixels
        positive_pixels += targets.sum().item()  # Sum of positive pixels
    
    pos_ratio = positive_pixels / total_pixels
    neg_ratio = 1 - pos_ratio
    alpha = neg_ratio  # Weight for positive class = proportion of negative class
    print(f"Proporção de positivos: {pos_ratio:.4f}, Alpha sugerido: {alpha:.4f}")
    return alpha