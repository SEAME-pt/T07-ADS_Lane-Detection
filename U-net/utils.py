import torch
import torchvision
import os
import random
from dataset import LaneDataset
from torch.utils.data import DataLoader

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
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
    num_workers=4,
    pin_memory=True,
):
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


def check_accuracy(loader, model, device = "cuda"):
    
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            # y = y.to(device).unsqueeze(1)
            y = y.to(device)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()


def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        # torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")
        torchvision.utils.save_image(y, f"{folder}{idx}.png")

    model.train()
    

def save_prediction_visualization(model, loader, folder="saved_images/", device="cuda"):
    model.eval()
    os.makedirs(folder, exist_ok=True)
    
    # Seleciona uma imagem aleatória do loader
    x, y = random.choice(loader.dataset)
    x = x.unsqueeze(0).to(device=device)  # Adiciona dimensão de batch
    y = y.unsqueeze(0)
    
    with torch.no_grad():
        preds = torch.sigmoid(model(x))
        preds_binary = (preds > 0.5).float()
    
    # Expandir os canais para garantir compatibilidade
    if preds_binary.shape[1] == 1:
        preds_binary = preds_binary.expand(-1, 3, -1, -1)
    if preds.shape[1] == 1:
        preds = preds.expand(-1, 3, -1, -1)
    if y.shape[1] == 1:
        y = y.expand(-1, 3, -1, -1)
    
    # Criar grid de imagens lado a lado
    images = torch.cat([x.cpu(), preds_binary.cpu(), preds.cpu()], dim=3)  # Concatena ao longo da largura
    
    # Salvar a imagem resultante
    torchvision.utils.save_image(images, f"{folder}/result.png")
    
    model.train()



