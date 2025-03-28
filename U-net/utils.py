import torch
import torchvision
from dataset import LaneDataset
from torch.utils.data import DataLoader
import logging
import os

logging.basicConfig(filename='training_log.log', level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

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
            mean_dice = dice_score / len(loader)

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
    print(f"Dice score: {mean_dice:.4f}")
    logging.info(f"Dice Score: {mean_dice:.4f}")
    
    model.train()
    return mean_dice


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


def save_predictions_as_imgs_1(loader, model, epoch, folder="saved_images/", device="cuda"):
    model.eval()
    
    # Cria o diretório se não existir
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    for idx, (x, y) in enumerate(loader):
        # Salva apenas a cada 10 batches
        print("")
        if idx % 10 == 0:
            x = x.to(device=device)
            y = y.to(device=device).float()
            
            with torch.no_grad():
                preds = torch.sigmoid(model(x))
                preds = (preds > 0.5).float()
            
            # Pega a primeira amostra do batch
            img_original = x[0]  # [3, H, W]
            # print(f"Image: {img_original.shape}, tipo: {img_original.dtype}")
            
            # Ground truth (1 canal) -> converte para 3 canais
            mask = y[0]  # [1, H, W] ou [H, W]
            # print(f"Máscara before: {mask.shape}, tipo: {mask.dtype}")
            if mask.dim() == 2:  # Se for [H, W], adiciona dimensão de canal
                mask = mask.unsqueeze(0)  # [1, H, W]
                # print(f"Máscara after: {mask.shape}, tipo: {mask.dtype}")
            mask_rgb = mask.repeat(3, 1, 1)  # [3, H, W]
            
            # Previsão (1 canal) -> converte para 3 canais
            pred = preds[0]  # [1, H, W]
            # print(f"Máscara before: {pred.shape}, tipo: {pred.dtype}")
            if pred.dim() == 2:  # Se for [H, W], adiciona dimensão de canal
                pred = pred.unsqueeze(0)  # [1, H, W]
                # print(f"Máscara before: {pred.shape}, tipo: {pred.dtype}")
            pred_rgb = pred.repeat(3, 1, 1)  # [3, H, W]
            
            # Define o espaço (faixa preta de 5 pixels de largura)
            space_width = 10
            # space = torch.zeros(3, img_original.size(1), space_width, device=device)  # [3, H, 5]
            space = torch.ones(3, img_original.size(1), space_width, device=device)
            
            # Concatena horizontalmente com espaços: original | espaço | máscara | espaço | previsão
            combined = torch.cat(
                (img_original, space, mask_rgb, space, pred_rgb), dim=2
            )  # [3, H, 3*W + 2*space_width]
            
            # Salva a imagem combinada
            torchvision.utils.save_image(
                combined, f"{folder}/combined_epoch{epoch}_batch{idx}.png"
            )
    
    model.train()
    

def calculate_alpha(dataset_loader):
    total_pixels = 0
    positive_pixels = 0
    
    for _, targets in dataset_loader:
        total_pixels += targets.numel()  # Número total de pixels
        positive_pixels += targets.sum().item()  # Soma de pixels positivos
    
    pos_ratio = positive_pixels / total_pixels
    neg_ratio = 1 - pos_ratio
    alpha = neg_ratio  # Peso para a classe positiva = proporção da classe negativa
    print(f"Proporção de positivos: {pos_ratio:.4f}, Alpha sugerido: {alpha:.4f}")
    return alpha

