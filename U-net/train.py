import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
import torch.nn.functional as F
from dataset import train_transforms, val_transforms
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    save_prediction_visualization,
)

# Hyperparameters 
LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 160  
IMAGE_WIDTH = 240  
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_IMG_DIR = "data/train_little/"
TRAIN_MASK_DIR = "data/train_masks_little/"
VAL_IMG_DIR = "data/val_little/"
VAL_MASK_DIR = "data/val_masks_little/"



def train_fn_scaler(loader, model, optimizer, loss_fn_focal, loss_fn_dice, scaler): 
    model.train()
    loop = tqdm(loader)
    total_loss = 0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        # targets = targets.float().unsqueeze(1).to(device=DEVICE)
        targets = targets.float().to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast(): 
            predictions = model(data)
            loss_focal = loss_fn_focal(predictions, targets)
            loss_dice = loss_fn_dice(predictions, targets)
        
           # Combinação ponderada (mais peso para Dice Loss)
            loss = 0.7 * loss_focal + 1.3 * loss_dice  # Ajustado para priorizar sobreposição

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
    
    print(f"Mean loss: {total_loss / len(loader)}")
        

# Função de treino ajustada com novos pesos para as perdas
def train_fn(loader, model, optimizer, loss_fn_focal, loss_fn_dice):
    
    model.train()
    loop = tqdm(loader)
    total_loss = 0
    
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(DEVICE)
        targets = targets.to(DEVICE).float()
        
        # Forward
        predictions = model(data)
        
        # Calcula as perdas
        loss_focal = loss_fn_focal(predictions, targets)
        loss_dice = loss_fn_dice(predictions, targets)
        
        # Combinação ponderada (mais peso para Dice Loss)
        loss = 0.7 * loss_focal + 1.3 * loss_dice  # Ajustado para priorizar sobreposição
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    
    print(f"Mean loss: {total_loss / len(loader)}") #imprime loss média por epoch
        



# Focal Loss (ajustada com alpha mais alto para desbalanceamento)
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.95, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Aumentado para refletir o desbalanceamento (baseado em pos_weight=19)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = torch.sigmoid(inputs)
        p_t = targets * p_t + (1 - targets) * (1 - p_t)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_loss = alpha_weight * focal_weight * bce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-8):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice
    

    
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



# Main
def main():
    
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    
    # Definição das funções de perda
    loss_fn_focal = FocalLoss(alpha=0.95, gamma=2.0)  # Alpha ajustado para desbalanceamento
    loss_fn_dice = DiceLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

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

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    check_accuracy(val_loader, model, device=DEVICE)
    
	# scaler = torch.cuda.amp.GradScaler()
    scaler = None

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn_focal, loss_fn_dice)
        
        # Save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # Check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # Print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE
        )


if __name__ == "__main__":  
	
	main()
    

	#Visualizar resultado
	# model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    
	# load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
    
	# train_loader, val_loader = get_loaders(
    #     TRAIN_IMG_DIR,
    #     TRAIN_MASK_DIR,
    #     VAL_IMG_DIR,
    #     VAL_MASK_DIR,
    #     BATCH_SIZE,
    #     train_transforms,
    #     val_transforms,
    #     NUM_WORKERS,
    #     PIN_MEMORY,
    # )
	
	# save_prediction_visualization(model, val_loader, "saved_images_1/", DEVICE)
