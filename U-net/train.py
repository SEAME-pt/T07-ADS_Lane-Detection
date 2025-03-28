
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
import torch.nn.functional as F
from dataset import train_transforms, val_transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler
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

# Hyperparameters 
LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 128 #160  # 1280 originally
IMAGE_WIDTH = 256 #240  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
# PATIENCE = 5  # Para Early Stopping
TRAIN_IMG_DIR = "data/train_little/"
TRAIN_MASK_DIR = "data/train_masks_little/"
VAL_IMG_DIR = "data/val_little/"
VAL_MASK_DIR = "data/val_masks_little/"


# Configuração de logging para salvar métricas em um arquivo
logging.basicConfig(filename='training_log.log', level=logging.INFO, 
                    format='%(asctime)s - %(message)s')
logging.getLogger().addHandler(logging.StreamHandler()) # para aparecerem no terminal

# Função de treino ajustada
def train_fn(loader, model, optimizer, loss_fn_focal, loss_fn_dice, scaler=None, epoch=0):
    model.train()  # Ativa o modo de treino
    loop = tqdm(loader)
    total_loss = 0
    total_batches = len(loader)
    
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(DEVICE)
        targets = targets.to(DEVICE).float()
        
        # Forward com mixed precision (opcional)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                predictions = model(data)
                loss_focal = loss_fn_focal(predictions, targets)
                loss_dice = loss_fn_dice(predictions, targets)
                loss = 0.7 * loss_focal + 1.3 * loss_dice
        else:
            predictions = model(data)
            loss_focal = loss_fn_focal(predictions, targets)
            loss_dice = loss_fn_dice(predictions, targets)
            loss = 0.7 * loss_focal + 1.3 * loss_dice
        
        # Backward
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        loop.set_postfix(loss=total_loss / (batch_idx + 1))  # Média móvel da perda por batch feito
    
    mean_loss = total_loss / total_batches
    # print(f"Epoch {epoch} - Mean training loss: {mean_loss}")
    logging.info(f"Epoch {epoch} - Mean training loss: {mean_loss:.4f}")
    return mean_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.95, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Peso para a classe positiva
        self.gamma = gamma  # Fator de focalização
        self.reduction = reduction
        print(f"FocalLoss inicializada com alpha={self.alpha}, gamma={self.gamma}. "
              f"Certifique-se de que alpha reflete o desbalanceamento do dataset.")

    def forward(self, inputs, targets):
        # Verifica valores inválidos nos dados
        # if torch.isnan(inputs).any() or torch.isinf(inputs).any():
        #     raise ValueError("Inputs contêm NaN ou Inf!")
        # if torch.isnan(targets).any() or torch.isinf(targets).any():
        #     raise ValueError("Targets contêm NaN ou Inf!")

        if torch.isnan(inputs).any() or torch.isinf(inputs).any():
            logging.warning("Inputs contêm NaN ou Inf.")
        if torch.isnan(targets).any() or torch.isinf(targets).any():
            logging.warning("Targets contêm NaN ou Inf.")
            
		# se der este logging existe uma maneira de colocar valores pre-definidos
        
        # Limita os logits para evitar overflow (opcional, mas melhora estabilidade)
        inputs = torch.clamp(inputs, min=-100, max=100)  # Valores razoáveis para logits
        
        # Calcula a perda BCE com logits
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Probabilidades ajustadas
        p_t = torch.sigmoid(inputs)
        p_t = targets * p_t + (1 - targets) * (1 - p_t)  # Probabilidade corrigida
        
        # Fator de focalização: reduz peso de exemplos fáceis
        focal_weight = (1 - p_t) ** self.gamma
        
        # Peso baseado em alpha
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        
        # Perda final
        focal_loss = alpha_weight * focal_weight * bce_loss
        
        # Redução
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Dice Loss (já definida anteriormente)
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


# Função principal ajustada
def main():
    
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn_focal = FocalLoss(alpha=0.95, gamma=2.0)
    loss_fn_dice = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Adicionando Learning Rate Scheduler
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    # scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # Carrega os dados
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
        print("Modelo carregado com sucesso!")
        dice_score_saved = check_accuracy(val_loader, model, device=DEVICE)
        best_dice_score = dice_score_saved

    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None  # Mixed precision
    
    print(f'Dice score: {best_dice_score}')
  
    
    for epoch in range(NUM_EPOCHS):
        # Treinamento
        train_loss = train_fn(train_loader, model, optimizer, loss_fn_focal, loss_fn_dice, scaler, epoch)
        
        # Validação a cada época (ou a cada X batches, se preferir)
        dice_score= check_accuracy(val_loader, model, device=DEVICE)
        
        # Atualiza o scheduler com base na perda de validação
        # scheduler.step(dice_score)
        
		# Obtém a taxa de aprendizado atual
        # current_lr = scheduler.get_last_lr()
        # current_lr = scheduler.get_last_lr()[0]
        
        # print(
        #     f"Epoch {epoch}: Dice Score = {dice_score:.4f}, Learning rate = {current_lr:.6f}"
        #     )
        
        print(f"Epoch {epoch}: Dice Score = {dice_score:.4f}" )
        
        # Early Stopping
        if dice_score > best_dice_score:
            best_dice_score = dice_score
            # patience_counter = 0
            checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
            save_checkpoint(checkpoint, filename="model.pth.tar")
            print("Modelo salvo (melhor validação)!")
        # else:
        #     patience_counter += 1
        #     print(f"Patience: {patience_counter}/{PATIENCE}")
        #     if patience_counter >= PATIENCE:
        #         print("Early Stopping ativado!")
        #         break
        
        # Salva predições como imagens
        save_predictions_as_imgs_1(val_loader, model, epoch, folder="saved_images/", device=DEVICE)

if __name__ == "__main__":
    
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
    
	# alpha = calculate_alpha(train_loader)
    
    main()