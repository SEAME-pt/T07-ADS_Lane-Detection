import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2

class LaneDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, num_augmentations=2):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.num_augmentations = num_augmentations
        self.images = os.listdir(image_dir)
        self.total_samples = len(self.images) * (1 + self.num_augmentations if transform else 1)

        # Transformação básica para garantir tamanho fixo mesmo sem augmentações
        self.base_transform = A.Compose([
            # A.Resize(height=160, width=240),
            A.Resize(height=128, width=256),
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
            ToTensorV2()
        ])

    def __len__(self):
        return self.total_samples

    def __getitem__(self, index):
        samples_per_image = (1 + self.num_augmentations) if self.transform else 1
        img_idx = index // samples_per_image
        sample_idx = index % samples_per_image
        
        img_path = os.path.join(self.image_dir, self.images[img_idx])
        mask_path = os.path.join(self.mask_dir, self.images[img_idx].replace(".png", "_label.png"))
        
        # Carrega como NumPy
        image = np.array(Image.open(img_path).convert("RGB"))  # [H, W, 3]
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)  # [H, W]
        mask[mask == 255.0] = 1.0
        
        # Aplica transformações
        if self.transform is not None and sample_idx > 0:
            # Augmentações para versões transformadas
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]  # [C, H, W]
            mask = augmentations["mask"]    # [1, H, W] ou [H, W]
            # print(f"Imagem transform: {image.shape}, tipo: {image.dtype}")  # Deve ser torch.Tensor [C, H, W]
            # print(f"Máscara transform: {mask.shape}, tipo: {mask.dtype}")  
        else:
            # Apenas redimensiona para o tamanho fixo na imagem original
            augmentations = self.base_transform(image=image, mask=mask)
            image = augmentations["image"]  # [C, H, W]
            mask = augmentations["mask"]    # [1, H, W]
            # print(f"Imagem else: {image.shape}, tipo: {image.dtype}")  # Deve ser torch.Tensor [C, H, W]
            # print(f"Máscara else: {mask.shape}, tipo: {mask.dtype}")  
        
        # Garante formato correto da máscara
        if len(mask.shape) == 2:  # [H, W] -> [1, H, W]
            mask = mask.unsqueeze(0)
        mask = (mask > 0.5).float()
        
        # print(f"Imagem: {image.shape}, tipo: {image.dtype}")  # Deve ser torch.Tensor [C, H, W]
        # print(f"Máscara: {mask.shape}, tipo: {mask.dtype}")   # Deve ser torch.Tensor [1, H, W]
        # print(f"Valores únicos da máscara: {torch.unique(mask)}")
        return image, mask


# train_transforms = A.Compose([
#     # A.Resize(height=160, width=240),  # Redimensiona
#     A.Resize(height=128, width=256, interpolation=cv2.INTER_CUBIC),  # Redimensiona
#     A.HorizontalFlip(p=0.5),  # Espelha horizontalmente
#     A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),  # Normaliza
#     # A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.9),
#     #     A.OneOf([
#     #         A.Blur(blur_limit=3, p=0.5),
#     #         A.ColorJitter(p=0.5),
#     #     ], p=1.0),
#     A.RandomBrightnessContrast(p=0.5),  # Ajusta brilho/contraste
#     A.RandomGamma(p=0.5),  # Ajusta gama para destacar linhas
#     ToTensorV2(),  # Converte para tensor
# ])



train_transforms = A.Compose([
    A.Resize(height=128, width=256, interpolation=cv2.INTER_CUBIC),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.RandomGamma(p=0.5),
    A.RandomShadow(
        shadow_roi=(0.2, 0.2, 0.8, 0.8),
        num_shadows_limit=(2, 4),
        shadow_dimension=8,
        shadow_intensity_range=(0.3, 0.7),
        p=0.5
    ),
    A.RandomRain(
        slant_range=(-15, 15),
        drop_length=30,
        drop_width=2,
        drop_color=(180, 180, 180),
        blur_value=5,
        brightness_coefficient=0.8,
        p=0.3
    ),
    A.GaussNoise(std_range=(0.1, 0.2), p=0.3),
    A.RandomFog(fog_coef_range=(0.2, 0.5), alpha_coef=0.1, p=0.3),  # Atualizado conforme os docs
    A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
    ToTensorV2(),
])


# Transformações para validação
val_transforms = A.Compose([
    # A.Resize(height=160, width=240),  # Redimensiona
    A.Resize(height=128, width=256, interpolation=cv2.INTER_CUBIC),  # Redimensiona
    A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),  # Normaliza
    ToTensorV2(),  # Converte para tensor
])


train_ds_test = LaneDataset(
        image_dir="data/train_little",
        mask_dir="data/train_masks_little",
        transform=train_transforms,
    )

train_loader_test = DataLoader(
        train_ds_test,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
        shuffle=True,
    )

val_ds_test = LaneDataset(
        image_dir="data/val_little",
        mask_dir="data/val_masks_little",
        transform=val_transforms,
    )

val_loader_test = DataLoader(
        val_ds_test,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
        shuffle=False,
    )



def test():
    # Criando imagem e máscara como tensores PyTorch
    img_tensor = torch.randint(0, 256, (572, 572, 3), dtype=torch.uint8)  # Imagem RGB
    mask_tensor = torch.randint(0, 2, (572, 572), dtype=torch.uint8)       # Máscara binária
    
    print("Data:", img_tensor.shape, img_tensor.dtype)
    print("Targets:", mask_tensor.shape, mask_tensor.dtype)
    
    # Convertendo para arrays NumPy
    img = img_tensor.numpy()
    mask = mask_tensor.numpy()
    
    # Aplicando as transformações
    transformed = train_transforms(image=img, mask=mask)
    dataset = [(transformed["image"], transformed["mask"]) for _ in range(4)]
    loader = DataLoader(dataset, batch_size=4)
    data, targets = next(iter(loader))
    print("Data:", data.shape, data.dtype)
    print("Targets:", targets.shape, targets.dtype)
    

def test_1(loader):
    data, targets = next(iter(loader))
    print("Data:", data.shape, data.dtype)
    print("Targets:", targets.shape, targets.dtype)
    

def teste_2():
    # Define transformações com tamanho fixo
    transform = A.Compose([
        A.Resize(height=160, width=240),  # Tamanho fixo
        A.Rotate(limit=10),
        ToTensorV2()
    ])
    
    dataset = LaneDataset(
        image_dir="data/train_little",
        mask_dir="data/train_masks_little",
        transform=transform,
        num_augmentations=2,
    )
    
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    for batch_idx, (images, masks) in enumerate(loader):
        print(f"Batch {batch_idx}: Imagens {images.shape}, Máscaras {masks.shape}")

if __name__ == "__main__":
    test()
    # print("\n")
    # print('Com o loader \n')
    # test_1(train_loader_test)
    # print(f'tamanho dataset: {len(train_ds_test)}')
    # teste_2()
    