import os
import sys
from pathlib import Path
import pytest
import cv2
import numpy as np

print("Loading conftest.py for YOLO dataset tests")


@pytest.fixture(scope="session", autouse=True)
def add_project_root_to_syspath():
    """Adiciona o diretório raiz do projeto ao sys.path para permitir importações."""
    tests_dir = Path(__file__).resolve().parent
    project_root = tests_dir.parent
    sys.path.insert(0, str(project_root))


@pytest.fixture
def tmp_yolo_dataset(tmp_path: Path):
    """Cria um dataset YOLO sintético com imagens e labels para testes.

    Estrutura:
    - tmp_path/
        - train/
            - images/
            - labels/
        - valid/
            - images/
            - labels/
        - data.yaml

    Imagens: .jpg (RGB)
    Labels: .txt (formato YOLO com class_id, x_center, y_center, width, height)
    Inclui alguns casos de borda, como imagens sem labels.

    Returns:
        Path: Caminho para o diretório raiz do dataset temporário.
    """
    # Criar estrutura de diretórios
    train_image_dir = tmp_path / "train" / "images"
    train_label_dir = tmp_path / "train" / "labels"
    valid_image_dir = tmp_path / "valid" / "images"
    valid_label_dir = tmp_path / "valid" / "labels"
    train_image_dir.mkdir(parents=True, exist_ok=True)
    train_label_dir.mkdir(parents=True, exist_ok=True)
    valid_image_dir.mkdir(parents=True, exist_ok=True)
    valid_label_dir.mkdir(parents=True, exist_ok=True)

    # Criar arquivo data.yaml
    with open(tmp_path / "data.yaml", "w") as f:
        f.write("train: ./train/images\nvalid: ./valid/images\nnc: 3\nnames: ['50', '80', '100']\n")

    # Gerar imagens e labels sintéticos
    num_samples = 3
    img_height, img_width = 64, 96  # evitar conflito com bbox width/height

    for i in range(num_samples):
        # Criar imagem RGB
        img = (np.random.rand(img_height, img_width, 3) * 255).astype(np.uint8)
        img_path = train_image_dir / f"image_{i:03d}.jpg"
        cv2.imwrite(str(img_path), img)

        # Criar label YOLO com uma bounding box (classe 0 ou 1)
        label_path = train_label_dir / f"image_{i:03d}.txt"
        bboxes = [[0.5, 0.5, 0.2, 0.2, 0 if i % 2 == 0 else 1]]  # Classe alterna entre 0 e 1
        with open(label_path, "w") as f:
            for bbox in bboxes:
                x_center, y_center, box_w, box_h, class_id = bbox
                f.write(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")

        # Criar uma imagem sem label para testar casos de borda
        if i == 2:
            img_path = train_image_dir / f"image_no_label_{i:03d}.jpg"
            cv2.imwrite(str(img_path), img)

    # Criar uma imagem e label no diretório valid
    img_path = valid_image_dir / "image_valid_000.jpg"
    label_path = valid_label_dir / "image_valid_000.txt"
    img = (np.random.rand(img_height, img_width, 3) * 255).astype(np.uint8)
    cv2.imwrite(str(img_path), img)
    with open(label_path, "w") as f:
        f.write("0 0.5 0.5 0.2 0.2\n")

    return tmp_path



## Run pytest tests/test_utils.py -v --tb=long

## Relatório no terminal pytest tests/test_dataset.py --cov=dataset --cov-report=term

## Relatório html pytest tests/test_dataset.py --cov=dataset --cov-report=html (cool)

## pip install pytest & pip install pytest-cov & pip install pytest-html

## testar tudo pytest tests/ --cov=dataset --cov=train --cov=model --cov-report=term --cov-report=html -v --tb=long

## testar tudo e criar o relatório html pytest tests/ --cov=augment_dataset --cov=check_labels --cov=class_count --cov=rename_files --cov=train --cov-report=term --cov-report=html --html=report.html -v --tb=long
