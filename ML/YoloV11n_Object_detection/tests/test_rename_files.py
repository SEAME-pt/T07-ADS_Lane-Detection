import os
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rename_files import rename_image_label_pairs

def test_rename_files_basic(tmp_yolo_dataset: Path):
    """Testa a renomeação de pares de imagens e labels."""
    image_dir = tmp_yolo_dataset / "train" / "images"
    label_dir = tmp_yolo_dataset / "train" / "labels"
    rename_image_label_pairs(str(image_dir), str(label_dir))

    images = set(os.listdir(image_dir))
    labels = set(os.listdir(label_dir))
    expected_images = {"image00001.jpg", "image00002.jpg", "image00003.jpg", "image_no_label_002.jpg"}
    expected_labels = {"image00001.txt", "image00002.txt", "image00003.txt"}
    assert images == expected_images
    assert labels == expected_labels

def test_rename_files_empty_dir(tmp_path: Path):
    """Testa a renomeação em diretórios vazios."""
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    image_dir.mkdir()
    label_dir.mkdir()
    rename_image_label_pairs(str(image_dir), str(label_dir))
    assert len(os.listdir(image_dir)) == 0
    assert len(os.listdir(label_dir)) == 0

def test_rename_files_no_pairs(tmp_yolo_dataset: Path):
    """Testa quando não há pares correspondentes."""
    image_dir = tmp_yolo_dataset / "train" / "images"
    label_dir = tmp_yolo_dataset / "train" / "labels"
    # Remover pares válidos
    for f in os.listdir(label_dir):
        os.remove(os.path.join(label_dir, f))
    rename_image_label_pairs(str(image_dir), str(label_dir))
    images = set(os.listdir(image_dir))
    labels = set(os.listdir(label_dir))
    assert images == {"image_000.jpg", "image_001.jpg", "image_002.jpg", "image_no_label_002.jpg"}
    assert labels == set()