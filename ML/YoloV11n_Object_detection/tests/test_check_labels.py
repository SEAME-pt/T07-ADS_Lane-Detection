import os
import sys
from pathlib import Path
import pytest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import check_labels


def test_teste_labels(tmp_yolo_dataset: Path):
    """Testa a identificação de imagens sem anotações e anotações sem imagens."""
    images_dir = tmp_yolo_dataset / "train" / "images"
    labels_dir = tmp_yolo_dataset / "train" / "labels"

    missing_labels, missing_images = check_labels.check_labels(images_dir, labels_dir)

    assert missing_labels == {"image_no_label_002"}  # Imagem sem label
    assert missing_images == set()  # Nenhum label sem imagem


def test_teste_labels_empty_dirs(tmp_path: Path):
    """Testa diretórios vazios."""
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    missing_labels, missing_images = check_labels.check_labels(images_dir, labels_dir)

    assert missing_labels == set()
    assert missing_images == set()


def test_teste_labels_perfect_match(tmp_yolo_dataset: Path):
    """Testa quando todos os arquivos têm correspondência."""
    images_dir = tmp_yolo_dataset / "train" / "images"
    labels_dir = tmp_yolo_dataset / "train" / "labels"
    # Remover imagem sem label
    os.remove(images_dir / "image_no_label_002.jpg")

    missing_labels, missing_images = check_labels.check_labels(images_dir, labels_dir)

    assert missing_labels == set()
    assert missing_images == set()


def test_teste_labels_label_without_image(tmp_yolo_dataset: Path):
    """Testa quando há um label sem imagem correspondente."""
    images_dir = tmp_yolo_dataset / "train" / "images"
    labels_dir = tmp_yolo_dataset / "train" / "labels"
    # Adicionar um label sem imagem
    with open(labels_dir / "orphan.txt", "w") as f:
        f.write("0 0.5 0.5 0.2 0.2\n")

    missing_labels, missing_images = check_labels.check_labels(images_dir, labels_dir)

    assert missing_labels == {"image_no_label_002"}
    assert missing_images == {"orphan"}


def test_teste_labels_ignores_non_image_and_non_label(tmp_path: Path):
    """Arquivos que não são .jpg ou .txt devem ser ignorados."""
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    (images_dir / "not_an_image.png").write_text("fake")
    (labels_dir / "not_a_label.csv").write_text("fake")

    missing_labels, missing_images = check_labels.check_labels(images_dir, labels_dir)

    assert missing_labels == set()
    assert missing_images == set()
    

def test_main_block_execution(monkeypatch, tmp_path, capsys):
    """Testa o bloco `if __name__ == '__main__'` simulando execução do script."""
    # Criar a estrutura de diretórios esperada pelo script
    images_dir = tmp_path / "dataset" / "valid" / "images"
    labels_dir = tmp_path / "dataset" / "valid" / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    # Criar imagem sem label
    (images_dir / "img001.jpg").write_text("fake")

    monkeypatch.chdir(tmp_path)  # simula que estamos na raiz do dataset

    # Executa o script como se fosse `python teste_labels.py`
    import runpy
    runpy.run_module("check_labels", run_name="__main__")

    out, _ = capsys.readouterr()
    assert "Images without annotations:" in out
    assert "Annotations without images:" in out
