import pytest
from pathlib import Path
from collections import Counter
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import class_count  # agora importamos o módulo

def test_class_count(tmp_yolo_dataset: Path):
    """Testa a contagem de classes em arquivos de labels."""
    label_dir = tmp_yolo_dataset / "train" / "labels"
    result = class_count.class_count(label_dir)
    assert result == Counter({0: 2, 1: 1})

def test_class_count_empty_dir(tmp_path: Path):
    """Testa a contagem em um diretório vazio."""
    label_dir = tmp_path / "labels"
    label_dir.mkdir()
    result = class_count.class_count(label_dir)
    assert result == Counter()

def test_class_count_invalid_label(tmp_yolo_dataset: Path):
    """Testa o comportamento com um label inválido."""
    invalid_label_path = tmp_yolo_dataset / "train" / "labels" / "invalid.txt"
    with open(invalid_label_path, "w") as f:
        f.write("invalid data")

    with pytest.raises(ValueError):
        class_count.class_count(tmp_yolo_dataset / "train" / "labels")
