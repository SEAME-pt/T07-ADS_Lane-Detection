from pathlib import Path
import pytest
from PIL import Image
import yaml
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train import main

def make_data_yaml(base: Path, valid=True):
    """Cria o diretório de dataset e o data.yaml válido ou inválido."""
    train_images = base / "train" / "images"
    val_images = base / "valid" / "images"
    train_images.mkdir(parents=True, exist_ok=True)
    val_images.mkdir(parents=True, exist_ok=True)

    for img_dir in [train_images, val_images]:
        for i in range(2):
            img = Image.new("RGB", (32, 32), color=(0, 0, 0))
            img.save(img_dir / f"{i}.jpg")

    yaml_path = base / "data.yaml"
    if valid:
        yaml_content = {
            "train": str(train_images),
            "val": str(val_images),
            "nc": 3,
            "names": ["50", "80", "100"],
        }
    else:
        yaml_content = {"bad_key": "sem_campos_necessarios"}

    with open(yaml_path, "w") as f:
        yaml.safe_dump(yaml_content, f)
    return yaml_path

def make_dummy_dataset(base_path: Path):
    """Cria um dataset mínimo para o YOLO treinar."""
    train_img_dir = base_path / "train/images"
    valid_img_dir = base_path / "valid/images"
    train_img_dir.mkdir(parents=True)
    valid_img_dir.mkdir(parents=True)

    for dir_path in [train_img_dir, valid_img_dir]:
        img = Image.new("RGB", (32, 32), color=(0, 0, 0))
        img.save(dir_path / "img1.jpg")

    yaml_content = {
        "train": str(train_img_dir),
        "val": str(valid_img_dir),
        "nc": 1,
        "names": ["dummy"]
    }
    yaml_path = base_path / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f)
    return yaml_path

def run_main(argv):
    """Substitui sys.argv temporariamente para testar main()."""
    old_argv = sys.argv
    sys.argv = ["train.py", *argv]
    try:
        main()
        return {"returncode": 0}
    except (SystemExit, RuntimeError, FileNotFoundError) as e:
        return {"returncode": 1}  # Return non-zero for any error
    finally:
        sys.argv = old_argv

@patch("ultralytics.YOLO")
def test_train_runs_success(mock_yolo, tmp_path: Path):
    """Testa execução mínima do train.py com dataset válido."""
    yaml_path = make_dummy_dataset(tmp_path)
    
    mock_yolo_instance = MagicMock()
    mock_yolo.return_value = mock_yolo_instance
    mock_val_results = MagicMock()
    mock_val_results.box.map50 = 0.85
    mock_val_results.box.map = 0.65
    mock_val_results.box.maps = [0.60, 0.70, 0.65]
    mock_val_results.names = {0: "dummy"}
    mock_yolo_instance.val.return_value = mock_val_results
    
    with patch("train.model", mock_yolo_instance):
        result = run_main(["--data", str(yaml_path), "--epochs", "1", "--imgsz", "32", "--batch", "1"])
    
    assert result["returncode"] == 0
    mock_yolo_instance.train.assert_called_once_with(
        data=str(yaml_path),
        epochs=1,
        imgsz=32,
        batch=1,
        hsv_h=0.015,
        hsv_s=0.1,
        hsv_v=0.1,
        translate=0.1,
        scale=0.2,
        fliplr=0.5,
        mosaic=0.5,
        erasing=0.2,
        auto_augment=None,
        amp=True,
        device="cpu",
        workers=2,
        project="./models",
        name="yolo_object_lane",
        exist_ok=True,
        freeze=0,
        lr0=0.01,
        patience=0,
        weight_decay=0.0005,
        save_period=20,
        save=True
    )
    mock_yolo_instance.val.assert_called_once_with(
        data=str(yaml_path),
        imgsz=32,
        batch=16,
        device="cpu",
        plots=True
    )

@patch("ultralytics.YOLO")
def test_train_runs_success_yaml(mock_yolo, tmp_path: Path):
    """Testa execução com YAML válido, esperando falha devido a imagens incompatíveis."""
    yaml_path = make_data_yaml(tmp_path, valid=True)
    
    mock_yolo_instance = MagicMock()
    mock_yolo.return_value = mock_yolo_instance
    mock_yolo_instance.train.side_effect = RuntimeError("Invalid dataset")
    
    with patch("train.model", mock_yolo_instance):
        result = run_main(["--data", str(yaml_path), "--epochs", "1"])
    
    assert result["returncode"] != 0

@patch("ultralytics.YOLO")
def test_train_invalid_yaml(mock_yolo, tmp_path: Path):
    """Testa execução com YAML inválido."""
    yaml_path = make_data_yaml(tmp_path, valid=False)
    
    mock_yolo_instance = MagicMock()
    mock_yolo.return_value = mock_yolo_instance
    mock_yolo_instance.train.side_effect = RuntimeError("Invalid YAML format")
    
    with patch("train.model", mock_yolo_instance):
        result = run_main(["--data", str(yaml_path), "--epochs", "1"])
    
    assert result["returncode"] != 0

@patch("ultralytics.YOLO")
def test_train_missing_yaml(mock_yolo, tmp_path: Path):
    """Testa execução com YAML ausente."""
    yaml_path = tmp_path / "data.yaml"
    
    mock_yolo_instance = MagicMock()
    mock_yolo.return_value = mock_yolo_instance
    mock_yolo_instance.train.side_effect = FileNotFoundError("Dataset file not found")
    
    with patch("train.model", mock_yolo_instance):
        result = run_main(["--data", str(yaml_path), "--epochs", "1"])
    
    assert result["returncode"] != 0

@patch("ultralytics.YOLO")
def test_train_invalid_epochs(mock_yolo, tmp_path: Path):
    """Testa execução com argumento epochs inválido."""
    yaml_path = make_data_yaml(tmp_path, valid=True)
    
    mock_yolo_instance = MagicMock()
    mock_yolo.return_value = mock_yolo_instance
    
    with patch("train.model", mock_yolo_instance):
        result = run_main(["--data", str(yaml_path), "--epochs", "abc"])
    
    assert result["returncode"] != 0

@patch("ultralytics.YOLO")
def test_train_no_args(mock_yolo):
    """Testa execução sem argumentos, usando defaults."""
    mock_yolo_instance = MagicMock()
    mock_yolo.return_value = mock_yolo_instance
    mock_yolo_instance.train.side_effect = FileNotFoundError("Dataset file not found")
    
    with patch("train.model", mock_yolo_instance):
        result = run_main([])
    
    assert result["returncode"] != 0