import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
from PIL import Image

print("Loading conftest.py")

@pytest.fixture(scope="session", autouse=True)
def add_project_root_to_syspath() -> None:
    """Ensure the project directory is importable in tests."""
    
    tests_dir = Path(__file__).resolve().parent
    project_root = tests_dir.parent
    sys.path.insert(0, str(project_root))
    # print(f"sys.path: {sys.path}")
    # print(f"Project root: {project_root}")
    # print(f"dataset.py exists: {os.path.exists(os.path.join(project_root, 'dataset.py'))}")  # Verifica se dataset.py existe


@pytest.fixture()
def tmp_image_mask_dirs(tmp_path: Path) -> Tuple[str, str]:
    """Create a small synthetic image/mask dataset on disk.

    Images: RGB .png
    Masks: grayscale .png with suffix "_label.png" where values are {0, 255}
    """
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    num_samples = 4
    height, width = 64, 96

    for i in range(num_samples):
        # RGB image [H, W, 3] uint8
        img = (rng.random((height, width, 3)) * 255).astype(np.uint8)
        img_path = image_dir / f"image_{i:03d}.png"
        Image.fromarray(img, mode="RGB").save(str(img_path))

        # Binary mask [H, W] uint8 with values {0, 255}
        mask = rng.random((height, width))
        mask_bin = (mask > 0.5).astype(np.uint8) * 255
        mask_path = mask_dir / f"image_{i:03d}_label.png"
        Image.fromarray(mask_bin, mode="L").save(str(mask_path))

    return str(image_dir), str(mask_dir)

## Run pytest tests/test_utils.py -v --tb=long

## Relatório no terminal pytest tests/test_dataset.py --cov=dataset --cov-report=term

## Relatório html pytest tests/test_dataset.py --cov=dataset --cov-report=html (cool)

## pip install pytest & pip install pytest-cov & pip install pytest-html

## testar tudo pytest tests/ --cov=dataset --cov=train --cov=model --cov-report=term --cov-report=html -v --tb=long

## testar tudo e criar o relatório html pytest tests/ --cov=dataset --cov=model --cov=train --cov-report=html --html=report.html -v --tb=long