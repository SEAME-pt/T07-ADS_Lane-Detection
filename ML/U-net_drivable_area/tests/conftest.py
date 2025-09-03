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
    # print(f"dataset.py exists: {os.path.exists(os.path.join(project_root, 'dataset.py'))}")

@pytest.fixture()
def tmp_image_mask_dirs(tmp_path: Path) -> Tuple[str, str]:
    """Create a small synthetic image/mask dataset on disk.

    Images: RGB .jpg
    Masks: grayscale .png with suffix "_mask.png" where values are {0, 1, 2} for multiclass
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
        img_path = image_dir / f"image_{i:03d}.jpg"
        Image.fromarray(img, mode="RGB").save(str(img_path))

        # Multiclass mask [H, W] uint8 with values {0, 1, 2}
        mask = rng.integers(0, 3, (height, width)).astype(np.uint8)
        mask_path = mask_dir / f"image_{i:03d}_mask.png"
        Image.fromarray(mask, mode="L").save(str(mask_path))

    return str(image_dir), str(mask_dir)