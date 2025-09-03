import os
from pathlib import Path
from typing import Tuple, Set

def check_labels(images_dir: str | Path, labels_dir: str | Path) -> Tuple[Set[str], Set[str]]:
    """Check for images without corresponding labels and labels without corresponding images.

    Compares the filenames (without extensions) in the images and labels directories to identify
    mismatches, such as images without annotations or annotations without corresponding images.

    Args:
        images_dir (str | Path): Path to the directory containing image files (.jpg).
        labels_dir (str | Path): Path to the directory containing label files (.txt).

    Returns:
        Tuple[Set[str], Set[str]]: A tuple containing:
            - missing_labels: Set of image filenames (without extension) that lack corresponding label files.
            - missing_images: Set of label filenames (without extension) that lack corresponding image files.
    """
    # Collect image filenames (without .jpg extension)
    images = {f.split('.')[0] for f in os.listdir(images_dir) if f.endswith('.jpg')}
    # Collect label filenames (without .txt extension)
    labels = {f.split('.')[0] for f in os.listdir(labels_dir) if f.endswith('.txt')}
    # Identify images without labels
    missing_labels = images - labels
    # Identify labels without images
    missing_images = labels - images
    return missing_labels, missing_images

if __name__ == "__main__":
    """Entry point for running the label checking script.

    Checks for mismatches between images and labels in the specified directories and prints the results.

    Args:
        images_dir (str): Path to the directory containing validation images.
        labels_dir (str): Path to the directory containing validation labels.
    """
    images_dir = "dataset/valid/images"
    labels_dir = "dataset/valid/labels"

    missing_labels, missing_images = check_labels(images_dir, labels_dir)
    print("Images without annotations:", missing_labels)
    print("Annotations without images:", missing_images)