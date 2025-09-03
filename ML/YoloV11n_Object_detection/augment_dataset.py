import os
import cv2
import albumentations as A
import numpy as np
import shutil
from tqdm import tqdm

def read_yolo_label(label_path):
    """Read a YOLO annotation file and return bounding boxes.

    Reads a YOLO format label file and extracts bounding box coordinates and class IDs.

    Args:
        label_path (str): Path to the YOLO label file (.txt).

    Returns:
        list: List of bounding boxes, each as [x_center, y_center, width, height, class_id].
    """
    bboxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                bboxes.append([x_center, y_center, width, height, int(class_id)])
    return bboxes

def write_yolo_label(label_path, bboxes):
    """Write bounding boxes to a YOLO format label file.

    Saves bounding box coordinates and class IDs in YOLO format to a specified file.

    Args:
        label_path (str): Path to save the YOLO label file (.txt).
        bboxes (list): List of bounding boxes, each as [x_center, y_center, width, height, class_id].
    """
    with open(label_path, 'w') as f:
        for bbox in bboxes:
            x_center, y_center, width, height, class_id = bbox
            f.write(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def augment_dataset(input_dataset_dir, output_dataset_dir, base_num_augmentations=5, oversample_classes=[0, 1], oversample_factor=2, save_empty_labels=False):
    """Augment a YOLO dataset by generating new images and labels.

    Applies data augmentation to images and their corresponding YOLO labels, with optional oversampling
    for specified classes. Saves the augmented dataset to a new directory.

    Args:
        input_dataset_dir (str): Path to the original dataset directory.
        output_dataset_dir (str): Path to save the augmented dataset.
        base_num_augmentations (int, optional): Base number of augmentations per image. Defaults to 5.
        oversample_classes (list, optional): List of class IDs to oversample (e.g., [0, 1]). Defaults to [0, 1].
        oversample_factor (int, optional): Multiplier for augmentations of oversampled classes. Defaults to 2.
        save_empty_labels (bool, optional): If True, creates empty .txt files for images without valid bounding boxes. Defaults to False.
    """
    # Define augmentation pipeline for robustness in digit detection and real-world conditions
    transform = A.Compose([
        # Adjust brightness and contrast to simulate varying lighting conditions
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
        # Rotate image slightly to account for camera tilt or angle variations
        A.Rotate(limit=15, p=0.5),
        # Flip image horizontally to increase robustness to mirrored perspectives
        A.HorizontalFlip(p=0.5),
        # Randomly scale image to simulate varying object distances
        A.RandomScale(scale_limit=0.2, p=0.5),
        # Apply affine transformations (scale, translate, rotate) for geometric robustness
        A.Affine(scale=(0.9, 1.1), translate_percent=(0.1, 0.1), rotate=(-15, 15), p=0.5),
        # Adjust hue, saturation, and value to simulate color variations
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.5),
        # Apply Gaussian blur to simulate out-of-focus or low-quality images
        A.GaussianBlur(blur_limit=(3, 7), p=0.4),
        # Enhance local contrast to improve visibility of digits and details
        A.CLAHE(clip_limit=4.0, p=0.4),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_ids']))

    # Create directory structure for the augmented dataset
    os.makedirs(os.path.join(output_dataset_dir, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dataset_dir, 'train', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(output_dataset_dir, 'valid', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dataset_dir, 'valid', 'labels'), exist_ok=True)

    # Copy the data.yaml configuration file
    shutil.copy(os.path.join(input_dataset_dir, 'data.yaml'), output_dataset_dir)

    # Process images and labels for train and validation splits
    for split in ['train', 'valid']:
        image_dir = os.path.join(input_dataset_dir, split, 'images')
        label_dir = os.path.join(input_dataset_dir, split, 'labels')
        output_image_dir = os.path.join(output_dataset_dir, split, 'images')
        output_label_dir = os.path.join(output_dataset_dir, split, 'labels')

        if not os.path.exists(image_dir):
            print(f"Directory not found: {image_dir}")
            continue

        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

        for image_file in tqdm(image_files, desc=f"Processing {split}"):
            image_path = os.path.join(image_dir, image_file)
            label_path = os.path.join(label_dir, os.path.splitext(image_file)[0] + '.txt')

            # Load image
            image = cv2.imread(image_path)
            if image is None:
                # print(f"Error loading image: {image_path}") ## comentada para não aparecer no report
                continue

            # Load labels, if they exist
            bboxes = read_yolo_label(label_path)
            class_ids = [int(bbox[4]) for bbox in bboxes] if bboxes else []
            bboxes = [bbox[:4] for bbox in bboxes] if bboxes else []

            # Determine number of augmentations (oversample for specified classes)
            has_oversample = any(cid in oversample_classes for cid in class_ids)
            num_augs = base_num_augmentations * oversample_factor if has_oversample else base_num_augmentations

            # Copy original image and label, if labels exist or save_empty_labels is True
            if bboxes or save_empty_labels:
                shutil.copy(image_path, os.path.join(output_image_dir, image_file))
                if bboxes:
                    shutil.copy(label_path, os.path.join(output_label_dir, os.path.splitext(image_file)[0] + '.txt'))
                elif save_empty_labels:
                    open(os.path.join(output_label_dir, os.path.splitext(image_file)[0] + '.txt'), 'a').close()

            # Generate augmented images and labels
            for i in range(num_augs):
                # Apply augmentation
                augmented = transform(image=image, bboxes=bboxes, class_ids=class_ids)
                aug_image = augmented['image']
                aug_bboxes = augmented['bboxes']
                aug_class_ids = augmented['class_ids']

                # Combine bboxes with class IDs for YOLO format
                aug_bboxes_with_class = [
                    [bbox[0], bbox[1], bbox[2], bbox[3], class_id]
                    for bbox, class_id in zip(aug_bboxes, aug_class_ids)
                ]

                # Filter invalid bounding boxes (outside image or invalid dimensions)
                aug_bboxes_with_class = [
                    bbox for bbox in aug_bboxes_with_class
                    if 0 <= bbox[0] <= 1 and 0 <= bbox[1] <= 1 and bbox[2] > 0 and bbox[3] > 0
                ]

                # Save augmented image and label if there are valid bboxes or save_empty_labels is True
                if aug_bboxes_with_class or save_empty_labels:
                    aug_image_file = f"{os.path.splitext(image_file)[0]}_aug{i+1}{os.path.splitext(image_file)[1]}"
                    cv2.imwrite(os.path.join(output_image_dir, aug_image_file), aug_image)
                    aug_label_file = os.path.join(output_label_dir, f"{os.path.splitext(image_file)[0]}_aug{i+1}.txt")
                    if aug_bboxes_with_class:
                        write_yolo_label(aug_label_file, aug_bboxes_with_class)
                    elif save_empty_labels:
                        open(aug_label_file, 'a').close()

if __name__ == "__main__":
    """Entry point for running the dataset augmentation script.

    Configures and runs the dataset augmentation process with specified parameters.

    Args:
        input_dataset_dir (str): Path to the original dataset directory.
        output_dataset_dir (str): Path to save the augmented dataset.
        base_num_augmentations (int): Number of augmentations per original image.
        oversample_classes (list): List of class IDs to oversample (e.g., [8, 9]).
        oversample_factor (int): Multiplier for augmentations of oversampled classes.
        save_empty_labels (bool): If True, creates empty .txt files for images without bounding boxes.
    """
    input_dataset_dir = "./dataset_original"
    output_dataset_dir = "./dataset"
    base_num_augmentations = 6
    oversample_classes = [8, 9]
    oversample_factor = 2
    save_empty_labels = False

    augment_dataset(input_dataset_dir, output_dataset_dir, base_num_augmentations, oversample_classes, oversample_factor, save_empty_labels)
    print(f"Augmented dataset saved to: {output_dataset_dir}")