import os
from pathlib import Path
from ultralytics import YOLO
import argparse

# Define the base directory for storing model outputs, such as checkpoints and logs
project_dir = "./models"
# Set a unique name for this training experiment to organize output files
experiment_name = "yolo_object_lane"
# Create the project directory if it does not already exist, ensuring a safe destination for outputs
os.makedirs(project_dir, exist_ok=True)

# Initialize the YOLO model using pre-trained weights from 'yolo11n.pt'
# The 'yolo11n.pt' is a lightweight YOLOv11 nano model, optimized for fast inference and suitable for object detection tasks
model = YOLO("yolo11n.pt")

def main():
    """Train and validate a YOLO model for object detection with customizable parameters.

    This function configures a command-line argument parser to allow flexible specification of training parameters,
    trains the YOLO model using the Ultralytics library with a detailed set of hyperparameters, and validates the
    trained model on a specified dataset to evaluate performance metrics such as mAP. The training process includes
    data augmentation, checkpoint saving, and regularization to optimize model performance.

    Args:
        None: Arguments are parsed from the command line using argparse.

    Returns:
        None: Outputs are saved to the project directory, and validation results are printed to the console.
    """
    # Initialize an argument parser to capture command-line inputs for flexible training configuration
    parser = argparse.ArgumentParser(description="Train a YOLO model for object detection with customizable parameters")
    # Specify the path to the dataset YAML file, which defines paths to training/validation data and class names
    parser.add_argument("--data", type=str, default="./dataset/data.yaml", help="Path to the dataset YAML file specifying train, validation, and class details")
    # Define the total number of training epochs to balance model convergence and potential overfitting
    parser.add_argument("--epochs", type=int, default=300, help="Total number of training epochs to optimize the model")
    # Set the input image size for resizing images during training, ensuring consistency across inputs
    parser.add_argument("--imgsz", type=int, default=640, help="Image size in pixels (square, e.g., 640x640) for resizing input images")
    # Specify the batch size for training, balancing memory usage and gradient stability
    parser.add_argument("--batch", type=int, default=8, help="Number of images per batch during training")
    # Parse the provided command-line arguments into a namespace object
    args = parser.parse_args()

    # Train the YOLO model with a comprehensive set of hyperparameters to optimize performance
    results = model.train(
        # Path to the dataset YAML file, which includes paths to train/validation images and class definitions
        data=args.data,
        # Number of epochs to train the model, controlling the duration of training
        epochs=args.epochs,
        # Resize all input images to this square dimension (e.g., 640x640 pixels) for consistent processing
        imgsz=args.imgsz,
        # Apply hue augmentation within a small range to simulate slight color variations in images
        hsv_h=0.015,
        # Adjust saturation to enhance robustness to changes in lighting and color intensity
        hsv_s=0.1,
        # Modify brightness (value) to account for varying illumination conditions in real-world scenarios
        hsv_v=0.1,
        # Apply random translation (shifting) to images by up to 10% to improve spatial robustness
        translate=0.1,
        # Randomly scale images by up to 20% to simulate objects at varying distances from the camera
        scale=0.2,
        # Flip images horizontally with a 50% probability to augment dataset symmetry and robustness
        fliplr=0.5,
        # Apply mosaic augmentation (combining multiple images into one) with 50% probability to increase data diversity
        mosaic=0.5,
        # Use random erasing with 20% probability to simulate occlusions, improving robustness to partial object visibility
        erasing=0.2,
        # Disable auto-augmentation to rely solely on the specified augmentation parameters
        auto_augment=None,
        # Set the batch size for training, controlling the number of images processed per iteration
        batch=args.batch,
        # Enable Automatic Mixed Precision (AMP) to reduce memory usage and speed up training on compatible hardware
        amp=True,
        # Specify CPU as the training device (can be changed to GPU, e.g., 'cuda', if available for faster training)
        device="cpu",
        # Set the number of worker threads for data loading to optimize the data pipeline and reduce bottlenecks
        workers=2,
        # Define the directory where training outputs (checkpoints, logs, etc.) will be saved
        project=project_dir,
        # Specify the experiment name to organize output files within the project directory
        name=experiment_name,
        # Allow overwriting of existing experiment directory to avoid conflicts during repeated runs
        exist_ok=True,
        # Set to 0 to train all model layers, ensuring maximum flexibility and no frozen weights
        freeze=0,
        # Set the initial learning rate for the optimizer to control the step size in gradient descent
        lr0=0.01,
        # Disable early stopping by setting patience to 0, ensuring all specified epochs are completed
        patience=0,
        # Apply weight decay for regularization to prevent overfitting by penalizing large weights
        weight_decay=0.0005,
        # Save model checkpoints every 20 epochs to track progress and enable recovery
        save_period=20,
        # Enable saving of model checkpoints during training for later use or evaluation
        save=True
    )

    # Perform validation on the trained model to evaluate its performance on the validation dataset
    results_val = model.val(
        # Use the same dataset YAML file as in training to ensure consistency in data configuration
        data=args.data,
        # Maintain the same image size as training for consistent evaluation
        imgsz=args.imgsz,
        # Use a larger batch size for validation to speed up the evaluation process
        batch=16,
        # Use CPU for validation (can be changed to GPU, e.g., 'cuda', if available)
        device="cpu",
        # Generate visualization plots, such as confusion matrices and precision-recall curves, for performance analysis
        plots=True
    )

    # Display key validation metrics to summarize model performance
    print("Validation Results:")
    # Print mean Average Precision (mAP) at IoU=0.5, indicating detection performance at a moderate overlap threshold
    print(f"mAP@0.5: {results_val.box.map50:.4f}")
    # Print mAP averaged over IoU thresholds from 0.5 to 0.95, providing a comprehensive performance metric
    print(f"mAP@0.5:0.95: {results_val.box.map:.4f}")
    # Iterate through class names and display per-class mAP@0.5:0.95 for detailed performance breakdown
    for i, name in results_val.names.items():
        print(f"mAP@0.5:0.95 for {name}: {results_val.box.maps[i]:.4f}")

if __name__ == "__main__":
    """Entry point for executing the YOLO model training and validation pipeline.

    This block serves as the main entry point for the script, invoking the main function
    to configure, train, and validate the YOLO model for object detection tasks.

    Args:
        None

    Returns:
        None
    """
    main()