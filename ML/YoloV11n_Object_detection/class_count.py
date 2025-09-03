import os
from collections import Counter
from pathlib import Path

def class_count(label_dir: str | Path) -> Counter:
    """Count the occurrences of each class in YOLO label files.

    Reads all YOLO label files in the specified directory and counts how many times each class ID appears.

    Args:
        label_dir (str | Path): Path to the directory containing YOLO label files (.txt).

    Returns:
        Counter: A Counter object mapping class IDs to their occurrence counts.
    """
    # Convert input path to Path object for consistent handling
    label_dir = Path(label_dir)
    # Initialize Counter to track class occurrences
    class_counts = Counter()
    # Iterate through all files in the label directory
    for label_file in os.listdir(label_dir):
        # Open and read each label file
        with open(label_dir / label_file, 'r') as f:
            # Process each line in the file
            for line in f:
                # Extract class ID from the first column of the line
                class_id = int(line.split()[0])
                # Increment count for the class ID
                class_counts[class_id] += 1
    return class_counts

if __name__ == "__main__":
    """Entry point for running the class counting script.

    Executes the class counting function on a specified directory and prints the results.

    Args:
        label_dir (str): Path to the directory containing training label files.
    """
    # Count classes in the training labels directory
    counts = class_count("./dataset/train/labels")
    # Print the class counts
    print(counts)