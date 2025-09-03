import os

def rename_image_label_pairs(image_dir, label_dir):
    """Rename pairs of image and label files to a standardized format.

    Matches image and label files based on their base names (without extensions), renames them to a
    consistent format (e.g., image00001.jpg and image00001.txt), and reports any mismatches.

    Args:
        image_dir (str): Path to the directory containing image files.
        label_dir (str): Path to the directory containing label files.
    """
    # List all files in the image and label directories
    image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    label_files = [f for f in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, f))]
    
    # Filter files by valid image and label extensions
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
    image_files = [f for f in image_files if f.lower().endswith(image_extensions)]
    label_extensions = ('.txt',)
    label_files = [f for f in label_files if f.lower().endswith(label_extensions)]
    
    # Sort files to ensure consistent ordering
    image_files.sort()
    label_files.sort()
    
    # Create dictionaries mapping base names (without extensions) to full filenames
    image_basenames = {os.path.splitext(f)[0]: f for f in image_files}
    label_basenames = {os.path.splitext(f)[0]: f for f in label_files}
    
    # Find matching image-label pairs
    paired_files = []
    for base_name in image_basenames:
        if base_name in label_basenames:
            paired_files.append((image_basenames[base_name], label_basenames[base_name]))
        else:
            print(f"Warning: Image {image_basenames[base_name]} has no corresponding .txt file.")
    
    # Check for label files without corresponding images
    for base_name in label_basenames:
        if base_name not in image_basenames:
            print(f"Warning: Label file {label_basenames[base_name]} has no corresponding image.")
    
    # Rename matched image-label pairs to a standardized format
    for index, (image_name, label_name) in enumerate(paired_files, start=1):
        # Get the image file extension
        image_extension = os.path.splitext(image_name)[1]
        # Create new base name with 5-digit padding (e.g., image00001)
        new_name_base = f"image{index:05d}"
        new_image_name = f"{new_name_base}{image_extension}"
        new_label_name = f"{new_name_base}.txt"
        
        # Construct full file paths
        old_image_path = os.path.join(image_dir, image_name)
        new_image_path = os.path.join(image_dir, new_image_name)
        old_label_path = os.path.join(label_dir, label_name)
        new_label_path = os.path.join(label_dir, new_label_name)
        
        # Rename the image file
        try:
            os.rename(old_image_path, new_image_path)
            print(f"Renamed: {image_name} -> {new_image_name}")
        except Exception as e:
            print(f"Error renaming image {image_name}: {e}")
            continue  # Continue to next pair even if image renaming fails
        
        # Rename the label file
        try:
            os.rename(old_label_path, new_label_path)
            print(f"Renamed: {label_name} -> {new_label_name}")
        except Exception as e:
            print(f"Error renaming label {label_name}: {e}")
    
    # Report the total number of renamed pairs
    print(f"Total pairs renamed: {len(paired_files)}")

if __name__ == "__main__":
    """Entry point for running the file renaming script.

    Renames image and label file pairs in the specified directories to a standardized format.

    Args:
        image_dir (str): Path to the directory containing validation images.
        label_dir (str): Path to the directory containing validation labels.
    """
    # Specify directories for validation images and labels
    rename_image_label_pairs("dataset/valid/images", "dataset/valid/labels")
    # Optionally, specify directories for training images and labels (commented out)
    # rename_image_label_pairs("dataset/train/images", "dataset/train/labels")