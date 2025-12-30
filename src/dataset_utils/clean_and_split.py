import os
import shutil
import random
from pathlib import Path

# Configuration
DATASET_PATH = "/home/krzeslaav/Projects/datasets/dataset_segmentation (Copy)"
BLACKLIST = ["Rissbilder_for_Florian", "Sylvie_Chambon", "volker"]
MINI_TEST_SIZE = 100


def clean_dataset(base_path, blacklist):
    """Removes files containing blacklisted strings from train/test directories."""
    print(f"Cleaning dataset in {base_path}...")
    removed_count = 0

    # We need to clean both images and labels in train and test
    dirs_to_clean = [
        "train_img", "train_lab",
        "test_img", "test_lab"
    ]

    for subdir in dirs_to_clean:
        dir_path = Path(base_path) / subdir
        if not dir_path.exists():
            continue

        for file_path in dir_path.glob("*"):
            if file_path.is_file():
                # Case-insensitive check
                if any(bad_word.lower() in file_path.name.lower() for bad_word in blacklist):
                    try:
                        file_path.unlink()
                        removed_count += 1
                        # formatted output for log
                        # print(f"Removed: {file_path.name}")
                    except OSError as e:
                        print(f"Error deleting {file_path}: {e}")

    print(f"Cleanup complete. Removed {removed_count} files.")


def extract_mini_test(base_path, count):
    """Moves 'count' random images/labels from train to a new 'mini_test' folder."""
    base = Path(base_path)
    train_img_dir = base / "train_img"
    train_lab_dir = base / "train_lab"

    # New directories for mini calibration set
    mini_img_dir = base / "mini_test_img"
    mini_lab_dir = base / "mini_test_lab"

    mini_img_dir.mkdir(exist_ok=True)
    mini_lab_dir.mkdir(exist_ok=True)

    # Get all valid image files
    all_images = list(train_img_dir.glob("*"))
    valid_images = [img for img in all_images if img.suffix.lower() in [
        '.jpg', '.png', '.jpeg', '.bmp']]

    if len(valid_images) < count:
        print(
            f"Not enough training images to extract {count}. Found {len(valid_images)}.")
        return

    selected_images = random.sample(valid_images, count)
    print(f"Extracting {len(selected_images)} pairs to mini_test...")

    moved_count = 0
    for img_path in selected_images:
        # Resolve corresponding label
        # Trying exact name match first
        # Assuming .png labels based on context, checking existence
        lab_name = img_path.stem + ".png"
        lab_path = train_lab_dir / lab_name

        # If not found, try same extension as image (rare for masks but possible)
        if not lab_path.exists():
            lab_path = train_lab_dir / img_path.name

        if lab_path.exists():
            # Move both
            shutil.move(str(img_path), str(mini_img_dir / img_path.name))
            shutil.move(str(lab_path), str(mini_lab_dir / lab_path.name))
            moved_count += 1
        else:
            print(
                f"Warning: Label not found for {img_path.name}, skipping move.")

    print(f"Extraction complete. Moved {moved_count} image-label pairs.")


if __name__ == "__main__":
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Path {DATASET_PATH} does not exist.")
    else:
        clean_dataset(DATASET_PATH, BLACKLIST)
        extract_mini_test(DATASET_PATH, MINI_TEST_SIZE)
