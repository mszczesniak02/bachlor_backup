
from PIL import Image, ImageOps
import random
import shutil
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')

# Default configuration
# Try to find the dataset path
POSSIBLE_PATHS = [
    r"/content/datasets/multi/train_lab",
    r"/home/krzeslaav/Projects/datasets/dataset_segmentation/train_lab"
    # r"/content/datasets/DeepCrack/train_lab"
]


def find_dataset_root():
    for path in POSSIBLE_PATHS:
        if os.path.exists(path):
            # Extract root (parent of train_lab)
            return os.path.dirname(path)
    # If not found, return None
    return None


DATASET_ROOT = find_dataset_root()


def calculate_max_width(mask_path):
    """
    Calculates the maximum width of the crack in pixels using Distance Transform.
    """
    img_pil = Image.open(mask_path).convert('L')
    img = np.array(img_pil)

    # Binarize: Crack should be white (255), background black (0)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Distance Transform
    # Note: Using DIST_L2 (Euclidean distance) and mask size 5
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    # Max value in distance transform is half the max width
    max_dist = np.max(dist_transform)

    # Width = 2 * max_dist
    max_width = 2.0 * max_dist

    return max_width


def analyze_dataset(mask_dir):
    print(f"Analyzing masks in: {mask_dir}")

    if not os.path.exists(mask_dir):
        print(f"Error: Directory {mask_dir} not found.")
        return [], []

    mask_files = [
        f for f in os.listdir(mask_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        and not any(x in f.lower() for x in ['eugene', 'florian', 'volker'])
    ]

    widths = []
    files_with_width = []

    for mask_file in tqdm(mask_files, desc="Calculating Widths"):
        mask_path = os.path.join(mask_dir, mask_file)
        try:
            width = calculate_max_width(mask_path)
            widths.append(width)
            files_with_width.append((mask_file, width))
        except Exception as e:
            # print(f"Error processing {mask_file}: {e}")
            pass

    return widths, files_with_width


def plot_histogram(widths, output_file="width_hist.png", title_suffix=""):
    if not widths:
        print("No data to plot.")
        return

    plt.figure(figsize=(12, 6))

    # Filter out 0 width (no crack) for log scale or just visibility if dominated by 0
    non_zero_widths = [w for w in widths if w > 0]
    zeros = len(widths) - len(non_zero_widths)

    plt.hist(non_zero_widths, bins=50, color='skyblue',
             edgecolor='black', alpha=0.7)

    plt.title(
        f'Distribution of Max Crack Widths (Pixels) - {title_suffix}\nNon-zero: {len(non_zero_widths)}, Zeros: {zeros}')
    plt.xlabel('Max Width (Pixels)')
    plt.ylabel('Count')
    plt.grid(axis='y', alpha=0.3)

    # Visualizing the CHOSEN thresholds
    thresholds = [8.0, 14.0, 26.0]
    colors = ['green', 'orange', 'red']
    labels = ['Włosowe (8px)', 'Małe (14px)', 'Średnie (26px)']

    for th, col, lab in zip(thresholds, colors, labels):
        plt.axvline(th, color=col, linestyle='-', linewidth=2, label=lab)

    # Add colored text for regions (approximate positions)
    plt.text(4, plt.ylim()[1]*0.9, 'WŁOSOWE\n(<8px)',
             color='green', ha='center', fontsize=10, fontweight='bold')
    plt.text(11, plt.ylim()[1]*0.8, 'MAŁE\n(8-14px)',
             color='orange', ha='center', fontsize=10, fontweight='bold')
    plt.text(20, plt.ylim()[1]*0.9, 'ŚREDNIE\n(14-26px)',
             color='red', ha='center', fontsize=10, fontweight='bold')
    plt.text(35, plt.ylim()[1]*0.8, 'DUŻE\n(>26px)',
             color='darkred', ha='center', fontsize=10, fontweight='bold')

    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Histogram saved to {output_file}")
    plt.close()


def plot_class_distribution(counts, category_names, output_file="class_dist.png", title="Class Distribution"):
    plt.figure(figsize=(10, 6))

    values = []
    for i, name in enumerate(category_names):
        # Try retrieving by index first, then by name
        if i in counts:
            values.append(counts[i])
        elif name in counts:
            values.append(counts[name])
        else:
            values.append(0)

    bars = plt.bar(category_names, values, color=[
                   'green', 'orange', 'red', 'darkred'])

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.grid(axis='y', alpha=0.3)

    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height}',
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Class distribution plot saved to {output_file}")
    plt.close()


def categorize_crack_width(width, thresholds):
    """
    Categorizes crack based on max width in pixels.
    Categories: 1_wlosowe, 2_male, 3_srednie, 4_duze
    """
    if len(thresholds) != 3:
        thresholds = [8.0, 14.0, 26.0]

    if width < thresholds[0]:
        return 0, "wlosowe"
    elif width < thresholds[1]:
        return 1, "male"
    elif width < thresholds[2]:
        return 2, "srednie"
    else:
        return 3, "duze"


def create_categorized_dataset_width(mask_dir, output_base_dir, image_dir=None, thresholds=None):
    if int(cv2.__version__.split('.')[0]) < 3:
        print("Error: OpenCV version too old.")
        return {}

    category_names = ["1_wlosowe", "2_male", "3_srednie", "4_duze"]
    for cat_name in category_names:
        os.makedirs(os.path.join(output_base_dir, cat_name), exist_ok=True)

    mask_files = [
        f for f in os.listdir(mask_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        and not any(x in f.lower() for x in ['eugene', 'florian', 'volker'])
    ]
    category_counts = {i: 0 for i in range(4)}

    print(f"Categorizing {len(mask_files)} files from {mask_dir}...")

    for mask_file in tqdm(mask_files, desc="Categorizing"):
        mask_path = os.path.join(mask_dir, mask_file)
        try:
            width = calculate_max_width(mask_path)

            # Skip empty masks
            if width < 1.0:
                continue

            cat_id, _ = categorize_crack_width(width, thresholds)
            category_counts[cat_id] += 1

            # COPY LOGIC
            src_file = mask_path  # Default copy mask
            dest_name = mask_file

            if image_dir:
                # Try to find corresponding image
                image_source = os.path.join(image_dir, mask_file)
                # Handle potential extension mismatch (mask .png, img .jpg)
                # Handle potential extension mismatch (mask .png, img .jpg)
                # Removed JPG support as per user request to stick to PNG
                if not os.path.exists(image_source):
                    pass

                if os.path.exists(image_source):
                    src_file = image_source
                    dest_name = os.path.basename(image_source)
                else:
                    # If image not found, skip
                    continue

            # Force PNG extension for output
            dest_name = os.path.splitext(dest_name)[0] + '.png'
            dest_path = os.path.join(
                output_base_dir, category_names[cat_id], dest_name)

            # RESIZE LOGIC (256x256)
            # Load, resize (NEAREST for masks), and save
            try:
                with Image.open(src_file) as img:
                    img = img.convert('L')
                    # Binarize to remove JPEG artifacts (threshold at 127)
                    img = img.point(lambda p: 255 if p > 127 else 0)
                    img = img.resize((256, 256), resample=Image.NEAREST)
                    img.save(dest_path)
            except Exception as e:
                print(f"Error resizing/saving {src_file}: {e}")
                continue

        except Exception as e:
            continue

    return category_counts


def balance_dataset(dataset_dir, category_names):
    """
    Balances the dataset by augmenting minority classes to match the majority class count.
    """
    print(f"\nBalancing dataset in {dataset_dir}...")

    # 1. Count existing files
    counts = {}
    file_lists = {}

    max_count = 0

    for cat in category_names:
        cat_dir = os.path.join(dataset_dir, cat)
        files = [
            f for f in os.listdir(cat_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            and not any(x in f.lower() for x in ['eugene', 'florian', 'volker'])
        ]
        counts[cat] = len(files)
        file_lists[cat] = files
        if len(files) > max_count:
            max_count = len(files)

    print(f"Current counts: {counts}")
    print(f"Target count per class: {max_count}")

    # 2. Augment
    for cat in category_names:
        current_count = counts[cat]
        if current_count >= max_count:
            continue

        needed = max_count - current_count
        print(f"Augmenting {cat}: need {needed} more images...")

        cat_dir = os.path.join(dataset_dir, cat)
        # Filter out existing augmented files to avoid double-augmentation
        source_files = [f for f in file_lists[cat] if "_bal_" not in f]

        if not source_files:
            print(
                f"No source files found for {cat} (all are augmented?). Using all.")
            source_files = file_lists[cat]

        # Simple random sampling with replacement
        for i in tqdm(range(needed), desc=f"Balancing {cat}"):
            src_file = random.choice(source_files)
            src_path = os.path.join(cat_dir, src_file)

            try:
                # Force L mode for masks and Nearest Neighbor for NO ANTI-ALIASING
                img = Image.open(src_path).convert('L')

                # Binarize to remove JPEG artifacts (threshold at 127)
                img = img.point(lambda p: 255 if p > 127 else 0)

                # Ensure it is 256x256 (if source wasn't already processed, but safe to force)
                img = img.resize((256, 256), resample=Image.NEAREST)

                # Apply random augmentation
                aug_type = random.choice(
                    ['flip_lr', 'flip_tb', 'rotate90', 'rotate180', 'rotate270'])

                if aug_type == 'flip_lr':
                    aug_img = ImageOps.mirror(img)
                elif aug_type == 'flip_tb':
                    aug_img = ImageOps.flip(img)
                elif aug_type == 'rotate90':
                    aug_img = img.transpose(Image.ROTATE_90)
                elif aug_type == 'rotate180':
                    aug_img = img.transpose(Image.ROTATE_180)
                elif aug_type == 'rotate270':
                    aug_img = img.transpose(Image.ROTATE_270)
                else:
                    aug_img = img

                # Save as PNG to avoid JPEG artifacts on masks
                fname, _ = os.path.splitext(src_file)
                new_name = f"{fname}_bal_{i}.png"
                aug_img.save(os.path.join(cat_dir, new_name))

            except Exception as e:
                # print(f"Error augmenting {src_file}: {e}")
                pass

    # Re-count to return updated stats
    new_counts = {}
    for cat in category_names:
        cat_dir = os.path.join(dataset_dir, cat)
        new_counts[cat] = len(os.listdir(cat_dir))

    return new_counts


def main():
    if not DATASET_ROOT:
        print("Error: Could not find dataset root.")
        print(f"Checked paths: {POSSIBLE_PATHS}")
        return

    # Define thresholds based on previous analysis
    thresholds = [8.0, 14.0, 26.0]
    print(f"Using Max Width Thresholds: {thresholds}")

    # Process both TRAIN and TEST

    subsets = [
        {"name": "TRAIN", "mask_sub": "train_lab",
            "out_sub": "train_img", "balance": True},
        {"name": "TEST",  "mask_sub": "test_lab",
            "out_sub": "test_img", "balance": False}
    ]

    # Check execution environment (Colab vs Local) to determine output root
    if DATASET_ROOT.startswith("/content"):
        OUTPUT_ROOT = r"/content/datasets/classification_width"
    else:
        # Local: Create sibling directory to the input dataset
        parent_dir = os.path.dirname(DATASET_ROOT)
        OUTPUT_ROOT = os.path.join(parent_dir, "classification_width")

    category_names = ["1_wlosowe", "2_male", "3_srednie", "4_duze"]

    for subset in subsets:
        print(f"\n{'='*20} PROCESSING {subset['name']} DATASET {'='*20}")

        mask_dir = os.path.join(DATASET_ROOT, subset["mask_sub"])
        # User requested NO IMAGES loopkup, so we work on masks only
        # img_dir = os.path.join(DATASET_ROOT, subset["img_sub"])
        img_dir = None

        # Output directory
        output_dir = os.path.join(OUTPUT_ROOT, subset["out_sub"])

        # Clean output directory to ensure fresh start (removes old artifacts)
        if os.path.exists(output_dir):
            print(f"Cleaning existing directory: {output_dir}")
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # 1. Analyze and Plot Histogram
        widths, _ = analyze_dataset(mask_dir)
        hist_filename = f"hist_{subset['name'].lower()}.png"
        plot_histogram(widths, output_file=hist_filename,
                       title_suffix=subset['name'])

        # 2. Categorize and Create Dataset (Masks Only)
        counts = create_categorized_dataset_width(
            mask_dir, output_dir, image_dir=None, thresholds=thresholds)

        # 3. BALANCE (Only for Train)
        if subset.get("balance", False):
            print("Applying Class Balancing (Augmentation)...")
            counts = balance_dataset(output_dir, category_names)

        # 4. Plot Class Distribution (After balancing if applied)
        dist_filename = f"dist_{subset['name'].lower()}.png"
        plot_class_distribution(counts, category_names, output_file=dist_filename,
                                title=f"Class Distribution - {subset['name']}")

        print(f"Completed {subset['name']}. Output: {output_dir}")
        print(f"Final Counts: {counts}")


if __name__ == "__main__":
    main()
