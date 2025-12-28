
from skimage.morphology import skeletonize, remove_small_objects, remove_small_holes, closing
from skimage import data, morphology, measure
from skimage.util import invert
import cv2

import torch
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

import os
from PIL import Image


def preprocess_mask(mask_input, threshold: float = 0.5, min_size: int = 50, hole_threshold: int = 30) -> np.ndarray:
    """
    Ensures mask is binary and performs morphological cleanup (noise removal, hole filling, closing).
    Accepts mask_input as numpy array or file path.
    """
    if isinstance(mask_input, str):
        if not os.path.exists(mask_input):
            raise FileNotFoundError(f"Mask file not found: {mask_input}")
        # Load as grayscale using cv2
        mask = cv2.imread(mask_input, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            # Try via PIL if cv2 fails or for other formats
            try:
                mask = np.array(Image.open(mask_input).convert('L'))
            except Exception:
                raise ValueError(f"Could not load mask from {mask_input}")
    elif torch.is_tensor(mask_input):
        mask = mask_input.cpu().numpy()
    else:
        mask = mask_input

    # Normalize if needed (0-255 to 0-1)
    if mask.max() > 1:
        mask = mask / 255.0

    binary_mask = mask > threshold

    # Morphological cleanup
    # 1. Remove small artifacts (noise)
    mask_clean = remove_small_objects(binary_mask, min_size=min_size)

    # 2. Fill small holes
    mask_filled = remove_small_holes(mask_clean, area_threshold=hole_threshold)

    # 3. Close gaps (optional, small kernel)
    # Using a small 3x3 footprint for closing
    mask_closed = closing(mask_filled, footprint=np.ones((3, 3)))

    return mask_closed.astype(bool)


def calculate_basic_properties(binary_mask: np.ndarray):
    """
    Calculates basic region properties like area, perimeter, eccentricity, etc.
    """
    label_img = measure.label(binary_mask)
    regions = measure.regionprops(label_img)

    results = {
        "area_pixels": 0,
        "perimeter_pixels": 0,
        "bbox": None,
        "orientation": 0,
        "centroid": (0, 0),
        "eccentricity": 0,
        "axis_major_length": 0,
        "axis_minor_length": 0,
        "feret_diameter_max": 0,
        "solidity": 0,
        "extent": 0
    }

    if not regions:
        return results

    # Assuming the crack is the largest region for shape descriptors
    largest_region = max(regions, key=lambda r: r.area)

    results["area_pixels"] = sum(r.area for r in regions)
    results["perimeter_pixels"] = sum(r.perimeter for r in regions)

    # Properties of largest segment
    results["bbox"] = largest_region.bbox
    results["orientation"] = largest_region.orientation
    results["centroid"] = largest_region.centroid
    results["eccentricity"] = largest_region.eccentricity
    results["axis_major_length"] = largest_region.axis_major_length
    results["axis_minor_length"] = largest_region.axis_minor_length
    # Feret diameter requires convex hull which might not be computed by default in all versions,
    # but feret_diameter_max is standard in recent skimage
    try:
        results["feret_diameter_max"] = largest_region.feret_diameter_max
    except AttributeError:
        # Fallback for older skimage versions
        results["feret_diameter_max"] = 0

    results["solidity"] = largest_region.solidity
    results["extent"] = largest_region.extent

    return results


def get_skeleton(binary_mask: np.ndarray) -> np.ndarray:
    """
    Returns the skeleton of the binary mask.
    """
    return skeletonize(binary_mask)


def calculate_length(skeleton: np.ndarray) -> float:
    """
    Calculates the length of the crack based on the skeleton pixels.
    This is a rough approximation (pixel count). For more precision, we could correct for connectivity.
    """
    return np.sum(skeleton)


def calculate_width(binary_mask: np.ndarray, skeleton: np.ndarray):
    """
    Calculates width by using distance transform on the binary mask 
    and sampling it at the skeleton coordinates.
    Since skeleton is the medial axis, distance transform values there represent half-width.
    """
    # Distance transform: distance to nearest zero (background)
    # We want distance from skeleton to edge of crack.
    # In binary mask, 1 is crack, 0 is background.
    # ndimage.distance_transform_edt calculates distance to nearest background pixel.
    # So at center (skeleton), value is radius (half-width).

    dist_transform = ndimage.distance_transform_edt(binary_mask)

    # Filter distances only at skeleton points
    # Diameter = 2 * Radius
    skeleton_width_values = dist_transform[skeleton] * 2

    if len(skeleton_width_values) == 0:
        return {
            "mean_width": 0.0,
            "max_width": 0.0,
            "min_width": 0.0,
            "std_width": 0.0,
            "widths": []
        }, dist_transform

    stats = {
        "mean_width": np.mean(skeleton_width_values),
        "max_width": np.max(skeleton_width_values),
        "min_width": np.min(skeleton_width_values),
        "std_width": np.std(skeleton_width_values),
        "widths": skeleton_width_values
    }

    return stats, dist_transform


def analyze_crack_mask(mask: np.ndarray, pixel_size_mm: float = None):
    """
    Main analysis function.
    pixel_size_mm: if provided, converts pixels to physical units (mm).
    """
    binary_mask = preprocess_mask(mask)

    # 1. Basic Props
    basic_props = calculate_basic_properties(binary_mask)

    # 2. Skeleton & Length
    skeleton = get_skeleton(binary_mask)
    length_pixels = calculate_length(skeleton)

    # 3. Width
    width_stats, dist_map = calculate_width(binary_mask, skeleton)

    analysis_results = {
        "basic": basic_props,
        "length_pixels": length_pixels,
        "width_stats": width_stats
    }

    # Conversion if pixel size known
    if pixel_size_mm:
        analysis_results["length_mm"] = length_pixels * pixel_size_mm
        analysis_results["width_mean_mm"] = width_stats["mean_width"] * pixel_size_mm
        analysis_results["width_max_mm"] = width_stats["max_width"] * \
            pixel_size_mm
        analysis_results["area_mm2"] = basic_props["area_pixels"] * \
            (pixel_size_mm ** 2)

    return analysis_results, skeleton, dist_map


def visualize_analysis(image, mask: np.ndarray, skeleton: np.ndarray, dist_map: np.ndarray, results: dict, save_path: str = None):
    """
    Plots the analysis results: Original, Skeleton on Mask, Width/Distance Map.
    Accepts image as numpy array or file path (optional).
    """
    if isinstance(image, str) and os.path.exists(image):
        image = cv2.imread(image)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 4 if image is not None else 3, figsize=(
        20 if image is not None else 15, 5))

    idx = 0
    if image is not None:
        axes[idx].imshow(image)
        axes[idx].set_title("Original Image")
        axes[idx].axis('off')
        idx += 1

    # Original Image with Mask overlay (optional) or just Mask
    axes[idx].imshow(mask, cmap='gray')
    axes[idx].set_title("Binary Mask (Processed)")
    axes[idx].axis('off')
    idx += 1

    # Skeleton
    axes[idx].imshow(mask, cmap='gray', alpha=0.3)
    axes[idx].imshow(skeleton, cmap='jet', alpha=0.7)
    axes[idx].set_title(f"Skeleton (Len: {results['length_pixels']:.1f} px)")
    axes[idx].axis('off')
    idx += 1

    # Width Map
    # Mask dist map by binary_mask to clear background
    masked_dist_map = dist_map * mask
    im3 = axes[idx].imshow(masked_dist_map, cmap='magma')
    axes[idx].set_title(
        f"Width Map (Max: {results['width_stats']['max_width']:.1f} px)")
    axes[idx].axis('off')
    plt.colorbar(im3, ax=axes[idx], fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Analysis visualization saved to {save_path}")
    # plt.show() # Return figure object or show interactive if needed, but in pipeline better to save or return fig


if __name__ == "__main__":
    # Test with provided image
    image_path = "/home/krzeslaav/Projects/bachlor/src/final_prediction_pipeline/output_predictions/image_test_0_ensemble_binary.png"
    print(f"Running analytics test on {image_path}...")

    if os.path.exists(image_path):
        # Load image
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Create a dummy mask using simple thresholding (inverse, assuming dark cracks)
        # This is just for testing the analytics module standalone.
        # cracks are dark -> < threshold
        _, mask_thresh = cv2.threshold(
            img_gray, 80, 255, cv2.THRESH_BINARY_INV)
        mask_thresh = mask_thresh > 0  # make boolean

        results, skel, dmap = analyze_crack_mask(
            mask_thresh, pixel_size_mm=0.26)  # assumption: 0.26mm/px

        print("Analysis Results:")
        print(
            f"  Area: {results['basic']['area_pixels']} px ({results.get('area_mm2', 0):.2f} mm2)")
        print(
            f"  Length: {results['length_pixels']:.1f} px ({results.get('length_mm', 0):.2f} mm)")
        print(
            f"  Max Width: {results['width_stats']['max_width']:.2f} px ({results.get('width_max_mm', 0):.2f} mm)")
        print(f"  Solidity: {results['basic']['solidity']:.3f}")

        visualize_analysis(img_rgb, mask_thresh, skel, dmap,
                           results, save_path="analytics_output.png")
    else:
        print("Test image not found, using dummy data.")
        fake_mask = np.zeros((100, 100))
        fake_mask[20:80, 45:55] = 1
        fake_mask[50:60, 20:80] = 1

        results, skel, dmap = analyze_crack_mask(fake_mask)
        print("Results:", results['basic']
              ['area_pixels'], results['length_pixels'])
        visualize_analysis(None, fake_mask, skel, dmap, results)
