
import cv2
import numpy as np


def calculate_max_width_synthetic(img):
    """
    Calculates the maximum width of the crack in pixels using Distance Transform on a numpy array.
    """
    # Distance Transform
    dist_transform = cv2.distance_transform(img, cv2.DIST_L2, 5)
    max_dist = np.max(dist_transform)
    max_width = 2.0 * max_dist
    return max_width


def test():
    print("Testing width calculation...")

    # Create a 100x100 black image
    img = np.zeros((100, 100), dtype=np.uint8)

    # Draw a rectangle of known width 10 (from x=20 to x=30)
    # The width is 10 pixels.
    # Note: cv2.rectangle uses (x1, y1), (x2, y2).
    # If we fill from x=20 to x=30 (inclusive?), let's see.
    # 20 to 30 is 10 pixels.
    cv2.rectangle(img, (20, 0), (29, 99), 255, -1)

    # Check width manually
    # Row 50 slice
    row = img[50, :]
    # print(row[15:35])
    width_pixels = np.sum(row == 255)
    print(f"Actual width in pixels (by counting): {width_pixels}")

    calc_width = calculate_max_width_synthetic(img)
    print(f"Calculated Max Width (Distance Transform): {calc_width:.2f}")

    assert abs(
        calc_width - width_pixels) <= 1.5, f"Expected ~{width_pixels}, got {calc_width}"
    print("Test passed!")


if __name__ == "__main__":
    test()
