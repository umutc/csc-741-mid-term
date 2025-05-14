import cv2
import numpy as np


def rgb_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to grayscale.

    Args:
        image: NumPy array of shape (H, W, 3) in RGB format.

    Returns:
        Grayscale image as a 2D NumPy array.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be an RGB image with shape (H, W, 3).")
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def rgb_to_hsv(image: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to HSV color space.

    Args:
        image: NumPy array of shape (H, W, 3) in RGB format.

    Returns:
        HSV image as a NumPy array with shape (H, W, 3).
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be an RGB image with shape (H, W, 3).")
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV) 