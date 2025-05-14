import cv2
import numpy as np
from typing import Tuple


def opening(
    mask: np.ndarray,
    kernel_size: Tuple[int, int] = (5, 5)
) -> np.ndarray:
    """
    Apply morphological opening (erosion followed by dilation) to clean up a binary mask.

    Args:
        mask: 2D NumPy array (binary mask, values 0 or 255).
        kernel_size: Size of the structuring element (default 5x5).

    Returns:
        Cleaned binary mask as a 2D NumPy array.
    """
    if mask.ndim != 2:
        raise ValueError("Input mask must be a 2D binary array.")
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return opened_mask


def closing(
    mask: np.ndarray,
    kernel_size: Tuple[int, int] = (5, 5)
) -> np.ndarray:
    """
    Apply morphological closing (dilation followed by erosion) to fill holes in a binary mask.

    Args:
        mask: 2D NumPy array (binary mask, values 0 or 255).
        kernel_size: Size of the structuring element (default 5x5).

    Returns:
        Enhanced binary mask as a 2D NumPy array.
    """
    if mask.ndim != 2:
        raise ValueError("Input mask must be a 2D binary array.")
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return closed_mask 