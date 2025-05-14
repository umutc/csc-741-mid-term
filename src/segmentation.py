import cv2
import numpy as np
from typing import Tuple


def global_threshold(
    gray_image: np.ndarray,
    thresh: int = 128,
    max_value: int = 255
) -> np.ndarray:
    """
    Apply a global threshold to a grayscale image to produce a binary mask.

    Args:
        gray_image: 2D NumPy array representing a grayscale image.
        thresh: Threshold value to classify pixel intensity.
        max_value: Value to assign to pixels above the threshold (default 255).

    Returns:
        Binary mask as a 2D NumPy array with values 0 or max_value.
    """
    if gray_image.ndim != 2:
        raise ValueError("Input image must be a 2D grayscale array.")
    _, mask = cv2.threshold(gray_image, thresh, max_value, cv2.THRESH_BINARY)
    return mask


def otsu_threshold(
    gray_image: np.ndarray
) -> Tuple[np.ndarray, int]:
    """
    Apply Otsu's thresholding method to a grayscale image.

    Args:
        gray_image: 2D NumPy array representing a grayscale image.

    Returns:
        A tuple (binary_mask, optimal_threshold_value).
    """
    if gray_image.ndim != 2:
        raise ValueError("Input image must be a 2D grayscale array.")
    thresh, mask = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return mask, int(thresh)


def apply_threshold(
    gray_image: np.ndarray,
    method: str = 'otsu',
    global_thresh: int = 128,
    cleanup: bool = False,
    open_kernel: Tuple[int, int] = (5, 5),
    close_kernel: Tuple[int, int] = (5, 5)
) -> Tuple[np.ndarray, int]:
    """
    Apply a specified thresholding method and optionally perform morphological cleanup.

    Args:
        gray_image: 2D NumPy array (grayscale image).
        method: 'global', 'otsu', or 'adaptive'.
        global_thresh: Threshold for 'global' method.
        cleanup: If True, apply opening then closing to clean the mask.
        open_kernel: Kernel size for opening.
        close_kernel: Kernel size for closing.

    Returns:
        mask: Binary mask as a 2D NumPy array.
        thresh: Threshold value used (None for adaptive).
    """
    # Select thresholding method
    if method == 'global':
        mask = global_threshold(gray_image, thresh=global_thresh)
        thresh_used = global_thresh
    elif method == 'otsu':
        mask, thresh_used = otsu_threshold(gray_image)
    elif method == 'adaptive':
        from src.preprocessing import adaptive_threshold
        mask = adaptive_threshold(gray_image)
        thresh_used = None
    else:
        raise ValueError(f"Unsupported method '{method}'. Use 'global', 'otsu', or 'adaptive'.")

    # Optional cleanup
    if cleanup:
        from src.morphology import opening, closing
        mask = opening(mask, kernel_size=open_kernel)
        mask = closing(mask, kernel_size=close_kernel)

    return mask, thresh_used 