import cv2
import numpy as np
from typing import Tuple


def remove_hair(
    gray_image: np.ndarray,
    kernel_size: Tuple[int, int] = (31, 31),
    inpaint_radius: int = 1
) -> np.ndarray:
    """
    Remove hair artifacts from a grayscale image using black-hat morphological filtering and inpainting.

    Args:
        gray_image: 2D NumPy array (grayscale image).
        kernel_size: Size of the structuring element for black-hat (default 31x31).
        inpaint_radius: Radius for inpainting (default 1).

    Returns:
        Inpainted grayscale image with hair removed.
    """
    if gray_image.ndim != 2:
        raise ValueError("Input image must be a 2D grayscale array.")
    # Structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    # Black-hat to highlight dark hair on light skin
    blackhat = cv2.morphologyEx(gray_image, cv2.MORPH_BLACKHAT, kernel)
    # Threshold the blackhat image to get a hair mask
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    # Inpaint to remove hair
    inpainted = cv2.inpaint(gray_image, hair_mask, inpaint_radius, cv2.INPAINT_TELEA)
    return inpainted


def correct_illumination(
    gray_image: np.ndarray,
    kernel_size: Tuple[int, int] = (31, 31)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Correct non-uniform illumination using morphological opening (background estimation) and division.

    Args:
        gray_image: 2D NumPy array (grayscale image).
        kernel_size: Size of structuring element for background estimation (default 31x31).

    Returns:
        A tuple containing:
            - corrected_image: Illumination-corrected grayscale image (uint8).
            - background_estimate: The estimated background (uint8).
    """
    if gray_image.ndim != 2:
        raise ValueError("Input image must be a 2D grayscale array.")
    # Structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    # Estimate background via opening
    background_estimate_uint8 = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernel)
    
    background_float = background_estimate_uint8.astype(np.float32)
    gray_float = gray_image.astype(np.float32)
    
    # Ensure denominator for division is not zero and has a minimum value (e.g., 1.0)
    denominator = np.clip(background_float, 1.0, None)
    
    # Perform division with a more sensible scale (e.g., 128.0 or mean of original gray)
    # Using 128.0 to aim for a mid-range intensity output.
    # Alternative: scale = np.mean(gray_image) if gray_image.size > 0 else 128.0
    corrected_float = cv2.divide(gray_float, denominator, scale=128.0)
    
    # Clip final result to 0-255 and convert to uint8
    corrected_image_uint8 = np.clip(corrected_float, 0, 255).astype(np.uint8)
    
    return corrected_image_uint8, background_estimate_uint8


def adaptive_threshold(
    gray_image: np.ndarray,
    max_value: int = 255,
    method: int = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    threshold_type: int = cv2.THRESH_BINARY,
    block_size: int = 11,
    C: int = 2
) -> np.ndarray:
    """
    Apply adaptive thresholding to a grayscale image.

    Args:
        gray_image: 2D NumPy array (grayscale image).
        max_value: Maximum value to use with THRESH_BINARY.
        method: Adaptive method (GAUSSIAN or MEAN).
        threshold_type: cv2.THRESH_BINARY or cv2.THRESH_BINARY_INV.
        block_size: Size of neighborhood area (must be odd).
        C: Constant subtracted from the mean or weighted mean.

    Returns:
        Binary mask as a 2D NumPy array.
    """
    if gray_image.ndim != 2:
        raise ValueError("Input image must be a 2D grayscale array.")
    if block_size % 2 == 0 or block_size < 3:
        raise ValueError("block_size must be an odd number >= 3.")
    return cv2.adaptiveThreshold(
        gray_image,
        max_value,
        method,
        threshold_type,
        block_size,
        C
    ) 