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


def calculate_ida_channel(rgb_img: np.ndarray) -> np.ndarray:
    """
    Calculate the Ida (Darkness) channel for an RGB image.
    
    Args:
        rgb_img: 3D NumPy array, RGB image (H, W, 3)
        
    Returns:
        2D NumPy array representing the Ida channel where
        Ida = max(R,G,B) - min(R,G,B)
    """
    if rgb_img.ndim != 3 or rgb_img.shape[2] != 3:
        raise ValueError("Input must be an RGB image with shape (H, W, 3)")
    
    r_channel = rgb_img[:, :, 0].astype(np.float32)
    g_channel = rgb_img[:, :, 1].astype(np.float32)
    b_channel = rgb_img[:, :, 2].astype(np.float32)
    
    max_rgb = np.maximum(np.maximum(r_channel, g_channel), b_channel)
    min_rgb = np.minimum(np.minimum(r_channel, g_channel), b_channel)
    
    ida_channel = max_rgb - min_rgb
    
    return ida_channel.astype(np.uint8)