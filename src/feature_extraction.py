import cv2
import numpy as np
from typing import Dict, List, Union, Tuple, Optional, Any

# Import our mask application utility from histogram_utils
try:
    from src.histogram_utils import apply_mask_to_channel
except ImportError:
    # Fallback implementation if needed
    def apply_mask_to_channel(channel, mask, outside_value=None):
        """Fallback implementation of apply_mask_to_channel if import fails"""
        binary_mask = mask > 0
        if outside_value is None:
            return channel[binary_mask]
        return np.where(binary_mask, channel, outside_value)

# Import border and texture feature extraction functions if available
try:
    from src.border_texture_utils import (
        calculate_gradient_features,
        calculate_texture_features
    )
except ImportError:
    # Define stubs that return empty dictionaries if the imports fail
    def calculate_gradient_features(grayscale_img, mask, kernel_size=3):
        """Stub for gradient features if import fails"""
        print("Warning: border_texture_utils not found. Border features unavailable.")
        return {}
    
    def calculate_texture_features(grayscale_img, mask, kernel_size=7):
        """Stub for texture features if import fails"""
        print("Warning: border_texture_utils not found. Texture features unavailable.")
        return {}

def calculate_intensity_stats(
    grayscale_img: np.ndarray, 
    mask: np.ndarray
) -> Dict[str, float]:
    """
    Calculate basic statistical intensity features from a grayscale image within a mask.
    
    Args:
        grayscale_img: 2D NumPy array, grayscale image
        mask: 2D NumPy array, binary mask (0 for background, non-zero for foreground)
        
    Returns:
        Dictionary containing intensity statistics features
    """
    # Apply mask to get only lesion pixels
    pixels = apply_mask_to_channel(grayscale_img, mask)
    
    # Check if we have pixels to analyze
    if len(pixels) == 0:
        print("Warning: No pixels within mask for intensity calculations!")
        return {
            'intensity_mean': 0.0,
            'intensity_std': 0.0,
            'intensity_min': 0.0,
            'intensity_max': 0.0
        }
    
    # Calculate basic statistics
    mean_val = np.mean(pixels)
    std_val = np.std(pixels)
    min_val = np.min(pixels)
    max_val = np.max(pixels)
    
    return {
        'intensity_mean': float(mean_val),
        'intensity_std': float(std_val),
        'intensity_min': float(min_val),
        'intensity_max': float(max_val)
    }

def calculate_hsv_stats(
    hsv_img: np.ndarray, 
    mask: np.ndarray
) -> Dict[str, float]:
    """
    Calculate statistical features from each HSV channel within a mask.
    
    Args:
        hsv_img: 3D NumPy array, HSV image (H, S, V channels)
        mask: 2D NumPy array, binary mask (0 for background, non-zero for foreground)
        
    Returns:
        Dictionary containing HSV statistics features
    """
    if hsv_img.ndim != 3 or hsv_img.shape[2] != 3:
        raise ValueError("Expected HSV image with 3 channels")
    
    features = {}
    channel_names = ['hue', 'saturation', 'value']
    
    # Process each HSV channel
    for i, name in enumerate(channel_names):
        channel = hsv_img[:, :, i]
        pixels = apply_mask_to_channel(channel, mask)
        
        # Check if we have pixels to analyze
        if len(pixels) == 0:
            print(f"Warning: No pixels within mask for {name} calculations!")
            features[f'{name}_mean'] = 0.0
            features[f'{name}_std'] = 0.0
            continue
        
        # Calculate statistics
        mean_val = np.mean(pixels)
        std_val = np.std(pixels)
        
        # Special handling for hue (circular data)
        if i == 0:  # Hue channel
            # OpenCV: H is 0-179, S is 0-255, V is 0-255
            # Normalize hue to [0, 1] for circular statistics
            normalized_hue = pixels / 179.0 * 2 * np.pi
            
            # Calculate circular mean
            sin_sum = np.sum(np.sin(normalized_hue))
            cos_sum = np.sum(np.cos(normalized_hue))
            circular_mean = np.arctan2(sin_sum, cos_sum)
            
            # Convert back to OpenCV's range
            circular_mean = (circular_mean % (2 * np.pi)) / (2 * np.pi) * 179.0
            
            # Update the mean value with circular mean
            mean_val = circular_mean
        
        # Add to features dictionary
        features[f'{name}_mean'] = float(mean_val)
        features[f'{name}_std'] = float(std_val)
    
    return features

def calculate_all_intensity_features(
    gray_img: np.ndarray, 
    hsv_img: np.ndarray, 
    mask: np.ndarray
) -> Dict[str, float]:
    """
    Calculate all intensity-related features (grayscale and HSV) within a mask.
    
    Args:
        gray_img: 2D NumPy array, grayscale image
        hsv_img: 3D NumPy array, HSV image
        mask: 2D NumPy array, binary mask (0 for background, non-zero for foreground)
        
    Returns:
        Dictionary containing all intensity-related features
    """
    # Validate inputs
    if gray_img.ndim != 2:
        raise ValueError("gray_img must be a 2D array")
    
    if hsv_img.ndim != 3 or hsv_img.shape[2] != 3:
        raise ValueError("hsv_img must be a 3D array with 3 channels")
    
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")
    
    # Calculate all intensity features
    gray_features = calculate_intensity_stats(gray_img, mask)
    hsv_features = calculate_hsv_stats(hsv_img, mask)
    
    # Combine features
    all_features = {}
    all_features.update(gray_features)
    all_features.update(hsv_features)
    
    return all_features

def calculate_all_features(
    gray_img: np.ndarray, 
    hsv_img: np.ndarray, 
    mask: np.ndarray
) -> Dict[str, float]:
    """
    Calculate ALL features (intensity, border, texture) within a mask.
    
    This is the main feature extraction function that combines all feature types.
    
    Args:
        gray_img: 2D NumPy array, grayscale image
        hsv_img: 3D NumPy array, HSV image
        mask: 2D NumPy array, binary mask (0 for background, non-zero for foreground)
        
    Returns:
        Dictionary containing all available features
    """
    # Ensure inputs are correct format
    if gray_img.ndim != 2:
        raise ValueError("gray_img must be a 2D array")
    
    if hsv_img.ndim != 3 or hsv_img.shape[2] != 3:
        raise ValueError("hsv_img must be a 3D array with 3 channels")
    
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")
    
    # Calculate intensity features (grayscale and HSV)
    intensity_features = calculate_all_intensity_features(gray_img, hsv_img, mask)
    
    # Calculate border features (requires grayscale)
    try:
        border_features = calculate_gradient_features(gray_img, mask)
    except Exception as e:
        print(f"Warning: Error calculating border features: {e}")
        border_features = {}
    
    # Calculate texture features (requires grayscale)
    try:
        texture_features = calculate_texture_features(gray_img, mask)
    except Exception as e:
        print(f"Warning: Error calculating texture features: {e}")
        texture_features = {}
    
    # Combine all features
    all_features = {}
    all_features.update(intensity_features)
    all_features.update(border_features)
    all_features.update(texture_features)
    
    return all_features

def print_features(features: Dict[str, float], title: str = "Features"):
    """
    Pretty-print the extracted features.
    
    Args:
        features: Dictionary of feature name-value pairs
        title: Title for the feature section
    """
    print(f"\n{title}")
    print("=" * len(title))
    
    # Group features by type
    intensity_features = {k:v for k,v in features.items() if k.startswith('intensity_')}
    hsv_features = {k:v for k,v in features.items() if any(k.startswith(p) for p in ['hue_', 'saturation_', 'value_'])}
    border_features = {k:v for k,v in features.items() if k.startswith('border_')}
    texture_features = {k:v for k,v in features.items() if k.startswith('texture_')}
    
    # Print each group
    if intensity_features:
        print("\nIntensity Features (Grayscale):")
        for name, value in intensity_features.items():
            print(f"  {name.replace('intensity_', '')}: {value:.4f}")
    
    if hsv_features:
        print("\nColor Features (HSV):")
        
        # Further break down by channel
        hue_features = {k:v for k,v in hsv_features.items() if k.startswith('hue_')}
        saturation_features = {k:v for k,v in hsv_features.items() if k.startswith('saturation_')}
        value_features = {k:v for k,v in hsv_features.items() if k.startswith('value_')}
        
        if hue_features:
            print("  Hue:")
            for name, value in hue_features.items():
                print(f"    {name.replace('hue_', '')}: {value:.4f}")
                
        if saturation_features:
            print("  Saturation:")
            for name, value in saturation_features.items():
                print(f"    {name.replace('saturation_', '')}: {value:.4f}")
                
        if value_features:
            print("  Value:")
            for name, value in value_features.items():
                print(f"    {name.replace('value_', '')}: {value:.4f}")
    
    if border_features:
        print("\nBorder Features:")
        for name, value in border_features.items():
            print(f"  {name.replace('border_', '')}: {value:.4f}")
    
    if texture_features:
        print("\nTexture Features:")
        
        # Further break down texture features
        tophat_features = {k:v for k,v in texture_features.items() if k.startswith('texture_tophat_')}
        bottomhat_features = {k:v for k,v in texture_features.items() if k.startswith('texture_bottomhat_')}
        other_texture = {k:v for k,v in texture_features.items() if k.startswith('texture_') 
                       and not k.startswith('texture_tophat_') 
                       and not k.startswith('texture_bottomhat_')}
        
        if tophat_features:
            print("  Top Hat (Bright Details):")
            for name, value in tophat_features.items():
                print(f"    {name.replace('texture_tophat_', '')}: {value:.4f}")
        
        if bottomhat_features:
            print("  Bottom Hat (Dark Details):")
            for name, value in bottomhat_features.items():
                print(f"    {name.replace('texture_bottomhat_', '')}: {value:.4f}")
        
        if other_texture:
            print("  Other Texture Metrics:")
            for name, value in other_texture.items():
                print(f"    {name.replace('texture_', '')}: {value:.4f}")

# Example usage in case the script is run directly
if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    
    parser = argparse.ArgumentParser(description="Extract features from an image within a mask.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--mask", required=True, help="Path to the binary mask image.")
    args = parser.parse_args()
    
    # Load image and mask
    image = cv2.imread(args.image)
    if image is None:
        raise ValueError(f"Could not load image from {args.image}")
    
    # Convert to formats needed for feature extraction
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not load mask from {args.mask}")
    
    # Make mask binary if not already
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Calculate all features
    features = calculate_all_features(gray_img, hsv_img, mask)
    
    # Print the results
    print_features(features, title=f"All Features for {args.image}") 