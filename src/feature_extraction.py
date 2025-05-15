import cv2
import numpy as np
from typing import Dict, List, Union, Tuple, Optional, Any

# Import our mask application utility and histogram statistics from histogram_utils
try:
    from src.histogram_utils import (
        apply_mask_to_channel,
        calculate_histogram_stats,
        calculate_histogram_differences,
        calculate_histogram_range_ratios
    )
except ImportError:
    # Fallback implementation if needed
    def apply_mask_to_channel(channel, mask, outside_value=None):
        """Fallback implementation of apply_mask_to_channel if import fails"""
        binary_mask = mask > 0
        if outside_value is None:
            return channel[binary_mask]
        return np.where(binary_mask, channel, outside_value)
    
    # Fallback implementations of histogram statistics functions
    def calculate_histogram_stats(histogram, bin_edges):
        """Fallback for histogram stats if import fails"""
        print("Warning: histogram_utils not found. Advanced histogram stats unavailable.")
        return {}
    
    def calculate_histogram_differences(histogram):
        """Fallback for histogram differences if import fails"""
        print("Warning: histogram_utils not found. Histogram difference metrics unavailable.")
        return {}
    
    def calculate_histogram_range_ratios(histogram, bin_edges):
        """Fallback for histogram range ratios if import fails"""
        print("Warning: histogram_utils not found. Histogram range ratio metrics unavailable.")
        return {}

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
    rgb_img: np.ndarray,
    mask: np.ndarray
) -> Dict[str, float]:
    """
    Calculate ALL features (intensity, border, texture, and advanced features f1-f28) within a mask.
    
    This is the main feature extraction function that combines all feature types.
    
    Args:
        gray_img: 2D NumPy array, grayscale image
        hsv_img: 3D NumPy array, HSV image
        rgb_img: 3D NumPy array, RGB image (needed for Ida channel)
        mask: 2D NumPy array, binary mask (0 for background, non-zero for foreground)
        
    Returns:
        Dictionary containing all available features
    """
    # Ensure inputs are correct format
    if gray_img.ndim != 2:
        raise ValueError("gray_img must be a 2D array")
    
    if hsv_img.ndim != 3 or hsv_img.shape[2] != 3:
        raise ValueError("hsv_img must be a 3D array with 3 channels")
    
    if rgb_img.ndim != 3 or rgb_img.shape[2] != 3:
        raise ValueError("rgb_img must be a 3D array with 3 channels")
    
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
    
    # Calculate new advanced features (f1-f28)
    try:
        # Extract channels needed for feature calculation
        v_channel = hsv_img[:, :, 2]  # Value channel for Brightness features
        s_channel = hsv_img[:, :, 1]  # Saturation channel for Saturation features
        
        # Calculate Ida channel from RGB image
        from src.color_utils import calculate_ida_channel
        ida_channel = calculate_ida_channel(rgb_img)
        
        # Calculate feature sets
        brightness_features = calculate_brightness_features(v_channel, mask)  # f1-f9
        saturation_features = calculate_saturation_features(s_channel, mask)  # f10-f18
        darkness_features = calculate_darkness_features(ida_channel, mask)    # f19-f28
    except Exception as e:
        print(f"Warning: Error calculating advanced features: {e}")
        brightness_features = {}
        saturation_features = {}
        darkness_features = {}
    
    # Combine all features
    all_features = {}
    all_features.update(intensity_features)
    all_features.update(border_features)
    all_features.update(texture_features)
    all_features.update(brightness_features)
    all_features.update(saturation_features)
    all_features.update(darkness_features)
    
    return all_features


def calculate_brightness_features(v_channel: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """
    Calculate Brightness Features (f1-f9) based on the Value (V) channel.
    
    Args:
        v_channel: 2D NumPy array, V channel from HSV image
        mask: 2D NumPy array, binary mask (0 for background, non-zero for foreground)
        
    Returns:
        Dictionary containing brightness features (f1-f9)
    """
    # Apply mask and calculate histogram
    masked_pixels = apply_mask_to_channel(v_channel, mask)
    
    # Check if we have pixels to analyze
    if len(masked_pixels) == 0:
        print("Warning: No pixels within mask for brightness features!")
        return {
            'f1_brightness_mean': 0.0,
            'f2_brightness_std': 0.0, 
            'f3_brightness_skewness': 0.0,
            'f4_brightness_kurtosis': 0.0,
            'f5_brightness_entropy': 0.0,
            'f6_brightness_avg_diff': 0.0,
            'f7_brightness_sum_10_largest_diffs': 0.0,
            'f8_brightness_ratio_70_99_to_40_69': 0.0,
            'f9_brightness_ratio_20_39_to_0_19': 0.0
        }
    
    # Calculate histogram
    hist, bin_edges = np.histogram(masked_pixels, bins=256, range=(0, 256))
    
    # Calculate statistical metrics (f1-f5)
    stats = calculate_histogram_stats(hist, bin_edges)
    
    # Calculate histogram difference metrics (f6-f7)
    diff_metrics = calculate_histogram_differences(hist)
    
    # Calculate histogram range ratio metrics (f8-f9)
    ratio_metrics = calculate_histogram_range_ratios(hist, bin_edges)
    
    # Return formatted features
    return {
        'f1_brightness_mean': stats['mean'],
        'f2_brightness_std': stats['std_dev'],
        'f3_brightness_skewness': stats['skewness'],
        'f4_brightness_kurtosis': stats['kurtosis'],
        'f5_brightness_entropy': stats['entropy'],
        'f6_brightness_avg_diff': diff_metrics['avg_diff'],
        'f7_brightness_sum_10_largest_diffs': diff_metrics['sum_largest_10_diffs'],
        'f8_brightness_ratio_70_99_to_40_69': ratio_metrics['ratio_70_99_to_40_69'],
        'f9_brightness_ratio_20_39_to_0_19': ratio_metrics['ratio_20_39_to_0_19']
    }


def calculate_saturation_features(s_channel: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """
    Calculate Saturation Features (f10-f18) based on the Saturation (S) channel.
    
    Args:
        s_channel: 2D NumPy array, S channel from HSV image
        mask: 2D NumPy array, binary mask (0 for background, non-zero for foreground)
        
    Returns:
        Dictionary containing saturation features (f10-f18)
    """
    # Apply mask and calculate histogram
    masked_pixels = apply_mask_to_channel(s_channel, mask)
    
    # Check if we have pixels to analyze
    if len(masked_pixels) == 0:
        print("Warning: No pixels within mask for saturation features!")
        return {
            'f10_saturation_mean': 0.0,
            'f11_saturation_std': 0.0, 
            'f12_saturation_skewness': 0.0,
            'f13_saturation_kurtosis': 0.0,
            'f14_saturation_entropy': 0.0,
            'f15_saturation_avg_diff': 0.0,
            'f16_saturation_sum_10_largest_diffs': 0.0,
            'f17_saturation_ratio_70_99_to_40_69': 0.0,
            'f18_saturation_ratio_20_39_to_0_19': 0.0
        }
    
    # Calculate histogram
    hist, bin_edges = np.histogram(masked_pixels, bins=256, range=(0, 256))
    
    # Calculate statistical metrics (f10-f14)
    stats = calculate_histogram_stats(hist, bin_edges)
    
    # Calculate histogram difference metrics (f15-f16)
    diff_metrics = calculate_histogram_differences(hist)
    
    # Calculate histogram range ratio metrics (f17-f18)
    ratio_metrics = calculate_histogram_range_ratios(hist, bin_edges)
    
    # Return formatted features
    return {
        'f10_saturation_mean': stats['mean'],
        'f11_saturation_std': stats['std_dev'],
        'f12_saturation_skewness': stats['skewness'],
        'f13_saturation_kurtosis': stats['kurtosis'],
        'f14_saturation_entropy': stats['entropy'],
        'f15_saturation_avg_diff': diff_metrics['avg_diff'],
        'f16_saturation_sum_10_largest_diffs': diff_metrics['sum_largest_10_diffs'],
        'f17_saturation_ratio_70_99_to_40_69': ratio_metrics['ratio_70_99_to_40_69'],
        'f18_saturation_ratio_20_39_to_0_19': ratio_metrics['ratio_20_39_to_0_19']
    }


def calculate_darkness_features(ida_channel: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """
    Calculate Darkness Features (f19-f28) based on the Ida (Darkness) channel.
    
    Args:
        ida_channel: 2D NumPy array, Ida channel (max(R,G,B) - min(R,G,B))
        mask: 2D NumPy array, binary mask (0 for background, non-zero for foreground)
        
    Returns:
        Dictionary containing darkness features (f19-f28)
    """
    # Apply mask and calculate histogram
    masked_pixels = apply_mask_to_channel(ida_channel, mask)
    
    # Check if we have pixels to analyze
    if len(masked_pixels) == 0:
        print("Warning: No pixels within mask for darkness features!")
        return {
            'f19_darkness_mean': 0.0,
            'f20_darkness_std': 0.0, 
            'f21_darkness_skewness': 0.0,
            'f22_darkness_kurtosis': 0.0,
            'f23_darkness_entropy': 0.0,
            'f24_darkness_avg_diff': 0.0,
            'f25_darkness_sum_10_largest_diffs': 0.0,
            'f26_darkness_ratio_70_99_to_40_69': 0.0,
            'f27_darkness_ratio_20_39_to_0_19': 0.0,
            'f28_darkness_bbox_coverage': 0.0
        }
    
    # Calculate histogram
    hist, bin_edges = np.histogram(masked_pixels, bins=256, range=(0, 256))
    
    # Calculate statistical metrics (f19-f23)
    stats = calculate_histogram_stats(hist, bin_edges)
    
    # Calculate histogram difference metrics (f24-f25)
    diff_metrics = calculate_histogram_differences(hist)
    
    # Calculate histogram range ratio metrics (f26-f27)
    ratio_metrics = calculate_histogram_range_ratios(hist, bin_edges)
    
    # Calculate f28: Coverage of the lesion within the bounding box
    # Find bounding box coordinates
    y_indices, x_indices = np.where(mask > 0)
    
    if len(y_indices) > 0 and len(x_indices) > 0:
        # Calculate bounding box
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        
        # Calculate area of bbox and lesion
        bbox_area = (y_max - y_min + 1) * (x_max - x_min + 1)
        lesion_area = np.sum(mask > 0)
        
        # Calculate coverage ratio
        coverage = lesion_area / bbox_area if bbox_area > 0 else 0.0
    else:
        coverage = 0.0
    
    # Return formatted features
    return {
        'f19_darkness_mean': stats['mean'],
        'f20_darkness_std': stats['std_dev'],
        'f21_darkness_skewness': stats['skewness'],
        'f22_darkness_kurtosis': stats['kurtosis'],
        'f23_darkness_entropy': stats['entropy'],
        'f24_darkness_avg_diff': diff_metrics['avg_diff'],
        'f25_darkness_sum_10_largest_diffs': diff_metrics['sum_largest_10_diffs'],
        'f26_darkness_ratio_70_99_to_40_69': ratio_metrics['ratio_70_99_to_40_69'],
        'f27_darkness_ratio_20_39_to_0_19': ratio_metrics['ratio_20_39_to_0_19'],
        'f28_darkness_bbox_coverage': float(coverage)
    }


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
    
    # Advanced features (f1-f28)
    brightness_features = {k:v for k,v in features.items() if k.startswith('f') and int(k.split('_')[0][1:]) <= 9}
    saturation_features_adv = {k:v for k,v in features.items() if k.startswith('f') and 10 <= int(k.split('_')[0][1:]) <= 18}
    darkness_features = {k:v for k,v in features.items() if k.startswith('f') and 19 <= int(k.split('_')[0][1:]) <= 28}
    
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
    
    # Print advanced features (f1-f28)
    if brightness_features:
        print("\nBrightness Features (f1-f9):")
        for name, value in sorted(brightness_features.items(), key=lambda x: int(x[0].split('_')[0][1:])):
            print(f"  {name}: {value:.4f}")
    
    if saturation_features_adv:
        print("\nSaturation Features (f10-f18):")
        for name, value in sorted(saturation_features_adv.items(), key=lambda x: int(x[0].split('_')[0][1:])):
            print(f"  {name}: {value:.4f}")
    
    if darkness_features:
        print("\nDarkness Features (f19-f28):")
        for name, value in sorted(darkness_features.items(), key=lambda x: int(x[0].split('_')[0][1:])):
            print(f"  {name}: {value:.4f}")

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
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    
    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not load mask from {args.mask}")
    
    # Make mask binary if not already
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Calculate all features
    features = calculate_all_features(gray_img, hsv_img, rgb_img, mask)
    
    # Print the results
    print_features(features, title=f"All Features for {args.image}") 