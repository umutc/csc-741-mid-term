import cv2
import numpy as np
from typing import Dict, List, Union, Tuple, Optional

# Import our mask application utility from histogram_utils if available
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

def calculate_gradient_features(
    grayscale_img: np.ndarray, 
    mask: np.ndarray,
    kernel_size: int = 3
) -> Dict[str, float]:
    """
    Calculate border features based on morphological gradient within a mask.
    
    The morphological gradient is the difference between dilation and erosion,
    which highlights edges/borders in the image.
    
    Args:
        grayscale_img: 2D NumPy array, grayscale image
        mask: 2D NumPy array, binary mask (0 for background, non-zero for foreground)
        kernel_size: Size of the structuring element for morphological operations
        
    Returns:
        Dictionary containing gradient-based border features
    """
    # Create a structuring element
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, 
        (kernel_size, kernel_size)
    )
    
    # Apply morphological gradient (dilation - erosion)
    gradient = cv2.morphologyEx(
        grayscale_img, 
        cv2.MORPH_GRADIENT, 
        kernel
    )
    
    # Create a border mask (edge of the original mask)
    border_mask = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
    
    # Get gradient values within the border mask
    border_gradient = apply_mask_to_channel(gradient, border_mask)
    
    # Get gradient values within the entire mask for comparison
    full_gradient = apply_mask_to_channel(gradient, mask)
    
    # Check if we have pixels to analyze
    if len(border_gradient) == 0 or len(full_gradient) == 0:
        print("Warning: No pixels within mask for gradient calculations!")
        return {
            'border_gradient_mean': 0.0,
            'border_gradient_std': 0.0,
            'border_gradient_max': 0.0,
            'border_irregularity': 0.0
        }
    
    # Calculate statistics
    mean_gradient = np.mean(border_gradient)
    std_gradient = np.std(border_gradient)
    max_gradient = np.max(border_gradient)
    
    # Border irregularity: ratio of border pixels to mask area
    # Higher values indicate more irregular (complex) borders
    border_pixels = np.sum(border_mask > 0)
    mask_pixels = np.sum(mask > 0)
    border_irregularity = border_pixels / mask_pixels if mask_pixels > 0 else 0
    
    return {
        'border_gradient_mean': float(mean_gradient),
        'border_gradient_std': float(std_gradient),
        'border_gradient_max': float(max_gradient),
        'border_irregularity': float(border_irregularity)
    }

def calculate_texture_features(
    grayscale_img: np.ndarray, 
    mask: np.ndarray,
    kernel_size: int = 7
) -> Dict[str, float]:
    """
    Calculate texture features based on Top Hat and Bottom Hat transforms within a mask.
    
    Top Hat = Original - Opening (highlights bright details)
    Bottom Hat = Closing - Original (highlights dark details)
    
    Args:
        grayscale_img: 2D NumPy array, grayscale image
        mask: 2D NumPy array, binary mask (0 for background, non-zero for foreground)
        kernel_size: Size of the structuring element for morphological operations
        
    Returns:
        Dictionary containing texture features
    """
    # Create a structuring element (larger for texture analysis)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, 
        (kernel_size, kernel_size)
    )
    
    # Apply Top Hat transform (original - opening)
    # Highlights bright details smaller than the kernel
    tophat = cv2.morphologyEx(
        grayscale_img, 
        cv2.MORPH_TOPHAT, 
        kernel
    )
    
    # Apply Bottom Hat transform (closing - original)
    # Highlights dark details smaller than the kernel
    bottomhat = cv2.morphologyEx(
        grayscale_img, 
        cv2.MORPH_BLACKHAT,  # OpenCV calls bottom hat "blackhat"
        kernel
    )
    
    # Get values within the mask
    tophat_values = apply_mask_to_channel(tophat, mask)
    bottomhat_values = apply_mask_to_channel(bottomhat, mask)
    
    # Check if we have pixels to analyze
    if len(tophat_values) == 0 or len(bottomhat_values) == 0:
        print("Warning: No pixels within mask for texture calculations!")
        return {
            'texture_tophat_mean': 0.0,
            'texture_tophat_std': 0.0,
            'texture_bottomhat_mean': 0.0,
            'texture_bottomhat_std': 0.0,
            'texture_contrast_index': 0.0
        }
    
    # Calculate statistics
    tophat_mean = np.mean(tophat_values)
    tophat_std = np.std(tophat_values)
    bottomhat_mean = np.mean(bottomhat_values)
    bottomhat_std = np.std(bottomhat_values)
    
    # Texture contrast index: sum of top hat and bottom hat means
    # Higher values indicate more texture (both bright and dark details)
    texture_contrast = tophat_mean + bottomhat_mean
    
    return {
        'texture_tophat_mean': float(tophat_mean),
        'texture_tophat_std': float(tophat_std),
        'texture_bottomhat_mean': float(bottomhat_mean),
        'texture_bottomhat_std': float(bottomhat_std),
        'texture_contrast_index': float(texture_contrast)
    }

def calculate_all_border_texture_features(
    grayscale_img: np.ndarray, 
    mask: np.ndarray
) -> Dict[str, float]:
    """
    Calculate all border and texture features for an image within a mask.
    
    Args:
        grayscale_img: 2D NumPy array, grayscale image
        mask: 2D NumPy array, binary mask (0 for background, non-zero for foreground)
        
    Returns:
        Dictionary containing all calculated border and texture features
    """
    # Ensure inputs are correct format
    if grayscale_img.ndim != 2:
        raise ValueError("grayscale_img must be a 2D array")
    
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")
    
    # Calculate all features
    gradient_features = calculate_gradient_features(grayscale_img, mask)
    texture_features = calculate_texture_features(grayscale_img, mask)
    
    # Combine features
    all_features = {}
    all_features.update(gradient_features)
    all_features.update(texture_features)
    
    return all_features

def print_border_texture_features(features: Dict[str, float], title: str = "Border & Texture Features"):
    """
    Pretty-print the extracted border and texture features.
    
    Args:
        features: Dictionary of feature name-value pairs
        title: Title for the feature section
    """
    print(f"\n{title}")
    print("=" * len(title))
    
    # Group features by type
    border_features = {k:v for k,v in features.items() if k.startswith('border_')}
    tophat_features = {k:v for k,v in features.items() if k.startswith('texture_tophat_')}
    bottomhat_features = {k:v for k,v in features.items() if k.startswith('texture_bottomhat_')}
    other_texture = {k:v for k,v in features.items() if k.startswith('texture_') 
                    and not k.startswith('texture_tophat_') 
                    and not k.startswith('texture_bottomhat_')}
    
    # Print groups
    if border_features:
        print("\nBorder Features:")
        for name, value in border_features.items():
            print(f"  {name.replace('border_', '')}: {value:.4f}")
    
    if tophat_features or bottomhat_features or other_texture:
        print("\nTexture Features:")
        
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
    
    parser = argparse.ArgumentParser(description="Calculate border and texture features.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--mask", required=True, help="Path to the binary mask image.")
    args = parser.parse_args()
    
    # Load image and mask
    image = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image from {args.image}")
    
    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not load mask from {args.mask}")
    
    # Make mask binary if not already
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Calculate features
    features = calculate_all_border_texture_features(image, mask)
    
    # Print the results
    print_border_texture_features(features, title=f"Border & Texture Features for {args.image}")
    
    # Visualize the gradient and texture transforms
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    
    gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel_small)
    tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel_large)
    bottomhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel_large)
    
    # Apply mask
    masked_gradient = np.where(mask > 0, gradient, 0)
    masked_tophat = np.where(mask > 0, tophat, 0)
    masked_bottomhat = np.where(mask > 0, bottomhat, 0)
    
    # Create visualization
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    
    axs[0, 0].imshow(image, cmap='gray')
    axs[0, 0].set_title('Original Grayscale')
    axs[0, 0].axis('off')
    
    axs[0, 1].imshow(mask, cmap='gray')
    axs[0, 1].set_title('Binary Mask')
    axs[0, 1].axis('off')
    
    axs[0, 2].imshow(gradient, cmap='inferno')
    axs[0, 2].set_title('Morphological Gradient')
    axs[0, 2].axis('off')
    
    axs[1, 0].imshow(masked_gradient, cmap='inferno')
    axs[1, 0].set_title('Masked Gradient (Border)')
    axs[1, 0].axis('off')
    
    axs[1, 1].imshow(masked_tophat, cmap='viridis')
    axs[1, 1].set_title('Masked Top Hat (Bright Details)')
    axs[1, 1].axis('off')
    
    axs[1, 2].imshow(masked_bottomhat, cmap='plasma')
    axs[1, 2].set_title('Masked Bottom Hat (Dark Details)')
    axs[1, 2].axis('off')
    
    plt.tight_layout()
    plt.suptitle(f"Border & Texture Visualization for {args.image}", fontsize=16)
    plt.subplots_adjust(top=0.9)
    plt.show() 