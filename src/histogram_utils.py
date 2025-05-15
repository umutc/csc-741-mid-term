import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Union, Optional
from scipy import stats

def apply_mask_to_channel(
    channel: np.ndarray,
    mask: np.ndarray,
    outside_value: Optional[int] = None
) -> np.ndarray:
    """
    Apply a binary mask to an image channel, returning only the pixels 
    within the masked region.

    Args:
        channel: 2D NumPy array, the image channel (grayscale, H, S, or V).
        mask: 2D NumPy array, binary mask (0 for background, non-zero for foreground).
        outside_value: Optional value to set pixels outside the mask.
                      If None, returns a flattened array of only masked pixels.

    Returns:
        A masked version of the channel, either as a 2D image with outside_value
        where mask is 0, or as a 1D array containing only pixels inside the mask.
    """
    # Input validation
    if channel.ndim not in [2, 3]:
        raise ValueError("Channel must be a 2D or 3D array")
    if mask.ndim != 2:
        raise ValueError("Mask must be a 2D array")
    
    # If mask is not binary, convert it
    binary_mask = mask > 0
    
    # Case 1: Return a flattened array of only masked pixels
    if outside_value is None:
        if channel.ndim == 3:  # Handle case where channel is a multi-channel image
            result = []
            for i in range(channel.shape[2]):
                # Extract pixels where mask is True for this channel
                result.append(channel[:, :, i][binary_mask])
            return result
        else:  # 2D channel
            return channel[binary_mask]
    
    # Case 2: Return a 2D image with outside_value where mask is 0
    result = np.full_like(channel, outside_value)
    if channel.ndim == 3:  # Handle case where channel is a multi-channel image
        for i in range(channel.shape[2]):
            result[:, :, i] = np.where(binary_mask, channel[:, :, i], outside_value)
    else:  # 2D channel
        result = np.where(binary_mask, channel, outside_value)
        
    return result

def calc_histogram_grayscale(
    gray_img: np.ndarray,
    mask: np.ndarray,
    bins: int = 256,
    range_values: Tuple[int, int] = (0, 256)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate a histogram of a grayscale image within a mask.

    Args:
        gray_img: 2D NumPy array, grayscale image.
        mask: 2D NumPy array, binary mask (0 for background, non-zero for foreground).
        bins: Number of bins for the histogram.
        range_values: Tuple (min, max) of the range of values to consider.

    Returns:
        Tuple (hist, bin_edges) where hist is the histogram counts and 
        bin_edges are the bin boundaries.
    """
    # Get pixels inside the mask
    masked_pixels = apply_mask_to_channel(gray_img, mask)
    
    # Check if we have pixels to analyze
    if len(masked_pixels) == 0:
        print("Warning: No pixels within mask!")
        return np.zeros(bins), np.linspace(range_values[0], range_values[1], bins+1)
    
    # Calculate histogram
    hist, bin_edges = np.histogram(
        masked_pixels, 
        bins=bins, 
        range=range_values
    )
    
    return hist, bin_edges

def calc_histogram_hsv(
    hsv_img: np.ndarray,
    mask: np.ndarray,
    channel: str = 'h',
    bins: int = 180 if 'h' else 256,
    range_values: Optional[Tuple[int, int]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate a histogram of a selected channel from an HSV image within a mask.

    Args:
        hsv_img: 3D NumPy array, HSV image.
        mask: 2D NumPy array, binary mask (0 for background, non-zero for foreground).
        channel: Which HSV channel to use: 'h', 's', or 'v'.
        bins: Number of bins for the histogram.
        range_values: Tuple (min, max) of the range of values to consider.
                    If None, uses (0, 180) for H and (0, 256) for S and V.

    Returns:
        Tuple (hist, bin_edges) where hist is the histogram counts and 
        bin_edges are the bin boundaries.
    """
    # Validate input
    if hsv_img.ndim != 3 or hsv_img.shape[2] < 3:
        raise ValueError("hsv_img must be a 3-channel HSV image")
    
    # Determine channel index and range
    channel = channel.lower()
    if channel == 'h':
        channel_idx = 0
        if range_values is None:
            range_values = (0, 180)  # Typical OpenCV Hue range
    elif channel == 's':
        channel_idx = 1
        if range_values is None:
            range_values = (0, 256)
    elif channel == 'v':
        channel_idx = 2
        if range_values is None:
            range_values = (0, 256)
    else:
        raise ValueError("channel must be 'h', 's', or 'v'")
    
    # Extract the specified channel
    channel_data = hsv_img[:, :, channel_idx]
    
    # Get pixels inside the mask
    masked_pixels = apply_mask_to_channel(channel_data, mask)
    
    # Check if we have pixels to analyze
    if len(masked_pixels) == 0:
        print(f"Warning: No pixels within mask for {channel} channel!")
        return np.zeros(bins), np.linspace(range_values[0], range_values[1], bins+1)
    
    # Calculate histogram
    hist, bin_edges = np.histogram(
        masked_pixels, 
        bins=bins, 
        range=range_values
    )
    
    return hist, bin_edges

def calc_all_histograms(
    image: np.ndarray,
    mask: np.ndarray,
    is_hsv: bool = False
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Calculate histograms for grayscale and all HSV channels within a mask.

    Args:
        image: A NumPy array, either grayscale (2D) or color (3D).
        mask: 2D NumPy array, binary mask (0 for background, non-zero for foreground).
        is_hsv: Whether the input image is already in HSV format.

    Returns:
        Dictionary of histograms for 'gray', 'h', 's', and 'v', each containing
        a tuple of (hist, bin_edges).
    """
    results = {}
    
    # Check image dimensions
    if image.ndim == 2:  # Grayscale
        gray_img = image
        # Convert to HSV (need BGR/RGB first)
        rgb_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)
    elif image.ndim == 3:  # Color
        if image.shape[2] == 1:  # Single channel
            gray_img = image[:, :, 0]
            # Convert to HSV (need BGR/RGB first)
            rgb_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
            hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)
        else:  # Multi-channel
            if is_hsv:
                hsv_img = image
                # Convert to grayscale from HSV
                bgr_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
                gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
            else:
                # Assume BGR and convert to grayscale and HSV
                gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    else:
        raise ValueError("Unsupported image format")
    
    # Calculate grayscale histogram
    results['gray'] = calc_histogram_grayscale(gray_img, mask)
    
    # Calculate HSV histograms
    results['h'] = calc_histogram_hsv(hsv_img, mask, channel='h')
    results['s'] = calc_histogram_hsv(hsv_img, mask, channel='s')
    results['v'] = calc_histogram_hsv(hsv_img, mask, channel='v')
    
    return results

def plot_histograms(
    histograms: Dict[str, Tuple[np.ndarray, np.ndarray]],
    figsize: Tuple[int, int] = (12, 8),
    title_prefix: str = "Masked Histogram -"
) -> None:
    """
    Plot histograms for visual inspection.

    Args:
        histograms: Dictionary of histograms, as returned by calc_all_histograms.
        figsize: Size of the figure.
        title_prefix: Prefix for the subplot titles.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Define colors for each channel type
    colors = {
        'gray': 'black',
        'h': 'blue',
        's': 'green',
        'v': 'red'
    }
    
    # Flatten axes for simpler indexing
    axes = axes.flatten()
    
    # Plot each histogram
    for i, (channel, (hist, bin_edges)) in enumerate(histograms.items()):
        if i < 4:  # Just in case we have more histograms than subplots
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            axes[i].bar(bin_centers, hist, width=(bin_edges[1] - bin_edges[0]), 
                     color=colors.get(channel, 'gray'), alpha=0.7)
            axes[i].set_title(f"{title_prefix} {channel.upper()}")
            axes[i].set_xlabel("Intensity Value")
            axes[i].set_ylabel("Pixel Count")
            axes[i].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Example usage in case the script is run directly
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate and visualize masked histograms.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--mask", required=True, help="Path to the binary mask image.")
    args = parser.parse_args()
    
    # Load image and mask
    image = cv2.imread(args.image)
    if image is None:
        raise ValueError(f"Could not load image from {args.image}")
    
    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not load mask from {args.mask}")
    
    # Calculate histograms
    histograms = calc_all_histograms(image, mask)
    
    # Plot the results
    plot_histograms(histograms, title_prefix=f"Masked Histogram ({args.image}) -") 


def calculate_histogram_stats(histogram: np.ndarray, bin_edges: np.ndarray) -> Dict[str, float]:
    """
    Calculate statistical metrics from a histogram.
    
    Args:
        histogram: 1D NumPy array of histogram counts
        bin_edges: 1D NumPy array of bin edges
        
    Returns:
        Dictionary with statistical metrics: mean, std_dev, skewness, kurtosis, entropy
    """
    # Normalize histogram to get probability distribution
    total_pixels = np.sum(histogram)
    if total_pixels == 0:
        return {
            'mean': 0.0,
            'std_dev': 0.0,
            'skewness': 0.0,
            'kurtosis': 0.0,
            'entropy': 0.0
        }
    
    normalized_hist = histogram / total_pixels
    
    # Calculate bin centers for weighted calculations
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Calculate mean (1st moment)
    mean = np.sum(bin_centers * normalized_hist)
    
    # Calculate variance (2nd moment)
    variance = np.sum(((bin_centers - mean) ** 2) * normalized_hist)
    std_dev = np.sqrt(variance) if variance > 0 else 0.0
    
    # Calculate skewness (3rd moment)
    if std_dev > 0:
        skewness = np.sum(((bin_centers - mean) / std_dev) ** 3 * normalized_hist)
    else:
        skewness = 0.0
    
    # Calculate kurtosis (4th moment)
    if std_dev > 0:
        kurtosis = np.sum(((bin_centers - mean) / std_dev) ** 4 * normalized_hist) - 3.0  # -3 for Fisher's definition
    else:
        kurtosis = 0.0
    
    # Calculate entropy
    # Add a small epsilon to avoid log(0)
    epsilon = 1e-10
    entropy = -np.sum(normalized_hist * np.log2(normalized_hist + epsilon))
    
    return {
        'mean': float(mean),
        'std_dev': float(std_dev),
        'skewness': float(skewness),
        'kurtosis': float(kurtosis),
        'entropy': float(entropy)
    }


def calculate_histogram_differences(histogram: np.ndarray) -> Dict[str, float]:
    """
    Calculate metrics based on differences between adjacent histogram bins.
    
    Args:
        histogram: 1D NumPy array of histogram counts
        
    Returns:
        Dictionary with difference metrics: avg_diff, sum_largest_10_diffs
    """
    # Calculate differences between adjacent bins
    diffs = np.abs(np.diff(histogram))
    
    # Calculate average difference
    avg_diff = np.mean(diffs) if len(diffs) > 0 else 0.0
    
    # Calculate sum of 10 largest differences
    # If fewer than 10 differences, use all of them
    k = min(10, len(diffs))
    largest_diffs = np.sort(diffs)[-k:]
    sum_largest_k_diffs = np.sum(largest_diffs)
    
    return {
        'avg_diff': float(avg_diff),
        'sum_largest_10_diffs': float(sum_largest_k_diffs)
    }


def calculate_histogram_range_ratios(histogram: np.ndarray, bin_edges: np.ndarray) -> Dict[str, float]:
    """
    Calculate ratio metrics between different ranges of the histogram.
    
    Args:
        histogram: 1D NumPy array of histogram counts
        bin_edges: 1D NumPy array of bin edges
        
    Returns:
        Dictionary with ratio metrics: ratio_70_99_to_40_69, ratio_20_39_to_0_19
    """
    # Determine bin indices for the percentage ranges
    # 0-19%, 20-39%, 40-69%, 70-99%
    n_bins = len(histogram)
    idx_0_19 = slice(0, int(n_bins * 0.2))
    idx_20_39 = slice(int(n_bins * 0.2), int(n_bins * 0.4))
    idx_40_69 = slice(int(n_bins * 0.4), int(n_bins * 0.7))
    idx_70_99 = slice(int(n_bins * 0.7), n_bins)
    
    # Calculate sums for each range
    sum_0_19 = np.sum(histogram[idx_0_19])
    sum_20_39 = np.sum(histogram[idx_20_39])
    sum_40_69 = np.sum(histogram[idx_40_69])
    sum_70_99 = np.sum(histogram[idx_70_99])
    
    # Calculate ratios, handling division by zero
    ratio_70_99_to_40_69 = (sum_70_99 / sum_40_69) if sum_40_69 > 0 else 0.0
    ratio_20_39_to_0_19 = (sum_20_39 / sum_0_19) if sum_0_19 > 0 else 0.0
    
    return {
        'ratio_70_99_to_40_69': float(ratio_70_99_to_40_69),
        'ratio_20_39_to_0_19': float(ratio_20_39_to_0_19)
    }