# Mid-Term Project: Missing Components Analysis

## Overview

This document outlines the components that were not fully implemented in the mid-term project for CSC 741 (Digital Image Processing). The project successfully implemented Phases 1 and 2 as defined in the PRD (Product Requirements Document), but Phase 3 ("Advanced Feature Implementation") remains incomplete, with some partial implementation of Phase 4 ("Evaluation and Reporting").

## 1. Phase 3: Advanced Feature Implementation (f1-f28)

### 1.1 Ida (Darkness) Channel

**Status: Not Implemented**

The Ida (Darkness) channel, calculated as `max(R,G,B) - min(R,G,B)` per pixel, was not implemented in the current codebase. This channel is essential for calculating the Darkness Features (f19-f28).

**Implementation Requirements:**
- Create a new function, e.g., `calculate_ida_channel(rgb_img)` in `color_utils.py`
- The function should:
  - Accept a 3-channel RGB image as input
  - For each pixel, compute `max(R,G,B) - min(R,G,B)`
  - Return a single-channel image representing the Ida values

**Sample Implementation:**
```python
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
```

### 1.2 Brightness Features (f1-f9)

**Status: Not Implemented**

These features are based on the Value (V) channel of the HSV color space and include statistical moments, entropy, and histogram ratio metrics.

**Features to Implement:**
- **f1**: Mean (μ) of the V channel histogram
- **f2**: Standard Deviation (σ) of the V channel histogram
- **f3**: Skewness of the V channel histogram
- **f4**: Kurtosis of the V channel histogram
- **f5**: Entropy of the V channel histogram
- **f6**: Average of histogram differences for adjacent bins
- **f7**: Sum of 10 largest adjacent histogram bin differences
- **f8**: Ratio of the histogram range 70-99% to 40-69%
- **f9**: Ratio of the histogram range 20-39% to 0-19%

**Implementation Requirements:**
- Create a new function, e.g., `calculate_brightness_features(v_channel, mask)` in `feature_extraction.py`
- Enhance histogram calculation to include additional metrics (skewness, kurtosis, entropy)
- Implement specific ratio calculations for features f8-f9

### 1.3 Saturation Features (f10-f18)

**Status: Not Implemented**

These features are based on the Saturation (S) channel of the HSV color space and include statistical moments, entropy, and histogram ratio metrics, similar to the Brightness Features but applied to the S channel.

**Features to Implement:**
- **f10**: Mean (μ) of the S channel histogram
- **f11**: Standard Deviation (σ) of the S channel histogram
- **f12**: Skewness of the S channel histogram
- **f13**: Kurtosis of the S channel histogram
- **f14**: Entropy of the S channel histogram
- **f15**: Average of histogram differences for adjacent bins
- **f16**: Sum of 10 largest adjacent histogram bin differences
- **f17**: Ratio of the histogram range 70-99% to 40-69%
- **f18**: Ratio of the histogram range 20-39% to 0-19%

**Implementation Requirements:**
- Create a new function, e.g., `calculate_saturation_features(s_channel, mask)` in `feature_extraction.py`
- Apply similar calculations as for Brightness Features but to the S channel

### 1.4 Darkness Features (f19-f28)

**Status: Not Implemented**

These features are based on the Ida (Darkness) channel calculated as `max(R,G,B) - min(R,G,B)` and include statistical moments, entropy, and histogram ratio metrics.

**Features to Implement:**
- **f19**: Mean (μ) of the Ida channel histogram
- **f20**: Standard Deviation (σ) of the Ida channel histogram
- **f21**: Skewness of the Ida channel histogram
- **f22**: Kurtosis of the Ida channel histogram
- **f23**: Entropy of the Ida channel histogram
- **f24**: Average of histogram differences for adjacent bins
- **f25**: Sum of 10 largest adjacent histogram bin differences
- **f26**: Ratio of the histogram range 70-99% to 40-69%
- **f27**: Ratio of the histogram range 20-39% to 0-19%
- **f28**: Coverage of the lesion within the bounding box

**Implementation Requirements:**
- Create a new function, e.g., `calculate_darkness_features(ida_channel, mask)` in `feature_extraction.py`
- Apply similar calculations as for Brightness and Saturation Features but to the Ida channel
- For f28, calculate the ratio of lesion pixels to the total pixels in the bounding box

## 2. Statistical Functions for Feature Calculation

### 2.1 Higher-Order Statistical Moments

**Status: Partially Implemented**

Current implementation includes mean and standard deviation, but not skewness and kurtosis required for features f3/f4, f12/f13, and f21/f22.

**Implementation Requirements:**
- Implement skewness and kurtosis calculations for histogram data
- Use scipy.stats or implement custom functions based on statistical formulas

**Sample Implementation:**
```python
from scipy import stats

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
```

### 2.2 Histogram Difference Metrics

**Status: Not Implemented**

Features f6/f7, f15/f16, and f24/f25 require calculations based on differences between adjacent histogram bins.

**Implementation Requirements:**
- Implement functions to calculate the average difference between adjacent histogram bins
- Implement functions to find and sum the K largest differences between adjacent histogram bins

**Sample Implementation:**
```python
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
```

### 2.3 Histogram Range Ratio Metrics

**Status: Not Implemented**

Features f8/f9, f17/f18, and f26/f27 require calculations based on ratios of histogram ranges.

**Implementation Requirements:**
- Implement functions to calculate ratios between different percentage ranges of the histogram

**Sample Implementation:**
```python
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
```

## 3. Integration into Main Pipeline

### 3.1 Module Updates

The following modules need to be updated to incorporate the new features:

1. **color_utils.py**: Add the Ida channel calculation
2. **histogram_utils.py**: Enhance histogram calculations for new features
3. **feature_extraction.py**: Add new feature calculation functions

### 3.2 Feature Extraction Updates

The main feature extraction function should be updated to include the new features:

```python
def calculate_all_features(gray_img, hsv_img, rgb_img, mask):
    """Updated to include f1-f28 features"""
    # Existing intensity features
    intensity_features = calculate_intensity_stats(gray_img, mask)
    hsv_features = calculate_hsv_stats(hsv_img, mask)
    
    # Calculate Ida channel
    ida_channel = calculate_ida_channel(rgb_img)
    
    # Calculate new feature sets (f1-f28)
    v_channel = hsv_img[:, :, 2]  # Value channel for Brightness features
    s_channel = hsv_img[:, :, 1]  # Saturation channel for Saturation features
    
    brightness_features = calculate_brightness_features(v_channel, mask)  # f1-f9
    saturation_features = calculate_saturation_features(s_channel, mask)  # f10-f18
    darkness_features = calculate_darkness_features(ida_channel, mask)    # f19-f28
    
    # Combine all features
    all_features = {}
    all_features.update(intensity_features)
    all_features.update(hsv_features)
    all_features.update(brightness_features)
    all_features.update(saturation_features)
    all_features.update(darkness_features)
    
    return all_features
```

### 3.3 Visualization Updates

Update visualization scripts to display the new features:

- Create a new visualization script, e.g., `advanced_features_display.py`, to show features f1-f28
- Update `all_features_display.py` to include the new features
- Add histogram visualization for the Ida channel

## 4. Phase 4: Evaluation and Analysis

### 4.1 Feature Distinctiveness Analysis

**Status: Not Implemented for f1-f28**

The evaluation of how well features f1-f28 distinguish between benign and malignant lesions was not completed.

**Implementation Requirements:**
- Create a new script, e.g., `analyze_features.py`, to:
  - Load extracted features from multiple images
  - Group by benign/malignant labels
  - Generate comparative visualizations (box plots, histograms)
  - Calculate statistical significance metrics (t-tests, ANOVA)
  - Report on feature distinctiveness

### 4.2 Batch Processing Update

**Status: Partially Implemented, Needs Update for f1-f28**

The batch processing script needs to be updated to include the extraction of features f1-f28.

**Implementation Requirements:**
- Update `batch_processor.py` to:
  - Calculate and save features f1-f28 for all processed images
  - Include the new features in the output CSV/pickle files

## 5. Implementation Priority

Recommended priority order for implementing the missing components:

1. Ida Channel Calculation (`calculate_ida_channel`)
2. Statistical functions for higher-order moments, histogram differences, and range ratios
3. Brightness Features (f1-f9)
4. Saturation Features (f10-f18)
5. Darkness Features (f19-f28)
6. Integration into main feature extraction pipeline
7. Visualization updates for new features
8. Batch processing updates
9. Feature distinctiveness analysis

## 6. Conclusion

The project demonstrates solid implementation of Phases 1 and 2 from the PRD, establishing a robust foundation for image processing, segmentation, and basic feature extraction. The missing components, primarily in Phase 3, involve more advanced statistical feature extraction specifically targeted at the characteristics most relevant to skin lesion analysis.

Implementing these missing components would complete the project as originally defined in the PRD and potentially enhance the analytical capabilities of the pipeline for distinguishing between benign and malignant lesions.