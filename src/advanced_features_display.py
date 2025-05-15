import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Dict, List, Tuple, Optional, Any
import os

# Import our custom modules
try:
    from src.color_utils import rgb_to_hsv, calculate_ida_channel
    from src.feature_extraction import (
        calculate_brightness_features, 
        calculate_saturation_features, 
        calculate_darkness_features,
        print_features
    )
    from src.histogram_utils import calc_histogram_grayscale, plot_histograms
except ImportError:
    from color_utils import rgb_to_hsv, calculate_ida_channel
    from feature_extraction import (
        calculate_brightness_features, 
        calculate_saturation_features, 
        calculate_darkness_features,
        print_features
    )
    from histogram_utils import calc_histogram_grayscale, plot_histograms

def plot_advanced_features(
    image: np.ndarray,
    mask: np.ndarray,
    title: str = "Advanced Features",
    figsize: Tuple[int, int] = (18, 12)
) -> None:
    """
    Plot and display advanced features (f1-f28) for a given image and mask.
    
    Args:
        image: RGB image as a NumPy array
        mask: Binary mask as a NumPy array
        title: Title for the figure
        figsize: Size of the figure (width, height)
    """
    # Convert to HSV and extract channels
    hsv_img = rgb_to_hsv(image)
    v_channel = hsv_img[:, :, 2]
    s_channel = hsv_img[:, :, 1]
    
    # Calculate Ida channel
    ida_channel = calculate_ida_channel(image)
    
    # Calculate features
    brightness_features = calculate_brightness_features(v_channel, mask)
    saturation_features = calculate_saturation_features(s_channel, mask)
    darkness_features = calculate_darkness_features(ida_channel, mask)
    
    # Create figure with subplots
    fig, axs = plt.subplots(3, 3, figsize=figsize)
    
    # First row: Original image, HSV Value channel, Brightness histogram
    axs[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title("Original Image")
    axs[0, 0].axis('off')
    
    axs[0, 1].imshow(v_channel, cmap='gray')
    axs[0, 1].set_title("Value (V) Channel")
    axs[0, 1].axis('off')
    
    # Calculate V channel histogram
    v_hist, v_bins = calc_histogram_grayscale(v_channel, mask)
    bin_centers = (v_bins[:-1] + v_bins[1:]) / 2
    axs[0, 2].bar(bin_centers, v_hist, width=(v_bins[1] - v_bins[0]), color='darkred', alpha=0.7)
    axs[0, 2].set_title("V Channel Histogram")
    axs[0, 2].set_xlabel("Value")
    axs[0, 2].set_ylabel("Pixel Count")
    
    # Second row: Mask, Saturation channel, Saturation histogram
    axs[1, 0].imshow(mask, cmap='gray')
    axs[1, 0].set_title("Segmentation Mask")
    axs[1, 0].axis('off')
    
    axs[1, 1].imshow(s_channel, cmap='gray')
    axs[1, 1].set_title("Saturation (S) Channel")
    axs[1, 1].axis('off')
    
    # Calculate S channel histogram
    s_hist, s_bins = calc_histogram_grayscale(s_channel, mask)
    bin_centers = (s_bins[:-1] + s_bins[1:]) / 2
    axs[1, 2].bar(bin_centers, s_hist, width=(s_bins[1] - s_bins[0]), color='darkgreen', alpha=0.7)
    axs[1, 2].set_title("S Channel Histogram")
    axs[1, 2].set_xlabel("Saturation")
    axs[1, 2].set_ylabel("Pixel Count")
    
    # Third row: Masked image, Ida channel, Ida histogram
    # Create masked image
    masked_image = image.copy()
    for c in range(3):
        masked_image[:, :, c] = np.where(mask > 0, image[:, :, c], 0)
    
    axs[2, 0].imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
    axs[2, 0].set_title("Masked Image")
    axs[2, 0].axis('off')
    
    axs[2, 1].imshow(ida_channel, cmap='gray')
    axs[2, 1].set_title("Ida (Darkness) Channel")
    axs[2, 1].axis('off')
    
    # Calculate Ida channel histogram
    ida_hist, ida_bins = calc_histogram_grayscale(ida_channel, mask)
    bin_centers = (ida_bins[:-1] + ida_bins[1:]) / 2
    axs[2, 2].bar(bin_centers, ida_hist, width=(ida_bins[1] - ida_bins[0]), color='darkblue', alpha=0.7)
    axs[2, 2].set_title("Ida Channel Histogram")
    axs[2, 2].set_xlabel("Darkness")
    axs[2, 2].set_ylabel("Pixel Count")
    
    # Set overall title
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    # Add a text box with feature values
    all_features = {}
    all_features.update(brightness_features)
    all_features.update(saturation_features)
    all_features.update(darkness_features)
    
    # Format text for the feature summary
    feature_text = "ADVANCED FEATURES (f1-f28):\n\n"
    
    # Brightness features (f1-f9)
    feature_text += "Brightness Features (f1-f9):\n"
    for f in range(1, 10):
        key = f"f{f}_brightness_"
        for k in brightness_features.keys():
            if k.startswith(key):
                feature_text += f"{k}: {brightness_features[k]:.4f}\n"
    
    feature_text += "\nSaturation Features (f10-f18):\n"
    for f in range(10, 19):
        key = f"f{f}_saturation_"
        for k in saturation_features.keys():
            if k.startswith(key):
                feature_text += f"{k}: {saturation_features[k]:.4f}\n"
    
    feature_text += "\nDarkness Features (f19-f28):\n"
    for f in range(19, 29):
        key = f"f{f}_darkness_"
        for k in darkness_features.keys():
            if k.startswith(key):
                feature_text += f"{k}: {darkness_features[k]:.4f}\n"
    
    # Add a figure text outside the subplots
    plt.figtext(1.02, 0.5, feature_text, fontsize=10, 
               bbox=dict(facecolor='white', alpha=0.8),
               verticalalignment='center')
    
    plt.subplots_adjust(right=0.75)  # Make room for the text box
    plt.show()

    # Print features to console as well
    print("\nAdvanced Feature Summary:")
    print("------------------------")
    print_features(all_features, title=f"Advanced Features for {title}")
    
    return all_features

def main():
    """
    Main function to demonstrate advanced feature display.
    """
    parser = argparse.ArgumentParser(description="Display advanced features for an image with a mask.")
    parser.add_argument("--image_path", required=True, help="Path to the input image.")
    parser.add_argument("--mask_path", required=True, help="Path to the binary mask image.")
    parser.add_argument("--output_dir", help="Directory to save output visualizations.", default=None)
    args = parser.parse_args()
    
    # Read the image and mask
    image = cv2.imread(args.image_path)
    if image is None:
        raise ValueError(f"Could not read image from {args.image_path}")
    
    mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask from {args.mask_path}")
    
    # Convert to binary mask if needed
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Get image file name for the title
    image_name = os.path.basename(args.image_path)
    
    # Display advanced features
    features = plot_advanced_features(
        image, 
        mask, 
        title=f"Advanced Features - {image_name}"
    )
    
    # Save visualization if output directory is provided
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save histogram plots
        output_path = os.path.join(args.output_dir, f"{os.path.splitext(image_name)[0]}_advanced_features.png")
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
        
        # Save features to CSV
        features_path = os.path.join(args.output_dir, f"{os.path.splitext(image_name)[0]}_advanced_features.csv")
        with open(features_path, 'w') as f:
            f.write("feature,value\n")
            for name, value in features.items():
                f.write(f"{name},{value}\n")
        print(f"Saved features to {features_path}")

if __name__ == "__main__":
    main()