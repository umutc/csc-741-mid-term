import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.color_utils import rgb_to_hsv, calculate_ida_channel
from src.preprocessing import remove_hair
from src.segmentation import apply_threshold
import os

def create_advanced_features_visual(image_path, output_dir='img/advanced'):
    """Create a visualization of the advanced features for presentation."""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load and preprocess the image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_name = os.path.basename(image_path).split('.')[0]
    
    # 2. Process the image
    gray_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    hsv_img = rgb_to_hsv(img_rgb)
    
    # 3. Remove hair
    gray_hairless = remove_hair(gray_img)
    
    # 4. Create mask
    mask, _ = apply_threshold(gray_hairless, method='otsu', cleanup=True)
    
    # 5. Extract HSV channels and Ida channel
    h_channel = hsv_img[:, :, 0]
    s_channel = hsv_img[:, :, 1]
    v_channel = hsv_img[:, :, 2]
    ida_channel = calculate_ida_channel(img_rgb)
    
    # Create masked versions
    masked_ida = cv2.bitwise_and(ida_channel, ida_channel, mask=mask)
    masked_s = cv2.bitwise_and(s_channel, s_channel, mask=mask)
    masked_v = cv2.bitwise_and(v_channel, v_channel, mask=mask)
    
    # 6. Visualization for the presentation
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # First row - original image, mask and Ida channel
    axs[0, 0].imshow(img_rgb)
    axs[0, 0].set_title('Original Image', fontsize=14)
    axs[0, 0].axis('off')
    
    axs[0, 1].imshow(mask, cmap='gray')
    axs[0, 1].set_title('Segmentation Mask', fontsize=14)
    axs[0, 1].axis('off')
    
    axs[0, 2].imshow(ida_channel, cmap='viridis')
    axs[0, 2].set_title('Ida (Darkness) Channel', fontsize=14)
    axs[0, 2].axis('off')
    
    # Second row - Value, Saturation channels and advanced features histogram
    axs[1, 0].imshow(v_channel, cmap='gray')
    axs[1, 0].set_title('Value (V) Channel (f1-f9)', fontsize=14)
    axs[1, 0].axis('off')
    
    axs[1, 1].imshow(s_channel, cmap='plasma')
    axs[1, 1].set_title('Saturation (S) Channel (f10-f18)', fontsize=14)
    axs[1, 1].axis('off')
    
    # Create a histogram visualization for the third panel
    axs[1, 2].axis('off')
    # Create nested plots for the histograms
    hist_ax = fig.add_axes([0.68, 0.11, 0.25, 0.25])
    
    # Calculate histograms
    v_hist = cv2.calcHist([v_channel], [0], mask, [256], [0, 256])
    s_hist = cv2.calcHist([s_channel], [0], mask, [256], [0, 256])
    ida_hist = cv2.calcHist([ida_channel], [0], mask, [256], [0, 256])
    
    # Normalize for better visualization
    v_hist = v_hist / v_hist.max()
    s_hist = s_hist / s_hist.max() 
    ida_hist = ida_hist / ida_hist.max()
    
    # Plot histograms
    hist_ax.plot(v_hist, 'r', label='Value (f1-f9)')
    hist_ax.plot(s_hist, 'g', label='Saturation (f10-f18)')
    hist_ax.plot(ida_hist, 'b', label='Ida (f19-f28)')
    hist_ax.set_title('Channel Histograms', fontsize=12)
    hist_ax.legend(loc='upper right')
    hist_ax.set_xlim([0, 256])
    
    # Add text annotations for features
    fig.text(0.68, 0.42, "Advanced Features (f1-f28)", fontsize=14, weight='bold')
    
    # Brief explanation of features
    feature_text = """
    Brightness Features (f1-f9):
    - Statistical moments: mean, std, skewness, kurtosis
    - Entropy, histogram differences, range ratios
    
    Saturation Features (f10-f18):
    - Same metrics applied to Saturation channel
    
    Darkness Features (f19-f28):
    - Same metrics applied to Ida channel
    - Plus: bbox coverage ratio (f28)
    """
    
    fig.text(0.68, 0.38, feature_text, fontsize=12, va='top')
    
    # Add main title
    plt.suptitle(f'Advanced Features Visualization (f1-f28) - {img_name}', fontsize=16)
    
    # Save the figure
    output_path = os.path.join(output_dir, f"{img_name}_advanced_features.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"Advanced features visualization saved to: {output_path}")
    
    plt.close()
    
    return output_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create advanced features visualization for presentation")
    parser.add_argument("--image_path", required=True, help="Path to the input image file")
    parser.add_argument("--output_dir", default="img/advanced", help="Directory to save visualization outputs")
    
    args = parser.parse_args()
    create_advanced_features_visual(args.image_path, args.output_dir)