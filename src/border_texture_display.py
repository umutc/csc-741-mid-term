import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

from src.color_utils import rgb_to_grayscale
from src.preprocessing import remove_hair
from src.segmentation import apply_threshold
from src.border_texture_utils import calculate_all_border_texture_features, print_border_texture_features

def display_border_texture_features(image_path: str, output_dir=None):
    """
    Process an image through the pipeline, extract border and texture features,
    and display the results with appropriate visualizations.
    
    Args:
        image_path: Path to the input image.
        output_dir: Optional directory to save visualization outputs.
    """
    try:
        # Check if file exists
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # 1. Load the image (OpenCV loads as BGR by default)
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert to RGB for display & consistency
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # 2. Convert to grayscale
        try:
            gray_img = rgb_to_grayscale(img_rgb.copy())
        except Exception as e:
            print(f"Warning: Error in RGB color conversion: {e}")
            print("Falling back to OpenCV's direct conversion.")
            gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 3. Hair Removal (preprocessing step)
        try:
            gray_hairless = remove_hair(gray_img.copy())
            # Check if the result is a tuple and extract first element if needed
            if isinstance(gray_hairless, tuple):
                print("Note: hair removal returned a tuple, extracting first element.")
                gray_hairless = gray_hairless[0] if len(gray_hairless) > 0 else gray_img.copy()
        except Exception as e:
            print(f"Warning: Error in hair removal: {e}")
            print("Using original grayscale image.")
            gray_hairless = gray_img.copy()

        # 4. Segmentation - to get the mask
        try:
            # Ensure we're passing a valid NumPy array
            if isinstance(gray_hairless, tuple):
                gray_hairless_input = gray_hairless[0] if len(gray_hairless) > 0 else gray_img.copy()
            else:
                gray_hairless_input = gray_hairless.copy()
                
            # Using Otsu's method with cleanup by default
            mask, otsu_thresh = apply_threshold(
                gray_hairless_input,
                method='otsu',
                cleanup=True
            )
        except Exception as e:
            print(f"Warning: Error in thresholding: {e}")
            print("Using direct Otsu thresholding without cleanup.")
            try:
                # Try direct OpenCV Otsu as fallback
                _, mask = cv2.threshold(
                    gray_hairless_input, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
            except Exception as e2:
                print(f"Error in fallback thresholding: {e2}")
                # Last resort - basic threshold at mean
                thresh = int(np.mean(gray_hairless_input))
                _, mask = cv2.threshold(gray_hairless_input, thresh, 255, cv2.THRESH_BINARY)

        # 5. Extract border and texture features
        try:
            # Calculate border and texture features
            features = calculate_all_border_texture_features(gray_hairless, mask)
            
            # Print the features
            print_border_texture_features(features, title=f"Border & Texture Features for {os.path.basename(image_path)}")
        except Exception as e:
            print(f"Error calculating border and texture features: {e}")
            raise

        # 6. Prepare visualization
        # Create morphological transforms for visualization
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        # Border (gradient) visualization
        gradient = cv2.morphologyEx(gray_hairless, cv2.MORPH_GRADIENT, kernel_small)
        border_mask = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel_small)
        
        # Texture visualization
        tophat = cv2.morphologyEx(gray_hairless, cv2.MORPH_TOPHAT, kernel_large)
        bottomhat = cv2.morphologyEx(gray_hairless, cv2.MORPH_BLACKHAT, kernel_large)
        
        # Create masked versions
        masked_gradient = np.where(mask > 0, gradient, 0)
        masked_tophat = np.where(mask > 0, tophat, 0)
        masked_bottomhat = np.where(mask > 0, bottomhat, 0)
        
        # 7. Display Results - Main visualization
        fig1, ax1 = plt.subplots(2, 3, figsize=(15, 10))
        
        ax1[0, 0].imshow(img_rgb)
        ax1[0, 0].set_title('Original RGB Image')
        ax1[0, 0].axis('off')
        
        ax1[0, 1].imshow(gray_hairless, cmap='gray')
        ax1[0, 1].set_title('Preprocessed (Hair Removed)')
        ax1[0, 1].axis('off')
        
        ax1[0, 2].imshow(mask, cmap='gray')
        ax1[0, 2].set_title('Binary Mask')
        ax1[0, 2].axis('off')
        
        ax1[1, 0].imshow(gradient, cmap='inferno')
        ax1[1, 0].set_title('Morphological Gradient')
        ax1[1, 0].axis('off')
        
        ax1[1, 1].imshow(tophat, cmap='viridis')
        ax1[1, 1].set_title('Top Hat (Bright Details)')
        ax1[1, 1].axis('off')
        
        ax1[1, 2].imshow(bottomhat, cmap='plasma')
        ax1[1, 2].set_title('Bottom Hat (Dark Details)')
        ax1[1, 2].axis('off')
        
        plt.tight_layout()
        plt.suptitle(f"Border & Texture Analysis for: {os.path.basename(image_path)}", fontsize=16)
        plt.subplots_adjust(top=0.9)
        
        # 8. Display Results - Masked visualization
        fig2, ax2 = plt.subplots(2, 3, figsize=(15, 10))
        
        ax2[0, 0].imshow(img_rgb)
        ax2[0, 0].set_title('Original RGB Image')
        ax2[0, 0].axis('off')
        
        ax2[0, 1].imshow(border_mask, cmap='gray')
        ax2[0, 1].set_title('Border Pixels')
        ax2[0, 1].axis('off')
        
        ax2[0, 2].imshow(mask, cmap='gray')
        ax2[0, 2].set_title('Lesion Mask')
        ax2[0, 2].axis('off')
        
        ax2[1, 0].imshow(masked_gradient, cmap='inferno')
        ax2[1, 0].set_title('Masked Gradient (Border)')
        ax2[1, 0].axis('off')
        
        ax2[1, 1].imshow(masked_tophat, cmap='viridis')
        ax2[1, 1].set_title('Masked Top Hat')
        ax2[1, 1].axis('off')
        
        ax2[1, 2].imshow(masked_bottomhat, cmap='plasma')
        ax2[1, 2].set_title('Masked Bottom Hat')
        ax2[1, 2].axis('off')
        
        plt.tight_layout()
        plt.suptitle(f"Masked Transforms for: {os.path.basename(image_path)}", fontsize=16)
        plt.subplots_adjust(top=0.9)
        
        # 9. Create a text figure with the feature information
        fig3 = plt.figure(figsize=(10, 6))
        ax3 = fig3.add_subplot(111)
        ax3.axis('off')
        
        # Prepare feature text
        text = f"Border & Texture Features for {os.path.basename(image_path)}\n"
        text += "=" * len(text) + "\n\n"
        
        # Add border features
        text += "Border Features:\n"
        text += f"  gradient_mean: {features.get('border_gradient_mean', 0):.4f}\n"
        text += f"  gradient_std: {features.get('border_gradient_std', 0):.4f}\n"
        text += f"  gradient_max: {features.get('border_gradient_max', 0):.4f}\n"
        text += f"  irregularity: {features.get('border_irregularity', 0):.4f}\n\n"
        
        # Add texture features
        text += "Texture Features:\n"
        
        text += "  Top Hat (Bright Details):\n"
        text += f"    mean: {features.get('texture_tophat_mean', 0):.4f}\n"
        text += f"    std: {features.get('texture_tophat_std', 0):.4f}\n"
        
        text += "  Bottom Hat (Dark Details):\n"
        text += f"    mean: {features.get('texture_bottomhat_mean', 0):.4f}\n"
        text += f"    std: {features.get('texture_bottomhat_std', 0):.4f}\n"
        
        text += "  Summary Metrics:\n"
        text += f"    contrast_index: {features.get('texture_contrast_index', 0):.4f}\n"
        
        # Add the text to the figure
        ax3.text(0.05, 0.95, text, va='top', fontsize=12, family='monospace')
        
        # 10. Save visualization if output directory specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save transforms visualization
            transforms_path = os.path.join(
                output_dir, 
                f"{os.path.splitext(os.path.basename(image_path))[0]}_transforms.png"
            )
            plt.figure(fig1.number)  # Activate the first figure
            plt.savefig(transforms_path)
            print(f"Transforms visualization saved to: {transforms_path}")
            
            # Save masked transforms visualization
            masked_path = os.path.join(
                output_dir, 
                f"{os.path.splitext(os.path.basename(image_path))[0]}_masked_transforms.png"
            )
            plt.figure(fig2.number)  # Activate the second figure
            plt.savefig(masked_path)
            print(f"Masked transforms visualization saved to: {masked_path}")
            
            # Save feature text figure
            features_path = os.path.join(
                output_dir, 
                f"{os.path.splitext(os.path.basename(image_path))[0]}_border_texture_features.png"
            )
            plt.figure(fig3.number)  # Activate the third figure
            plt.savefig(features_path)
            print(f"Feature visualization saved to: {features_path}")
        
        # 11. Show all figures
        plt.show()
        
        print(f"Border and texture feature extraction completed for {image_path}.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and display border and texture features.")
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to the input image file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional directory to save visualization outputs."
    )
    args = parser.parse_args()
    display_border_texture_features(args.image_path, args.output_dir) 