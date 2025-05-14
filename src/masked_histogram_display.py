import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

from src.color_utils import rgb_to_grayscale, rgb_to_hsv
from src.preprocessing import remove_hair
from src.segmentation import apply_threshold
from src.histogram_utils import calc_all_histograms, plot_histograms

def display_masked_histograms(image_path: str, output_dir=None):
    """
    Process an image through the pipeline, calculate masked histograms,
    and display the results.
    
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
        
        # 2. Color Space Conversions
        try:
            gray_img = rgb_to_grayscale(img_rgb.copy())
            hsv_img = rgb_to_hsv(img_rgb.copy())
        except Exception as e:
            print(f"Warning: Error in RGB color conversion: {e}")
            print("Falling back to OpenCV's direct conversion.")
            gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

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

        # 5. Calculate histograms with the mask
        try:
            # RGB image with mask
            histograms = calc_all_histograms(img_rgb, mask)
        except Exception as e:
            print(f"Error calculating histograms: {e}")
            raise

        # 6. Display Results - First show the image and mask
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img_rgb)
        axes[0].set_title('Original RGB Image')
        axes[0].axis('off')
        
        axes[1].imshow(gray_hairless, cmap='gray')
        axes[1].set_title('Preprocessed (Hair Removed)')
        axes[1].axis('off')
        
        axes[2].imshow(mask, cmap='gray')
        axes[2].set_title(f'Binary Mask for Histogram')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save segmentation visualization if output directory specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            seg_output_path = os.path.join(
                output_dir, 
                f"{os.path.splitext(os.path.basename(image_path))[0]}_segmentation.png"
            )
            plt.savefig(seg_output_path)
            print(f"Segmentation visualization saved to: {seg_output_path}")
        
        plt.show()
        
        # 7. Now plot the histograms
        plot_histograms(
            histograms, 
            title_prefix=f"Masked Histogram: {os.path.basename(image_path)}"
        )
        
        # Save histogram visualization if output directory specified
        if output_dir:
            hist_output_path = os.path.join(
                output_dir, 
                f"{os.path.splitext(os.path.basename(image_path))[0]}_histograms.png"
            )
            plt.savefig(hist_output_path)
            print(f"Histogram visualization saved to: {hist_output_path}")
        
        print(f"Histogram calculation completed for {image_path}.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate and display masked histograms.")
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
    display_masked_histograms(args.image_path, args.output_dir) 