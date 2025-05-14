import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

from src.color_utils import rgb_to_grayscale, rgb_to_hsv
from src.preprocessing import remove_hair
from src.segmentation import apply_threshold
from src.feature_extraction import calculate_all_features, print_features

def display_all_features(image_path: str, output_dir=None):
    """
    Process an image through the pipeline, extract all features (intensity, border, texture),
    and display the results with comprehensive visualizations.
    
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
        
        # 2. Convert to grayscale and HSV
        try:
            gray_img = rgb_to_grayscale(img_rgb.copy())
        except Exception as e:
            print(f"Warning: Error in RGB to grayscale conversion: {e}")
            print("Falling back to OpenCV's direct conversion.")
            gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
        try:
            hsv_img = rgb_to_hsv(img_rgb.copy())
        except Exception as e:
            print(f"Warning: Error in RGB to HSV conversion: {e}")
            print("Falling back to OpenCV's direct conversion.")
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

        # 5. Extract ALL features
        try:
            # Calculate comprehensive features
            features = calculate_all_features(gray_hairless, hsv_img, mask)
            
            # Print the features
            print_features(features, title=f"All Features for {os.path.basename(image_path)}")
        except Exception as e:
            print(f"Error calculating features: {e}")
            import traceback
            traceback.print_exc()
            raise

        # 6. Prepare visualization elements
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
        
        # HSV channel visualization
        hue = hsv_img[:, :, 0]
        saturation = hsv_img[:, :, 1]
        value = hsv_img[:, :, 2]
        
        # Masked HSV channels
        masked_hue = np.where(mask > 0, hue, 0)
        masked_saturation = np.where(mask > 0, saturation, 0)
        masked_value = np.where(mask > 0, value, 0)
        
        # 7. Display Results - Main visualization (Preprocessing and Segmentation)
        fig1 = plt.figure(figsize=(15, 10))
        plt.suptitle(f"Preprocessing & Segmentation for: {os.path.basename(image_path)}", fontsize=16)
        
        # Grid for 2 rows, 3 columns
        ax1 = fig1.add_subplot(2, 3, 1)
        ax1.imshow(img_rgb)
        ax1.set_title('Original RGB Image')
        ax1.axis('off')
        
        ax2 = fig1.add_subplot(2, 3, 2)
        ax2.imshow(gray_img, cmap='gray')
        ax2.set_title('Grayscale Image')
        ax2.axis('off')
        
        ax3 = fig1.add_subplot(2, 3, 3)
        ax3.imshow(gray_hairless, cmap='gray')
        ax3.set_title('After Hair Removal')
        ax3.axis('off')
        
        ax4 = fig1.add_subplot(2, 3, 4)
        ax4.imshow(mask, cmap='gray')
        ax4.set_title('Binary Mask')
        ax4.axis('off')
        
        ax5 = fig1.add_subplot(2, 3, 5)
        # Create a composite RGB segmentation visualization
        segmentation_vis = np.zeros_like(img_rgb)
        segmentation_vis[:, :, 1] = np.where(mask > 0, 255, 0)  # Green mask
        ax5.imshow(segmentation_vis)
        ax5.set_title('Segmentation Overlay (Green)')
        ax5.axis('off')
        
        ax6 = fig1.add_subplot(2, 3, 6)
        # Create an overlay of the mask on the original image
        overlay = img_rgb.copy()
        overlay[mask == 0] = overlay[mask == 0] // 2  # Darken non-mask regions
        ax6.imshow(overlay)
        ax6.set_title('Original with Segmentation')
        ax6.axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # 8. Display Results - Border and Texture Features visualization
        fig2 = plt.figure(figsize=(15, 10))
        plt.suptitle(f"Border & Texture Features for: {os.path.basename(image_path)}", fontsize=16)
        
        # Grid for 2 rows, 3 columns
        ax1 = fig2.add_subplot(2, 3, 1)
        ax1.imshow(gradient, cmap='inferno')
        ax1.set_title('Morphological Gradient')
        ax1.axis('off')
        
        ax2 = fig2.add_subplot(2, 3, 2)
        ax2.imshow(masked_gradient, cmap='inferno')
        ax2.set_title('Masked Gradient (Border)')
        ax2.axis('off')
        
        ax3 = fig2.add_subplot(2, 3, 3)
        ax3.imshow(border_mask, cmap='gray')
        ax3.set_title('Border Pixels')
        ax3.axis('off')
        
        ax4 = fig2.add_subplot(2, 3, 4)
        ax4.imshow(tophat, cmap='viridis')
        ax4.set_title('Top Hat (Bright Details)')
        ax4.axis('off')
        
        ax5 = fig2.add_subplot(2, 3, 5)
        ax5.imshow(bottomhat, cmap='plasma')
        ax5.set_title('Bottom Hat (Dark Details)')
        ax5.axis('off')
        
        ax6 = fig2.add_subplot(2, 3, 6)
        # Create a visualization of both masks combined
        combined = np.zeros_like(gray_img)
        combined = np.where(mask > 0, 100, 0)
        combined = np.where(border_mask > 0, 255, combined)
        ax6.imshow(combined, cmap='gray')
        ax6.set_title('Mask & Border Combined')
        ax6.axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # 9. Display Results - HSV Features visualization
        fig3 = plt.figure(figsize=(15, 10))
        plt.suptitle(f"HSV Channel Analysis for: {os.path.basename(image_path)}", fontsize=16)
        
        # Grid for 2 rows, 3 columns 
        ax1 = fig3.add_subplot(2, 3, 1)
        ax1.imshow(hue, cmap='hsv')
        ax1.set_title('Hue Channel')
        ax1.axis('off')
        
        ax2 = fig3.add_subplot(2, 3, 2)
        ax2.imshow(saturation, cmap='plasma')
        ax2.set_title('Saturation Channel')
        ax2.axis('off')
        
        ax3 = fig3.add_subplot(2, 3, 3)
        ax3.imshow(value, cmap='gray')
        ax3.set_title('Value Channel')
        ax3.axis('off')
        
        ax4 = fig3.add_subplot(2, 3, 4)
        ax4.imshow(masked_hue, cmap='hsv')
        ax4.set_title('Masked Hue')
        ax4.axis('off')
        
        ax5 = fig3.add_subplot(2, 3, 5)
        ax5.imshow(masked_saturation, cmap='plasma')
        ax5.set_title('Masked Saturation')
        ax5.axis('off')
        
        ax6 = fig3.add_subplot(2, 3, 6)
        ax6.imshow(masked_value, cmap='gray')
        ax6.set_title('Masked Value')
        ax6.axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # 10. Create a text figure with the feature information
        fig4 = plt.figure(figsize=(15, 9))
        ax7 = fig4.add_subplot(111)
        ax7.axis('off')
        
        # Prepare feature text
        text = f"FEATURES EXTRACTED FROM {os.path.basename(image_path)}\n"
        text += "=" * len(text) + "\n\n"
        
        # Add intensity features
        intensity_features = {k:v for k,v in features.items() if k.startswith('intensity_')}
        if intensity_features:
            text += "INTENSITY FEATURES (Grayscale):\n"
            for name, value in intensity_features.items():
                text += f"  {name.replace('intensity_', '')}: {value:.4f}\n"
            text += "\n"
        
        # Add HSV features
        hsv_features = {k:v for k,v in features.items() if any(k.startswith(p) for p in ['hue_', 'saturation_', 'value_'])}
        if hsv_features:
            text += "COLOR FEATURES (HSV):\n"
            
            # Break down by channel
            hue_features = {k:v for k,v in hsv_features.items() if k.startswith('hue_')}
            saturation_features = {k:v for k,v in hsv_features.items() if k.startswith('saturation_')}
            value_features = {k:v for k,v in hsv_features.items() if k.startswith('value_')}
            
            if hue_features:
                text += "  Hue:\n"
                for name, value in hue_features.items():
                    text += f"    {name.replace('hue_', '')}: {value:.4f}\n"
                    
            if saturation_features:
                text += "  Saturation:\n"
                for name, value in saturation_features.items():
                    text += f"    {name.replace('saturation_', '')}: {value:.4f}\n"
                    
            if value_features:
                text += "  Value:\n"
                for name, value in value_features.items():
                    text += f"    {name.replace('value_', '')}: {value:.4f}\n"
            text += "\n"
        
        # Add border features
        border_features = {k:v for k,v in features.items() if k.startswith('border_')}
        if border_features:
            text += "BORDER FEATURES:\n"
            for name, value in border_features.items():
                text += f"  {name.replace('border_', '')}: {value:.4f}\n"
            text += "\n"
        
        # Add texture features
        texture_features = {k:v for k,v in features.items() if k.startswith('texture_')}
        if texture_features:
            text += "TEXTURE FEATURES:\n"
            
            # Break down texture features
            tophat_features = {k:v for k,v in texture_features.items() if k.startswith('texture_tophat_')}
            bottomhat_features = {k:v for k,v in texture_features.items() if k.startswith('texture_bottomhat_')}
            other_texture = {k:v for k,v in texture_features.items() if k.startswith('texture_') 
                          and not k.startswith('texture_tophat_') 
                          and not k.startswith('texture_bottomhat_')}
            
            if tophat_features:
                text += "  Top Hat (Bright Details):\n"
                for name, value in tophat_features.items():
                    text += f"    {name.replace('texture_tophat_', '')}: {value:.4f}\n"
            
            if bottomhat_features:
                text += "  Bottom Hat (Dark Details):\n"
                for name, value in bottomhat_features.items():
                    text += f"    {name.replace('texture_bottomhat_', '')}: {value:.4f}\n"
            
            if other_texture:
                text += "  Summary Metrics:\n"
                for name, value in other_texture.items():
                    text += f"    {name.replace('texture_', '')}: {value:.4f}\n"
        
        # Add the text to the figure - use a monospace font for alignment
        ax7.text(0.05, 0.95, text, va='top', fontsize=14, family='monospace')
        fig4.tight_layout()
        
        # 11. Save visualization if output directory specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save each visualization figure
            preprocessing_path = os.path.join(
                output_dir, 
                f"{os.path.splitext(os.path.basename(image_path))[0]}_preprocessing.png"
            )
            plt.figure(fig1.number)
            plt.savefig(preprocessing_path)
            print(f"Preprocessing visualization saved to: {preprocessing_path}")
            
            border_texture_path = os.path.join(
                output_dir, 
                f"{os.path.splitext(os.path.basename(image_path))[0]}_border_texture.png"
            )
            plt.figure(fig2.number)
            plt.savefig(border_texture_path)
            print(f"Border & texture visualization saved to: {border_texture_path}")
            
            hsv_path = os.path.join(
                output_dir, 
                f"{os.path.splitext(os.path.basename(image_path))[0]}_hsv_analysis.png"
            )
            plt.figure(fig3.number)
            plt.savefig(hsv_path)
            print(f"HSV channels visualization saved to: {hsv_path}")
            
            features_path = os.path.join(
                output_dir, 
                f"{os.path.splitext(os.path.basename(image_path))[0]}_features_summary.png"
            )
            plt.figure(fig4.number)
            plt.savefig(features_path)
            print(f"Features summary saved to: {features_path}")
            
            # Save features as text file
            features_txt_path = os.path.join(
                output_dir, 
                f"{os.path.splitext(os.path.basename(image_path))[0]}_features.txt"
            )
            with open(features_txt_path, 'w') as f:
                f.write(text)
            print(f"Features saved as text to: {features_txt_path}")
        
        # 12. Show all figures
        plt.show()
        
        print(f"Feature extraction and visualization completed for {image_path}.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and display all features from an image.")
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
    display_all_features(args.image_path, args.output_dir) 