import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# Import functions using more robust approach with fallbacks
try:
    # Try the main imports first
    from src.color_utils import rgb_to_grayscale, rgb_to_hsv
except ImportError:
    # Provide fallback implementations if modules don't exist
    print("Warning: Could not import from src.color_utils. Using fallback implementations.")
    def rgb_to_grayscale(img):
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img
    
    def rgb_to_hsv(img):
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        return img

try:
    from src.preprocessing import remove_hair, correct_illumination
except ImportError:
    # Provide fallback implementations
    print("Warning: Could not import from src.preprocessing. Using fallback implementations.")
    def remove_hair(img):
        print("Note: remove_hair function not available. Using original image.")
        return img
    
    def correct_illumination(img):
        print("Note: correct_illumination function not available. Using original image.")
        return img

try:
    from src.segmentation import apply_threshold, otsu_threshold
except ImportError:
    from src.segmentation import otsu_threshold
    # Fallback implementation if apply_threshold doesn't exist
    print("Warning: Could not import apply_threshold. Using direct Otsu thresholding.")
    
    def apply_threshold(img, method='otsu', cleanup=False, **kwargs):
        if method != 'otsu':
            print(f"Warning: Method '{method}' not available in fallback. Using Otsu.")
        mask, thresh = otsu_threshold(img)
        # No cleanup in fallback
        if cleanup:
            print("Note: Cleanup option not available in fallback implementation.")
        return mask, thresh

# Import new components
try:
    from src.histogram_utils import calc_all_histograms
except ImportError:
    print("Warning: Could not import histogram utilities. Histogram calculation will be skipped.")
    calc_all_histograms = None

try:
    from src.feature_extraction import calculate_all_intensity_features, print_features
except ImportError:
    print("Warning: Could not import feature extraction utilities. Feature extraction will be skipped.")
    calculate_all_intensity_features = None
    print_features = None


def display_full_pipeline(image_path: str, output_dir=None):
    """
    Loads an image, applies the full preprocessing and segmentation pipeline,
    calculates histograms and features, and displays intermediate and final results.

    Args:
        image_path: Path to the input image.
        output_dir: Optional directory to save the visualization. If None, just display.
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
        
        # Manual fallback to grayscale if needed (just in case)
        gray_img_fallback = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 2. Color Space Conversions
        # Try with RGB, but have BGR fallback ready
        try:
            # First try with RGB input (copy to prevent modification)
            gray_img = rgb_to_grayscale(img_rgb.copy())
            hsv_img = rgb_to_hsv(img_rgb.copy())
        except Exception as e:
            print(f"Warning: Error in RGB color conversion: {e}")
            print("Falling back to OpenCV's direct conversion.")
            gray_img = gray_img_fallback
            hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # 3. Preprocessing (on grayscale image)
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
            
        # IGNORING illumination correction as requested
        print("Note: Skipping illumination correction step as requested.")
        gray_corrected = gray_hairless.copy()  # Skip illumination correction
            
        # 4. Segmentation (on preprocessed grayscale image)
        try:
            # Ensure gray_corrected is a valid NumPy array before calling copy()
            if isinstance(gray_corrected, tuple):
                gray_corrected_input = gray_corrected[0] if len(gray_corrected) > 0 else gray_img.copy()
            else:
                gray_corrected_input = gray_corrected.copy()
                
            # Using Otsu's method with cleanup by default
            final_mask, otsu_thresh = apply_threshold(
                gray_corrected_input,
                method='otsu',
                cleanup=True
            )
        except Exception as e:
            print(f"Warning: Error in thresholding: {e}")
            print("Using direct Otsu thresholding without cleanup.")
            try:
                # Ensure gray_corrected is a valid NumPy array before calling copy()
                if isinstance(gray_corrected, tuple):
                    gray_corrected_input = gray_corrected[0] if len(gray_corrected) > 0 else gray_img.copy()
                else:
                    gray_corrected_input = gray_corrected.copy()
                    
                final_mask, otsu_thresh = otsu_threshold(gray_corrected_input)
            except Exception as e2:
                print(f"Error in otsu_threshold: {e2}")
                # Last resort - basic threshold at mean
                if isinstance(gray_corrected, tuple):
                    gray_corrected_input = gray_corrected[0] if len(gray_corrected) > 0 else gray_img.copy()
                else:
                    gray_corrected_input = gray_corrected

                thresh = int(np.mean(gray_corrected_input))
                _, final_mask = cv2.threshold(gray_corrected_input, thresh, 255, cv2.THRESH_BINARY)
                otsu_thresh = thresh

        # 5. Calculate histograms and features (NEW)
        histograms = None
        features = None
        
        if calc_all_histograms is not None:
            try:
                print("\nCalculating histograms within the mask region...")
                histograms = calc_all_histograms(img_rgb, final_mask)
                print("Histogram calculation complete.")
            except Exception as e:
                print(f"Error in histogram calculation: {e}")
                histograms = None
        
        if calculate_all_intensity_features is not None:
            try:
                print("\nExtracting intensity features from the masked region...")
                features = calculate_all_intensity_features(gray_img, hsv_img, final_mask)
                if print_features is not None:
                    print_features(features, title=f"Intensity Features for {os.path.basename(image_path)}")
                else:
                    print("Features extracted:", features)
            except Exception as e:
                print(f"Error in feature extraction: {e}")
                features = None

        # For display, ensure all images are valid arrays, not tuples
        def ensure_array(img, fallback):
            if isinstance(img, tuple):
                return img[0] if len(img) > 0 else fallback
            return img

        gray_img_display = ensure_array(gray_img, gray_img_fallback)
        hsv_img_display = ensure_array(hsv_img, cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV))
        gray_hairless_display = ensure_array(gray_hairless, gray_img_display)
        gray_corrected_display = ensure_array(gray_corrected, gray_hairless_display)
        final_mask_display = ensure_array(final_mask, np.zeros_like(gray_img_display))

        # 6. Display Results
        # Main pipeline visualization
        fig1, axes1 = plt.subplots(2, 3, figsize=(18, 10))
        fig1.suptitle(f'Full Pipeline Stages for: {os.path.basename(image_path)}', fontsize=16)

        axes1[0, 0].imshow(img_rgb)
        axes1[0, 0].set_title('Original RGB Image')
        axes1[0, 0].axis('off')

        axes1[0, 1].imshow(gray_img_display, cmap='gray')
        axes1[0, 1].set_title('Grayscale')
        axes1[0, 1].axis('off')
        
        # Check if HSV is 3-channel before displaying specific channel
        if len(hsv_img_display.shape) == 3 and hsv_img_display.shape[2] >= 3:
            axes1[0, 2].imshow(hsv_img_display[:, :, 0], cmap='hsv')  # Hue channel
            axes1[0, 2].set_title('HSV (Hue Channel)')
        else:
            axes1[0, 2].imshow(hsv_img_display, cmap='viridis')  # Display whatever we have
            axes1[0, 2].set_title('HSV (Format Issue - Using Viridis)')
        axes1[0, 2].axis('off')

        axes1[1, 0].imshow(gray_hairless_display, cmap='gray')
        axes1[1, 0].set_title('Hair Removed (Grayscale)')
        axes1[1, 0].axis('off')

        axes1[1, 1].imshow(gray_corrected_display, cmap='gray')
        axes1[1, 1].set_title('After Hair Removal (Illumination Correction Skipped)')
        axes1[1, 1].axis('off')

        axes1[1, 2].imshow(final_mask_display, cmap='gray')
        axes1[1, 2].set_title(f'Final Mask (Otsu Thresh: {otsu_thresh})')
        axes1[1, 2].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout for suptitle
        
        # 7. Create additional figures for features and histograms (if calculated)
        
        # Features figure (if available)
        fig2 = None
        if features is not None:
            fig2 = plt.figure(figsize=(10, 6))
            ax2 = fig2.add_subplot(111)
            ax2.axis('off')
            
            # Prepare feature text
            text = f"Intensity Features for {os.path.basename(image_path)}\n"
            text += "=" * len(text) + "\n\n"
            
            # Add grayscale intensity features
            text += "Grayscale Intensity:\n"
            text += f"  Mean: {features.get('intensity_mean', 0):.4f}\n"
            text += f"  Std: {features.get('intensity_std', 0):.4f}\n\n"
            
            # Add HSV features
            text += "HSV Channels:\n"
            
            text += "  Hue (H):\n"
            text += f"    Mean: {features.get('h_mean', 0):.4f}\n"
            text += f"    Std: {features.get('h_std', 0):.4f}\n"
            
            text += "  Saturation (S):\n"
            text += f"    Mean: {features.get('s_mean', 0):.4f}\n"
            text += f"    Std: {features.get('s_std', 0):.4f}\n"
            
            text += "  Value (V):\n"
            text += f"    Mean: {features.get('v_mean', 0):.4f}\n"
            text += f"    Std: {features.get('v_std', 0):.4f}\n"
            
            # Add the text to the figure
            ax2.text(0.05, 0.95, text, va='top', fontsize=12, family='monospace')
            
            fig2.tight_layout()
        
        # Histogram figure (if histograms were calculated)
        fig3 = None
        if histograms is not None:
            fig3 = plt.figure(figsize=(12, 8))
            fig3.suptitle(f'Channel Histograms (Masked Region) for: {os.path.basename(image_path)}', fontsize=14)
            
            # Define colors and subplot positions for each channel
            channels = [
                ('gray', 'Grayscale', 'black', 1),
                ('h', 'Hue (H)', 'blue', 2),
                ('s', 'Saturation (S)', 'green', 3),
                ('v', 'Value (V)', 'red', 4)
            ]
            
            for channel, title, color, pos in channels:
                if channel in histograms:
                    hist, bin_edges = histograms[channel]
                    ax = fig3.add_subplot(2, 2, pos)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    ax.bar(bin_centers, hist, width=(bin_edges[1]-bin_edges[0]), color=color, alpha=0.7)
                    ax.set_title(f'{title} Histogram')
                    ax.set_xlabel('Intensity Value')
                    ax.set_ylabel('Pixel Count')
                    ax.grid(alpha=0.3)
            
            fig3.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
        
        # Save figures if output directory specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save main pipeline figure
            pipeline_path = os.path.join(
                output_dir, 
                f"{os.path.splitext(os.path.basename(image_path))[0]}_pipeline.png"
            )
            plt.figure(fig1.number)
            plt.savefig(pipeline_path)
            print(f"Pipeline visualization saved to: {pipeline_path}")
            
            # Save features figure if available
            if fig2 is not None:
                features_path = os.path.join(
                    output_dir, 
                    f"{os.path.splitext(os.path.basename(image_path))[0]}_features.png"
                )
                plt.figure(fig2.number)
                plt.savefig(features_path)
                print(f"Features visualization saved to: {features_path}")
            
            # Save histograms figure if available
            if fig3 is not None:
                histograms_path = os.path.join(
                    output_dir, 
                    f"{os.path.splitext(os.path.basename(image_path))[0]}_histograms.png"
                )
                plt.figure(fig3.number)
                plt.savefig(histograms_path)
                print(f"Histograms visualization saved to: {histograms_path}")
        
        # Show all figures
        plt.show()

        print(f"Full pipeline completed for {image_path}.")
        print(f"Otsu threshold determined: {otsu_thresh}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ImportError as e:
        print(f"Error: Missing a module. Please ensure all dependencies are installed. {e}")
        print("Try running: pip install opencv-python numpy matplotlib")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Display full image processing pipeline results.")
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
        help="Optional directory to save visualization (in addition to displaying)."
    )
    args = parser.parse_args()
    display_full_pipeline(args.image_path, args.output_dir) 