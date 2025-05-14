import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
from src.segmentation import otsu_threshold # Assuming otsu_threshold is in src.segmentation

def display_otsu_segmentation(image_path: str):
    """
    Loads an image, applies Otsu's thresholding, and displays the results.

    Args:
        image_path: Path to the input image.
    """
    try:
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image from {image_path}")
            return

        # Convert to grayscale
        if len(img.shape) == 3 and img.shape[2] == 3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif len(img.shape) == 2:
            gray_img = img # Already grayscale
        else:
            print("Error: Image format not supported for grayscale conversion.")
            return

        # Apply Otsu's thresholding
        mask, optimal_thresh = otsu_threshold(gray_img)

        # Display the results
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(gray_img, cmap='gray')
        axes[0].set_title('Original Grayscale Image')
        axes[0].axis('off')

        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title(f"Otsu\'s Mask (Thresh: {optimal_thresh})")
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Display Otsu\'s thresholding results.")
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to the input image file."
    )
    args = parser.parse_args()
    display_otsu_segmentation(args.image_path) 