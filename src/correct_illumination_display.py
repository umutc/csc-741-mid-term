import cv2
import matplotlib.pyplot as plt
from src.preprocessing import correct_illumination
import numpy as np


def main(image_path, output_path=None, kernel_size=(31, 31), debug=False):
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    corrected_image, background_estimate = correct_illumination(gray, kernel_size=kernel_size)

    if debug:
        print(f"[DEBUG] Original gray min/max/mean: {gray.min()}/{gray.max()}/{gray.mean():.2f}")
        print(f"[DEBUG] Background estimate min/max/mean: {background_estimate.min()}/{background_estimate.max()}/{background_estimate.mean():.2f}")
        print(f"[DEBUG] Corrected image min/max/mean: {corrected_image.min()}/{corrected_image.max()}/{corrected_image.mean():.2f}")
        cv2.imwrite('debug_background.png', background_estimate) # Save the uint8 background
        print("[DEBUG] Saved background_estimate as debug_background.png")

    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(corrected_image, cv2.COLOR_GRAY2BGR))
    else:
        fig, axes = plt.subplots(1, 3 if debug else 2, figsize=(15, 5))
        axes[0].imshow(gray, cmap='gray')
        axes[0].set_title('Original Grayscale')
        axes[0].axis('off')
        axes[1].imshow(corrected_image, cmap='gray')
        axes[1].set_title('After Illumination Correction')
        axes[1].axis('off')
        if debug:
            axes[2].imshow(background_estimate, cmap='gray')
            axes[2].set_title('Background Estimate (Debug)')
            axes[2].axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Correct illumination in a grayscale image and display the result.")
    parser.add_argument('image_path', help='Path to the input image file')
    parser.add_argument('--output', help='Path to save the corrected image', default=None)
    parser.add_argument('--kernel-size', nargs=2, type=int, metavar=('W', 'H'), default=[31, 31], help='Kernel size for background estimation (default: 31 31)')
    parser.add_argument('--debug', action='store_true', help='Save and display background estimate for debugging')
    args = parser.parse_args()
    main(args.image_path, args.output, tuple(args.kernel_size), args.debug) 