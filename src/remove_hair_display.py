import cv2
import matplotlib.pyplot as plt
from src.preprocessing import remove_hair


def main(image_path, output_path=None):
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hairless = remove_hair(gray)

    if output_path:
        # Save as BGR for compatibility
        cv2.imwrite(output_path, cv2.cvtColor(hairless, cv2.COLOR_GRAY2BGR))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(gray, cmap='gray')
        axes[0].set_title('Original Grayscale')
        axes[0].axis('off')
        axes[1].imshow(hairless, cmap='gray')
        axes[1].set_title('After Hair Removal')
        axes[1].axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Remove hair from a grayscale image and display the result.")
    parser.add_argument('image_path', help='Path to the input image file')
    parser.add_argument('--output', help='Path to save the hairless image', default=None)
    args = parser.parse_args()
    main(args.image_path, args.output) 