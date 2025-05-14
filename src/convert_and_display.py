import cv2
import matplotlib.pyplot as plt
from src.color_utils import rgb_to_grayscale, rgb_to_hsv


def convert_and_display(image_path: str) -> None:
    """
    Load an image, convert to RGB, apply grayscale and HSV conversions,
    and display original, grayscale, H, S, V, and full HSV (as RGB).

    Args:
        image_path: Path to the input image file.
    """
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = rgb_to_grayscale(rgb) # Uses COLOR_RGB2GRAY internally
    hsv = rgb_to_hsv(rgb) # Uses COLOR_RGB2HSV internally
    
    h, s, v = cv2.split(hsv)
    
    # For display, convert full HSV back to RGB
    hsv_display_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title('Original RGB')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(gray, cmap='gray')
    axes[0, 1].set_title('Grayscale (from RGB)')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(hsv_display_rgb)
    axes[0, 2].set_title('HSV (converted to RGB for display)')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(h, cmap='gray')
    axes[1, 0].set_title('H Channel (Hue)')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(s, cmap='gray')
    axes[1, 1].set_title('S Channel (Saturation)')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(v, cmap='gray')
    axes[1, 2].set_title('V Channel (Value/Intensity)')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Display an image in RGB, Grayscale, H, S, V, and full HSV (converted to RGB) representations."
    )
    parser.add_argument('image_path', help='Path to the input image file')
    args = parser.parse_args()
    convert_and_display(args.image_path) 