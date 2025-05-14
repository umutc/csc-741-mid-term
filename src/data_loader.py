import os
import cv2
import numpy as np
import pandas as pd
from typing import Dict, Tuple


def load_images_from_directory(
    directory: str,
    extensions: Tuple[str, ...] = ('.jpg', '.jpeg')
) -> Dict[str, np.ndarray]:
    """
    Load all JPEG images from the given directory into a dict of RGB NumPy arrays.

    Args:
        directory: Path to the folder containing image files.
        extensions: Tuple of file extensions to include (default: '.jpg', '.jpeg').

    Returns:
        A dict mapping each filename (without extension) to its RGB image array.
    """
    images: Dict[str, np.ndarray] = {}
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Image directory not found: {directory}")
    for filename in os.listdir(directory):
        if not filename.lower().endswith(extensions):
            continue
        filepath = os.path.join(directory, filename)
        try:
            img = cv2.imread(filepath, cv2.IMREAD_COLOR)
            if img is None:
                print(f"Warning: Failed to read image: {filepath}")
                continue
            # Convert from BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            key = os.path.splitext(filename)[0]
            images[key] = img
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    return images


def load_metadata(csv_path: str) -> pd.DataFrame:
    """
    Read metadata CSV into a Pandas DataFrame.

    Args:
        csv_path: Path to the CSV file (e.g., 'train.csv').

    Returns:
        A DataFrame containing the metadata.
    """
    try:
        df = pd.read_csv(csv_path)
        return df
    except FileNotFoundError:
        print(f"Metadata file not found: {csv_path}")
        raise
    except Exception as e:
        print(f"Error reading CSV {csv_path}: {e}")
        raise


def load_data(
    image_dir: str,
    csv_path: str
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    """
    Load images and metadata, with error handling.

    Args:
        image_dir: Directory containing JPEG images.
        csv_path: Path to metadata CSV file.

    Returns:
        A tuple (images_dict, metadata_df).
    """
    try:
        images = load_images_from_directory(image_dir)
    except Exception as e:
        print(f"Failed to load images: {e}")
        raise

    try:
        metadata = load_metadata(csv_path)
    except Exception as e:
        print(f"Failed to load metadata: {e}")
        raise

    return images, metadata 