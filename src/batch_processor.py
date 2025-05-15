import argparse
import cv2
import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm # For progress bar
from typing import Dict, List, Tuple, Optional, Any

# Import pipeline components
from src.data_loader import load_metadata # To get image IDs and labels
from src.color_utils import rgb_to_grayscale, rgb_to_hsv, calculate_ida_channel
from src.preprocessing import remove_hair # Assuming correct_illumination might be skipped or optional
from src.segmentation import apply_threshold
from src.feature_extraction import calculate_all_features

# For visualizations during batch processing
from src.all_features_display import display_all_features
from src.advanced_features_display import plot_advanced_features

def process_single_image(image_path: str, image_id: str, output_vis_dir: Optional[str] = None) -> Optional[Dict[str, any]]:
    """
    Applies the full processing pipeline to a single image and extracts features.

    Args:
        image_path: Path to the input image.
        image_id: Unique identifier for the image (e.g., ISIC_xxxxxxx).
        output_vis_dir: Optional directory to save detailed visualization outputs for this image.

    Returns:
        A dictionary containing extracted features, or None if processing fails.
    """
    try:
        # 1. Load the image (OpenCV loads as BGR by default)
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"Warning: Could not load image {image_id} from {image_path}. Skipping.")
            return None
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 2. Color Space Conversions
        gray_img = rgb_to_grayscale(img_rgb.copy())
        hsv_img = rgb_to_hsv(img_rgb.copy())

        # 3. Preprocessing (Hair Removal)
        gray_hairless = remove_hair(gray_img.copy())
        if isinstance(gray_hairless, tuple):
            gray_hairless = gray_hairless[0] if len(gray_hairless) > 0 else gray_img.copy()

        # 4. Segmentation (Otsu with cleanup)
        mask, _ = apply_threshold(gray_hairless.copy(), method='otsu', cleanup=True)

        # 5. Feature Extraction
        # Note: calculate_all_features expects preprocessed grayscale, original HSV, and original RGB
        features = calculate_all_features(gray_hairless, hsv_img, img_rgb, mask)
        features['image_id'] = image_id # Add image_id for tracking

        # 6. Optional: Save detailed visualization for this image
        if output_vis_dir:
            vis_save_path = os.path.join(output_vis_dir, image_id)
            os.makedirs(vis_save_path, exist_ok=True)
            try:
                # Save standard feature visualization
                print(f"Saving standard visualization for {image_id} to {vis_save_path}...")
                display_all_features(image_path, output_dir=vis_save_path)
                
                # Save advanced feature visualization
                print(f"Saving advanced features visualization for {image_id} to {vis_save_path}...")
                try:
                    plot_advanced_features(
                        img_bgr,  # plot_advanced_features expects BGR format
                        mask,
                        title=f"Advanced Features - {image_id}",
                    )
                    plt.savefig(os.path.join(vis_save_path, f"{image_id}_advanced_features.png"))
                    plt.close()
                except Exception as e:
                    print(f"Warning: Could not save advanced visualization for {image_id}: {e}")
            except Exception as e:
                print(f"Warning: Could not save visualization for {image_id}: {e}")
        
        return features

    except Exception as e:
        print(f"Error processing image {image_id} at {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def batch_process_images(
    image_dir: str,
    metadata_path: str,
    output_features_path: str,
    output_vis_dir: Optional[str] = None,
    num_images: Optional[int] = None,
    image_ids_subset: Optional[List[str]] = None,
    save_interval: int = 100 # Save features every N images
):
    """
    Processes a batch of images, extracts features, and saves them.

    Args:
        image_dir: Directory containing the input images (e.g., 'data/jpeg').
        metadata_path: Path to the CSV file containing image metadata (e.g., 'data/train.csv').
        output_features_path: Path to save the extracted features (e.g., 'data/output/extracted_features.csv').
        output_vis_dir: Optional. Directory to save detailed visualizations for each processed image.
        num_images: Optional. Number of images to process from the dataset. Processes all if None.
        image_ids_subset: Optional. A list of specific image_ids to process. Overrides num_images if provided.
        save_interval: How often to save the features DataFrame to disk during processing.
    """
    print(f"Starting batch processing...")
    print(f"Image directory: {image_dir}")
    print(f"Metadata path: {metadata_path}")
    print(f"Output features path: {output_features_path}")
    if output_vis_dir:
        print(f"Output visualization directory: {output_vis_dir}")
        os.makedirs(output_vis_dir, exist_ok=True)
    
    # Load metadata
    metadata_df = load_metadata(metadata_path)
    if metadata_df is None or metadata_df.empty:
        print("Error: Could not load metadata or metadata is empty. Aborting.")
        return

    print(f"Loaded metadata with {len(metadata_df)} entries.")

    all_features_list = []
    processed_count = 0
    start_time = time.time()

    # Determine which images to process
    if image_ids_subset:
        target_ids = image_ids_subset
        print(f"Processing a specific subset of {len(target_ids)} image IDs.")
    elif num_images is not None:
        target_ids = metadata_df['image_name'].tolist()[:num_images]
        print(f"Processing the first {num_images} images from metadata.")
    else:
        target_ids = metadata_df['image_name'].tolist()
        print(f"Processing all {len(target_ids)} images from metadata.")

    # Use tqdm for a progress bar
    for image_id in tqdm(target_ids, desc="Processing Images"):
        image_filename = f"{image_id}.jpg" # Assuming .jpg extension
        image_file_path = os.path.join(image_dir, image_filename)

        if not os.path.exists(image_file_path):
            # Try .png as a fallback for some datasets or manual additions
            image_filename_png = f"{image_id}.png"
            image_file_path_png = os.path.join(image_dir, image_filename_png)
            if os.path.exists(image_file_path_png):
                image_file_path = image_file_path_png
            else:
                print(f"Warning: Image file for {image_id} not found at {image_file_path} or as .png. Skipping.")
                continue
        
        # Process the single image
        # Determine if visualizations should be saved for this specific image
        current_output_vis_dir = os.path.join(output_vis_dir, image_id) if output_vis_dir else None
        
        features = process_single_image(image_file_path, image_id, current_output_vis_dir)

        if features is not None:
            # Get corresponding label and other metadata
            meta_row = metadata_df[metadata_df['image_name'] == image_id]
            if not meta_row.empty:
                features['target'] = meta_row['target'].values[0]
                # Add other relevant metadata if needed, e.g., sex, age_approx
                if 'sex' in meta_row.columns: features['sex'] = meta_row['sex'].values[0]
                if 'age_approx' in meta_row.columns: features['age_approx'] = meta_row['age_approx'].values[0]
                if 'anatom_site_general_challenge' in meta_row.columns: 
                    features['anatom_site'] = meta_row['anatom_site_general_challenge'].values[0]
            else:
                print(f"Warning: Metadata not found for {image_id}. Label will be missing.")
                features['target'] = -1 # Or some other placeholder for missing label
            
            all_features_list.append(features)
            processed_count += 1

            # Save features periodically
            if processed_count % save_interval == 0 and all_features_list:
                features_df_intermediate = pd.DataFrame(all_features_list)
                try:
                    # Attempt to save as CSV, create directory if it doesn't exist
                    output_dir_intermediate = os.path.dirname(output_features_path)
                    if output_dir_intermediate and not os.path.exists(output_dir_intermediate):
                        os.makedirs(output_dir_intermediate, exist_ok=True)
                    features_df_intermediate.to_csv(output_features_path, index=False)
                    print(f"Successfully saved intermediate features for {processed_count} images to {output_features_path}")
                except Exception as e:
                    print(f"Error saving intermediate features to CSV: {e}. Trying pickle instead.")
                    try:
                        features_df_intermediate.to_pickle(output_features_path.replace('.csv', '.pkl'))
                        print(f"Successfully saved intermediate features to pickle: {output_features_path.replace('.csv', '.pkl')}")
                    except Exception as e_pkl:
                        print(f"Error saving intermediate features to pickle: {e_pkl}")

    # Final save of all features
    if all_features_list:
        features_df = pd.DataFrame(all_features_list)
        print(f"\nProcessed {processed_count} images in total.")
        print(f"Extracted features DataFrame shape: {features_df.shape}")
        
        # Ensure output directory exists for the final save
        final_output_dir = os.path.dirname(output_features_path)
        if final_output_dir and not os.path.exists(final_output_dir):
            os.makedirs(final_output_dir, exist_ok=True)
            print(f"Created output directory: {final_output_dir}")

        try:
            features_df.to_csv(output_features_path, index=False)
            print(f"Successfully saved all extracted features to {output_features_path}")
        except Exception as e_csv:
            print(f"Error saving final features to CSV: {e_csv}. Trying pickle.")
            pickle_path = output_features_path.replace('.csv', '.pkl')
            try:
                features_df.to_pickle(pickle_path)
                print(f"Successfully saved all extracted features to {pickle_path} (pickle format)")
            except Exception as e_pkl:
                print(f"CRITICAL: Failed to save features to both CSV and Pickle: {e_pkl}")
                print("You can try accessing the 'all_features_list' in a debugger if running interactively.")
    else:
        print("No features were extracted. Check logs for errors.")

    end_time = time.time()
    print(f"Batch processing completed in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process images for feature extraction.")
    parser.add_argument(
        "--image_dir", 
        type=str, 
        default="data/jpeg", 
        help="Directory containing input images."
    )
    parser.add_argument(
        "--metadata_path", 
        type=str, 
        default="data/train.csv", 
        help="Path to metadata CSV file."
    )
    parser.add_argument(
        "--output_features_path", 
        type=str, 
        default="data/output/extracted_features.csv", 
        help="Path to save the extracted features (CSV or PKL)."
    )
    parser.add_argument(
        "--output_vis_dir", 
        type=str, 
        default=None, # "data/output/batch_visualizations"
        help="Optional. Directory to save detailed visualizations for each processed image. Subfolders per image_id will be created."
    )
    parser.add_argument(
        "--num_images", 
        type=int, 
        default=None, 
        help="Number of images to process. Processes all if not specified."
    )
    parser.add_argument(
        "--image_ids",
        nargs='+', # Allows multiple image IDs to be passed
        default=None,
        help="A list of specific image_ids (e.g., ISIC_0015719 ISIC_0052212) to process. Overrides --num_images."
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=50,
        help="How often to save the features DataFrame to disk during processing (every N images)."
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist for the main features file
    main_output_dir = os.path.dirname(args.output_features_path)
    if main_output_dir and not os.path.exists(main_output_dir):
        os.makedirs(main_output_dir, exist_ok=True)
        print(f"Created main output directory: {main_output_dir}")

    batch_process_images(
        image_dir=args.image_dir,
        metadata_path=args.metadata_path,
        output_features_path=args.output_features_path,
        output_vis_dir=args.output_vis_dir,
        num_images=args.num_images,
        image_ids_subset=args.image_ids,
        save_interval=args.save_interval
    ) 