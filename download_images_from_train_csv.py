import os
import pandas as pd
import shutil

# Paths
csv_path = 'data/train.csv'  # Adjust if your csv is elsewhere
src_dir = '/kaggle/input/siim-isic-melanoma-classification/jpeg/train'
dst_dir = 'data/input'

# Create destination directory if it doesn't exist
os.makedirs(dst_dir, exist_ok=True)

# Read the CSV
df = pd.read_csv(csv_path, sep=';')  # Use sep=';' as in your file

# Loop through image names and copy
for image_name in df['image_name']:
    src = os.path.join(src_dir, f"{image_name}.jpg")
    dst = os.path.join(dst_dir, f"{image_name}.jpg")
    if os.path.exists(src):
        shutil.copy(src, dst)
    else:
        print(f"Warning: {src} does not exist.")

print("Copying complete.") 