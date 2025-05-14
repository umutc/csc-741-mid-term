# 🔬 Skin Lesion Analysis Pipeline

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/) 
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE) 
<!-- Add other badges if applicable, e.g., build status, code coverage -->

**A classical (non-deep learning) image processing toolkit for skin lesion segmentation and feature extraction using dermoscopic images.**

This project, developed for CSC 741 (Digital Image Processing), demonstrates the application of fundamental image processing techniques to the challenge of analyzing skin lesions. It provides a modular pipeline for researchers and students to explore segmentation and feature extraction from dermoscopic images.

---

## ✨ Features

*   **🖼️ Data Loading:** Handles loading of JPEG images and associated CSV metadata.
*   **🎨 Color Space Conversion:** Converts images between RGB, Grayscale, and HSV.
*   **🧹 Preprocessing:** Includes hair removal (inpainting) and optional illumination correction.
*   **🎯 Segmentation:** Implements Otsu's thresholding for lesion segmentation with morphological cleanup options.
*   **📐 Feature Extraction:** Computes a rich set of features from the segmented lesion:
    *   **Intensity:** Mean, Std Dev, Min, Max (Grayscale)
    *   **Color:** Mean, Std Dev (HSV channels, with circular stats for Hue)
    *   **Border:** Gradient Mean/Std/Max, Irregularity Index
    *   **Texture:** Top-Hat Mean/Std, Bottom-Hat Mean/Std, Contrast Index
*   **📊 Visualization:** Provides multiple scripts to visualize intermediate steps, final segmentation, feature maps (gradient, texture), and comprehensive feature summaries.
*   **⚙️ Batch Processing:** Enables processing multiple images, extracting features, associating labels, and saving results to CSV/Pickle for further analysis.

---

## 🔧 Project Structure

```
.
├── data/                     # Dataset (Input images, CSV, Output results)
│   ├── input/                # (Create this) Placeholder for input images if not using data/jpeg
│   ├── jpeg/                 # Default directory for Kaggle dataset images
│   ├── output/               # Default directory for saved results (masks, features, visualizations)
│   └── train.csv             # Metadata from Kaggle dataset
├── src/                      # Source code modules
│   ├── __init__.py
│   ├── data_loader.py        # Image and metadata loading
│   ├── color_utils.py        # Color space conversions
│   ├── preprocessing.py      # Hair removal, illumination correction
│   ├── segmentation.py       # Thresholding methods (Otsu)
│   ├── morphology.py         # Basic morphological operations (used internally)
│   ├── histogram_utils.py    # Masked histogram calculations
│   ├── feature_extraction.py # Intensity and color feature calculations
│   ├── border_texture_utils.py # Border (gradient) and texture (Top/Bottom Hat) features
│   ├── verify_imports.py     # Environment check script
│   ├── convert_and_display.py # Display script for color conversions
│   ├── remove_hair_display.py # Display script for hair removal
│   ├── correct_illumination_display.py # Display script for illumination correction
│   ├── apply_threshold_display.py # Display script for basic thresholding
│   ├── otsu_display.py         # Display script specifically for Otsu's method
│   ├── masked_histogram_display.py # Display script for masked histograms
│   ├── intensity_features_display.py # Display script for intensity/color features
│   ├── border_texture_display.py # Display script for border/texture features
│   ├── all_features_display.py # Comprehensive visualization of pipeline and all features
│   └── batch_processor.py    # Script for processing multiple images
├── scripts/                  # Supporting scripts
│   └── object_detection_prd.md # Project Requirements Document
├── .gitignore
├── LICENSE                   # Project license file (e.g., MIT)
├── requirements.txt          # Python package dependencies
└── README.md                 # This file
```

---

## 🚀 Getting Started

### Prerequisites

*   Python 3.9+
*   `pip` (Python package installer)
*   (Optional) Kaggle account and API token (`kaggle.json`) for automatic dataset download.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url> # Replace with your repository URL
    cd skin-lesion-analysis-pipeline # Or your chosen directory name
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(This installs OpenCV, NumPy, Pandas, Matplotlib, Pillow, tqdm, etc.)*

3.  **Verify environment:**
    ```bash
    python3 -m src.verify_imports
    ```
    *(Should confirm all necessary packages are installed)*

### Data Setup

1.  **Download the Dataset:** (Requires Kaggle API setup or manual download)
    ```bash
    # Make sure kaggle.json is set up (see https://www.kaggle.com/docs/api)
    pip install kaggle # If not already installed

    # Download competition files to 'data/' directory
    kaggle competitions download -c siim-isic-melanoma-classification -p data

    # Unzip the training images into data/jpeg
    unzip data/siim-isic-melanoma-classification.zip -d data
    unzip data/jpeg.zip -d data/ # Adjust if zip structure is different
    # Ensure you have data/train.csv and images in data/jpeg/
    ```
    *Alternatively, manually download `train.csv` and the `jpeg/` directory from the Kaggle competition page and place them inside the `data/` folder.*

2.  **(Optional) Create Output Directory:** The scripts often save to `data/output/`. Create it if needed:
    ```bash
    mkdir -p data/output
    ```

---

## 💡 Usage Examples

**Important:** Run all commands from the project root directory using `python3 -m src.<script_name>`. This ensures Python can find the modules correctly. Replace `data/input/ISIC_0015719.jpg` (or similar) with a valid path to an image in your `data/jpeg/` directory.

### Individual Component Visualization

*   **Verify Imports:**
    ```bash
    python3 -m src.verify_imports
    ```
*   **Color Conversion:**
    ```bash
    python3 -m src.convert_and_display --image_path data/jpeg/ISIC_0015719.jpg
    ```
*   **Hair Removal:**
    ```bash
    python3 -m src.remove_hair_display --image_path data/jpeg/ISIC_0015719.jpg --output_dir data/output
    ```
*   **Illumination Correction:**
    ```bash
    python3 -m src.correct_illumination_display --image_path data/jpeg/ISIC_0015719.jpg --output_dir data/output
    ```
*   **Otsu Segmentation:**
    ```bash
    python3 -m src.otsu_display --image_path data/jpeg/ISIC_0015719.jpg --output_dir data/output
    ```
*   **Masked Histograms:**
    ```bash
    python3 -m src.masked_histogram_display --image_path data/jpeg/ISIC_0015719.jpg --output_dir data/output
    ```
*   **Intensity/Color Features:**
    ```bash
    python3 -m src.intensity_features_display --image_path data/jpeg/ISIC_0015719.jpg --output_dir data/output
    ```
*   **Border/Texture Features:**
    ```bash
    python3 -m src.border_texture_display --image_path data/jpeg/ISIC_0015719.jpg --output_dir data/output
    ```

### Comprehensive Pipeline Visualization (Single Image)

*   **Run Full Pipeline & Display/Save All Features & Visuals:**
    ```bash
    # Display only
    python3 -m src.all_features_display --image_path data/jpeg/ISIC_0015719.jpg

    # Display and save all visualizations to data/output/some_image_id/
    python3 -m src.all_features_display --image_path data/jpeg/ISIC_0015719.jpg --output_dir data/output/ISIC_0015719
    ```
    *(This is the recommended script for visualizing the complete results for one image, fulfilling Task 9 requirements)*

### Batch Processing (Multiple Images)

*   **Process First 20 Images & Save Features:**
    ```bash
    python3 -m src.batch_processor --num_images 20 --output_features_path data/output/features_first_20.csv
    ```
*   **Process Specific Images & Save Features + Visuals:**
    ```bash
    python3 -m src.batch_processor --image_ids ISIC_0015719 ISIC_0052212 \
        --output_features_path data/output/features_specific.csv \
        --output_vis_dir data/output/visuals_specific
    ```
*   **Process All Images & Save Features:**
    ```bash
    # This can take a long time! Saves features to data/output/extracted_features.csv
    python3 -m src.batch_processor
    ```
    *(Use `--output_vis_dir` cautiously when processing all images, as it generates many files.)*

---

## 📈 Next Steps & Future Work

Based on the original project roadmap (`scripts/object_detection_prd.md`), potential next steps include:

1.  **Feature Analysis:**
    *   Implement a script (`src/analyze_features.py`) to load `extracted_features.csv`.
    *   Generate comparative plots (box plots, histograms) for each feature, grouped by benign/malignant labels.
    *   Calculate summary statistics to assess feature distinctiveness.
2.  **Classifier Training:**
    *   Use the extracted features and labels to train machine learning models (e.g., SVM, Random Forest, Logistic Regression) to classify lesions.
    *   Evaluate classifier performance using appropriate metrics (Accuracy, Precision, Recall, F1-score, AUC).
3.  **Advanced Techniques:**
    *   Explore more sophisticated segmentation methods (e.g., active contours, watershed).
    *   Implement more complex texture features (e.g., GLCM, LBP).
    *   Investigate deep learning approaches for segmentation and classification.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*This README provides a guide to setting up and running the Skin Lesion Analysis Pipeline. For more technical details and the original project plan, refer to `scripts/object_detection_prd.md`.*  