# Cardiac Pathologies Detection using CNN

This repository contains the source code for training and evaluating Convolutional Neural Networks (CNNs) for the detection of aortic calcification and other cardiac pathologies from JPEG images. It supports both a custom CNN architecture and transfer learning with various pre-trained models (ResNet, VGG, etc.).

## Features

* **K-Fold Cross-Validation**: Robust evaluation using Stratified K-Fold.
* **Transfer Learning**: Easy switching between different pre-trained backbones.
* **Grad-CAM Visualization**: Explainability maps to visualize model focus.
* **Comprehensive Metrics**: Automated generation of confusion matrices, ROC curves, and classification reports.

## Installation

1. Clone the repository:

   ```bash
   git clone [URL to Version Control System]
   cd [Repository Name]
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training with K-Fold Cross-Validation (Default)

```bash
python Hospital_training.py --data_dir /path/to/dataset --k_folds 5
```

### Single Split Training

```bash
python Hospital_training.py --data_dir /path/to/dataset --mode single
```

### Transfer Learning Experiments

```bash
python Hospital_training.py --data_dir /path/to/dataset --mode transfer
```

```

## Inference / Evaluation

The `evaluate_cardiac_pathology.py` script allows you to run inference on new images (DICOM or standard formats) using a trained model. It generates a JSON output compatible with the Java FHIR Converter.

### Usage

```bash
python evaluate_cardiac_pathology.py [input_path] [model_path] [config_path] [options]
```

### Arguments

*   `input_path`: Path to the input image file (DICOM/JPG/PNG) or directory containing images.
*   `model_path`: Path to the trained Keras model (`.h5` file).
*   `config_path`: Path to the configuration JSON file (containing `img_width`, `img_height`, `Labels`, `Classification`).
*   `-o`, `--output_path`: (Optional) Directory to save the output JSON files. Defaults to current directory.
*   `--use_labels`: (Optional) If set, saves a copy of the processed image with detected lung bounding boxes drawn.

### Example

```bash
python evaluate_cardiac_pathology.py \
    ./data/test_image.dcm \
    ./MODELS/model_fold_1.h5 \
    ./config.json \
    -o ./output_results
```

### Output

For each input image, the script generates a JSON file (e.g., `image_name.json`) containing:
*   **Classification**: Predicted class (e.g., "AORTIC_CALCIFICATION").
*   **PredictionAccuracy**: Confidence score.
*   **PredictionProbabilities**: Raw probabilities for all classes.
*   **DicomInfo**: Metadata extracted from the DICOM file (PatientID, StudyInstanceUID, etc.).
*   **PredictionDateTime**: Timestamp of the inference.

This JSON output is designed to be directly consumed by the Java FHIR Converter.

## Software Availability Statement

* **Source code available from**: [URL to Version Control System (such as GitHub)]
* **Archived software available from**: [DOI where archived source code can be accessed (such as Zenodo)]
* **License**: GNU General Public License v3.0 (GPL-3.0)

## Author

* **Maria Pisani** - *mpisani@ftgm.it*

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0) - see the LICENSE file for details.
