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
python Hospital_training_cleaned.py --data_dir /path/to/dataset --k_folds 5
```

### Single Split Training

```bash
python Hospital_training_cleaned.py --data_dir /path/to/dataset --mode single
```

### Transfer Learning Experiments

```bash
python Hospital_training_cleaned.py --data_dir /path/to/dataset --mode transfer
```

## Software Availability Statement

* **Source code available from**: [URL to Version Control System (such as GitHub)]
* **Archived software available from**: [DOI where archived source code can be accessed (such as Zenodo)]
* **License**: GNU General Public License v3.0 (GPL-3.0)

## Author

* **Maria Pisani** - *mpisani@ftgm.it*

## License

This project is licensed under the GNU GPL V 3.0 License - see the LICENSE file for details.
