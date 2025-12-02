# Imaging Study CLI Converter

[![Java](https://img.shields.io/badge/Java-21-orange.svg)](https://openjdk.org/)
[![FHIR](https://img.shields.io/badge/FHIR-R5-blue.svg)](https://hl7.org/fhir/R5/)
[![HAPI FHIR](https://img.shields.io/badge/HAPI%20FHIR-6.10.1-green.svg)](https://hapifhir.io/)
[![License](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

A lightweight, standalone command-line tool for converting imaging study JSON data (with AI predictions) to FHIR R5 bundles using HAPI FHIR.

## 🎯 Overview

This tool converts imaging study data with AI analysis results into HL7 FHIR R5 compliant bundles. It includes both a Java-based FHIR converter and a Python-based AI model training component for cardiac pathologies detection. Designed for:

- **Mirth Connect integration** - Easy to call from JavaScript channels
- **Batch processing** - Shell/PowerShell automation
- **Docker containers** - Small footprint (~10MB vs 170MB+)
- **CI/CD pipelines** - Fast startup, single-purpose tool
- **Embedded systems** - Minimal resource requirements

## ✨ Features

### FHIR Conversion
- ✅ Converts imaging study JSON to FHIR R5 Bundle
- ✅ Creates Patient resource from DICOM metadata
- ✅ Creates ImagingStudy with series and instances
- ✅ Creates Device resources (imaging equipment)
- ✅ Creates AI Device and DiagnosticReport for AI predictions
- ✅ Creates Provenance resource for AI workflow tracking
- ✅ Creates Library resource for training data provenance
- ✅ Validates JSON input structure
- ✅ Pretty-printed FHIR JSON output
- ✅ Minimal dependencies (only 4!)
- ✅ Fast startup (<1 second)
- ✅ Small JAR size (~10MB)

### AI Model Training
- ✅ CNN training for cardiac pathologies detection (aortic calcification)
- ✅ K-Fold Cross-Validation for robust evaluation
- ✅ Transfer Learning with pre-trained models (ResNet, VGG, MobileNet, etc.)
- ✅ Grad-CAM visualization for model explainability
- ✅ Comprehensive metrics (confusion matrices, ROC curves, classification reports)
- ✅ Support for custom CNN architectures
- ✅ Data augmentation and preprocessing
- ✅ Model comparison and performance analysis

## 📋 Requirements

### For FHIR Conversion
- **Java 21** or higher
- **Maven 3.6+** (for building)

### For AI Model Training
- **Python 3.8+** with TensorFlow 2.0+
- See `Python/requirements.txt` for complete dependencies

## 🚀 Quick Start

### Build

```bash
# Clone or download this repository
cd imaging-study-cli-converter

# Build the executable JAR
./build-cli.sh

# Or manually with Maven
mvn clean package
```

### Run

```bash
# Using the wrapper script (recommended)
./run-imaging-cli.sh examples/sample_input.json output.json

# Or directly with Java
java -jar target/imaging-study-cli.jar examples/sample_input.json output.json R5
```

### Verify

```bash
# Check the output
cat output.json | jq '.resourceType'
# Should output: "Bundle"

cat output.json | jq '.entry[].resource.resourceType'
# Should show: Patient, ImagingStudy, Device, DiagnosticReport, Library, Provenance
```

## 📖 Usage

### Basic Command

```bash
java -jar target/imaging-study-cli.jar <input.json> <output.json> [fhirVersion]
```

### Parameters

1. **input.json** (required) - Path to input imaging study JSON file
2. **output.json** (required) - Path to output FHIR bundle JSON file
3. **fhirVersion** (optional) - FHIR version (default: R5)

### Examples

```bash
# Basic conversion
java -jar target/imaging-study-cli.jar input/study.json output/fhir_bundle.json

# Explicit FHIR version
java -jar target/imaging-study-cli.jar input/study.json output/fhir_bundle.json R5

# Using wrapper script
./run-imaging-cli.sh input/study.json output/fhir_bundle.json
```

## 📝 Input JSON Format

The tool expects a JSON file with imaging study data and AI predictions:

```json
{
  "AIModelVersion": "ItalyHospital_Aortic_Calc_v1.0_20251014",
  "ProcessingTimestamp": "2025-10-17T13:21:00.741826+02:00",
  "TrainingDataVersion": "MIMIC-CXR-Dataset-v2.0",
  "Classification": "NORMAL",
  "PredictionAccuracy": 0.9636083841323853,
  "ClassificationEncoding": {
    "Labels": [0, 1],
    "ClassificationMap": {
      "0": "AORTIC_CALCIFICATION",
      "1": "NORMAL"
    }
  },
  "PredictionProbabilities": [0.03639, 0.96361],
  "DicomInfo": {
    "PatientName": "DOE^JOHN",
    "PatientID": "123456",
    "PatientBirthDate": "19740424",
    "PatientSex": "M",
    "StudyInstanceUID": "1.2.826.0.1.3680043.8.291...",
    "Modality": "DX",
    "AcquisitionDateTime": "2025-10-14T07:27:38.278000",
    ...
  }
}
```

See `examples/sample_input.json` for a complete example.

### Required Fields

- `Classification` - AI classification result
- `PredictionAccuracy` - Prediction confidence (0.0-1.0)
- `DicomInfo` - DICOM metadata object
  - `PatientID` - Patient identifier
  - `PatientName` OR (`PatientGivenName` + `PatientFamilyName`)

## 📦 Output FHIR Bundle

The tool generates a FHIR R5 Bundle (type: collection) containing:

1. **Patient** - Patient demographics from DICOM
2. **ImagingStudy** - Study, series, and instance information
3. **Device** (Imaging Equipment) - Scanner/modality device
4. **Device** (AI System) - AI model as a device
5. **DiagnosticReport** - AI classification results
6. **Library** - Training data provenance (if available)
7. **Provenance** - AI inference workflow tracking

All resources are properly linked with FHIR references.

## 🔌 Integration Examples

### Mirth Connect

```javascript
// In a Mirth Connect destination transformer
var inputJson = "/tmp/imaging_study.json";
var outputFhir = "/tmp/fhir_bundle.json";

var command = [
    "java", "-jar", 
    "/opt/converters/imaging-study-cli.jar",
    inputJson, outputFhir, "R5"
];

var process = java.lang.Runtime.getRuntime().exec(command);
var exitCode = process.waitFor();

if (exitCode === 0) {
    var fhirBundle = FileUtil.read(outputFhir);
    // Process FHIR bundle...
} else {
    logger.error("Conversion failed");
}
```

### Shell Script Batch Processing

```bash
#!/bin/bash
for file in input/*.json; do
    filename=$(basename "$file" .json)
    ./run-imaging-cli.sh "$file" "output/${filename}_fhir.json"
done
```

### Docker

```dockerfile
FROM eclipse-temurin:21-jre-alpine
WORKDIR /app
COPY target/imaging-study-cli.jar /app/
ENTRYPOINT ["java", "-jar", "/app/imaging-study-cli.jar"]
```

```bash
docker build -t imaging-study-cli .
docker run -v $(pwd)/data:/data imaging-study-cli \
    /data/input.json /data/output.json
```

## 📊 Performance

- **Startup time**: < 1 second
- **Conversion time**: ~100-500ms per file
- **Memory usage**: ~50-100MB heap
- **JAR size**: ~10MB (with dependencies)

## 🗂️ Project Structure

```
imaging-study-cli-converter/
├── README.md                           # This file
├── LICENSE                             # License file
├── pom.xml                             # Maven build configuration
├── build-cli.sh                        # Build script
├── run-imaging-cli.sh                  # Wrapper script
├── src/
│   └── main/
│       ├── java/com/fhirconverter/
│       │   ├── ImagingStudyConverter.java   # Core converter
│       │   └── ImagingStudyCli.java         # CLI entry point
│       └── resources/
│           └── simplelogger.properties      # Logging config
├── Python/                             # AI Model Training
│   ├── README.md                       # Training documentation
│   ├── requirements.txt                # Python dependencies
│   └── Hospital_training.py            # CNN training script
└── examples/
    └── sample_input.json                   # Example input
```



## 🔧 Dependencies

This CLI version uses minimal dependencies:

- **HAPI FHIR Base** (6.10.1) - Core FHIR functionality
- **HAPI FHIR R5 Structures** (6.10.1) - FHIR R5 resource models
- **Jackson Databind** (2.15.3) - JSON parsing
- **SLF4J Simple** (2.0.9) - Logging

Total dependency size: ~10MB

## 🐛 Troubleshooting

### Java Version Issues

```bash
# Check Java version
java -version
# Should show Java 21 or higher

# Install Java 21 if needed
# Ubuntu/Debian: sudo apt install openjdk-21-jdk
# Mac: brew install openjdk@21
# Windows: Download from https://adoptium.net/
```

### Memory Issues

```bash
# Increase heap size if needed
java -Xmx512m -jar target/imaging-study-cli.jar input.json output.json
```

### Validation Errors

The CLI tool does NOT include FHIR validation. To validate output:

```bash
# Use the official FHIR validator
java -jar validator_cli.jar output.json -version 5.0
```

## 🤖 AI Model Training

The project includes a complete Python-based CNN training pipeline for cardiac pathologies detection (specifically aortic calcification). This component trains the AI models whose predictions are then converted to FHIR by the Java converter.

### Features

- **K-Fold Cross-Validation** - Robust model evaluation with stratified splits
- **Transfer Learning** - Support for multiple pre-trained architectures:
  - ResNet50, ResNet152V2
  - VGG16, VGG19
  - MobileNetV2
  - DenseNet201
  - InceptionV3, InceptionResNetV2
  - Xception, NASNetLarge
- **Grad-CAM Visualization** - Explainability heatmaps showing model focus areas
- **Comprehensive Metrics** - Confusion matrices, ROC curves, classification reports
- **Custom CNN Architecture** - Lightweight custom model option
- **Data Augmentation** - Rotation, zoom, shift, and flip transformations

### Setup

```bash
cd Python

# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
# K-Fold Cross-Validation (default, 5 folds)
python Hospital_training.py --data_dir /path/to/dataset --k_folds 5

# Single training run (train/val/test split)
python Hospital_training.py --data_dir /path/to/dataset --mode single

# Transfer learning experiments (compares multiple models)
python Hospital_training.py --data_dir /path/to/dataset --mode transfer

# Custom parameters
python Hospital_training.py \
    --data_dir /path/to/dataset \
    --output_dir /path/to/results \
    --img_width 400 \
    --img_height 400 \
    --batch_size 20 \
    --epochs 100 \
    --mode kfold
```

### Dataset Structure

The training script expects images organized in subdirectories by class:

```
dataset/
├── AORTIC_CALCIFICATION/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── NORMAL/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

### Output

The training script generates:
- **Models** - Saved in `.h5` format (Keras/TensorFlow)
- **Metrics** - CSV files with accuracy, loss, and timing
- **Plots** - Training curves, confusion matrices, ROC curves
- **Grad-CAM** - Visualization heatmaps for model interpretability
- **Summary Reports** - Text summaries of training results

For complete documentation, see `Python/README.md`.

## 🧠 AI Inference

The project also includes an evaluation script (`evaluate_cardiac_pathology.py`) to run the trained models on new data and generate the JSON input required by the Java FHIR Converter.

### Features
- **DICOM Support**: Automatically extracts metadata from DICOM files
- **Lung Detection**: Uses computer vision to detect and crop lungs
- **Batch Processing**: Can process single files or entire directories
- **JSON Generation**: Outputs the exact JSON format needed for FHIR conversion

### Usage

```bash
cd Python

# Run inference on a single DICOM file
python evaluate_cardiac_pathology.py \
    /path/to/image.dcm \
    /path/to/model.h5 \
    /path/to/config.json \
    -o /path/to/output_dir
```

This will generate a JSON file in the output directory that can be passed to the Java CLI converter:

```bash
# Convert the AI result to FHIR
java -jar ../target/imaging-study-cli.jar \
    /path/to/output_dir/image.json \
    /path/to/output_dir/fhir_bundle.json
```

## 🔄 Exit Codes

- **0** - Success
- **1** - Invalid command-line arguments
- **2** - Conversion error (file I/O, JSON parsing, FHIR generation)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [HAPI FHIR](https://hapifhir.io/)
- Follows [HL7 FHIR R5](https://hl7.org/fhir/R5/) specification
- Designed for [Mirth Connect](https://www.nextgen.com/products-and-services/mirth-connect-integration-engine-downloads) integration

## 📧 Support

For issues or questions:
- Check the examples in `examples/` folder
- Open an issue on GitHub

## 🚀 Version History

- **1.0.0** (2025-12-01) - Initial release
  - Core imaging study conversion
  - AI provenance tracking
  - Minimal dependencies
  - Docker-ready
  - Mirth Connect integration examples

## 👨‍💻 Authors

**Paolo Marcheschi**  
Email: [paolo.marcheschi@ftgm.it](mailto:paolo.marcheschi@ftgm.it)

**Maria Pisani**  
Email: [mpisani@ftgm.it](mailto:mpisani@ftgm.it)

---

**Made with ❤️ for the FHIR community**
