# Quick Setup Guide

This guide will help you get started with the Imaging Study CLI Converter in 5 minutes.

## ⚡ 5-Minute Quick Start

### Step 1: Verify Prerequisites (30 seconds)

```bash
# Check Java version (must be 21+)
java -version

# Check Maven (must be 3.6+)
mvn -version
```

If you don't have Java 21 or Maven, install them first.

### Step 2: Navigate to Project (10 seconds)

```bash
cd imaging-study-cli-converter
```

### Step 3: Build the Project (2 minutes)

```bash
# Build using the provided script
./build-cli.sh

# Or manually with Maven
mvn clean package
```

This will create `target/imaging-study-cli.jar` (~10MB)

### Step 4: Test with Sample Data (30 seconds)

```bash
# Run with the included sample
./run-imaging-cli.sh examples/sample_input.json output.json

# Or directly with Java
java -jar target/imaging-study-cli.jar examples/sample_input.json output.json R5
```

### Step 5: Verify Output (30 seconds)

```bash
# Check that output was created
ls -lh output.json

# View the FHIR Bundle (requires jq)
cat output.json | jq '.resourceType'
# Should output: "Bundle"

# See all resources in the bundle
cat output.json | jq '.entry[].resource.resourceType'
# Should show: Patient, ImagingStudy, Device, DiagnosticReport, Library, Provenance
```

## ✅ You're Done!

The converter is now ready to use. Here's what you can do next:

### Use with Your Own Data

```bash
./run-imaging-cli.sh /path/to/your/input.json /path/to/output.json
```

### Deploy to Docker

```bash
# Build Docker image
docker build -t imaging-study-cli .

# Run with Docker
docker run -v $(pwd)/data:/data imaging-study-cli \
    /data/input.json /data/output.json R5
```

### Integrate with Mirth Connect

See `README.md` for Mirth Connect integration examples.

### Batch Processing

```bash
# Process all JSON files in a directory
for file in input/*.json; do
    filename=$(basename "$file" .json)
    ./run-imaging-cli.sh "$file" "output/${filename}_fhir.json"
done
```

### Train AI Models (Optional)

If you want to train your own AI models for cardiac pathologies detection:

```bash
cd Python

# Install Python dependencies
pip install -r requirements.txt

# Train with K-Fold Cross-Validation
python Hospital_training.py --data_dir /path/to/your/dataset --k_folds 5

# Or run a single training experiment
python Hospital_training.py --data_dir /path/to/your/dataset --mode single
```

### Run AI Inference (Optional)

To generate JSON input for the converter from new images:

```bash
cd Python

# Run inference on a DICOM file
python evaluate_cardiac_pathology.py \
    /path/to/image.dcm \
    /path/to/model.h5 \
    /path/to/config.json \
    -o /path/to/output_dir
```

See `Python/README.md` for detailed training and inference documentation.

## 📖 Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check `README.md` for common patterns
- Review [examples/sample_input.json](examples/sample_input.json) for input format
- See [CONTRIBUTING.md](CONTRIBUTING.md) if you want to contribute

## 🆘 Troubleshooting

### "Command not found: java"

Install Java 21:
```bash
# Ubuntu/Debian
sudo apt install openjdk-21-jdk

# Mac
brew install openjdk@21

# Windows
# Download from https://adoptium.net/
```

### "Command not found: mvn"

Install Maven:
```bash
# Ubuntu/Debian
sudo apt install maven

# Mac
brew install maven

# Windows
# Download from https://maven.apache.org/download.cgi
```

### "Permission denied: ./build-cli.sh"

Make scripts executable:
```bash
chmod +x build-cli.sh run-imaging-cli.sh
```

### Build Fails

Try cleaning and rebuilding:
```bash
mvn clean
mvn package
```

### "Missing required field" Error

Check your input JSON has:
- `Classification`
- `PredictionAccuracy`
- `DicomInfo.PatientID`
- `DicomInfo.PatientName` (or GivenName + FamilyName)

## 📞 Get Help

- Check the [README.md](README.md)
- Review `README.md`
- Look at [examples/sample_input.json](examples/sample_input.json)
- See [CONTRIBUTING.md](CONTRIBUTING.md) for reporting issues

---

**Happy Converting! 🎉**
