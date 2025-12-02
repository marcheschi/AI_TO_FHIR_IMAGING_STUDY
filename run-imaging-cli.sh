#!/bin/bash
# Wrapper script for running Imaging Study CLI Converter

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JAR_PATH="$SCRIPT_DIR/target/imaging-study-cli.jar"

# Check if JAR exists
if [ ! -f "$JAR_PATH" ]; then
    echo "Error: JAR file not found at $JAR_PATH"
    echo "Please build the project first:"
    echo "  ./build-cli.sh"
    exit 1
fi

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <inputJsonFile> <outputFhirJsonFile> [fhirVersion]"
    echo ""
    echo "Arguments:"
    echo "  inputJsonFile       - Path to input imaging study JSON file"
    echo "  outputFhirJsonFile  - Path to output FHIR bundle JSON file"
    echo "  fhirVersion         - FHIR version (default: R5)"
    echo ""
    echo "Example:"
    echo "  $0 input/study.json output/fhir_bundle.json"
    echo "  $0 input/study.json output/fhir_bundle.json R5"
    exit 1
fi

# Run the converter
java -jar "$JAR_PATH" "$@"
exit_code=$?

# Report result
if [ $exit_code -eq 0 ]; then
    echo ""
    echo "✓ Conversion successful!"
    echo "Output: $2"
else
    echo ""
    echo "✗ Conversion failed with exit code: $exit_code"
fi

exit $exit_code
