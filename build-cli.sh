#!/bin/bash
# Build script for Imaging Study CLI Converter

set -e  # Exit on error

echo "=========================================="
echo "Building Imaging Study CLI Converter"
echo "=========================================="
echo ""

# Check Java version
echo "Checking Java version..."
java -version 2>&1 | head -1
echo ""

# Check Maven
echo "Checking Maven..."
mvn -version | head -1
echo ""

# Clean and build
echo "Building with minimal dependencies..."
mvn clean package -f pom.xml

echo ""
echo "=========================================="
echo "Build Complete!"
echo "=========================================="
echo ""
echo "Executable JAR: target/imaging-study-cli.jar"
echo "Size: $(du -h target/imaging-study-cli.jar | cut -f1)"
echo ""
echo "Usage:"
echo "  java -jar target/imaging-study-cli.jar <input.json> <output.json> [R5]"
echo ""
echo "Or use the wrapper script:"
echo "  ./run-imaging-cli.sh <input.json> <output.json>"
echo ""
