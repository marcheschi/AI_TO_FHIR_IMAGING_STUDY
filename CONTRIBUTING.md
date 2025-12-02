# Contributing to Imaging Study CLI Converter

Thank you for your interest in contributing to the Imaging Study CLI Converter! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Reporting Bugs](#reporting-bugs)
- [Feature Requests](#feature-requests)

## Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow:

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/imaging-study-cli-converter.git
   cd imaging-study-cli-converter
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/imaging-study-cli-converter.git
   ```

## Development Setup

### Prerequisites

**For Java/FHIR Converter:**
- Java 21 or higher
- Maven 3.6+
- Git
- A Java IDE (IntelliJ IDEA, Eclipse, VS Code, etc.)

**For Python/AI Training:**
- Python 3.8 or higher
- TensorFlow 2.0+
- pip for package management
- (Optional) GPU support for faster training

### Build the Project

```bash
# Build the project
./build-cli.sh

# Or manually
mvn clean package

# Run tests
mvn test

# Run the converter
./run-imaging-cli.sh examples/sample_input.json output.json
```

### Project Structure

```
imaging-study-cli-converter/
├── src/main/java/com/fhirconverter/
│   ├── ImagingStudyConverter.java    # Core conversion logic
│   └── ImagingStudyCli.java          # CLI entry point
├── src/main/resources/
│   └── simplelogger.properties       # Logging configuration
├── Python/                            # AI Model Training & Inference
│   ├── README.md                     # Python documentation
│   ├── requirements.txt              # Python dependencies
│   ├── Hospital_training.py          # CNN training script
│   └── evaluate_cardiac_pathology.py # Inference script
├── examples/                          # Example files
└── pom.xml                            # Maven configuration
```

## How to Contribute

### Types of Contributions

We welcome various types of contributions:

1. **Bug Fixes** - Fix issues in the code (Java or Python)
2. **Features** - Add new functionality to converter or training
3. **Documentation** - Improve or add documentation
4. **Examples** - Add example input files or use cases
5. **Tests** - Add or improve test coverage
6. **Performance** - Optimize code performance
7. **AI Models** - Improve training algorithms or add new architectures
8. **Datasets** - Contribute training datasets (with proper licensing)

### Contribution Workflow

1. **Create a branch** for your work:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. **Make your changes** following the coding standards

3. **Test your changes**:
   ```bash
   mvn clean test
   ./run-imaging-cli.sh examples/sample_input.json test_output.json
   ```

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Brief description of changes"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request** on GitHub

## Coding Standards

### Java Code Style

- **Java Version**: Use Java 21 features where appropriate
- **Formatting**: Follow standard Java conventions
  - Indentation: 4 spaces (no tabs)
  - Line length: Max 120 characters
  - Braces: Opening brace on same line
- **Naming**:
  - Classes: PascalCase (e.g., `ImagingStudyConverter`)
  - Methods: camelCase (e.g., `createPatient`)
  - Constants: UPPER_SNAKE_CASE (e.g., `DICOM_ORG_ROOT`)
  - Variables: camelCase (e.g., `patientId`)

### Code Quality

- **Comments**: Add JavaDoc for public methods
- **Error Handling**: Use appropriate exception handling
- **Logging**: Use SLF4J for logging
  - DEBUG: Detailed diagnostic information
  - INFO: General informational messages
  - WARN: Warning messages
  - ERROR: Error messages

### Example JavaDoc

/**
 * Creates a Patient resource from DICOM patient information.
 * 
 * @param imagingData The JSON node containing imaging study data
 * @return A FHIR Patient resource
 * @throws IllegalArgumentException if required patient data is missing
 */
private Patient createPatient(JsonNode imagingData) {
    // Implementation
}
```

### Python Code Style

- **Python Version**: Use Python 3.8+ features
- **Formatting**: Follow PEP 8 conventions
  - Indentation: 4 spaces (no tabs)
  - Line length: Max 100 characters (120 for comments)
  - Use `black` formatter if possible
- **Naming**:
  - Classes: PascalCase (e.g., `CustomCNN`)
  - Functions: snake_case (e.g., `load_and_preprocess_images`)
  - Constants: UPPER_SNAKE_CASE (e.g., `SEED`, `DEFAULT_BATCH_SIZE`)
  - Variables: snake_case (e.g., `img_width`)
- **Type Hints**: Use type hints for function signatures where appropriate
- **Docstrings**: Use Google-style or NumPy-style docstrings

### Example Python Docstring

```python
def create_custom_model(img_width: int, img_height: int, num_classes: int, 
                       activation: str = 'softmax') -> keras.Model:
    """
    Creates a custom CNN model for image classification.
    
    Args:
        img_width: Target image width in pixels
        img_height: Target image height in pixels
        num_classes: Number of output classes
        activation: Activation function for output layer (default: 'softmax')
    
    Returns:
        Compiled Keras model ready for training
    """
    # Implementation
    pass
```

## Testing

### Running Tests

```bash
# Run all tests
mvn test

# Run specific test
mvn test -Dtest=ImagingStudyConverterTest

# Run with coverage
mvn clean test jacoco:report
```

### Writing Tests

- Place tests in `src/test/java/com/fhirconverter/`
- Use JUnit 5 for testing
- Test file naming: `*Test.java`
- Aim for good test coverage of new code

### Test Example

```java
@Test
void testCreatePatient() {
    // Given
    String jsonInput = "{ ... }";
    
    // When
    Patient patient = converter.createPatient(jsonInput);
    
    // Then
    assertNotNull(patient);
    assertEquals("123456", patient.getIdElement().getIdPart());
}
```

### Testing Python Code

```bash
# Test the training script with a small dataset
python Hospital_training.py \
    --data_dir test_data \
    --epochs 2 \
    --batch_size 4 \
    --mode single

# Verify model output format
python -c "
import tensorflow as tf
model = tf.keras.models.load_model('MODELS/single_run_model.h5')
print(model.summary())
"
```

**Python Testing Guidelines:**
- Test with small datasets and few epochs for quick validation
- Verify model architecture and output shapes
- Check that all output files are generated correctly
- Validate metrics files (CSV) can be parsed
- Test with different modes (kfold, single, transfer)

## Submitting Changes

### Pull Request Guidelines

1. **Title**: Clear, concise description of changes
   - Good: "Add support for FHIR R4 output"
   - Bad: "Updates"

2. **Description**: Include:
   - What changed and why
   - Related issue numbers (if any)
   - Testing performed
   - Breaking changes (if any)

3. **Commits**: 
   - Keep commits focused and atomic
   - Write clear commit messages
   - Reference issues in commits: "Fixes #123"

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] Tested with sample data
- [ ] Added/updated unit tests
- [ ] Verified FHIR output validity

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings introduced
```

## Reporting Bugs

### Before Reporting

1. Check existing issues
2. Verify you're using the latest version
3. Test with the sample input file

### Bug Report Template

```markdown
**Description**
Clear description of the bug

**To Reproduce**
Steps to reproduce:
1. Run command '...'
2. With input file '...'
3. See error

**Expected Behavior**
What you expected to happen

**Actual Behavior**
What actually happened

**Environment**
- OS: [e.g., Ubuntu 22.04]
- Java Version: [e.g., Java 21]
- CLI Version: [e.g., 1.0.0]

**Input File**
Attach or paste the input JSON (sanitize sensitive data)

**Error Output**
```
Paste error output here
```

**Additional Context**
Any other relevant information
```

## Feature Requests

### Feature Request Template

```markdown
**Feature Description**
Clear description of the proposed feature

**Use Case**
Why is this feature needed? What problem does it solve?

**Proposed Solution**
How should this feature work?

**Alternatives Considered**
Other approaches you've considered

**Additional Context**
Any other relevant information
```

## Development Tips

### Debugging

```bash
# Enable debug logging
java -Dorg.slf4j.simpleLogger.defaultLogLevel=debug \
    -jar target/imaging-study-cli.jar input.json output.json

# Run with Java debugger
java -agentlib:jdwp=transport=dt_socket,server=y,suspend=y,address=5005 \
    -jar target/imaging-study-cli.jar input.json output.json
```

### Performance Profiling

```bash
# Run with JVM profiling
java -XX:+PrintGCDetails -XX:+PrintGCTimeStamps \
    -jar target/imaging-study-cli.jar input.json output.json
```

### Code Review Checklist

Before submitting a PR, review:

- [ ] Code compiles without warnings
- [ ] All tests pass
- [ ] New code has tests
- [ ] Documentation is updated
- [ ] No sensitive data in commits
- [ ] Code follows style guidelines
- [ ] Commit messages are clear
- [ ] PR description is complete

## Resources

- [FHIR R5 Specification](https://hl7.org/fhir/R5/)
- [HAPI FHIR Documentation](https://hapifhir.io/hapi-fhir/docs/)
- [Java 21 Documentation](https://docs.oracle.com/en/java/javase/21/)
- [Maven Documentation](https://maven.apache.org/guides/)

## Questions?

If you have questions:

- Review existing issues
- Ask in a new issue with the "question" label

## License

By contributing, you agree that your contributions will be licensed under the GNU General Public License v3.0 (GPL-3.0).

---

Thank you for contributing to the Imaging Study CLI Converter! 🎉
