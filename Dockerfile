FROM eclipse-temurin:21-jre-alpine

# Metadata
LABEL maintainer="Imaging Study CLI Converter"
LABEL description="Lightweight CLI tool for converting imaging study JSON to FHIR R5 bundles"
LABEL version="1.0.0"

# Set working directory
WORKDIR /app

# Copy the JAR file
COPY target/imaging-study-cli.jar /app/converter.jar

# Create directories for input/output
RUN mkdir -p /data/input /data/output

# Set the entrypoint
ENTRYPOINT ["java", "-jar", "/app/converter.jar"]

# Default command (shows usage)
CMD ["--help"]

# Usage:
# docker build -t imaging-study-cli .
# docker run -v $(pwd)/data:/data imaging-study-cli /data/input.json /data/output.json R5
