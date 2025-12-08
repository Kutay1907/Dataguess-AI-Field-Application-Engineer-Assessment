#!/bin/bash

# Define image name
IMAGE_NAME="cv-advanced-assessment"

print_green() {
  echo -e "\033[0;32m$1\033[0m"
}

print_yellow() {
  echo -e "\033[0;33m$1\033[0m"
}

# Ensure Docker Buildx is available
if ! docker buildx version > /dev/null 2>&1; then
    echo "Error: Docker Buildx is not available. Please install/enable it (included in Docker Desktop)."
    exit 1
fi

print_yellow "Building Docker image for linux/amd64 (Intel) on current machine..."
print_yellow "This may take longer due to emulation."

# Build and Load (so it appears in local images, though running it might fail on Mac)
# Use --load to import into local Docker daemon, but warn that it won't run on M1
docker buildx build --platform linux/amd64 \
  -t ${IMAGE_NAME}:amd64 \
  -f api/docker/Dockerfile \
  --load .

if [ $? -eq 0 ]; then
    print_green "Build Successful!"
    print_green "Image Tag: ${IMAGE_NAME}:amd64"
    echo ""
    print_yellow "NOTE: This image is built for Intel processors."
    print_yellow "If you try to run it on your Mac (M1/M2), it might be slow or crash due to emulation."
    print_yellow "Push this image to your registry to deploy to Render/Intel Servers."
else
    echo "Build Failed."
    exit 1
fi
