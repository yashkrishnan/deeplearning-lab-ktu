#!/bin/bash
# Download dataset for Lab 1 (Lite): Image Processing
# Copies 200 Face Mask images from labs_full

echo "Lab 1 (Lite): Copying Face Mask images (200 images)..."

# Create data directory
mkdir -p data/sample_images

# Check if source exists
if [ -d "../../labs_full/lab_01_image_processing/data/sample_images" ]; then
    # Copy first 200 images
    ls ../../labs_full/lab_01_image_processing/data/sample_images/*.jpg ../../labs_full/lab_01_image_processing/data/sample_images/*.png 2>/dev/null | head -200 | xargs -I {} cp {} data/sample_images/
    
    count=$(ls data/sample_images/ | wc -l)
    echo "✓ Copied $count images to data/sample_images/"
else
    echo "⚠ Source not found at ../../labs_full/lab_01_image_processing/data/sample_images/"
    echo "Please ensure labs_full datasets are downloaded first."
    exit 1
fi

echo "Dataset ready for Lab 1 (Lite)!"

# Made with Bob
