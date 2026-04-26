#!/bin/bash
# Download dataset for Lab 4 (Lite): Labeling Tools
# Copies 200 road sign images from labs_full

echo "Lab 4 (Lite): Copying road sign images (200 images)..."

# Create data directory
mkdir -p data/practice_images

# Check if source exists
if [ -d "../../labs_full/lab_04_labeling_tools/data/practice_images" ]; then
    # Copy first 200 images and their annotations
    ls ../../labs_full/lab_04_labeling_tools/data/practice_images/*.jpg ../../labs_full/lab_04_labeling_tools/data/practice_images/*.png 2>/dev/null | head -200 | while read img; do
        cp "$img" data/practice_images/
        basename=$(basename "$img")
        # Copy corresponding annotation files if they exist
        for ext in txt xml json; do
            annotation="${img%.*}.$ext"
            if [ -f "$annotation" ]; then
                cp "$annotation" data/practice_images/
            fi
        done
    done
    
    count=$(ls data/practice_images/*.jpg data/practice_images/*.png 2>/dev/null | wc -l)
    echo "✓ Copied $count images with annotations to data/practice_images/"
else
    echo "⚠ Source not found at ../../labs_full/lab_04_labeling_tools/data/practice_images/"
    echo "Please ensure labs_full datasets are downloaded first."
    exit 1
fi

echo "Dataset ready for Lab 4 (Lite)!"

# Made with Bob
