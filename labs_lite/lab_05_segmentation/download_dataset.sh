#!/bin/bash
# Download dataset for Lab 5 (Lite): Image Segmentation
# Copies 300 Pet images with trimaps from labs_full

echo "Lab 5 (Lite): Copying Pet dataset subset (300 images)..."

# Create data directories
mkdir -p data/pets/images
mkdir -p data/pets/annotations/trimaps

# Check if source exists
if [ -d "../../labs_full/lab_05_segmentation/data/pets/images" ]; then
    # Copy first 300 images and their corresponding trimaps
    ls ../../labs_full/lab_05_segmentation/data/pets/images/*.jpg 2>/dev/null | head -300 | while read img; do
        cp "$img" data/pets/images/
        basename=$(basename "$img" .jpg)
        if [ -f "../../labs_full/lab_05_segmentation/data/pets/annotations/trimaps/${basename}.png" ]; then
            cp "../../labs_full/lab_05_segmentation/data/pets/annotations/trimaps/${basename}.png" data/pets/annotations/trimaps/
        fi
    done
    
    img_count=$(ls data/pets/images/ | wc -l)
    mask_count=$(ls data/pets/annotations/trimaps/ | wc -l)
    echo "✓ Copied $img_count images and $mask_count trimaps"
else
    echo "⚠ Source not found at ../../labs_full/lab_05_segmentation/data/pets/"
    echo "Please ensure labs_full datasets are downloaded first."
    exit 1
fi

echo "Dataset ready for Lab 5 (Lite)!"

# Made with Bob
