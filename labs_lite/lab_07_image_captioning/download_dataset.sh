#!/bin/bash
# Download dataset for Lab 7 (Lite): Image Captioning
# Copies 1000 Flickr8k images with captions from labs_full

echo "Lab 7 (Lite): Copying Flickr8k dataset subset (1000 images)..."

# Create data directory
mkdir -p data/flickr8k/Images

# Check if source exists
if [ -d "../../labs_full/lab_07_image_captioning/data/flickr8k" ]; then
    # Copy first 1000 images
    ls ../../labs_full/lab_07_image_captioning/data/flickr8k/Images/*.jpg 2>/dev/null | head -1000 | xargs -I {} cp {} data/flickr8k/Images/
    
    # Copy captions file
    if [ -f "../../labs_full/lab_07_image_captioning/data/flickr8k/captions.txt" ]; then
        cp ../../labs_full/lab_07_image_captioning/data/flickr8k/captions.txt data/flickr8k/
        echo "✓ Copied captions.txt"
    fi
    
    count=$(ls data/flickr8k/Images/ | wc -l)
    echo "✓ Copied $count images to data/flickr8k/Images/"
else
    echo "⚠ Source not found at ../../labs_full/lab_07_image_captioning/data/flickr8k/"
    echo "Please ensure labs_full datasets are downloaded first."
    exit 1
fi

echo "Dataset ready for Lab 7 (Lite)!"

# Made with Bob
