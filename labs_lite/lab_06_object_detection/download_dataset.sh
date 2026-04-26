#!/bin/bash
# Download dataset for Lab 6 (Lite): Object Detection
# Copies 500 VOC images with annotations from labs_full

echo "Lab 6 (Lite): Copying VOC dataset subset (500 images)..."

# Create data directories
mkdir -p data/voc/VOC2012/JPEGImages
mkdir -p data/voc/VOC2012/Annotations

# Check if source exists
if [ -d "../../labs_full/lab_06_object_detection/data/voc/VOC2012" ]; then
    # Copy first 500 images and annotations using find to avoid argument list too long
    find ../../labs_full/lab_06_object_detection/data/voc/VOC2012/JPEGImages -name "*.jpg" -type f | head -500 | while read img; do
        cp "$img" data/voc/VOC2012/JPEGImages/
        basename=$(basename "$img" .jpg)
        if [ -f "../../labs_full/lab_06_object_detection/data/voc/VOC2012/Annotations/${basename}.xml" ]; then
            cp "../../labs_full/lab_06_object_detection/data/voc/VOC2012/Annotations/${basename}.xml" data/voc/VOC2012/Annotations/
        fi
    done
    
    img_count=$(ls data/voc/VOC2012/JPEGImages/ | wc -l)
    ann_count=$(ls data/voc/VOC2012/Annotations/ | wc -l)
    echo "✓ Copied $img_count images and $ann_count annotations"
else
    echo "⚠ Source not found at ../../labs_full/lab_06_object_detection/data/voc/VOC2012/"
    echo "Please ensure labs_full datasets are downloaded first."
    exit 1
fi

echo "Dataset ready for Lab 6 (Lite)!"

# Made with Bob
