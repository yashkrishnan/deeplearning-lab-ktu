#!/bin/bash

# Lab 06: Object Detection - PASCAL VOC 2012 Dataset Download Script

echo "================================================"
echo "Lab 06: Object Detection Dataset Download"
echo "================================================"
echo ""
echo "Dataset: PASCAL VOC 2012"
echo "Source: Direct download"
echo "Size: ~2GB"
echo ""

# Create data directory
mkdir -p data/voc

# Download dataset
echo "Downloading PASCAL VOC 2012 dataset..."
cd data/voc

wget -q --show-progress http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

if [ $? -eq 0 ]; then
    echo "Download successful!"
    echo "Extracting dataset..."
    tar -xf VOCtrainval_11-May-2012.tar
    rm VOCtrainval_11-May-2012.tar
    echo "Dataset ready at: data/voc/VOC2012/"
    echo "✓ Lab 06 dataset download complete!"
else
    echo "✗ Download failed. Please check your internet connection."
    exit 1
fi

# Made with Bob
