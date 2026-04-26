#!/bin/bash

# Lab 01: Image Processing - Face Mask Dataset Download Script
# Downloads the Face Mask Detection dataset from Kaggle

echo "=========================================="
echo "Lab 01: Image Processing Dataset Download"
echo "=========================================="
echo ""
echo "Dataset: Face Mask Detection"
echo "Source: Kaggle"
echo "Size: ~100MB"
echo ""

# Check if kaggle is installed
if ! command -v kaggle &> /dev/null; then
    echo "Error: Kaggle CLI not found!"
    echo "Please install it with: pip install kaggle"
    echo "And configure your API credentials in ~/.kaggle/kaggle.json"
    exit 1
fi

# Create data directory
mkdir -p data/sample_images

# Download dataset
echo "Downloading Face Mask Detection dataset..."
cd data/sample_images

kaggle datasets download -d omkargurav/face-mask-dataset

if [ $? -eq 0 ]; then
    echo "Download successful!"
    echo "Extracting dataset..."
    unzip -q face-mask-dataset.zip
    rm face-mask-dataset.zip
    echo "Dataset ready at: data/sample_images/"
    echo "✓ Lab 01 dataset download complete!"
else
    echo "✗ Download failed. Please check your Kaggle credentials."
    exit 1
fi

# Made with Bob
