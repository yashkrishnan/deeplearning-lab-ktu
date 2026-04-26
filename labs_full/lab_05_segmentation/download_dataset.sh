#!/bin/bash

# Lab 05: Image Segmentation - Oxford-IIIT Pet Dataset Download Script

echo "=================================================="
echo "Lab 05: Image Segmentation Dataset Download"
echo "=================================================="
echo ""
echo "Dataset: Oxford-IIIT Pet Dataset"
echo "Source: Kaggle"
echo "Size: ~800MB"
echo ""

# Check if kaggle is installed
if ! command -v kaggle &> /dev/null; then
    echo "Error: Kaggle CLI not found!"
    echo "Please install it with: pip install kaggle"
    echo "And configure your API credentials in ~/.kaggle/kaggle.json"
    exit 1
fi

# Create data directory
mkdir -p data/pets

# Download dataset
echo "Downloading Oxford-IIIT Pet Dataset..."
cd data/pets

kaggle datasets download -d tanlikesmath/the-oxfordiiit-pet-dataset

if [ $? -eq 0 ]; then
    echo "Download successful!"
    echo "Extracting dataset..."
    unzip -q the-oxfordiiit-pet-dataset.zip
    rm the-oxfordiiit-pet-dataset.zip
    echo "Dataset ready at: data/pets/"
    echo "✓ Lab 05 dataset download complete!"
else
    echo "✗ Download failed. Please check your Kaggle credentials."
    exit 1
fi

# Made with Bob
