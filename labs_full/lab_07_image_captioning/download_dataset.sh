#!/bin/bash

# Lab 07: Image Captioning - Flickr8k Dataset Download Script

echo "=============================================="
echo "Lab 07: Image Captioning Dataset Download"
echo "=============================================="
echo ""
echo "Dataset: Flickr8k"
echo "Source: Kaggle"
echo "Size: ~1GB"
echo ""

# Check if kaggle is installed
if ! command -v kaggle &> /dev/null; then
    echo "Error: Kaggle CLI not found!"
    echo "Please install it with: pip install kaggle"
    echo "And configure your API credentials in ~/.kaggle/kaggle.json"
    exit 1
fi

# Create data directory
mkdir -p data/flickr8k

# Download dataset
echo "Downloading Flickr8k dataset..."
cd data/flickr8k

kaggle datasets download -d adityajn105/flickr8k

if [ $? -eq 0 ]; then
    echo "Download successful!"
    echo "Extracting dataset..."
    unzip -q flickr8k.zip
    rm flickr8k.zip
    echo "Dataset ready at: data/flickr8k/"
    echo "✓ Lab 07 dataset download complete!"
else
    echo "✗ Download failed. Please check your Kaggle credentials."
    exit 1
fi

# Made with Bob
