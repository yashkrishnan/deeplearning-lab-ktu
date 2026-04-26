#!/bin/bash

# Lab 03: Batch Normalization & Dropout - Dataset Download Script
# Uses CIFAR-10 dataset (same as Lab 02)

echo "======================================================="
echo "Lab 03: Batch Normalization & Dropout Dataset Download"
echo "======================================================="
echo ""
echo "Dataset: CIFAR-10"
echo "Source: Direct download"
echo "Size: ~170MB"
echo ""

# Create data directory
mkdir -p data

# Download dataset
echo "Downloading CIFAR-10 dataset..."
cd data

wget -q --show-progress https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

if [ $? -eq 0 ]; then
    echo "Download successful!"
    echo "Extracting dataset..."
    tar -xzf cifar-10-python.tar.gz
    rm cifar-10-python.tar.gz
    echo "Dataset ready at: data/cifar-10-batches-py/"
    echo "✓ Lab 03 dataset download complete!"
else
    echo "✗ Download failed. Please check your internet connection."
    exit 1
fi

# Made with Bob
