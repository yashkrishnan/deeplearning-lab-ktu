#!/bin/bash

# Lab 10: Sequence to Sequence - English-French Translation Dataset Download Script

echo "====================================================="
echo "Lab 10: Sequence to Sequence Dataset Download"
echo "====================================================="
echo ""
echo "Dataset: English-French Translation"
echo "Source: Kaggle"
echo "Size: ~50MB"
echo ""

# Check if kaggle is installed
if ! command -v kaggle &> /dev/null; then
    echo "Error: Kaggle CLI not found!"
    echo "Please install it with: pip install kaggle"
    echo "And configure your API credentials in ~/.kaggle/kaggle.json"
    exit 1
fi

# Create data directory
mkdir -p data/translation

# Download dataset
echo "Downloading English-French Translation dataset..."
cd data/translation

kaggle datasets download -d devicharith/language-translation-englishfrench

if [ $? -eq 0 ]; then
    echo "Download successful!"
    echo "Extracting dataset..."
    unzip -q language-translation-englishfrench.zip
    rm language-translation-englishfrench.zip
    echo "Dataset ready at: data/translation/"
    echo "✓ Lab 10 dataset download complete!"
else
    echo "✗ Download failed. Please check your Kaggle credentials."
    exit 1
fi

# Made with Bob
