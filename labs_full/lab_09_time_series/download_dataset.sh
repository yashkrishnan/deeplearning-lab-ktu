#!/bin/bash

# Lab 09: Time Series Forecasting - Energy Consumption Dataset Download Script

echo "====================================================="
echo "Lab 09: Time Series Forecasting Dataset Download"
echo "====================================================="
echo ""
echo "Dataset: Hourly Energy Consumption"
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
mkdir -p data/energy

# Download dataset
echo "Downloading Hourly Energy Consumption dataset..."
cd data/energy

kaggle datasets download -d robikscube/hourly-energy-consumption

if [ $? -eq 0 ]; then
    echo "Download successful!"
    echo "Extracting dataset..."
    unzip -q hourly-energy-consumption.zip
    rm hourly-energy-consumption.zip
    echo "Dataset ready at: data/energy/"
    echo "✓ Lab 09 dataset download complete!"
else
    echo "✗ Download failed. Please check your Kaggle credentials."
    exit 1
fi

# Made with Bob
