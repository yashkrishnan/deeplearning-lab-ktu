#!/bin/bash
# Download dataset for Lab 9 (Lite): Time Series
# Copies energy consumption data from labs_full

echo "Lab 9 (Lite): Copying energy consumption data..."

# Create data directory
mkdir -p data/energy

# Check if source exists
if [ -d "../../labs_full/lab_09_time_series/data/energy" ]; then
    # Copy CSV file (script will sample internally to 2000 steps)
    cp ../../labs_full/lab_09_time_series/data/energy/*.csv data/energy/ 2>/dev/null || true
    
    count=$(ls data/energy/*.csv 2>/dev/null | wc -l)
    echo "✓ Copied $count CSV file(s) to data/energy/"
    echo "  (Script will sample to 2000 time steps)"
else
    echo "⚠ Source not found at ../../labs_full/lab_09_time_series/data/energy/"
    echo "Please ensure labs_full datasets are downloaded first."
    exit 1
fi

echo "Dataset ready for Lab 9 (Lite)!"

# Made with Bob
