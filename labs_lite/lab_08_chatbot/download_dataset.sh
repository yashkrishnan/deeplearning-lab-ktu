#!/bin/bash
# Download dataset for Lab 8 (Lite): Chatbot
# Copies Cornell Movie Dialogs from labs_full

echo "Lab 8 (Lite): Copying Cornell dialogs..."

# Create data directory
mkdir -p data/cornell

# Check if source exists
if [ -d "../../labs_full/lab_08_chatbot/data/cornell" ]; then
    # Copy dialog files (script will sample internally to 2000 pairs)
    cp ../../labs_full/lab_08_chatbot/data/cornell/*.txt data/cornell/ 2>/dev/null || true
    
    count=$(ls data/cornell/*.txt 2>/dev/null | wc -l)
    echo "✓ Copied $count dialog files to data/cornell/"
    echo "  (Script will sample to 2000 conversation pairs)"
else
    echo "⚠ Source not found at ../../labs_full/lab_08_chatbot/data/cornell/"
    echo "Please ensure labs_full datasets are downloaded first."
    exit 1
fi

echo "Dataset ready for Lab 8 (Lite)!"

# Made with Bob
