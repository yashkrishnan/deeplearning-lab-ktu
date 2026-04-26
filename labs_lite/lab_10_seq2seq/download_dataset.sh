#!/bin/bash
# Download dataset for Lab 10 (Lite): Seq2Seq Translation
# Copies English-French translation data from labs_full

echo "Lab 10 (Lite): Copying translation data..."

# Create data directory
mkdir -p data/translation

# Check if source exists
if [ -d "../../labs_full/lab_10_seq2seq/data/translation" ]; then
    # Copy CSV file (script will sample internally to 2000 pairs)
    cp ../../labs_full/lab_10_seq2seq/data/translation/*.csv data/translation/ 2>/dev/null || true
    
    count=$(ls data/translation/*.csv 2>/dev/null | wc -l)
    echo "✓ Copied $count CSV file(s) to data/translation/"
    echo "  (Script will sample to 2000 translation pairs)"
else
    echo "⚠ Source not found at ../../labs_full/lab_10_seq2seq/data/translation/"
    echo "Please ensure labs_full datasets are downloaded first."
    exit 1
fi

echo "Dataset ready for Lab 10 (Lite)!"

# Made with Bob
