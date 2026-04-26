#!/bin/bash

# Lab 08: Chatbot - Cornell Movie Dialogs Dataset Download Script

echo "========================================"
echo "Lab 08: Chatbot Dataset Download"
echo "========================================"
echo ""
echo "Dataset: Cornell Movie-Dialogs Corpus"
echo "Source: Direct download"
echo "Size: ~10MB"
echo ""

# Create data directory
mkdir -p data/cornell

# Download dataset
echo "Downloading Cornell Movie-Dialogs Corpus..."
cd data/cornell

wget -q --show-progress http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip

if [ $? -eq 0 ]; then
    echo "Download successful!"
    echo "Extracting dataset..."
    unzip -q cornell_movie_dialogs_corpus.zip
    mv cornell\ movie-dialogs\ corpus/* .
    rmdir cornell\ movie-dialogs\ corpus
    rm cornell_movie_dialogs_corpus.zip
    echo "Dataset ready at: data/cornell/"
    echo "✓ Lab 08 dataset download complete!"
else
    echo "✗ Download failed. Please check your internet connection."
    exit 1
fi

# Made with Bob
