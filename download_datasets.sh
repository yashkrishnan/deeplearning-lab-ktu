#!/bin/bash

# Deep Learning Lab - Dataset Download Script
# This script downloads recommended Kaggle datasets for all labs

set -e  # Exit on error

echo "=========================================="
echo "  DEEP LEARNING LAB - DATASET DOWNLOADER"
echo "=========================================="
echo ""

# Check if Kaggle CLI is installed
if ! command -v kaggle &> /dev/null; then
    echo "❌ Error: Kaggle CLI is not installed"
    echo "Install it with: pip install kaggle"
    echo ""
    echo "Then configure your API credentials:"
    echo "1. Go to https://www.kaggle.com/account"
    echo "2. Create New API Token"
    echo "3. Place kaggle.json in ~/.kaggle/"
    exit 1
fi

# Check if Kaggle is configured
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "❌ Error: Kaggle API not configured"
    echo "Please download kaggle.json from https://www.kaggle.com/account"
    echo "and place it in ~/.kaggle/"
    exit 1
fi

echo "✓ Kaggle CLI is installed and configured"
echo ""

# Function to download and extract dataset
download_dataset() {
    local lab_name=$1
    local dataset_id=$2
    local target_dir=$3
    local description=$4
    
    echo "----------------------------------------"
    echo "📦 $lab_name: $description"
    echo "----------------------------------------"
    
    if [ -d "$target_dir" ] && [ "$(ls -A $target_dir)" ]; then
        echo "⚠️  Dataset already exists in $target_dir"
        read -p "Do you want to re-download? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "⏭️  Skipping..."
            echo ""
            return
        fi
        rm -rf "$target_dir"
    fi
    
    mkdir -p "$target_dir"
    cd "$target_dir"
    
    echo "⬇️  Downloading from Kaggle..."
    if kaggle datasets download -d "$dataset_id" --quiet; then
        echo "📂 Extracting..."
        unzip -q *.zip 2>/dev/null || echo "⚠️  No zip file to extract"
        rm -f *.zip
        echo "✅ Complete!"
    else
        echo "❌ Download failed"
        echo "   Try downloading manually from: https://www.kaggle.com/datasets/$dataset_id"
    fi
    
    cd - > /dev/null
    echo ""
}

# Ask user which datasets to download
echo "Which datasets would you like to download?"
echo ""
echo "1. Essential only (Labs 1-3, auto-downloaded)"
echo "2. Recommended (Labs 5-10, ~5 GB)"
echo "3. All datasets (~20 GB)"
echo "4. Custom selection"
echo ""
read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo ""
        echo "✓ Labs 1-3 use auto-downloaded datasets (CIFAR-10)"
        echo "✓ No manual downloads needed!"
        echo ""
        echo "Run the labs directly:"
        echo "  cd lab1_image_processing && python3 image_processing.py"
        echo "  cd lab2_cifar10_classifiers && python3 cifar10_classifiers.py"
        echo "  cd lab3_batchnorm_dropout && python3 batchnorm_dropout_study.py"
        exit 0
        ;;
    2)
        DOWNLOAD_RECOMMENDED=true
        ;;
    3)
        DOWNLOAD_ALL=true
        ;;
    4)
        CUSTOM_SELECTION=true
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "Starting downloads..."
echo ""

# Lab 4: Image Labeling Tools (Sample images)
if [ "$DOWNLOAD_ALL" = true ] || [ "$CUSTOM_SELECTION" = true ]; then
    read -p "Download Lab 4 sample images? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        download_dataset "Lab 4" "andrewmvd/road-sign-detection" \
            "lab4_labeling_tools/sample_images" "Road Sign Detection"
    fi
fi

# Lab 5: Image Segmentation
if [ "$DOWNLOAD_RECOMMENDED" = true ] || [ "$DOWNLOAD_ALL" = true ] || [ "$CUSTOM_SELECTION" = true ]; then
    if [ "$CUSTOM_SELECTION" = true ]; then
        read -p "Download Lab 5 segmentation dataset? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "⏭️  Skipping Lab 5"
            echo ""
        else
            download_dataset "Lab 5" "tanlikesmath/the-oxfordiiit-pet-dataset" \
                "lab5_segmentation/data/pets" "Oxford-IIIT Pet Dataset (~800 MB)"
        fi
    else
        download_dataset "Lab 5" "tanlikesmath/the-oxfordiiit-pet-dataset" \
            "lab5_segmentation/data/pets" "Oxford-IIIT Pet Dataset (~800 MB)"
    fi
fi

# Lab 6: Object Detection
if [ "$DOWNLOAD_RECOMMENDED" = true ] || [ "$DOWNLOAD_ALL" = true ] || [ "$CUSTOM_SELECTION" = true ]; then
    if [ "$CUSTOM_SELECTION" = true ]; then
        read -p "Download Lab 6 object detection dataset? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "⏭️  Skipping Lab 6"
            echo ""
        else
            echo "Note: Pascal VOC dataset may not be available directly"
            echo "Alternative: Download from http://host.robots.ox.ac.uk/pascal/VOC/"
        fi
    fi
fi

# Lab 7: Image Captioning
if [ "$DOWNLOAD_RECOMMENDED" = true ] || [ "$DOWNLOAD_ALL" = true ] || [ "$CUSTOM_SELECTION" = true ]; then
    if [ "$CUSTOM_SELECTION" = true ]; then
        read -p "Download Lab 7 image captioning dataset? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "⏭️  Skipping Lab 7"
            echo ""
        else
            download_dataset "Lab 7" "adityajn105/flickr8k" \
                "lab7_image_captioning/data/flickr8k" "Flickr8k Dataset (~1 GB)"
        fi
    else
        download_dataset "Lab 7" "adityajn105/flickr8k" \
            "lab7_image_captioning/data/flickr8k" "Flickr8k Dataset (~1 GB)"
    fi
fi

# Lab 8: Chatbot
if [ "$DOWNLOAD_RECOMMENDED" = true ] || [ "$DOWNLOAD_ALL" = true ] || [ "$CUSTOM_SELECTION" = true ]; then
    if [ "$CUSTOM_SELECTION" = true ]; then
        read -p "Download Lab 8 chatbot dataset? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "⏭️  Skipping Lab 8"
            echo ""
        else
            download_dataset "Lab 8" "rajathmc/cornell-moviedialog-corpus" \
                "lab8_chatbot/data/cornell" "Cornell Movie Dialogs (~10 MB)"
        fi
    else
        download_dataset "Lab 8" "rajathmc/cornell-moviedialog-corpus" \
            "lab8_chatbot/data/cornell" "Cornell Movie Dialogs (~10 MB)"
    fi
fi

# Lab 9: Time Series
if [ "$DOWNLOAD_RECOMMENDED" = true ] || [ "$DOWNLOAD_ALL" = true ] || [ "$CUSTOM_SELECTION" = true ]; then
    if [ "$CUSTOM_SELECTION" = true ]; then
        read -p "Download Lab 9 time series dataset? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "⏭️  Skipping Lab 9"
            echo ""
        else
            download_dataset "Lab 9" "robikscube/hourly-energy-consumption" \
                "lab9_time_series/data/energy" "Energy Consumption (~50 MB)"
        fi
    else
        download_dataset "Lab 9" "robikscube/hourly-energy-consumption" \
            "lab9_time_series/data/energy" "Energy Consumption (~50 MB)"
    fi
fi

# Lab 10: Seq2Seq
if [ "$DOWNLOAD_RECOMMENDED" = true ] || [ "$DOWNLOAD_ALL" = true ] || [ "$CUSTOM_SELECTION" = true ]; then
    if [ "$CUSTOM_SELECTION" = true ]; then
        read -p "Download Lab 10 seq2seq dataset? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "⏭️  Skipping Lab 10"
            echo ""
        else
            download_dataset "Lab 10" "dhruvildave/en-fr-translation-dataset" \
                "lab10_seq2seq/data/translation" "EN-FR Translation (~50 MB)"
        fi
    else
        download_dataset "Lab 10" "dhruvildave/en-fr-translation-dataset" \
            "lab10_seq2seq/data/translation" "EN-FR Translation (~50 MB)"
    fi
fi

echo "=========================================="
echo "  DOWNLOAD SUMMARY"
echo "=========================================="
echo ""
echo "✅ Dataset download complete!"
echo ""
echo "Downloaded datasets:"
find . -type d -name "data" -exec du -sh {} \; 2>/dev/null | grep -v "^0" || echo "  (Check individual lab directories)"
echo ""
echo "Next steps:"
echo "1. Navigate to a lab directory"
echo "2. Read the README.md for instructions"
echo "3. Run the lab program"
echo ""
echo "Example:"
echo "  cd lab7_image_captioning"
echo "  cat README.md"
echo ""
echo "Note: Labs 1-3 work without manual dataset downloads!"
echo "=========================================="


