#!/bin/bash

# Interactive Dataset Download Script for Deep Learning Labs 1-10
# Features: Smart skip detection, progress tracking, user confirmation, error handling

set -e  # Exit on error

# Color codes for better visibility
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Summary tracking
declare -a DOWNLOADED_LABS
declare -a SKIPPED_LABS
declare -a FAILED_LABS

echo -e "${CYAN}=========================================="
echo "Deep Learning Labs - Dataset Downloader"
echo -e "==========================================${NC}"
echo ""
echo "This script will download datasets for Labs 1-10"
echo "Total estimated size: ~6-7 GB"
echo ""
echo -e "${YELLOW}⚠️  Requirements:${NC}"
echo "  - Kaggle API credentials at ~/.kaggle/kaggle.json"
echo "  - Sufficient disk space (~7 GB free)"
echo "  - Stable internet connection"
echo ""

# Check if kaggle is installed
if ! command -v kaggle &> /dev/null; then
    echo -e "${RED}❌ Error: Kaggle CLI not found${NC}"
    echo "Install with: pip install kaggle"
    exit 1
fi

# Check if kaggle credentials exist
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo -e "${RED}❌ Error: Kaggle credentials not found${NC}"
    echo "Please set up Kaggle API credentials at ~/.kaggle/kaggle.json"
    echo "See: https://github.com/Kaggle/kaggle-api#api-credentials"
    exit 1
fi

echo -e "${GREEN}✅ Kaggle CLI found and credentials configured${NC}"
echo ""

read -p "Press Enter to continue or Ctrl+C to cancel..."
echo ""

# Function to check if directory has files
check_existing_data() {
    local data_dir=$1
    if [ -d "$data_dir" ] && [ "$(ls -A $data_dir 2>/dev/null)" ]; then
        return 0  # Has files
    else
        return 1  # Empty or doesn't exist
    fi
}

# Function to get directory size
get_dir_size() {
    local dir=$1
    if [ -d "$dir" ]; then
        du -sh "$dir" 2>/dev/null | cut -f1
    else
        echo "0B"
    fi
}

# Function to download and extract dataset
download_dataset() {
    local lab_num=$1
    local lab_name=$2
    local lab_dir=$3
    local data_subdir=$4
    local kaggle_dataset=$5
    local estimated_size=$6
    local is_optional=$7
    
    local full_data_path="$lab_dir/data/$data_subdir"
    
    echo -e "${CYAN}=========================================="
    echo -e "Lab $lab_num: $lab_name"
    echo -e "==========================================${NC}"
    echo -e "Dataset: ${BLUE}$kaggle_dataset${NC}"
    echo -e "Estimated size: ${YELLOW}$estimated_size${NC}"
    if [ "$is_optional" = "true" ]; then
        echo -e "Status: ${YELLOW}Optional${NC}"
    fi
    echo ""
    
    # Check if data already exists
    if check_existing_data "$full_data_path"; then
        local current_size=$(get_dir_size "$full_data_path")
        echo -e "${GREEN}✅ Data already exists!${NC}"
        echo -e "Location: $full_data_path"
        echo -e "Current size: $current_size"
        echo ""
        
        read -p "Do you want to re-download? (y/N): " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${YELLOW}⏭️  Skipping Lab $lab_num${NC}"
            SKIPPED_LABS+=("Lab $lab_num: $lab_name (already exists - $current_size)")
            echo ""
            return 0
        fi
        
        echo -e "${YELLOW}🗑️  Cleaning existing files...${NC}"
        rm -rf "$full_data_path"/*
    fi
    
    # Ask for confirmation before downloading
    if [ "$is_optional" = "true" ]; then
        read -p "Download this optional dataset? (y/N): " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${YELLOW}⏭️  Skipping Lab $lab_num (optional)${NC}"
            SKIPPED_LABS+=("Lab $lab_num: $lab_name (skipped - optional)")
            echo ""
            return 0
        fi
    else
        read -p "Download Lab $lab_num dataset? (Y/n): " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Nn]$ ]]; then
            echo -e "${YELLOW}⏭️  Skipping Lab $lab_num${NC}"
            SKIPPED_LABS+=("Lab $lab_num: $lab_name (skipped by user)")
            echo ""
            return 0
        fi
    fi
    
    # Create directory structure
    echo -e "${BLUE}📁 Creating directory structure...${NC}"
    mkdir -p "$full_data_path"
    cd "$full_data_path"
    
    # Download dataset
    echo -e "${BLUE}⬇️  Downloading from Kaggle...${NC}"
    if kaggle datasets download "$kaggle_dataset" -q 2>&1 | grep -v "NotOpenSSLWarning"; then
        echo -e "${GREEN}✅ Download complete${NC}"
        
        # Extract zip files
        if ls *.zip 1> /dev/null 2>&1; then
            echo -e "${BLUE}📂 Extracting files...${NC}"
            for zipfile in *.zip; do
                echo "  Extracting: $zipfile"
                unzip -q "$zipfile"
            done
            echo -e "${GREEN}✅ Extraction complete (zip files preserved)${NC}"
        fi
        
        # Get final size
        cd - > /dev/null
        local final_size=$(get_dir_size "$full_data_path")
        local file_count=$(find "$full_data_path" -type f | wc -l | xargs)
        
        echo -e "${GREEN}✅ Lab $lab_num dataset ready!${NC}"
        echo -e "📁 Location: $full_data_path"
        echo -e "📊 Size: $final_size"
        echo -e "📄 Files: $file_count"
        
        DOWNLOADED_LABS+=("Lab $lab_num: $lab_name ($final_size, $file_count files)")
    else
        echo -e "${RED}❌ Failed to download Lab $lab_num dataset${NC}"
        FAILED_LABS+=("Lab $lab_num: $lab_name")
        cd - > /dev/null
        return 1
    fi
    
    echo ""
    return 0
}

# Lab 1: Image Processing (Optional)
download_dataset \
    "1" \
    "Image Processing - Face Mask Dataset" \
    "lab_01_image_processing" \
    "sample_images" \
    "ashishjangra27/face-mask-12k-images-dataset" \
    "~1 GB" \
    "true"

# Lab 2: CIFAR-10 (Auto-downloaded by script)
echo -e "${CYAN}=========================================="
echo -e "Lab 2: CIFAR-10 Classifiers"
echo -e "==========================================${NC}"
if check_existing_data "lab_02_cifar10_classifiers/data"; then
    size=$(get_dir_size "lab_02_cifar10_classifiers/data")
    echo -e "${GREEN}✅ Data already exists ($size)${NC}"
    echo -e "${YELLOW}ℹ️  This dataset is auto-downloaded by the lab script${NC}"
    SKIPPED_LABS+=("Lab 2: CIFAR-10 Classifiers (auto-downloaded - $size)")
else
    echo -e "${YELLOW}ℹ️  This dataset will be auto-downloaded when you run the lab script${NC}"
    SKIPPED_LABS+=("Lab 2: CIFAR-10 Classifiers (auto-downloaded by script)")
fi
echo ""

# Lab 3: Batch Normalization (Auto-downloaded by script)
echo -e "${CYAN}=========================================="
echo -e "Lab 3: Batch Normalization & Dropout"
echo -e "==========================================${NC}"
if check_existing_data "lab_03_batchnorm_dropout/data"; then
    size=$(get_dir_size "lab_03_batchnorm_dropout/data")
    echo -e "${GREEN}✅ Data already exists ($size)${NC}"
    echo -e "${YELLOW}ℹ️  This dataset is auto-downloaded by the lab script${NC}"
    SKIPPED_LABS+=("Lab 3: Batch Normalization & Dropout (auto-downloaded - $size)")
else
    echo -e "${YELLOW}ℹ️  This dataset will be auto-downloaded when you run the lab script${NC}"
    SKIPPED_LABS+=("Lab 3: Batch Normalization & Dropout (auto-downloaded by script)")
fi
echo ""

# Lab 4: Image Labeling Tools
download_dataset \
    "4" \
    "Image Labeling - Road Sign Detection" \
    "lab_04_labeling_tools" \
    "practice_images" \
    "andrewmvd/road-sign-detection" \
    "~100 MB" \
    "false"

# Lab 5: Image Segmentation
download_dataset \
    "5" \
    "Image Segmentation - Oxford-IIIT Pets" \
    "lab_05_segmentation" \
    "pets" \
    "tanlikesmath/the-oxfordiiit-pet-dataset" \
    "~800 MB" \
    "false"

# Lab 6: Object Detection
download_dataset \
    "6" \
    "Object Detection - Pascal VOC 2012" \
    "lab_06_object_detection" \
    "voc" \
    "huanghanchina/pascal-voc-2012" \
    "~3.6 GB" \
    "false"

# Lab 7: Image Captioning
download_dataset \
    "7" \
    "Image Captioning - Flickr8k" \
    "lab_07_image_captioning" \
    "flickr8k" \
    "adityajn105/flickr8k" \
    "~1 GB" \
    "false"

# Lab 8: Chatbot
download_dataset \
    "8" \
    "Chatbot - Cornell Movie Dialogs" \
    "lab_08_chatbot" \
    "cornell" \
    "rajathmc/cornell-moviedialog-corpus" \
    "~10 MB" \
    "false"

# Lab 9: Time Series
download_dataset \
    "9" \
    "Time Series - Hourly Energy Consumption" \
    "lab_09_time_series" \
    "energy" \
    "robikscube/hourly-energy-consumption" \
    "~50 MB" \
    "false"

# Lab 10: Seq2Seq
download_dataset \
    "10" \
    "Seq2Seq - EN-FR Translation" \
    "lab_10_seq2seq" \
    "translation" \
    "dhruvildave/en-fr-translation-dataset" \
    "~50 MB (extracts to ~7.8 GB)" \
    "false"

# Print Summary
echo -e "${CYAN}=========================================="
echo "Download Summary"
echo -e "==========================================${NC}"
echo ""

if [ ${#DOWNLOADED_LABS[@]} -gt 0 ]; then
    echo -e "${GREEN}✅ Successfully Downloaded (${#DOWNLOADED_LABS[@]}):${NC}"
    for lab in "${DOWNLOADED_LABS[@]}"; do
        echo -e "  ${GREEN}✓${NC} $lab"
    done
    echo ""
fi

if [ ${#SKIPPED_LABS[@]} -gt 0 ]; then
    echo -e "${YELLOW}⏭️  Skipped (${#SKIPPED_LABS[@]}):${NC}"
    for lab in "${SKIPPED_LABS[@]}"; do
        echo -e "  ${YELLOW}○${NC} $lab"
    done
    echo ""
fi

if [ ${#FAILED_LABS[@]} -gt 0 ]; then
    echo -e "${RED}❌ Failed (${#FAILED_LABS[@]}):${NC}"
    for lab in "${FAILED_LABS[@]}"; do
        echo -e "  ${RED}✗${NC} $lab"
    done
    echo ""
fi

# Calculate total downloaded size
echo -e "${CYAN}📊 Storage Usage:${NC}"
for lab_dir in lab_*/data; do
    if [ -d "$lab_dir" ]; then
        lab_name=$(dirname "$lab_dir")
        size=$(get_dir_size "$lab_dir")
        if [ "$size" != "0B" ]; then
            echo "  $lab_name: $size"
        fi
    fi
done
echo ""

echo -e "${GREEN}=========================================="
echo "Download Process Complete!"
echo -e "==========================================${NC}"
echo ""
echo -e "${CYAN}🎯 Next Steps:${NC}"
echo "  1. Check DATASET_DOWNLOAD_STATUS.md for details"
echo "  2. Run individual lab scripts to verify datasets"
echo "  3. Refer to each lab's README.md for usage instructions"
echo ""
echo -e "${YELLOW}💡 Tip:${NC} You can re-run this script anytime to download missing datasets"
echo ""


