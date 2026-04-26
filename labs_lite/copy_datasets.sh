#!/bin/bash
# Script to copy datasets from original labs to labs_lite

echo "Copying datasets to labs_lite directories..."
echo

# Lab 1 - Copy more Face Mask images
echo "Lab 1: Copying Face Mask images (200 images)..."
mkdir -p lab_01_image_processing/data/sample_images
if [ -d "../labs_full/lab_01_image_processing/data/sample_images" ]; then
    # Copy first 200 images
    ls ../labs_full/lab_01_image_processing/data/sample_images/*.jpg ../labs_full/lab_01_image_processing/data/sample_images/*.png 2>/dev/null | head -200 | xargs -I {} cp {} lab_01_image_processing/data/sample_images/
    echo "  ✓ Face Mask images copied (200 images)"
else
    echo "  ⚠ Source not found, skipping"
fi
echo

# Lab 2 - Uses CIFAR-10 (will be downloaded by script)
echo "Lab 2: CIFAR-10 dataset"
echo "  ℹ Dataset will be downloaded automatically when running the script"
echo

# Lab 3 - Uses CIFAR-10 (will be downloaded by script)
echo "Lab 3: CIFAR-10 dataset"
echo "  ℹ Dataset will be downloaded automatically when running the script"
echo

# Lab 4 - Copy more road sign images
echo "Lab 4: Copying road sign images and annotations (200 images)..."
mkdir -p lab_04_labeling_tools/data/practice_images
if [ -d "../labs_full/lab_04_labeling_tools/data/practice_images" ]; then
    # Copy first 200 images and their annotations
    ls ../labs_full/lab_04_labeling_tools/data/practice_images/*.jpg ../labs_full/lab_04_labeling_tools/data/practice_images/*.png 2>/dev/null | head -200 | while read img; do
        cp "$img" lab_04_labeling_tools/data/practice_images/
        basename=$(basename "$img")
        # Copy corresponding annotation files if they exist
        for ext in txt xml json; do
            annotation="${img%.*}.$ext"
            if [ -f "$annotation" ]; then
                cp "$annotation" lab_04_labeling_tools/data/practice_images/
            fi
        done
    done
    echo "  ✓ Road sign images and annotations copied (200 images)"
else
    echo "  ⚠ Source not found, skipping"
fi
echo

# Lab 5 - Copy more Pet images
echo "Lab 5: Copying Pet dataset subset (300 images)..."
mkdir -p lab_05_segmentation/data/pets/images
mkdir -p lab_05_segmentation/data/pets/annotations/trimaps
if [ -d "../labs_full/lab_05_segmentation/data/pets/images" ]; then
    # Copy first 300 images and their corresponding trimaps
    ls ../labs_full/lab_05_segmentation/data/pets/images/*.jpg | head -300 | while read img; do
        cp "$img" lab_05_segmentation/data/pets/images/
        basename=$(basename "$img" .jpg)
        if [ -f "../labs_full/lab_05_segmentation/data/pets/annotations/trimaps/${basename}.png" ]; then
            cp "../labs_full/lab_05_segmentation/data/pets/annotations/trimaps/${basename}.png" lab_05_segmentation/data/pets/annotations/trimaps/
        fi
    done
    echo "  ✓ Pet dataset subset copied (300 images)"
else
    echo "  ⚠ Source not found, skipping"
fi
echo

# Lab 6 - Copy more VOC images
echo "Lab 6: Copying VOC dataset subset (500 images)..."
mkdir -p lab_06_object_detection/data/voc/VOC2012/JPEGImages
mkdir -p lab_06_object_detection/data/voc/VOC2012/Annotations
if [ -d "../labs_full/lab_06_object_detection/data/voc/VOC2012" ]; then
    # Copy first 500 images and annotations using find to avoid argument list too long
    find ../labs_full/lab_06_object_detection/data/voc/VOC2012/JPEGImages -name "*.jpg" -type f | head -500 | while read img; do
        cp "$img" lab_06_object_detection/data/voc/VOC2012/JPEGImages/
        basename=$(basename "$img" .jpg)
        if [ -f "../labs_full/lab_06_object_detection/data/voc/VOC2012/Annotations/${basename}.xml" ]; then
            cp "../labs_full/lab_06_object_detection/data/voc/VOC2012/Annotations/${basename}.xml" lab_06_object_detection/data/voc/VOC2012/Annotations/
        fi
    done
    echo "  ✓ VOC dataset subset copied (500 images)"
else
    echo "  ⚠ Source not found, skipping"
fi
echo

# Lab 7 - Copy subset of Flickr8k dataset
echo "Lab 7: Copying Flickr8k dataset subset (1000 images)..."
mkdir -p lab_07_image_captioning/data/flickr8k/Images
if [ -d "../labs_full/lab_07_image_captioning/data/flickr8k" ]; then
    # Copy first 1000 images
    ls ../labs_full/lab_07_image_captioning/data/flickr8k/Images/*.jpg | head -1000 | xargs -I {} cp {} lab_07_image_captioning/data/flickr8k/Images/
    # Copy captions file
    if [ -f "../labs_full/lab_07_image_captioning/data/flickr8k/captions.txt" ]; then
        cp ../labs_full/lab_07_image_captioning/data/flickr8k/captions.txt lab_07_image_captioning/data/flickr8k/
    fi
    echo "  ✓ Flickr8k subset copied (1000 images)"
else
    echo "  ⚠ Source not found, skipping"
fi
echo

# Lab 8 - Copy subset of Cornell dialogs
echo "Lab 8: Copying Cornell dialogs subset (2000 pairs)..."
mkdir -p lab_08_chatbot/data/cornell
if [ -d "../labs_full/lab_08_chatbot/data/cornell" ]; then
    # Copy dialog files (script will sample internally)
    cp ../labs_full/lab_08_chatbot/data/cornell/*.txt lab_08_chatbot/data/cornell/ 2>/dev/null || true
    echo "  ✓ Cornell dialogs copied (will be sampled to 2000 pairs)"
else
    echo "  ⚠ Source not found, skipping"
fi
echo

# Lab 9 - Copy subset of energy data
echo "Lab 9: Copying energy consumption data subset (2000 steps)..."
mkdir -p lab_09_time_series/data/energy
if [ -d "../labs_full/lab_09_time_series/data/energy" ]; then
    # Copy CSV file (script will sample internally)
    cp ../labs_full/lab_09_time_series/data/energy/*.csv lab_09_time_series/data/energy/ 2>/dev/null || true
    echo "  ✓ Energy data copied (will be sampled to 2000 steps)"
else
    echo "  ⚠ Source not found, skipping"
fi
echo

# Lab 10 - Copy subset of translation data
echo "Lab 10: Copying translation dataset subset (2000 pairs)..."
mkdir -p lab_10_seq2seq/data/translation
if [ -d "../labs_full/lab_10_seq2seq/data/translation" ]; then
    # Copy CSV file (script will sample internally)
    cp ../labs_full/lab_10_seq2seq/data/translation/*.csv lab_10_seq2seq/data/translation/ 2>/dev/null || true
    echo "  ✓ Translation data copied (will be sampled to 2000 pairs)"
else
    echo "  ⚠ Source not found, skipping"
fi
echo

echo "Dataset copying complete!"
echo
echo "Summary:"
echo "  • Lab 1: Face Mask images (200 images)"
echo "  • Lab 2-3: CIFAR-10 (auto-download)"
echo "  • Lab 4: Road sign images (200 images)"
echo "  • Lab 5: Pet dataset subset (300 images)"
echo "  • Lab 6: VOC dataset subset (500 images)"
echo "  • Lab 7: Flickr8k subset (1000 images)"
echo "  • Lab 8: Cornell dialogs (2000 pairs)"
echo "  • Lab 9: Energy data (2000 steps)"
echo "  • Lab 10: Translation data (2000 pairs)"
echo
echo "Note: All datasets are subsets of the full datasets from labs_full/"


