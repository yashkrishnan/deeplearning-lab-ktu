# Dataset Guide for Deep Learning Labs

## 📊 Overview

This guide provides information on datasets for each lab, including Kaggle datasets and alternatives.

## 🔧 Setup Kaggle API

### Step 1: Install Kaggle CLI
```bash
pip install kaggle
```

### Step 2: Get Kaggle API Credentials
1. Go to https://www.kaggle.com/account
2. Scroll to "API" section
3. Click "Create New API Token"
4. Download `kaggle.json`
5. Place it in the correct location:

```bash
# On macOS/Linux
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# On Windows
mkdir %USERPROFILE%\.kaggle
move %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\
```

## 📁 Lab-Specific Datasets

### Lab 1: Image Processing
**Dataset**: Not required - generates synthetic images
**Alternative**: Use your own images

Optional datasets for practice:
```bash
# Sample images dataset
kaggle datasets download -d ashishjangra27/face-mask-12k-images-dataset
unzip face-mask-12k-images-dataset.zip -d dl-lab/lab_01_image_processing/sample_images/
```

### Lab 2: CIFAR-10 Classifiers
**Dataset**: CIFAR-10 (automatically downloaded by torchvision)
**Size**: ~170 MB
**Classes**: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)

The program automatically downloads CIFAR-10:
```python
# Automatic download in code
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
```

Manual download (optional):
```bash
cd dl-lab/lab_02_cifar10_classifiers
mkdir -p data
# Dataset will be downloaded automatically on first run
```

### Lab 3: Batch Normalization & Dropout
**Dataset**: CIFAR-10 (same as Lab 2)
**Size**: ~170 MB

Uses the same CIFAR-10 dataset, automatically downloaded.

### Lab 4: Image Labeling Tools
**Dataset**: Sample images for annotation practice

Recommended datasets:
```bash
# Object detection practice
kaggle datasets download -d andrewmvd/road-sign-detection
unzip road-sign-detection.zip -d dl-lab/lab_04_labeling_tools/practice_images/

# Or use COCO sample
kaggle datasets download -d awsaf49/coco-2017-dataset
```

### Lab 5: Image Segmentation
**Recommended Datasets**:

#### 1. Carvana Image Masking (Car Segmentation)
```bash
cd dl-lab/lab_05_segmentation
kaggle competitions download -c carvana-image-masking-challenge
unzip carvana-image-masking-challenge.zip -d data/carvana/
```
- **Size**: ~5 GB
- **Type**: Binary segmentation
- **Use**: Car vs background

#### 2. Oxford-IIIT Pet Dataset
```bash
kaggle datasets download -d tanlikesmath/the-oxfordiiit-pet-dataset
unzip the-oxfordiiit-pet-dataset.zip -d data/pets/
```
- **Size**: ~800 MB
- **Type**: Semantic segmentation
- **Use**: Pet segmentation

#### 3. Cityscapes (Street Scenes)
```bash
# Requires registration on Cityscapes website
# Alternative: Use subset
kaggle datasets download -d dansbecker/cityscapes-image-pairs
unzip cityscapes-image-pairs.zip -d data/cityscapes/
```

### Lab 6: Object Detection
**Recommended Datasets**:

#### 1. COCO Dataset (Subset)
```bash
cd dl-lab/lab_06_object_detection
kaggle datasets download -d awsaf49/coco-2017-dataset
unzip coco-2017-dataset.zip -d data/coco/
```
- **Size**: ~25 GB (full), ~5 GB (subset)
- **Classes**: 80 object categories
- **Format**: COCO JSON

#### 2. Pascal VOC
```bash
kaggle datasets download -d734b7bcb7ef13a045cbdd007a3c19b899d0c992b/pascal-voc-2012
unzip pascal-voc-2012.zip -d data/voc/
```
- **Size**: ~2 GB
- **Classes**: 20 object categories
- **Format**: XML annotations

#### 3. Open Images (Subset)
```bash
kaggle datasets download -d c1f8c3a7e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1/open-images-v6
# Or use smaller subset
kaggle datasets download -d google/open-images-dataset
```

### Lab 7: Image Captioning
**Recommended Datasets**:

#### 1. Flickr8k
```bash
cd dl-lab/lab_07_image_captioning
kaggle datasets download -d adityajn105/flickr8k
unzip flickr8k.zip -d data/flickr8k/
```
- **Size**: ~1 GB
- **Images**: 8,000
- **Captions**: 5 per image

#### 2. Flickr30k
```bash
kaggle datasets download -d hsankesara/flickr-image-dataset
unzip flickr-image-dataset.zip -d data/flickr30k/
```
- **Size**: ~4 GB
- **Images**: 30,000
- **Captions**: 5 per image

#### 3. MS COCO Captions
```bash
kaggle datasets download -d awsaf49/coco-2017-dataset
# Extract captions from annotations
```

### Lab 8: Chatbot
**Recommended Datasets**:

#### 1. Cornell Movie Dialogs
```bash
cd dl-lab/lab_08_chatbot
kaggle datasets download -d rajathmc/cornell-moviedialog-corpus
unzip cornell-moviedialog-corpus.zip -d data/cornell/
```
- **Size**: ~10 MB
- **Conversations**: 220,579
- **Type**: Movie dialogues

#### 2. Ubuntu Dialogue Corpus
```bash
kaggle datasets download -d rtatman/ubuntu-dialogue-corpus
unzip ubuntu-dialogue-corpus.zip -d data/ubuntu/
```
- **Size**: ~500 MB
- **Type**: Technical support conversations

#### 3. DailyDialog
```bash
kaggle datasets download -d csanhueza/the-reddit-climate-change-dataset
# Or search for "daily dialog dataset"
```

### Lab 9: Time Series Forecasting
**Recommended Datasets**:

#### 1. Stock Market Data
```bash
cd dl-lab/lab_09_time_series
kaggle datasets download -d borismarjanovic/price-volume-data-for-all-us-stocks-etfs
unzip price-volume-data-for-all-us-stocks-etfs.zip -d data/stocks/
```
- **Size**: ~800 MB
- **Type**: Stock prices
- **Features**: Open, High, Low, Close, Volume

#### 2. Energy Consumption
```bash
kaggle datasets download -d robikscube/hourly-energy-consumption
unzip hourly-energy-consumption.zip -d data/energy/
```
- **Size**: ~50 MB
- **Type**: Hourly energy usage
- **Period**: Multiple years

#### 3. Weather Data
```bash
kaggle datasets download -d selfishgene/historical-hourly-weather-data
unzip historical-hourly-weather-data.zip -d data/weather/
```
- **Size**: ~300 MB
- **Features**: Temperature, humidity, pressure, wind

#### 4. Sales Forecasting
```bash
kaggle datasets download -d c/rossmann-store-sales
unzip rossmann-store-sales.zip -d data/sales/
```

### Lab 10: Sequence to Sequence
**Recommended Datasets**:

#### 1. Machine Translation (English-French)
```bash
cd dl-lab/lab_10_seq2seq
kaggle datasets download -d dhruvildave/en-fr-translation-dataset
unzip en-fr-translation-dataset.zip -d data/translation/
```
- **Size**: ~50 MB
- **Pairs**: 135,842 sentence pairs

#### 2. Text Summarization (CNN/DailyMail)
```bash
kaggle datasets download -d gowrishankarp/newspaper-text-summarization-cnn-dailymail
unzip newspaper-text-summarization-cnn-dailymail.zip -d data/summarization/
```
- **Size**: ~500 MB
- **Articles**: 300,000+

#### 3. Multi30k (Multilingual)
```bash
kaggle datasets download -d google/multi30k-dataset
unzip multi30k-dataset.zip -d data/multi30k/
```

## 🚀 Quick Download Script

Create a script to download all datasets:

```bash
#!/bin/bash
# download_datasets.sh

echo "Downloading datasets for Deep Learning Labs..."

# Lab 2 & 3: CIFAR-10 (auto-downloaded)
echo "✓ CIFAR-10 will be downloaded automatically"

# Lab 5: Segmentation
echo "Downloading segmentation datasets..."
cd dl-lab/lab_05_segmentation
mkdir -p data
kaggle datasets download -d tanlikesmath/the-oxfordiiit-pet-dataset
unzip -q the-oxfordiiit-pet-dataset.zip -d data/pets/
rm the-oxfordiiit-pet-dataset.zip

# Lab 6: Object Detection
echo "Downloading object detection datasets..."
cd ../lab_06_object_detection
mkdir -p data
kaggle datasets download -d734b7bcb7ef13a045cbdd007a3c19b899d0c992b/pascal-voc-2012
unzip -q pascal-voc-2012.zip -d data/voc/
rm pascal-voc-2012.zip

# Lab 7: Image Captioning
echo "Downloading image captioning datasets..."
cd ../lab_07_image_captioning
mkdir -p data
kaggle datasets download -d adityajn105/flickr8k
unzip -q flickr8k.zip -d data/flickr8k/
rm flickr8k.zip

# Lab 8: Chatbot
echo "Downloading chatbot datasets..."
cd ../lab_08_chatbot
mkdir -p data
kaggle datasets download -d rajathmc/cornell-moviedialog-corpus
unzip -q cornell-moviedialog-corpus.zip -d data/cornell/
rm cornell-moviedialog-corpus.zip

# Lab 9: Time Series
echo "Downloading time series datasets..."
cd ../lab_09_time_series
mkdir -p data
kaggle datasets download -d robikscube/hourly-energy-consumption
unzip -q hourly-energy-consumption.zip -d data/energy/
rm hourly-energy-consumption.zip

# Lab 10: Seq2Seq
echo "Downloading seq2seq datasets..."
cd ../lab_10_seq2seq
mkdir -p data
kaggle datasets download -d dhruvildave/en-fr-translation-dataset
unzip -q en-fr-translation-dataset.zip -d data/translation/
rm en-fr-translation-dataset.zip

echo "✓ All datasets downloaded successfully!"
```

Save and run:
```bash
chmod +x download_datasets.sh
./download_datasets.sh
```

## 📊 Dataset Summary

| Lab | Dataset | Size (Estimated) | Actual Size | Status | Download Method |
|-----|---------|------------------|-------------|--------|-----------------|
| Lab 1 | Face Mask 12K | ~1 GB | 689 MB | ✅ Downloaded | Optional (Kaggle) |
| Lab 2 | CIFAR-10 | 170 MB | 340 MB | ✅ Downloaded | Auto (PyTorch) |
| Lab 3 | CIFAR-10 | 170 MB | 340 MB | ✅ Downloaded | Auto (PyTorch) |
| Lab 4 | Road Sign Detection | ~100 MB | 441 MB | ✅ Downloaded | Kaggle |
| Lab 5 | Oxford-IIIT Pets | ~800 MB | 3.0 GB | ✅ Downloaded | Kaggle |
| Lab 6 | Pascal VOC 2012 | ~3.6 GB | 7.5 GB | ✅ Downloaded | Kaggle |
| Lab 7 | Flickr8k | ~1 GB | 2.1 GB | ✅ Downloaded | Kaggle |
| Lab 8 | Cornell Dialogs | ~10 MB | 51 MB | ✅ Downloaded | Kaggle |
| Lab 9 | Energy Consumption | ~50 MB | 56 MB | ✅ Downloaded | Kaggle |
| Lab 10 | EN-FR Translation | ~50 MB | 10 GB | ✅ Downloaded | Kaggle |

## 💾 Storage Requirements

**Current Total Usage**: ~24.5 GB (all datasets downloaded)

**Breakdown by Lab**:
- Lab 1 (Face Mask): 689 MB
- Lab 2 (CIFAR-10): 340 MB
- Lab 3 (CIFAR-10): 340 MB
- Lab 4 (Road Signs): 441 MB
- Lab 5 (Pets): 3.0 GB
- Lab 6 (Pascal VOC): 7.5 GB
- Lab 7 (Flickr8k): 2.1 GB
- Lab 8 (Cornell): 51 MB
- Lab 9 (Energy): 56 MB
- Lab 10 (Translation): 10 GB

**Note**: Actual sizes are larger than estimated due to extracted files and multiple formats (images, annotations, etc.)

## 🔍 Alternative Data Sources

### If Kaggle is not available:

1. **CIFAR-10**: Automatically downloaded by PyTorch
2. **ImageNet**: https://image-net.org/
3. **COCO**: https://cocodataset.org/
4. **Pascal VOC**: http://host.robots.ox.ac.uk/pascal/VOC/
5. **Flickr**: https://www.flickr.com/services/api/
6. **UCI ML Repository**: https://archive.ics.uci.edu/ml/

## 🎯 Current Status

All datasets have been downloaded and are ready to use:
- **Lab 1**: Face Mask dataset (689 MB) ✅
- **Lab 2**: CIFAR-10 (340 MB) ✅
- **Lab 3**: CIFAR-10 (340 MB) ✅
- **Lab 4**: Road Sign Detection (441 MB) ✅
- **Lab 5**: Oxford-IIIT Pets (3.0 GB) ✅
- **Lab 6**: Pascal VOC 2012 (7.5 GB) ✅
- **Lab 7**: Flickr8k (2.1 GB) ✅
- **Lab 8**: Cornell Movie Dialogs (51 MB) ✅
- **Lab 9**: Energy Consumption (56 MB) ✅
- **Lab 10**: EN-FR Translation (10 GB) ✅

**Total Storage Used**: ~24.5 GB

## 📝 Notes

1. **All Datasets Downloaded**: All 10 labs now have their datasets ready
2. **Folder Structure**: Labs renamed to `lab_01_*`, `lab_02_*`, etc. for better organization
3. **Storage Used**: ~24.5 GB total across all labs
4. **Ready to Use**: All labs can now be run without additional downloads
5. **Kaggle API**: Was used for downloading datasets (credentials at ~/.kaggle/kaggle.json)

## 🆘 Troubleshooting

**Issue**: Need to re-download a dataset
```bash
# Remove the specific lab's data directory
rm -rf lab_XX_*/data/*

# Run the interactive download script
./download_datasets_interactive.sh
```

**Issue**: Check dataset integrity
```bash
# Verify all datasets are present
for lab in lab_*/data; do
    echo "$(dirname $lab): $(du -sh $lab 2>/dev/null | cut -f1)"
done
```

**Issue**: Running out of space
```bash
# Current usage: ~24.5 GB
# Consider removing optional Lab 1 dataset (689 MB) if needed
rm -rf lab_01_image_processing/data/*
```

## ✅ Verification

All datasets verified and ready:
```bash
# Current status (as of last check):
lab_01_image_processing: 689M ✅
lab_02_cifar10_classifiers: 340M ✅
lab_03_batchnorm_dropout: 340M ✅
lab_04_labeling_tools: 441M ✅
lab_05_segmentation: 3.0G ✅
lab_06_object_detection: 7.5G ✅
lab_07_image_captioning: 2.1G ✅
lab_08_chatbot: 51M ✅
lab_09_time_series: 56M ✅
lab_10_seq2seq: 10G ✅
```

---

**Note**: Most labs are designed to work with or without external datasets. Labs 1-3 are fully functional without any manual downloads.