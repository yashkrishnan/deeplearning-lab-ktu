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
unzip face-mask-12k-images-dataset.zip -d dl-lab/lab1_image_processing/sample_images/
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
cd dl-lab/lab2_cifar10_classifiers
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
unzip road-sign-detection.zip -d dl-lab/lab4_labeling_tools/practice_images/

# Or use COCO sample
kaggle datasets download -d awsaf49/coco-2017-dataset
```

### Lab 5: Image Segmentation
**Recommended Datasets**:

#### 1. Carvana Image Masking (Car Segmentation)
```bash
cd dl-lab/lab5_segmentation
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
cd dl-lab/lab6_object_detection
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
cd dl-lab/lab7_image_captioning
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
cd dl-lab/lab8_chatbot
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
cd dl-lab/lab9_time_series
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
cd dl-lab/lab10_seq2seq
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
cd dl-lab/lab5_segmentation
mkdir -p data
kaggle datasets download -d tanlikesmath/the-oxfordiiit-pet-dataset
unzip -q the-oxfordiiit-pet-dataset.zip -d data/pets/
rm the-oxfordiiit-pet-dataset.zip

# Lab 6: Object Detection
echo "Downloading object detection datasets..."
cd ../lab6_object_detection
mkdir -p data
kaggle datasets download -d734b7bcb7ef13a045cbdd007a3c19b899d0c992b/pascal-voc-2012
unzip -q pascal-voc-2012.zip -d data/voc/
rm pascal-voc-2012.zip

# Lab 7: Image Captioning
echo "Downloading image captioning datasets..."
cd ../lab7_image_captioning
mkdir -p data
kaggle datasets download -d adityajn105/flickr8k
unzip -q flickr8k.zip -d data/flickr8k/
rm flickr8k.zip

# Lab 8: Chatbot
echo "Downloading chatbot datasets..."
cd ../lab8_chatbot
mkdir -p data
kaggle datasets download -d rajathmc/cornell-moviedialog-corpus
unzip -q cornell-moviedialog-corpus.zip -d data/cornell/
rm cornell-moviedialog-corpus.zip

# Lab 9: Time Series
echo "Downloading time series datasets..."
cd ../lab9_time_series
mkdir -p data
kaggle datasets download -d robikscube/hourly-energy-consumption
unzip -q hourly-energy-consumption.zip -d data/energy/
rm hourly-energy-consumption.zip

# Lab 10: Seq2Seq
echo "Downloading seq2seq datasets..."
cd ../lab10_seq2seq
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

| Lab | Dataset | Size | Auto-Download | Manual Download |
|-----|---------|------|---------------|-----------------|
| Lab 1 | Synthetic | - | ✅ Yes | - |
| Lab 2 | CIFAR-10 | 170 MB | ✅ Yes | Optional |
| Lab 3 | CIFAR-10 | 170 MB | ✅ Yes | Optional |
| Lab 4 | Sample Images | Varies | ❌ No | Required |
| Lab 5 | Pet Dataset | 800 MB | ❌ No | Recommended |
| Lab 6 | Pascal VOC | 2 GB | ❌ No | Recommended |
| Lab 7 | Flickr8k | 1 GB | ❌ No | Recommended |
| Lab 8 | Cornell Dialogs | 10 MB | ❌ No | Recommended |
| Lab 9 | Energy Data | 50 MB | ❌ No | Recommended |
| Lab 10 | EN-FR Translation | 50 MB | ❌ No | Recommended |

## 💾 Storage Requirements

**Minimum** (Labs 1-3 only): ~500 MB  
**Recommended** (All labs with small datasets): ~5 GB  
**Full** (All labs with large datasets): ~20 GB

## 🔍 Alternative Data Sources

### If Kaggle is not available:

1. **CIFAR-10**: Automatically downloaded by PyTorch
2. **ImageNet**: https://image-net.org/
3. **COCO**: https://cocodataset.org/
4. **Pascal VOC**: http://host.robots.ox.ac.uk/pascal/VOC/
5. **Flickr**: https://www.flickr.com/services/api/
6. **UCI ML Repository**: https://archive.ics.uci.edu/ml/

## 🎯 Quick Start

For immediate start without downloads:
- **Lab 1**: No dataset needed ✅
- **Lab 2**: Auto-downloads CIFAR-10 ✅
- **Lab 3**: Auto-downloads CIFAR-10 ✅
- **Labs 4-10**: Use sample data or follow README instructions

## 📝 Notes

1. **Kaggle API**: Required for automated downloads
2. **Storage**: Ensure sufficient disk space
3. **Network**: Large datasets require good internet connection
4. **Alternatives**: Most labs work with smaller sample datasets
5. **Privacy**: Some datasets require acceptance of terms

## 🆘 Troubleshooting

**Issue**: Kaggle API not authenticated
```bash
# Check if kaggle.json exists
ls ~/.kaggle/kaggle.json

# If not, download from Kaggle account settings
```

**Issue**: Download fails
```bash
# Try manual download from Kaggle website
# Or use alternative datasets
```

**Issue**: Insufficient storage
```bash
# Use smaller datasets
# Or download only required labs
```

## ✅ Verification

Check if datasets are downloaded:
```bash
# Check all data directories
find dl-lab -type d -name "data" -exec du -sh {} \;
```

---

**Note**: Most labs are designed to work with or without external datasets. Labs 1-3 are fully functional without any manual downloads.