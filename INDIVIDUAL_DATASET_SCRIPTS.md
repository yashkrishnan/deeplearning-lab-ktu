# Individual Dataset Download Scripts

## Overview

Each lab now has its own `download_dataset.sh` script located in its respective directory. This allows you to download datasets individually as needed.

## Script Locations

| Lab | Script Location | Dataset | Size | Source |
|-----|----------------|---------|------|--------|
| Lab 01 | `labs_full/lab_01_image_processing/download_dataset.sh` | Face Mask Detection | ~100MB | Kaggle |
| Lab 02 | `labs_full/lab_02_cifar10_classifiers/download_dataset.sh` | CIFAR-10 | ~170MB | Direct |
| Lab 03 | `labs_full/lab_03_batchnorm_dropout/download_dataset.sh` | CIFAR-10 | ~170MB | Direct |
| Lab 04 | `labs_full/lab_04_labeling_tools/download_dataset.sh` | Generated | N/A | Generated |
| Lab 05 | `labs_full/lab_05_segmentation/download_dataset.sh` | Oxford-IIIT Pets | ~800MB | Kaggle |
| Lab 06 | `labs_full/lab_06_object_detection/download_dataset.sh` | PASCAL VOC 2012 | ~2GB | Direct |
| Lab 07 | `labs_full/lab_07_image_captioning/download_dataset.sh` | Flickr8k | ~1GB | Kaggle |
| Lab 08 | `labs_full/lab_08_chatbot/download_dataset.sh` | Cornell Movie Dialogs | ~10MB | Direct |
| Lab 09 | `labs_full/lab_09_time_series/download_dataset.sh` | Energy Consumption | ~50MB | Kaggle |
| Lab 10 | `labs_full/lab_10_seq2seq/download_dataset.sh` | English-French | ~50MB | Kaggle |

## Usage

### Download Single Dataset

```bash
# Example: Download dataset for Lab 02
cd labs_full/lab_02_cifar10_classifiers
./download_dataset.sh
```

### Download Multiple Datasets

```bash
# Download datasets for labs 1, 2, and 5
cd labs_full/lab_01_image_processing && ./download_dataset.sh && cd ../..
cd labs_full/lab_02_cifar10_classifiers && ./download_dataset.sh && cd ../..
cd labs_full/lab_05_segmentation && ./download_dataset.sh && cd ../..
```

### Download All Datasets

```bash
# Run all download scripts from project root
for lab in labs_full/lab_*/download_dataset.sh; do
    (cd "$(dirname "$lab")" && ./download_dataset.sh)
done
```

Or use the existing combined script:
```bash
./download_all_datasets.sh
```

## Prerequisites

### For Kaggle Datasets (Labs 01, 05, 07, 09, 10)

1. **Install Kaggle CLI**:
   ```bash
   pip install kaggle
   ```

2. **Configure API Credentials**:
   - Go to https://www.kaggle.com/account
   - Click "Create New API Token"
   - Save `kaggle.json` to `~/.kaggle/`
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

### For Direct Downloads (Labs 02, 03, 06, 08)

- No special setup required
- Just need `wget` (pre-installed on most systems)

### For Generated Datasets (Lab 04)

- No download needed
- Images generated when running the lab script

## Script Details

### Lab 01: Face Mask Detection
- **Dataset**: Face Mask Detection from Kaggle
- **Size**: ~100MB
- **Contents**: Images with/without face masks
- **Use**: Image processing operations

### Lab 02 & 03: CIFAR-10
- **Dataset**: CIFAR-10 from Toronto
- **Size**: ~170MB each
- **Contents**: 60,000 32x32 color images in 10 classes
- **Use**: Classification and regularization studies

### Lab 04: Image Labeling
- **Dataset**: Generated sample images
- **Size**: N/A
- **Contents**: Created by labeling_demo.py
- **Use**: Annotation tool demonstration

### Lab 05: Oxford-IIIT Pets
- **Dataset**: Pet images with segmentation masks
- **Size**: ~800MB
- **Contents**: 37 pet breeds with pixel-level masks
- **Use**: Semantic segmentation with UNet

### Lab 06: PASCAL VOC 2012
- **Dataset**: Object detection benchmark
- **Size**: ~2GB
- **Contents**: Images with bounding box annotations
- **Use**: YOLO-style object detection

### Lab 07: Flickr8k
- **Dataset**: Images with captions
- **Size**: ~1GB
- **Contents**: 8,000 images with 5 captions each
- **Use**: Image captioning with CNN+RNN

### Lab 08: Cornell Movie Dialogs
- **Dataset**: Movie conversation corpus
- **Size**: ~10MB
- **Contents**: 220,000+ conversational exchanges
- **Use**: Chatbot training with BiLSTM

### Lab 09: Energy Consumption
- **Dataset**: Hourly energy usage data
- **Size**: ~50MB
- **Contents**: Time series data from multiple regions
- **Use**: LSTM/GRU forecasting

### Lab 10: English-French Translation
- **Dataset**: Parallel translation corpus
- **Size**: ~50MB
- **Contents**: English-French sentence pairs
- **Use**: Seq2Seq with attention mechanism

## Troubleshooting

### Kaggle Authentication Error
```
Error: Kaggle CLI not found!
```
**Solution**: Install kaggle CLI and configure credentials (see Prerequisites)

### Download Failed
```
✗ Download failed. Please check your internet connection.
```
**Solution**: 
- Check internet connection
- Verify source URL is accessible
- Try again later if server is down

### Permission Denied
```
bash: ./download_dataset_lab01.sh: Permission denied
```
**Solution**: Make script executable
```bash
chmod +x download_dataset_lab01.sh
```

### Disk Space Error
**Solution**: Check available disk space
```bash
df -h .
```
Ensure you have enough space for the dataset size listed above.

## Benefits of Individual Scripts

1. **Selective Downloads**: Download only what you need
2. **Faster Setup**: Skip large datasets you won't use
3. **Bandwidth Management**: Download during off-peak hours
4. **Easier Debugging**: Isolate download issues per lab
5. **Flexible Workflow**: Download as you progress through labs

## Integration with Web Interface

These scripts can be called from the Advanced tab in the web interface for UI-based dataset management.

## See Also

- `download_all_datasets.sh` - Download all datasets at once
- `download_datasets_interactive.sh` - Interactive selection menu
- `DATASETS.md` - Complete dataset documentation
- `DATASET_DOWNLOAD_STATUS.md` - Download status tracking