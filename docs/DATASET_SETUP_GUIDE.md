# Dataset Setup Guide - Step by Step

## Quick Summary

**Good News**: Labs 1-3 work immediately without any dataset downloads!
- Lab 1: Generates synthetic images
- Lab 2 & 3: Auto-download CIFAR-10 dataset

**For Labs 4-10**: Follow this guide to download datasets from Kaggle.

## Prerequisites

Before downloading datasets, you need:
1. Python 3.8+ installed
2. pip package manager
3. Kaggle account (free)
4. 5-20 GB free disk space (depending on datasets)

## Step-by-Step Setup

### Step 1: Install Dependencies

```bash
cd dl-lab
pip install -r requirements.txt
```

This installs:
- Kaggle CLI
- PyTorch, TensorFlow
- OpenCV, scikit-learn
- And all other required packages

**Verify installation:**
```bash
kaggle --version
# Should output: Kaggle API 1.5.x
```

### Step 2: Get Kaggle API Credentials

#### 2.1 Create Kaggle Account
- Go to https://www.kaggle.com
- Sign up for free (if you don't have an account)

#### 2.2 Generate API Token
1. Go to https://www.kaggle.com/account
2. Scroll down to "API" section
3. Click **"Create New API Token"**
4. This downloads `kaggle.json` file

#### 2.3 Install API Token

**On macOS/Linux:**
```bash
# Create .kaggle directory
mkdir -p ~/.kaggle

# Move the downloaded file
mv ~/Downloads/kaggle.json ~/.kaggle/

# Set correct permissions (important!)
chmod 600 ~/.kaggle/kaggle.json
```

**On Windows:**
```cmd
# Create .kaggle directory
mkdir %USERPROFILE%\.kaggle

# Move the downloaded file
move %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\
```

**Verify setup:**
```bash
kaggle datasets list
# Should show a list of datasets (not an error)
```

### Step 3: Download Datasets

Now you're ready to download datasets!

#### Option A: Interactive Script (Recommended)
```bash
cd dl-lab
./download_datasets.sh
```

You'll see:
```
==========================================
  DEEP LEARNING LAB - DATASET DOWNLOADER
==========================================

SUCCESS: Kaggle CLI is installed and configured

Which datasets would you like to download?

1. Essential only (Labs 1-3, auto-downloaded)
2. Recommended (Labs 5-10, ~5 GB)
3. All datasets (~20 GB)
4. Custom selection

Enter choice (1-4):
```

**Recommendations:**
- Choose **1** if you only want to run Labs 1-3 (no downloads needed)
- Choose **2** for a good balance (~5 GB, covers most labs)
- Choose **3** if you have space and want everything
- Choose **4** to select specific labs

#### Option B: Manual Download

Follow instructions in `DATASETS.md` for manual downloads.

### Step 4: Verify Downloads

Check what was downloaded:
```bash
cd dl-lab
find . -type d -name "data" -exec du -sh {} \;
```

You should see directories like:
```
800M    ./lab_05_segmentation/data
1.0G    ./lab_07_image_captioning/data
10M     ./lab_08_chatbot/data
50M     ./lab_09_time_series/data
50M     ./lab_10_seq2seq/data
```

## Running Labs After Setup

### Labs 1-3 (No Setup Needed)
```bash
# Lab 1: Image Processing
cd lab_01_image_processing
python3 image_processing.py

# Lab 2: CIFAR-10 Classifiers
cd ../lab_02_cifar10_classifiers
python3 cifar10_classifiers.py

# Lab 3: Batch Normalization & Dropout
cd ../lab_03_batchnorm_dropout
python3 batchnorm_dropout_study.py
```

### Labs 4-10 (After Dataset Download)
Each lab has a README.md with specific instructions:
```bash
cd lab_07_image_captioning
cat README.md  # Read instructions
# Then implement based on the guide
```

## Troubleshooting

### Issue 1: "Kaggle CLI is not installed"
```bash
pip install kaggle
# Or
pip install -r requirements.txt
```

### Issue 2: "Kaggle API not configured"
```bash
# Check if kaggle.json exists
ls ~/.kaggle/kaggle.json

# If not, download from https://www.kaggle.com/account
# Then move to ~/.kaggle/
```

### Issue 3: "Permission denied" on kaggle.json
```bash
chmod 600 ~/.kaggle/kaggle.json
```

### Issue 4: "Dataset download failed"
- Check internet connection
- Verify Kaggle credentials
- Try manual download from Kaggle website
- Some datasets may require accepting terms on Kaggle

### Issue 5: "Insufficient storage"
```bash
# Check available space
df -h

# Download only essential datasets (option 1 or 2)
# Or download one lab at a time (option 4)
```

### Issue 6: Download is slow
- Large datasets take time (be patient)
- Use option 2 (recommended) instead of option 3 (all)
- Download during off-peak hours

## Dataset Size Reference

| Lab | Dataset | Size | Required? |
|-----|---------|------|-----------|
| Lab 1 | Synthetic | - | Auto |
| Lab 2 | CIFAR-10 | 170 MB | Auto |
| Lab 3 | CIFAR-10 | 170 MB | Auto |
| Lab 4 | Road Signs | Varies | Optional |
| Lab 5 | Pet Dataset | 800 MB | Recommended |
| Lab 6 | Pascal VOC | 2 GB | Recommended |
| Lab 7 | Flickr8k | 1 GB | Recommended |
| Lab 8 | Cornell Dialogs | 10 MB | Recommended |
| Lab 9 | Energy Data | 50 MB | Recommended |
| Lab 10 | Translation | 50 MB | Recommended |

**Legend:**
- Auto: Downloaded automatically by the program
- Optional: Lab works without it (uses examples in README)
- Recommended: Needed for full implementation

## Quick Start Paths

### Path 1: Immediate Start (No Downloads)
```bash
cd dl-lab
pip install -r requirements.txt
cd lab_01_image_processing
python3 image_processing.py
```
**Time**: 5 minutes setup + 30 seconds execution

### Path 2: Essential Labs (Auto-Download)
```bash
cd dl-lab
pip install -r requirements.txt
./run_all_labs.sh
```
**Time**: 5 minutes setup + 8 minutes execution

### Path 3: Full Experience (With Datasets)
```bash
cd dl-lab
pip install -r requirements.txt
# Setup Kaggle (see Step 2)
./download_datasets.sh  # Choose option 2
# Then run individual labs
```
**Time**: 10 minutes setup + 30 minutes download + varies per lab

## Additional Resources

- **Kaggle API Docs**: https://github.com/Kaggle/kaggle-api
- **Dataset Guide**: See `DATASETS.md` in this directory
- **Getting Started**: See `GETTING_STARTED.md`
- **Lab Instructions**: Each lab has its own README.md

## Verification Checklist

Before running labs, verify:
- [ ] Python 3.8+ installed (`python3 --version`)
- [ ] Dependencies installed (`pip list | grep torch`)
- [ ] Kaggle CLI installed (`kaggle --version`)
- [ ] Kaggle credentials configured (`ls ~/.kaggle/kaggle.json`)
- [ ] Sufficient disk space (`df -h`)
- [ ] Internet connection (for downloads)

## You're Ready!

Once setup is complete:
1. Start with Lab 1 (no downloads needed)
2. Progress through Labs 2-3 (auto-download)
3. Download datasets for Labs 4-10 as needed
4. Follow each lab's README for specific instructions

**Happy Learning!**

---

**Need Help?**
- Check `DATASETS.md` for detailed dataset information
- Review `GETTING_STARTED.md` for general setup
- Read individual lab READMEs for specific instructions