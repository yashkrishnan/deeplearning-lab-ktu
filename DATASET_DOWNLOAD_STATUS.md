# Dataset Download Status

## Summary
Based on the Dataset Summary in DATASETS.md, here's the current status of datasets for each lab:

## ✅ Already Available
- **Lab 2**: CIFAR-10 Classifiers - ✅ Auto-downloaded (170 MB) - Present in `lab_02_cifar10_classifiers/data/`
- **Lab 3**: Batch Normalization & Dropout - ✅ Auto-downloaded (170 MB) - Present in `lab_03_batchnorm_dropout/data/`

## ✅ Successfully Downloaded
- **Lab 4**: Image Labeling Tools - ✅ Road Sign Detection dataset downloaded to `lab_04_labeling_tools/data/practice_images/`
- **Lab 5**: Image Segmentation - ✅ Oxford-IIIT Pet Dataset (800 MB) downloaded to `lab_05_segmentation/data/pets/`
- **Lab 8**: Chatbot - ✅ Cornell Movie Dialogs (41 MB) downloaded to `lab_08_chatbot/data/cornell/`
- **Lab 9**: Time Series - ✅ Hourly Energy Consumption (44 MB) downloaded to `lab_09_time_series/data/energy/`
- **Lab 10**: Seq2Seq - ✅ EN-FR Translation (7.8 GB) downloaded to `lab_10_seq2seq/data/translation/`

## ⏳ Pending Download
- **Lab 1**: Image Processing - Face Mask 12K Images (optional practice dataset) - `lab_01_image_processing/data/sample_images/`
- **Lab 6**: Object Detection - Pascal VOC 2012 (3.6 GB) - `lab_06_object_detection/data/voc/`
- **Lab 7**: Image Captioning - Flickr8k (1 GB) - `lab_07_image_captioning/data/flickr8k/`

## 📝 Download Instructions

### Option 1: Download All Labs (Complete Setup)
```bash
chmod +x download_complete_datasets.sh
./download_complete_datasets.sh
```

This will download datasets for ALL labs (1-10). Total size: ~6-7 GB
**Note**: Lab 1 dataset is optional but recommended for practice with real images.

### Option 2: Download Remaining Labs Only (6-10)
```bash
chmod +x download_remaining_datasets.sh
./download_remaining_datasets.sh
```

This will download Labs 6-10 sequentially. Total size: ~4.8 GB

### Option 3: Download Individual Labs
```bash
# Lab 1: Image Processing (optional practice dataset)
cd lab_01_image_processing && mkdir -p data/sample_images && cd data/sample_images
kaggle datasets download ashishjangra27/face-mask-12k-images-dataset
unzip face-mask-12k-images-dataset.zip && rm face-mask-12k-images-dataset.zip
cd ../../..

# Lab 6: Object Detection (3.6 GB)
cd lab_06_object_detection && mkdir -p data/voc && cd data/voc
kaggle datasets download huanghanchina/pascal-voc-2012
unzip pascal-voc-2012.zip && rm pascal-voc-2012.zip
cd ../../..

# Lab 7: Image Captioning (1 GB)
cd lab_07_image_captioning && mkdir -p data/flickr8k && cd data/flickr8k
kaggle datasets download adityajn105/flickr8k
unzip flickr8k.zip && rm flickr8k.zip
cd ../../..

# Lab 8: Chatbot (10 MB)
cd lab_08_chatbot && mkdir -p data/cornell && cd data/cornell
kaggle datasets download rajathmc/cornell-moviedialog-corpus
unzip cornell-moviedialog-corpus.zip && rm cornell-moviedialog-corpus.zip
cd ../../..

# Lab 9: Time Series (50 MB)
cd lab_09_time_series && mkdir -p data/energy && cd data/energy
kaggle datasets download robikscube/hourly-energy-consumption
unzip hourly-energy-consumption.zip && rm hourly-energy-consumption.zip
cd ../../..

# Lab 10: Seq2Seq (50 MB)
cd lab_10_seq2seq && mkdir -p data/translation && cd data/translation
kaggle datasets download dhruvildave/en-fr-translation-dataset
unzip en-fr-translation-dataset.zip && rm en-fr-translation-dataset.zip
cd ../../..
```

## 📊 Storage Requirements
- **Currently Used**: ~2.6 GB (Labs 2-5)
- **Lab 1 (Optional)**: ~1 GB (Face Mask dataset)
- **Remaining**: ~4.8 GB (Labs 6-10)
- **Total with Lab 1**: ~6-7 GB
- **Total without Lab 1**: ~5.8 GB

## ⚠️ Notes
1. Large downloads (especially Lab 6) may take time depending on internet speed
2. Ensure you have sufficient disk space (~5 GB free)
3. Kaggle API credentials must be configured at `~/.kaggle/kaggle.json`
4. Downloads can be interrupted and resumed by running the commands again

## 🔍 Verify Downloads
```bash
# Check all data directories
for lab in lab{4..10}*; do
    if [ -d "$lab/data" ]; then
        echo "$lab: $(du -sh $lab/data 2>/dev/null | cut -f1)"
    else
        echo "$lab: No data directory"
    fi
done
```

## 📚 Dataset Details
Refer to `DATASETS.md` for:
- Detailed dataset descriptions
- Alternative datasets
- Manual download instructions
- Troubleshooting tips