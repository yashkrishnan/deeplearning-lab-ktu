# Lightweight Deep Learning Labs

Fast-running versions of all 10 deep learning labs optimized for training on regular laptops without GPU. **All labs now use real datasets** instead of synthetic data for authentic learning experience.

## Quick Start

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Navigate to labs_lite
cd labs_lite

# 3. Run a specific lab
cd lab_01_image_processing
python3 image_processing_lite.py

# Or run all labs at once
./run_all_labs.sh
```

## What's New in Version 2.0

### Real Datasets
All labs now use **real-world datasets** instead of synthetic data:
- Lab 1: Face Mask images
- Lab 2-3: CIFAR-10 (auto-download)
- Lab 4: Road sign detection images
- Lab 5: Oxford-IIIT Pet dataset
- Lab 6: Pascal VOC 2012
- Lab 7: Flickr8k image captions
- Lab 8: Cornell Movie Dialogs
- Lab 9: AEP Energy consumption data
- Lab 10: English-French translations

See [../docs/DATASETS.md](../docs/DATASETS.md) for complete dataset details.

## Lab Overview

| Lab | Topic | Dataset | Samples | Runtime | Status |
|-----|-------|---------|---------|---------|--------|
| 1 | Image Processing | Face Mask | 39 images | ~1-2s | ✅ Real Data |
| 2 | CIFAR-10 Classifiers | CIFAR-10 | 5,000 | ~2min | ✅ Real Data |
| 3 | BatchNorm & Dropout | CIFAR-10 | 10,000 | ~3min | ✅ Real Data |
| 4 | Labeling Tools | Road Signs | Multiple | ~3-5s | ✅ Real Data |
| 5 | Segmentation | Pet Images | 100 train | ~25s | ✅ Real Data |
| 6 | Object Detection | VOC 2012 | 200 train | ~2min | ✅ Real Data |
| 7 | Image Captioning | Flickr8k | 1,000 train | ~3min | ✅ Real Data |
| 8 | Chatbot | Cornell Dialogs | 2,000 pairs | ~2min | ✅ Real Data |
| 9 | Time Series | Energy Data | 2,000 steps | ~1min | ✅ Real Data |
| 10 | Seq2Seq Translation | EN-FR Pairs | 2,000 pairs | ~2min | ✅ Real Data |

**Total Runtime**: ~15-20 minutes for all labs

## Key Features

### Real-World Data
- Authentic datasets from research and industry
- Learn data preprocessing and handling
- Experience real-world challenges
- Portfolio-worthy results

### Optimized for Speed
- Reduced dataset sizes (10-20% of original)
- Fewer epochs (10 vs 30-50)
- Smaller batch sizes
- Simplified architectures
- CPU-optimized

### Educational Focus
- Core concepts preserved
- Clear code structure
- Comprehensive comments
- Visual outputs
- Performance metrics

## Prerequisites

### System Requirements
- **CPU**: Any modern processor (no GPU required)
- **RAM**: 4-8GB recommended
- **Storage**: ~5GB for datasets
- **OS**: macOS, Linux, or Windows

### Software Requirements
```bash
# Python 3.8+
python3 --version

# Virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dataset Requirements
Datasets are referenced from the main lab directories. Ensure datasets are downloaded:

```bash
# Check dataset availability
ls ../labs_full/lab_01_image_processing/data/sample_images/  # Lab 1
ls ../labs_full/lab_04_labeling_tools/data/practice_images/  # Lab 4
ls ../labs_full/lab_05_segmentation/data/pets/               # Lab 5
ls ../labs_full/lab_06_object_detection/data/voc/VOC2012/    # Lab 6
ls ../labs_full/lab_07_image_captioning/data/flickr8k/       # Lab 7
ls ../labs_full/lab_08_chatbot/data/cornell/                 # Lab 8
ls ../labs_full/lab_09_time_series/data/energy/              # Lab 9
ls ../labs_full/lab_10_seq2seq/data/translation/             # Lab 10
```

Labs 2 and 3 auto-download CIFAR-10 via torchvision.

## Running the Labs

### Option 1: Run All Labs
```bash
cd labs_lite
./run_all_labs.sh
```

This will:
- Run all 10 labs sequentially
- Save outputs to each lab's `output/` directory
- Display timing and performance metrics
- Generate a summary report

### Option 2: Run Individual Labs
```bash
cd labs_lite/lab_XX_name
python3 name_lite.py
```

Example:
```bash
cd labs_lite/lab_06_object_detection
python3 object_detection_lite.py
```

### Option 3: Run Specific Labs
```bash
# Run labs 1-5 only
for i in {1..5}; do
  cd lab_0${i}_*
  python3 *_lite.py
  cd ..
done
```

## Output Structure

Each lab generates outputs in its `output/` directory:

```
labs_lite/
├── lab_01_image_processing/
│   └── output/
│       ├── 0_original_image.png
│       ├── 1_histogram_equalization.png
│       └── ...
├── lab_02_cifar10_classifiers/
│   └── output/
│       ├── classifier_comparison.png
│       ├── confusion_matrix_*.png
│       └── training_history.png
└── ...
```

## Performance Expectations

### Training Time
- **Fast** (< 2 min): Labs 1, 4
- **Medium** (2-6 min): Labs 2, 3, 5, 9
- **Longer** (6-12 min): Labs 6, 7, 8, 10

### Memory Usage
- **Low** (< 2GB): Labs 1, 2, 3, 4
- **Medium** (2-4GB): Labs 5, 6, 9
- **Higher** (4-6GB): Labs 7, 8, 10

### Accuracy
- Expect 5-15% lower accuracy vs full datasets
- Core concepts and learning objectives preserved
- Sufficient for educational purposes

## Data Reduction Strategies

### 1. Sample Reduction
- Random sampling maintains distribution
- Typical reduction: 80-95% fewer samples

### 2. Vocabulary Reduction
- Keep most frequent words only
- Reduces model size and training time

### 3. Sequence Length Limiting
- Filter/truncate long sequences
- Faster processing, less memory

### 4. Image Resizing
- Smaller images (128x128 or 224x224)
- Maintains visual information

### 5. Epoch Reduction
- 10 epochs (from 30-50)
- Significantly faster training

## Troubleshooting

### Dataset Not Found
```bash
# Verify dataset exists
ls ../lab_XX_name/data/

# Check error message for exact path
# Ensure datasets were downloaded correctly
```

### Out of Memory
```python
# Reduce batch size in the script
BATCH_SIZE = 8  # Try smaller values: 4, 2, 1
```

### Slow Training
- Expected on CPU
- Consider running overnight for longer labs
- Close other applications to free resources
- GPU will significantly speed up training

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Or install specific package
pip install torch torchvision pandas
```

## Documentation

- **[../docs/GETTING_STARTED.md](../docs/GETTING_STARTED.md)**: Get started guide
- **[../docs/DATASETS.md](../docs/DATASETS.md)**: Complete dataset information
- **[../docs/DATASET_SETUP_GUIDE.md](../docs/DATASET_SETUP_GUIDE.md)**: Dataset download guide
- **[../docs/WEB_INTERFACE_GUIDE.md](../docs/WEB_INTERFACE_GUIDE.md)**: Web interface usage
- **[../labs_full/README.md](../labs_full/README.md)**: Full labs guide for comparison

## Comparison: Lite vs Original

| Aspect | Original Labs | Lite Labs |
|--------|--------------|-----------|
| **Dataset** | Full size | 10-20% sampled |
| **Epochs** | 30-50 | 10 |
| **Batch Size** | 32-64 | 8-32 |
| **Runtime** | 2-6 hours | 30-45 minutes |
| **GPU** | Recommended | Not required |
| **Memory** | 8-16GB | 4-8GB |
| **Accuracy** | Optimal | Good (5-15% lower) |
| **Learning** | Complete | Core concepts |

## Benefits of Real Datasets

1. **Authentic Experience**: Work with real-world data challenges
2. **Practical Skills**: Learn data preprocessing and handling
3. **Better Understanding**: See how models perform on actual data
4. **Portfolio Projects**: Results can be showcased
5. **Research Relevance**: Datasets used in academic research
6. **Problem Solving**: Debug real data issues
7. **Industry Preparation**: Similar to production scenarios

## Tips for Success

1. **Start Small**: Run Lab 1 first to verify setup
2. **Monitor Resources**: Watch CPU and memory usage
3. **Save Outputs**: Keep generated visualizations
4. **Experiment**: Try different hyperparameters
5. **Compare Results**: Run multiple labs to see patterns
6. **Read Code**: Understand implementation details
7. **Check Outputs**: Verify results make sense

## Next Steps

After completing the lite labs:

1. **Run Original Labs**: Try full datasets with GPU
2. **Experiment**: Modify hyperparameters
3. **Extend**: Add new features or models
4. **Compare**: Analyze lite vs original results
5. **Apply**: Use techniques on your own projects

## Contributing

Found an issue or have suggestions?
- Report bugs in the main repository
- Suggest optimizations
- Share your results
- Improve documentation

## License

Same as the main deep learning labs repository.

## Acknowledgments

- Original lab authors
- Dataset providers (VOC, Flickr8k, Cornell, etc.)
- PyTorch and TensorFlow communities
- Open source contributors

---

**Version**: 2.0 (Real Datasets)  
**Last Updated**: 2026-04-16  
**Maintained by**: Deep Learning Lab Team

For questions or issues, refer to the main repository documentation.