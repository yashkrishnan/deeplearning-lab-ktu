# Full Deep Learning Labs

Complete, production-ready implementations of all 10 deep learning labs with full datasets and optimal training configurations. Designed for GPU-accelerated training to achieve state-of-the-art results.

## Quick Start

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Navigate to labs_full
cd labs_full

# 3. Download datasets (see Dataset Setup below)
cd lab_01_image_processing
./download_dataset.sh
cd ..

# 4. Run a specific lab
cd lab_01_image_processing
python3 image_processing.py

# Or run all labs at once
./run_all_labs.sh
```

## Lab Overview

| Lab | Topic | Dataset | Samples | Runtime (GPU) | Runtime (CPU) |
|-----|-------|---------|---------|---------------|---------------|
| 1 | Image Processing | Face Mask | 853 images | ~5-10s | ~30s |
| 2 | CIFAR-10 Classifiers | CIFAR-10 | 50,000 | ~15min | ~2-3hrs |
| 3 | BatchNorm & Dropout | CIFAR-10 | 50,000 | ~20min | ~3-4hrs |
| 4 | Labeling Tools | Road Signs | Multiple | ~10-15s | ~1min |
| 5 | Segmentation | Pet Images | 3,680 train | ~30min | ~4-6hrs |
| 6 | Object Detection | VOC 2012 | 5,717 train | ~1-2hrs | ~8-12hrs |
| 7 | Image Captioning | Flickr8k | 6,000 train | ~1-2hrs | ~6-10hrs |
| 8 | Chatbot | Cornell Dialogs | 220,579 pairs | ~1-2hrs | ~6-10hrs |
| 9 | Time Series | Energy Data | 19,735 steps | ~30min | ~2-3hrs |
| 10 | Seq2Seq Translation | EN-FR Pairs | 135,842 pairs | ~1-2hrs | ~6-10hrs |

**Total Runtime**: 
- **GPU**: ~6-10 hours
- **CPU**: ~40-60 hours (not recommended)

## Key Features

### Production-Ready
- Full datasets for optimal accuracy
- State-of-the-art architectures
- Comprehensive hyperparameter tuning
- Extensive training (30-50 epochs)
- Professional evaluation metrics

### Research Quality
- Reproducible results
- Detailed logging and checkpointing
- Visualization of training progress
- Model comparison and analysis
- Publication-ready outputs

### Educational Value
- Complete implementations
- Extensive documentation
- Best practices demonstrated
- Real-world applications
- Industry-standard techniques

## Prerequisites

### System Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM (highly recommended)
  - CUDA 11.0+ and cuDNN installed
  - For best results: RTX 3060 or better
- **CPU**: Multi-core processor (if no GPU)
- **RAM**: 16GB+ recommended (32GB for Labs 6-8)
- **Storage**: ~50GB for all datasets
- **OS**: Linux (recommended), macOS, or Windows with WSL2

### Software Requirements
```bash
# Python 3.8+
python3 --version

# Virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify GPU availability (PyTorch)
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Dataset Setup

Each lab has its own dataset download script located in the lab directory:

```bash
# Download individual lab datasets
cd labs_full/lab_01_image_processing
./download_dataset.sh

cd ../lab_02_cifar10_classifiers
# CIFAR-10 auto-downloads via torchvision

cd ../lab_05_segmentation
./download_dataset.sh

# And so on for each lab...
```

See [docs/DATASETS.md](../docs/DATASETS.md) and [docs/DATASET_SETUP_GUIDE.md](../docs/DATASET_SETUP_GUIDE.md) for complete dataset information.

## Running the Labs

### Option 1: Run All Labs
```bash
cd labs_full
./run_all_labs.sh
```

This will:
- Run all 10 labs sequentially
- Save outputs to each lab's `output/` directory
- Display timing and performance metrics
- Generate comprehensive reports
- Save model checkpoints

### Option 2: Run Individual Labs
```bash
cd labs_full/lab_XX_name
python3 name.py
```

Example:
```bash
cd labs_full/lab_06_object_detection
python3 object_detection.py
```

### Option 3: Use Web Interface
```bash
# From project root
cd web_interface
python3 app.py

# Open browser to http://localhost:5001
# Navigate to "Full Labs" tab
```

## Output Structure

Each lab generates comprehensive outputs:

```
labs_full/
├── lab_01_image_processing/
│   └── output/
│       ├── 0_original_image.png
│       ├── 1_histogram_equalization.png
│       ├── 2_thresholding.png
│       └── ...
├── lab_02_cifar10_classifiers/
│   └── output/
│       ├── classifier_comparison.png
│       ├── confusion_matrix_*.png
│       ├── training_history.png
│       ├── best_model_*.pth
│       └── training_log.txt
└── ...
```

## Performance Expectations

### Training Time (GPU)
- **Fast** (< 30 min): Labs 1, 4, 9
- **Medium** (30-60 min): Labs 2, 3, 5
- **Longer** (1-2 hrs): Labs 6, 7, 8, 10

### Training Time (CPU)
- Not recommended for Labs 2, 3, 5-10
- Expect 10-20x longer training times
- Consider using cloud GPU services

### Memory Requirements
- **Low** (< 4GB VRAM): Labs 1, 2, 3, 4
- **Medium** (4-8GB VRAM): Labs 5, 9
- **High** (8GB+ VRAM): Labs 6, 7, 8, 10

### Expected Accuracy
- Lab 2: 85-92% (CIFAR-10 classification)
- Lab 3: 88-94% (with BatchNorm/Dropout)
- Lab 5: 85-90% IoU (segmentation)
- Lab 6: 70-80% mAP (object detection)
- Lab 7: 15-25 BLEU (image captioning)
- Lab 8: 0.6-0.8 perplexity (chatbot)
- Lab 9: < 0.1 MSE (time series)
- Lab 10: 25-35 BLEU (translation)

## Training Strategies

### 1. Full Dataset Training
- Use all available data
- 30-50 epochs for convergence
- Learning rate scheduling
- Early stopping with patience

### 2. Data Augmentation
- Random crops and flips
- Color jittering
- Mixup/Cutmix (where applicable)
- Increases model robustness

### 3. Advanced Architectures
- ResNet, VGG for classification
- U-Net for segmentation
- Faster R-CNN for detection
- Transformer models for NLP

### 4. Hyperparameter Tuning
- Grid search or random search
- Learning rate: 1e-4 to 1e-2
- Batch size: 32-128 (GPU dependent)
- Optimizer: Adam, SGD with momentum

### 5. Model Checkpointing
- Save best model based on validation
- Resume training from checkpoints
- Model versioning and comparison

## Troubleshooting

### CUDA Out of Memory
```python
# Reduce batch size
BATCH_SIZE = 16  # Try: 8, 4, or even 2

# Enable gradient checkpointing
torch.utils.checkpoint.checkpoint(...)

# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler
```

### Slow Training
```bash
# Verify GPU is being used
nvidia-smi

# Check CUDA availability
python3 -c "import torch; print(torch.cuda.is_available())"

# Monitor GPU utilization
watch -n 1 nvidia-smi
```

### Dataset Download Issues
```bash
# For Kaggle datasets, ensure API credentials are set
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"

# Or place kaggle.json in ~/.kaggle/
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Import Errors
```bash
# Reinstall with GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify installation
python3 -c "import torch; print(torch.__version__)"
```

## Documentation

- **[docs/DATASETS.md](../docs/DATASETS.md)**: Complete dataset information
- **[docs/DATASET_SETUP_GUIDE.md](../docs/DATASET_SETUP_GUIDE.md)**: Dataset download guide
- **[docs/INSTALLATION_GUIDE.md](../docs/INSTALLATION_GUIDE.md)**: Setup instructions
- **[docs/GETTING_STARTED.md](../docs/GETTING_STARTED.md)**: Beginner's guide
- **[docs/WEB_INTERFACE_GUIDE.md](../docs/WEB_INTERFACE_GUIDE.md)**: Web interface usage
- **[docs/ADVANCED_TAB_GUIDE.md](../docs/ADVANCED_TAB_GUIDE.md)**: Advanced setup tasks
- **Individual Lab READMEs**: Detailed lab-specific documentation in each lab directory

## Comparison: Full vs Lite Labs

| Aspect | Full Labs | Lite Labs |
|--------|-----------|-----------|
| **Dataset** | Complete | 10-20% sampled |
| **Epochs** | 30-50 | 10 |
| **Batch Size** | 32-128 | 8-32 |
| **Runtime (GPU)** | 6-10 hours | 30-45 minutes |
| **Runtime (CPU)** | 40-60 hours | 30-45 minutes |
| **GPU** | Required | Not required |
| **Memory** | 16-32GB RAM | 4-8GB RAM |
| **VRAM** | 8GB+ | Not needed |
| **Accuracy** | Optimal | Good (5-15% lower) |
| **Use Case** | Production/Research | Learning/Testing |

## Advanced Features

### 1. Distributed Training
```python
# Multi-GPU training
python3 -m torch.distributed.launch --nproc_per_node=2 train.py
```

### 2. Mixed Precision Training
```python
# Faster training with AMP
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

### 3. TensorBoard Integration
```bash
# Monitor training in real-time
tensorboard --logdir=output/logs
```

### 4. Model Export
```python
# Export to ONNX for deployment
torch.onnx.export(model, dummy_input, "model.onnx")
```

## Best Practices

1. **Start with Lite Labs**: Test setup and understand concepts
2. **Monitor Resources**: Watch GPU memory and utilization
3. **Use Checkpoints**: Save progress regularly
4. **Validate Early**: Check results after 1-2 epochs
5. **Log Everything**: Track metrics, hyperparameters, and outputs
6. **Version Control**: Use git for code and DVC for data
7. **Document Results**: Keep notes on experiments
8. **Compare Models**: Run multiple configurations

## Tips for Success

1. **GPU Setup**: Ensure CUDA is properly installed
2. **Batch Size**: Start large, reduce if OOM errors occur
3. **Learning Rate**: Use learning rate finder
4. **Data Loading**: Use multiple workers for faster loading
5. **Validation**: Monitor validation metrics to prevent overfitting
6. **Experimentation**: Try different architectures and hyperparameters
7. **Reproducibility**: Set random seeds for consistent results

## Next Steps

After completing the full labs:

1. **Experiment**: Modify architectures and hyperparameters
2. **Extend**: Add new features or datasets
3. **Optimize**: Profile and improve performance
4. **Deploy**: Export models for production use
5. **Research**: Implement latest papers and techniques
6. **Share**: Publish results and contribute improvements

## Contributing

Contributions welcome:
- Report bugs and issues
- Suggest improvements
- Add new labs or features
- Improve documentation
- Share your results

## License

Same as the main deep learning labs repository.

## Acknowledgments

- Dataset providers (CIFAR, VOC, Flickr8k, Cornell, etc.)
- PyTorch and TensorFlow communities
- Research paper authors
- Open source contributors
- Academic institutions

---

**Version**: 1.0  
**Last Updated**: 2026-04-25  
**Maintained by**: Deep Learning Lab Team

For questions or issues, refer to the main repository documentation or open an issue.