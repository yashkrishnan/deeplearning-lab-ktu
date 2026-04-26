# Getting Started with Deep Learning Labs

## Quick Start Guide

Welcome to the Deep Learning Lab collection! This guide will help you get started quickly.

## Prerequisites

- **Python 3.8 or higher**
- **pip** package manager
- **8GB RAM minimum** (16GB recommended)
- **GPU optional** (CUDA-compatible for faster training)

## Installation

### Step 1: Clone or Download

If you haven't already, navigate to the `dl-lab` directory:
```bash
cd dl-lab
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

This will install:
- PyTorch & TorchVision
- TensorFlow & Keras
- OpenCV
- scikit-learn
- Matplotlib & Seaborn
- And more...

### Step 4: Verify Installation

```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
```

## Lab Structure

```
dl-lab/
├── README.md                          # Main overview
├── GETTING_STARTED.md                 # This file
├── requirements.txt                   # All dependencies
├── run_all_labs.sh                    # Run all labs script
│
├── lab_01_image_processing/             # Basic image operations
│   ├── README.md
│   ├── image_processing.py
│   └── outputs/
│
├── lab_02_cifar10_classifiers/          # Classification comparison
│   ├── README.md
│   ├── cifar10_classifiers.py
│   └── outputs/
│
├── lab_03_batchnorm_dropout/            # Regularization study
│   ├── README.md
│   ├── batchnorm_dropout_study.py
│   └── outputs/
│
├── lab_04_labeling_tools/               # Annotation guide
│   └── README.md
│
├── lab_05_segmentation/                 # Image segmentation
│   └── README.md
│
├── lab_06_object_detection/             # Object detection
│   └── README.md
│
├── lab_07_image_captioning/             # Image to text
│   └── README.md
│
├── lab_08_chatbot/                      # Conversational AI
│   └── README.md
│
├── lab_09_time_series/                  # Time series forecasting
│   └── README.md
│
└── lab_10_seq2seq/                     # Sequence to sequence
    └── README.md
```

## Running Individual Labs

### Lab 1: Image Processing
```bash
cd lab_01_image_processing
python3 image_processing.py
```
**Time**: ~30 seconds  
**Output**: 6 visualization images in `outputs/`

### Lab 2: CIFAR-10 Classifiers
```bash
cd lab_02_cifar10_classifiers
python3 cifar10_classifiers.py
```
**Time**: ~3-4 minutes  
**Output**: Comparison plots and confusion matrices

### Lab 3: Batch Normalization & Dropout
```bash
cd lab_03_batchnorm_dropout
python3 batchnorm_dropout_study.py
```
**Time**: ~3-4 minutes  
**Output**: Training curves and performance comparison

### Labs 4-10
These labs are primarily educational with comprehensive README files:
- **Lab 4**: Guide to image labeling tools
- **Lab 5**: Image segmentation techniques
- **Lab 6**: Object detection methods
- **Lab 7**: Image captioning with RNNs/LSTMs
- **Lab 8**: Chatbot implementation
- **Lab 9**: Time series forecasting
- **Lab 10**: Sequence-to-sequence learning

Each lab's README contains:
- Detailed explanations
- Code examples
- Implementation guides
- Best practices
- Resources

## Running All Labs

To run all executable labs sequentially:

```bash
# Make script executable (first time only)
chmod +x run_all_labs.sh

# Run all labs
./run_all_labs.sh
```

## What to Expect

### Lab 1 Output:
```
SUCCESS: Sample image created
SUCCESS: Histogram equalization complete
SUCCESS: Thresholding complete
SUCCESS: Edge detection complete
SUCCESS: Data augmentation complete
SUCCESS: Morphological operations complete
```

### Lab 2 Output:
```
Training KNN...
  SUCCESS: Test Accuracy: 38.20%
Training SVM...
  SUCCESS: Test Accuracy: 42.50%
Training Neural Network...
  SUCCESS: Test Accuracy: 52.80%
```

### Lab 3 Output:
```
Training Baseline...
  SUCCESS: Final validation accuracy: 45.30%
Training Batch Normalization...
  SUCCESS: Final validation accuracy: 52.60%
Training Dropout...
  SUCCESS: Final validation accuracy: 49.20%
Training BatchNorm + Dropout...
  SUCCESS: Final validation accuracy: 54.70%
```

## Learning Path

### Beginner Path:
1. **Lab 1**: Image Processing Basics
2. **Lab 2**: Classification Algorithms
3. **Lab 4**: Data Annotation
4. **Lab 3**: Regularization Techniques

### Intermediate Path:
5. **Lab 5**: Image Segmentation
6. **Lab 6**: Object Detection
7. **Lab 9**: Time Series Forecasting

### Advanced Path:
8. **Lab 7**: Image Captioning
9. **Lab 8**: Chatbot Development
10. **Lab 10**: Seq2Seq Learning

## Troubleshooting

### Issue: Import Errors
```bash
# Reinstall packages
pip install --upgrade -r requirements.txt
```

### Issue: CUDA Out of Memory
```python
# In the code, change device to CPU
device = torch.device('cpu')
```

### Issue: Slow Training
```python
# Use smaller datasets or reduce epochs
# Most labs have use_subset=True option
```

### Issue: Permission Denied (run_all_labs.sh)
```bash
chmod +x run_all_labs.sh
```

## Tips for Success

1. **Start with Lab 1**: It's the easiest and fastest
2. **Read the READMEs**: Each lab has detailed documentation
3. **Experiment**: Modify parameters and see what happens
4. **Use GPU**: If available, training will be much faster
5. **Take Notes**: Document your observations
6. **Ask Questions**: Use the resources provided in each lab

## Additional Resources

### Online Courses:
- [Fast.ai Practical Deep Learning](https://course.fast.ai/)
- [Stanford CS231n](http://cs231n.stanford.edu/)
- [DeepLearning.AI Specialization](https://www.deeplearning.ai/)

### Books:
- "Deep Learning" by Goodfellow, Bengio, and Courville
- "Hands-On Machine Learning" by Aurélien Géron
- "Deep Learning with Python" by François Chollet

### Documentation:
- [PyTorch Documentation](https://pytorch.org/docs/)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [OpenCV Documentation](https://docs.opencv.org/)

## Contributing

Found an issue or want to improve a lab?
1. Document the issue or improvement
2. Test your changes
3. Share your modifications

## Lab Completion Checklist

- [ ] Lab 1: Image Processing [Done]
- [ ] Lab 2: CIFAR-10 Classifiers [Done]
- [ ] Lab 3: Batch Normalization & Dropout [Done]
- [ ] Lab 4: Image Labeling Tools (Read)
- [ ] Lab 5: Image Segmentation (Read)
- [ ] Lab 6: Object Detection (Read)
- [ ] Lab 7: Image Captioning (Read)
- [ ] Lab 8: Chatbot (Read)
- [ ] Lab 9: Time Series Forecasting (Read)
- [ ] Lab 10: Sequence to Sequence (Read)

## Next Steps

After completing these labs:

1. **Build Your Own Project**: Apply what you learned
2. **Kaggle Competitions**: Test your skills
3. **Research Papers**: Read latest developments
4. **Open Source**: Contribute to projects
5. **Share Knowledge**: Teach others

## Support

For questions or issues:
- Check the lab-specific README
- Review the troubleshooting section
- Consult online documentation
- Join deep learning communities

## Congratulations!

You're now ready to start your deep learning journey. Begin with Lab 1 and work your way through. Each lab builds on previous concepts, so take your time and enjoy learning!

---

**Happy Learning!**

Last Updated: March 2026