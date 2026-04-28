# Deep Learning Lab Activities

A comprehensive collection of educational deep learning programs demonstrating fundamental concepts in computer vision, natural language processing, and sequence modeling.

## 🚀 Quick Start - Web Interface (Recommended)

The easiest way to run and visualize the labs is through the web interface:

### 1. Install Dependencies

```bash
pip install -r requirements.txt
cd web_interface
pip install -r requirements.txt
```

### 2. Start the Web Server

```bash
cd web_interface
python app.py
```

### 3. Open in Browser

Navigate to `http://localhost:5000` in your web browser.

### 4. Run Labs

- Browse all 10 labs from the dashboard
- Click on any lab card to open its interface
- Click "Run Lab" to execute
- View real-time console output
- See generated visualizations automatically

**Features:**
- Interactive web dashboard with Full Labs, Lite Labs, and Documentation sections
- Real-time console output display
- Automatic visualization of results
- Run/Stop controls for each lab
- Built-in documentation viewer
- Modern dark theme with responsive design

For detailed web interface documentation, see [web_interface/README.md](web_interface/README.md)

## 📚 Overview

This repository contains 10 self-contained lab activities designed for academic learning. Each lab includes:
- Well-commented, production-quality code
- Clear learning objectives
- Sample datasets or data generation
- Training and evaluation pipelines
- Visualization of results
- Execution time under 5 minutes on standard hardware

## 🗂️ Lab Activities

### Lab 1: Basic Image Processing Operations
**Directory:** `labs_full/lab_01_image_processing/` | **Lite:** `labs_lite/lab_01_image_processing/`
- Histogram equalization
- Thresholding techniques
- Edge detection (Sobel, Canny)
- Data augmentation
- Morphological operations

### Lab 2: CIFAR-10 Classifiers
**Directory:** `labs_full/lab_02_cifar10_classifiers/` | **Lite:** `labs_lite/lab_02_cifar10_classifiers/`
- KNN classifier implementation
- 3-layer neural network classifier
- SVM/Softmax classifier comparison

### Lab 3: Batch Normalization and Dropout Study
**Directory:** `labs_full/lab_03_batchnorm_dropout/` | **Lite:** `labs_lite/lab_03_batchnorm_dropout/`
- Effect of batch normalization on training
- Impact of dropout on overfitting
- Comparative analysis with visualizations

### Lab 4: Image Labeling Tools Demonstration
**Directory:** `labs_full/lab_04_labeling_tools/` | **Lite:** `labs_lite/lab_04_labeling_tools/`
- Synthetic image generation with annotations
- Format conversions (COCO, YOLO, Pascal VOC)
- Annotation visualization
- Statistics generation

### Lab 5: Image Segmentation with UNet
**Directory:** `labs_full/lab_05_segmentation/` | **Lite:** `labs_lite/lab_05_segmentation/`
- UNet architecture implementation
- Comparison with simple CNN baseline
- Evaluation metrics (IoU, Dice coefficient)
- Synthetic segmentation dataset

### Lab 6: Object Detection with YOLO
**Directory:** `labs_full/lab_06_object_detection/` | **Lite:** `labs_lite/lab_06_object_detection/`
- YOLO-style object detector
- Bounding box prediction and classification
- Non-Maximum Suppression (NMS)
- Detection visualization

### Lab 7: Image Captioning with RNN/LSTM
**Directory:** `labs_full/lab_07_image_captioning/` | **Lite:** `labs_lite/lab_07_image_captioning/`
- CNN encoder + RNN/LSTM decoder
- Comparison between Vanilla RNN and LSTM
- Caption generation
- Synthetic image-caption pairs

### Lab 8: Chatbot with Bi-directional LSTM
**Directory:** `labs_full/lab_08_chatbot/` | **Lite:** `labs_lite/lab_08_chatbot/`
- Bi-directional LSTM for intent classification
- Comparison with unidirectional LSTM
- Response generation
- Interactive demo

### Lab 9: Time Series Forecasting with LSTM
**Directory:** `labs_full/lab_09_time_series/` | **Lite:** `labs_lite/lab_09_time_series/`
- LSTM and GRU for forecasting
- Multi-step ahead prediction
- Comparison with baseline
- Evaluation metrics (MSE, MAE, RMSE, MAPE)

### Lab 10: Sequence to Sequence Learning
**Directory:** `labs_full/lab_10_seq2seq/` | **Lite:** `labs_lite/lab_10_seq2seq/`
- Encoder-decoder architecture
- Attention mechanism
- Machine translation (synthetic language)
- Beam search decoding

## 💻 Alternative: Command Line

If you prefer running labs directly from the command line:

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training

### Installation

1. Clone or download this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running Individual Labs

Each lab is self-contained with an executable Python program. Navigate to the lab directory and run:

```bash
# Lab 1: Image Processing
cd labs_full/lab_01_image_processing
python image_processing.py

# Lab 2: CIFAR-10 Classifiers
cd labs_full/lab_02_cifar10_classifiers
python cifar10_classifiers.py

# Lab 3: Batch Normalization & Dropout
cd labs_full/lab_03_batchnorm_dropout
python batchnorm_dropout_study.py

# Lab 4: Labeling Tools Demo
cd labs_full/lab_04_labeling_tools
python labeling_demo.py

# Lab 5: Image Segmentation
cd labs_full/lab_05_segmentation
python segmentation_demo.py

# Lab 6: Object Detection
cd labs_full/lab_06_object_detection
python object_detection_demo.py

# Lab 7: Image Captioning
cd labs_full/lab_07_image_captioning
python image_captioning_demo.py

# Lab 8: Chatbot
cd labs_full/lab_08_chatbot
python chatbot_demo.py

# Lab 9: Time Series Forecasting
cd labs_full/lab_09_time_series
python time_series_demo.py

# Lab 10: Sequence to Sequence
cd labs_full/lab_10_seq2seq
python seq2seq_demo.py
```

Or run all labs sequentially:
```bash
cd labs_full
bash run_all_labs.sh
```

For lightweight versions with faster execution:
```bash
cd labs_lite
bash run_all_labs.sh
```

**Note:** Each program creates an `output/` directory in its lab folder with visualizations and results.

## 📦 Dependencies

Core libraries used across labs:
- **PyTorch**: Deep learning framework
- **TensorFlow/Keras**: Alternative framework for some labs
- **OpenCV**: Image processing
- **NumPy**: Numerical computations
- **Matplotlib/Seaborn**: Visualization
- **scikit-learn**: Classical ML algorithms
- **Pillow**: Image handling
- **pandas**: Data manipulation

See `requirements.txt` for complete list with versions.

## 📊 Expected Execution Times

All labs are designed to complete within 5 minutes on standard hardware (CPU):

| Lab | Program | Execution Time | Notes |
|-----|---------|----------------|-------|
| Lab 1 | `image_processing.py` | < 30 seconds | Image processing operations |
| Lab 2 | `cifar10_classifiers.py` | 2-3 minutes | KNN, SVM, 3-layer NN training |
| Lab 3 | `batchnorm_dropout_study.py` | 2-3 minutes | 4 model variants comparison |
| Lab 4 | `labeling_demo.py` | < 30 seconds | Synthetic data generation |
| Lab 5 | `segmentation_demo.py` | 3-4 minutes | UNet + baseline training |
| Lab 6 | `object_detection_demo.py` | 3-4 minutes | YOLO-style detector training |
| Lab 7 | `image_captioning_demo.py` | 3-4 minutes | RNN + LSTM training |
| Lab 8 | `chatbot_demo.py` | 2-3 minutes | BiLSTM + LSTM training |
| Lab 9 | `time_series_demo.py` | 2-3 minutes | LSTM + GRU forecasting |
| Lab 10 | `seq2seq_demo.py` | 3-4 minutes | Seq2Seq + Attention training |

**Total time for all labs:** ~25-30 minutes

**With GPU:** Execution times can be 2-5x faster depending on GPU model.

**Lite versions:** Execute 2-3x faster with reduced dataset sizes and epochs.

## 🎓 Learning Objectives

By completing these labs, you will:
1. Understand fundamental image processing techniques
2. Implement and compare different classification approaches
3. Master regularization techniques in neural networks
4. Learn data annotation workflows for computer vision
5. Build semantic and instance segmentation models
6. Implement modern object detection architectures
7. Create image captioning systems with RNNs and LSTMs
8. Develop conversational AI with bi-directional LSTMs
9. Apply deep learning to time series problems
10. Understand sequence-to-sequence architectures

## 📝 Repository Structure

```
dl-lab/
├── web_interface/         # Web-based interface (recommended)
│   ├── app.py            # Flask application
│   ├── templates/        # HTML templates
│   └── requirements.txt  # Web interface dependencies
├── labs_full/            # Full-featured lab programs
│   └── lab_XX_name/
│       ├── README.md     # Lab-specific instructions
│       ├── program.py    # Main executable
│       └── output/       # Generated results (created on run)
├── labs_lite/            # Lightweight versions (faster execution)
│   └── lab_XX_name/
│       ├── program_lite.py
│       └── output/
├── docs/                 # Additional documentation
└── requirements.txt      # Main dependencies
```

**All programs:**
- Are fully self-contained and executable
- Generate synthetic data or download datasets as needed
- Create visualizations and save results in `output/` directory
- Include comprehensive comments and documentation
- Print progress and results to console
- Available in both full and lite versions

## 🔧 Troubleshooting

### Common Issues

**Import Errors:**
```bash
pip install --upgrade -r requirements.txt
```

**CUDA Out of Memory:**
- Reduce batch size in the code
- Use CPU mode by setting `device = 'cpu'`

**Slow Execution:**
- Ensure you're using GPU if available
- Reduce dataset size or number of epochs
- Check system resources

## 📚 Documentation

For more detailed information, see:
- [Web Interface Guide](web_interface/README.md) - Complete web interface documentation
- [Installation Guide](docs/INSTALLATION_GUIDE.md) - Detailed setup instructions
- [Dataset Setup Guide](docs/DATASET_SETUP_GUIDE.md) - Dataset download and management
- [Getting Started](docs/GETTING_STARTED.md) - Beginner's guide
- [Advanced Tab Guide](docs/ADVANCED_TAB_GUIDE.md) - Advanced features

## 📚 Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [OpenCV Tutorials](https://docs.opencv.org/master/d9/df8/tutorial_root.html)
- [Deep Learning Book](https://www.deeplearningbook.org/)

## 🤝 Contributing

This is an educational repository. Feel free to:
- Report issues or bugs
- Suggest improvements
- Add more examples or variations

## 📄 License

This project is created for educational purposes. Feel free to use and modify for learning.

## ✨ Acknowledgments

These labs are designed for academic learning and incorporate best practices from:
- Stanford CS231n: Convolutional Neural Networks
- Stanford CS224n: Natural Language Processing
- Fast.ai courses
- PyTorch and TensorFlow tutorials

---

**Note:** All programs are designed to be self-contained and educational. They prioritize clarity and learning over production optimization.

**Last Updated:** March 2026