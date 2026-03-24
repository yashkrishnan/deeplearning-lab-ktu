# Deep Learning Lab Activities

A comprehensive collection of educational deep learning programs demonstrating fundamental concepts in computer vision, natural language processing, and sequence modeling.

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
**Directory:** `lab_01_image_processing/`
- Histogram equalization
- Thresholding techniques
- Edge detection (Sobel, Canny)
- Data augmentation
- Morphological operations

### Lab 2: CIFAR-10 Classifiers
**Directory:** `lab_02_cifar10_classifiers/`
- KNN classifier implementation
- 3-layer neural network classifier
- SVM/Softmax classifier comparison

### Lab 3: Batch Normalization and Dropout Study
**Directory:** `lab_03_batchnorm_dropout/`
- Effect of batch normalization on training
- Impact of dropout on overfitting
- Comparative analysis with visualizations

### Lab 4: Image Labeling Tools Demonstration
**Directory:** `lab_04_labeling_tools/`
**Program:** `labeling_demo.py`
- Synthetic image generation with annotations
- Format conversions (COCO, YOLO, Pascal VOC)
- Annotation visualization
- Statistics generation

### Lab 5: Image Segmentation with UNet
**Directory:** `lab_05_segmentation/`
**Program:** `segmentation_demo.py`
- UNet architecture implementation
- Comparison with simple CNN baseline
- Evaluation metrics (IoU, Dice coefficient)
- Synthetic segmentation dataset

### Lab 6: Object Detection with YOLO
**Directory:** `lab_06_object_detection/`
**Program:** `object_detection_demo.py`
- YOLO-style object detector
- Bounding box prediction and classification
- Non-Maximum Suppression (NMS)
- Detection visualization

### Lab 7: Image Captioning with RNN/LSTM
**Directory:** `lab_07_image_captioning/`
**Program:** `image_captioning_demo.py`
- CNN encoder + RNN/LSTM decoder
- Comparison between Vanilla RNN and LSTM
- Caption generation
- Synthetic image-caption pairs

### Lab 8: Chatbot with Bi-directional LSTM
**Directory:** `lab_08_chatbot/`
**Program:** `chatbot_demo.py`
- Bi-directional LSTM for intent classification
- Comparison with unidirectional LSTM
- Response generation
- Interactive demo

### Lab 9: Time Series Forecasting with LSTM
**Directory:** `lab_09_time_series/`
**Program:** `time_series_demo.py`
- LSTM and GRU for forecasting
- Multi-step ahead prediction
- Comparison with baseline
- Evaluation metrics (MSE, MAE, RMSE, MAPE)

### Lab 10: Sequence to Sequence Learning
**Directory:** `lab_10_seq2seq/`
**Program:** `seq2seq_demo.py`
- Encoder-decoder architecture
- Attention mechanism
- Machine translation (synthetic language)
- Beam search decoding

## 🚀 Quick Start

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
cd lab_01_image_processing
python image_processing.py

# Lab 2: CIFAR-10 Classifiers
cd lab_02_cifar10_classifiers
python cifar10_classifiers.py

# Lab 3: Batch Normalization & Dropout
cd lab_03_batchnorm_dropout
python batchnorm_dropout_study.py

# Lab 4: Labeling Tools Demo
cd lab_04_labeling_tools
python labeling_demo.py

# Lab 5: Image Segmentation
cd lab_05_segmentation
python segmentation_demo.py

# Lab 6: Object Detection
cd lab_06_object_detection
python object_detection_demo.py

# Lab 7: Image Captioning
cd lab_07_image_captioning
python image_captioning_demo.py

# Lab 8: Chatbot
cd lab_08_chatbot
python chatbot_demo.py

# Lab 9: Time Series Forecasting
cd lab_09_time_series
python time_series_demo.py

# Lab 10: Sequence to Sequence
cd lab_10_seq2seq
python seq2seq_demo.py
```

Or run all labs sequentially:
```bash
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

## 📝 Lab Structure

Each lab directory contains:
```
labX_name/
├── README.md              # Lab-specific instructions and theory
├── program_name.py        # Executable Python program
└── output/               # Generated visualizations and results (created on run)
```

**All programs:**
- Are fully self-contained and executable
- Generate synthetic data (no external downloads needed for Labs 4-10)
- Create visualizations and save results in `output/` directory
- Include comprehensive comments and documentation
- Print progress and results to console

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