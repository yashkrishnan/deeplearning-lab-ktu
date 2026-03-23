# Lab 1: Basic Image Processing Operations

## 📋 Overview

This lab demonstrates fundamental image processing techniques that are essential for computer vision and deep learning applications. You'll learn how to enhance images, detect edges, augment data, and apply morphological transformations.

## 🎯 Learning Objectives

By completing this lab, you will:
1. Understand histogram equalization for contrast enhancement
2. Master various thresholding techniques for image segmentation
3. Apply edge detection algorithms (Sobel, Canny, Laplacian)
4. Implement data augmentation techniques for deep learning
5. Use morphological operations for image analysis

## 🔧 Prerequisites

- Python 3.8+
- OpenCV (cv2)
- NumPy
- Matplotlib

## 📦 Installation

Install required packages:
```bash
pip install opencv-python numpy matplotlib
```

## 🚀 Running the Lab

Execute the main program:
```bash
python image_processing.py
```

Expected execution time: **< 30 seconds**

## 📊 Operations Demonstrated

### 1. Histogram Equalization
- **Purpose**: Enhance image contrast by redistributing pixel intensities
- **Use Case**: Improve visibility in poorly lit images
- **Output**: Original vs. equalized image with histograms

### 2. Thresholding Techniques
- **Binary Threshold**: Simple threshold at value 127
- **Otsu's Method**: Automatic threshold selection
- **Adaptive Mean**: Local threshold based on mean
- **Adaptive Gaussian**: Local threshold with Gaussian weighting
- **Use Case**: Segment objects from background

### 3. Edge Detection
- **Sobel**: Gradient-based edge detection (X, Y, Combined)
- **Canny**: Multi-stage edge detection with hysteresis
- **Laplacian**: Second derivative-based edge detection
- **Use Case**: Object boundary detection, feature extraction

### 4. Data Augmentation
Techniques to artificially expand training datasets:
- **Rotation**: Rotate image by 30 degrees
- **Flipping**: Horizontal and vertical flips
- **Scaling**: Zoom in/out (1.2x scale)
- **Translation**: Shift image position
- **Brightness**: Adjust image brightness
- **Blur**: Apply Gaussian blur
- **Noise**: Add random Gaussian noise
- **Use Case**: Improve model generalization

### 5. Morphological Operations
Shape-based image processing:
- **Erosion**: Shrink objects, remove noise
- **Dilation**: Expand objects, fill holes
- **Opening**: Erosion + Dilation (remove noise)
- **Closing**: Dilation + Erosion (fill holes)
- **Gradient**: Outline objects
- **Top Hat**: Extract small bright features
- **Black Hat**: Extract small dark features
- **Use Case**: Noise removal, shape analysis

## 📁 Output Files

After running the program, check the `outputs/` directory:

```
outputs/
├── 0_original_image.png              # Sample input image
├── 1_histogram_equalization.png      # Contrast enhancement results
├── 2_thresholding.png                # Various thresholding methods
├── 3_edge_detection.png              # Edge detection algorithms
├── 4_data_augmentation.png           # Augmentation techniques
└── 5_morphological_operations.png    # Morphological transforms
```

## 💡 Key Concepts

### Histogram Equalization
```python
equalized = cv2.equalizeHist(image)
```
Redistributes pixel intensities to span the full range [0, 255], improving contrast.

### Thresholding
```python
# Binary threshold
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Otsu's automatic threshold
_, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Adaptive threshold
adaptive = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
```

### Edge Detection
```python
# Sobel
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

# Canny
edges = cv2.Canny(img, threshold1=50, threshold2=150)

# Laplacian
laplacian = cv2.Laplacian(img, cv2.CV_64F)
```

### Data Augmentation
```python
# Rotation
M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
rotated = cv2.warpAffine(img, M, (w, h))

# Flip
flipped = cv2.flip(img, 1)  # 1=horizontal, 0=vertical

# Brightness
brightened = cv2.convertScaleAbs(img, alpha=1.3, beta=30)
```

### Morphological Operations
```python
kernel = np.ones((5, 5), np.uint8)

erosion = cv2.erode(img, kernel, iterations=1)
dilation = cv2.dilate(img, kernel, iterations=1)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
```

## 🔍 Understanding the Results

### Histogram Equalization
- **Before**: Histogram concentrated in narrow range
- **After**: Histogram spread across full range
- **Effect**: Better contrast and visibility

### Thresholding
- **Binary**: Simple but sensitive to lighting
- **Otsu**: Automatic, works well for bimodal histograms
- **Adaptive**: Best for varying lighting conditions

### Edge Detection
- **Sobel**: Good for gradient-based edges, directional
- **Canny**: Most accurate, multi-stage algorithm
- **Laplacian**: Sensitive to noise, detects all edges

### Data Augmentation
- Increases dataset diversity
- Prevents overfitting
- Improves model robustness
- Essential for deep learning with limited data

### Morphological Operations
- **Erosion**: Removes small white noise
- **Dilation**: Fills small holes
- **Opening**: Removes noise while preserving shape
- **Closing**: Fills holes while preserving shape

## 🎓 Exercises

Try modifying the code to:

1. **Experiment with parameters**:
   - Change threshold values
   - Adjust Canny edge detection thresholds
   - Modify kernel sizes for morphological operations

2. **Load your own images**:
   ```python
   img = cv2.imread('your_image.jpg', cv2.IMREAD_GRAYSCALE)
   ```

3. **Combine operations**:
   - Apply histogram equalization before edge detection
   - Use morphological operations after thresholding
   - Chain multiple augmentations

4. **Create custom augmentations**:
   - Implement perspective transformation
   - Add salt-and-pepper noise
   - Apply color jittering (for color images)

## 📚 Additional Resources

- [OpenCV Documentation](https://docs.opencv.org/)
- [Image Processing Fundamentals](https://en.wikipedia.org/wiki/Digital_image_processing)
- [Morphological Operations](https://docs.opencv.org/master/d9/d61/tutorial_py_morphological_ops.html)
- [Edge Detection Algorithms](https://docs.opencv.org/master/da/d22/tutorial_py_canny.html)

## 🐛 Troubleshooting

**Issue**: Import error for cv2
```bash
pip install opencv-python
```

**Issue**: Display issues with matplotlib
```bash
pip install matplotlib --upgrade
```

**Issue**: Images not saving
- Check write permissions for `outputs/` directory
- Ensure sufficient disk space

## ✅ Expected Output

```
============================================================
  LAB 1: BASIC IMAGE PROCESSING OPERATIONS
============================================================

Generating sample image...
  ✓ Sample image created: (400, 400)

1. Applying Histogram Equalization...
  ✓ Histogram equalization complete
  ✓ Saved to: outputs/1_histogram_equalization.png

2. Applying Thresholding Techniques...
  ✓ Thresholding complete
  ✓ Saved to: outputs/2_thresholding.png

3. Applying Edge Detection...
  ✓ Edge detection complete
  ✓ Saved to: outputs/3_edge_detection.png

4. Applying Data Augmentation...
  ✓ Data augmentation complete
  ✓ Saved to: outputs/4_data_augmentation.png

5. Applying Morphological Operations...
  ✓ Morphological operations complete
  ✓ Saved to: outputs/5_morphological_operations.png

============================================================
  SUMMARY
============================================================
✓ All operations completed successfully!
✓ Total execution time: 2.34 seconds
✓ Output files saved to: outputs/
```

## 🎯 Next Steps

After completing this lab:
1. Review the generated images in the `outputs/` folder
2. Understand how each operation transforms the image
3. Experiment with different parameters
4. Apply these techniques to your own images
5. Move on to Lab 2: CIFAR-10 Classifiers

---

**Note**: This lab focuses on grayscale images for simplicity. The same operations can be applied to color images by processing each channel separately or converting to appropriate color spaces.