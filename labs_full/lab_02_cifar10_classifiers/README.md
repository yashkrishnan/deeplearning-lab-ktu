# Lab 2: CIFAR-10 Classifiers Comparison

## 📋 Overview

This lab implements and compares three different classification approaches on the CIFAR-10 dataset:
1. **K-Nearest Neighbors (KNN)** - Classical ML approach
2. **Support Vector Machine (SVM)** - Classical ML with linear kernel
3. **3-Layer Neural Network** - Deep learning approach

You'll learn the strengths and weaknesses of each approach and understand when to use classical ML vs. deep learning.

## 🎯 Learning Objectives

By completing this lab, you will:
1. Understand different classification paradigms
2. Implement KNN and SVM classifiers
3. Build a simple neural network from scratch
4. Compare classical ML vs. deep learning performance
5. Analyze confusion matrices and classification metrics
6. Understand the trade-offs between accuracy and training time

## 🔧 Prerequisites

- Python 3.8+
- PyTorch
- scikit-learn
- NumPy
- Matplotlib
- Seaborn

## 📦 Installation

Install required packages:
```bash
pip install torch torchvision scikit-learn numpy matplotlib seaborn
```

## 🚀 Running the Lab

Execute the main program:
```bash
python cifar10_classifiers.py
```

Expected execution time: **3-4 minutes** (using subset of data)

## 📊 CIFAR-10 Dataset

**CIFAR-10** is a popular image classification dataset containing:
- **60,000** color images (32x32 pixels)
- **10 classes**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **50,000** training images
- **10,000** test images

For faster execution, this lab uses:
- **5,000** training samples
- **1,000** test samples

## 🤖 Classifiers Implemented

### 1. K-Nearest Neighbors (KNN)

**Algorithm:**
- Stores all training examples
- Classifies new samples based on k nearest neighbors
- Uses Euclidean distance in feature space

**Parameters:**
- k = 5 neighbors
- Distance metric: Euclidean

**Pros:**
- Simple and intuitive
- No training phase
- Works well with small datasets

**Cons:**
- Slow prediction time
- Memory intensive
- Sensitive to irrelevant features

**Code:**
```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
```

### 2. Support Vector Machine (SVM)

**Algorithm:**
- Finds optimal hyperplane to separate classes
- Maximizes margin between classes
- Uses kernel trick for non-linear boundaries

**Parameters:**
- Kernel: Linear
- C = 1.0 (regularization)

**Pros:**
- Effective in high-dimensional spaces
- Memory efficient
- Works well with clear margin of separation

**Cons:**
- Slow training on large datasets
- Sensitive to feature scaling
- Difficult to interpret

**Code:**
```python
from sklearn.svm import SVC

svm = SVC(kernel='linear', C=1.0)
svm.fit(X_train, y_train)
predictions = svm.predict(X_test)
```

### 3. 3-Layer Neural Network

**Architecture:**
```
Input Layer:  3072 neurons (32x32x3 flattened)
Hidden Layer 1: 512 neurons + ReLU
Hidden Layer 2: 256 neurons + ReLU
Output Layer: 10 neurons (softmax)
```

**Training:**
- Optimizer: Adam
- Learning rate: 0.001
- Loss: Cross-entropy
- Epochs: 10
- Batch size: 64

**Pros:**
- Learns hierarchical features
- Scalable to large datasets
- Can capture complex patterns

**Cons:**
- Requires more data
- Longer training time
- Needs hyperparameter tuning

**Code:**
```python
import torch.nn as nn

class ThreeLayerNN(nn.Module):
    def __init__(self):
        super(ThreeLayerNN, self).__init__()
        self.fc1 = nn.Linear(3072, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(-1, 3072)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

## 📁 Output Files

After running the program, check the `outputs/` directory:

```
outputs/
├── training_history.png                    # NN training curves
├── confusion_matrix_knn.png               # KNN confusion matrix
├── confusion_matrix_svm.png               # SVM confusion matrix
├── confusion_matrix_neural_network.png    # NN confusion matrix
└── classifier_comparison.png              # Performance comparison
```

## 📈 Expected Results

### Typical Performance (on subset):

| Classifier | Accuracy | Training Time |
|------------|----------|---------------|
| KNN        | ~35-40%  | ~5-10 seconds |
| SVM        | ~40-45%  | ~30-60 seconds |
| Neural Network | ~50-55% | ~60-90 seconds |

**Note:** Accuracies are lower than full dataset due to using subset for faster execution.

### Full Dataset Performance:

| Classifier | Accuracy | Training Time |
|------------|----------|---------------|
| KNN        | ~35%     | ~2-3 minutes |
| SVM        | ~40%     | ~10-15 minutes |
| Neural Network | ~55-60% | ~5-10 minutes |

## 🔍 Understanding the Results

### Confusion Matrix
Shows which classes are confused with each other:
- **Diagonal**: Correct predictions
- **Off-diagonal**: Misclassifications
- Common confusions: cat↔dog, automobile↔truck

### Training History (Neural Network)
- **Loss curve**: Should decrease over epochs
- **Accuracy curve**: Should increase over epochs
- Plateauing indicates convergence

### Classifier Comparison
- **KNN**: Fast training, moderate accuracy
- **SVM**: Slower training, better than KNN
- **Neural Network**: Best accuracy, reasonable training time

## 💡 Key Insights

### When to Use Each Classifier:

**KNN:**
- Small datasets
- Quick prototyping
- When interpretability is important
- No training time constraints

**SVM:**
- Medium-sized datasets
- Clear class separation
- High-dimensional data
- When accuracy is more important than speed

**Neural Network:**
- Large datasets
- Complex patterns
- When highest accuracy is needed
- GPU available for training

### Why Neural Networks Perform Better:

1. **Feature Learning**: Automatically learns relevant features
2. **Non-linearity**: Multiple layers capture complex patterns
3. **Scalability**: Performance improves with more data
4. **Flexibility**: Can be adapted to various tasks

## 🎓 Exercises

Try modifying the code to:

1. **Experiment with KNN:**
   ```python
   # Try different k values
   for k in [1, 3, 5, 7, 10]:
       knn = KNeighborsClassifier(n_neighbors=k)
       # Train and evaluate
   ```

2. **Try different SVM kernels:**
   ```python
   # RBF kernel
   svm = SVC(kernel='rbf', C=1.0, gamma='scale')
   
   # Polynomial kernel
   svm = SVC(kernel='poly', degree=3, C=1.0)
   ```

3. **Modify neural network architecture:**
   ```python
   # Add more layers
   self.fc1 = nn.Linear(3072, 1024)
   self.fc2 = nn.Linear(1024, 512)
   self.fc3 = nn.Linear(512, 256)
   self.fc4 = nn.Linear(256, 10)
   
   # Add dropout
   self.dropout = nn.Dropout(0.5)
   ```

4. **Use full dataset:**
   ```python
   classifier = CIFAR10Classifiers(use_subset=False)
   ```

5. **Implement data augmentation:**
   ```python
   transform = transforms.Compose([
       transforms.RandomHorizontalFlip(),
       transforms.RandomCrop(32, padding=4),
       transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
   ])
   ```

## 📚 Additional Resources

- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Understanding CNNs](https://cs231n.github.io/)

## 🐛 Troubleshooting

**Issue**: CUDA out of memory
```python
# Use CPU instead
device = torch.device('cpu')
```

**Issue**: Slow training
```python
# Use smaller subset
classifier = CIFAR10Classifiers(use_subset=True)
```

**Issue**: Low accuracy
- Try full dataset instead of subset
- Increase number of epochs
- Add data augmentation
- Use a deeper network (CNN)

**Issue**: Import errors
```bash
pip install torch torchvision scikit-learn
```

## ✅ Expected Output

```
============================================================
  LAB 2: CIFAR-10 CLASSIFIERS COMPARISON
============================================================

Loading CIFAR-10 dataset...
  ✓ Training samples: 5000
  ✓ Test samples: 1000

Preparing data for classical ML classifiers...
  ✓ Data prepared (flattened to vectors)

============================================================
1. K-NEAREST NEIGHBORS (KNN) CLASSIFIER
============================================================
Training KNN (k=5)...
  ✓ Training completed in 8.45 seconds
  ✓ Test Accuracy: 38.20%
  ✓ Confusion matrix saved

============================================================
2. SVM CLASSIFIER (LINEAR KERNEL)
============================================================
Training SVM...
  ✓ Training completed in 45.23 seconds
  ✓ Test Accuracy: 42.50%
  ✓ Confusion matrix saved

============================================================
3. 3-LAYER NEURAL NETWORK
============================================================
Training Neural Network...
  Architecture: 3072 -> 512 -> 256 -> 10
  Epoch [2/10] - Loss: 1.8234, Accuracy: 35.60%
  Epoch [4/10] - Loss: 1.6543, Accuracy: 42.30%
  Epoch [6/10] - Loss: 1.5234, Accuracy: 47.80%
  Epoch [8/10] - Loss: 1.4123, Accuracy: 51.20%
  Epoch [10/10] - Loss: 1.3456, Accuracy: 53.40%
  ✓ Training completed in 78.90 seconds
  ✓ Test Accuracy: 52.80%
  ✓ Training history saved
  ✓ Confusion matrix saved

============================================================
COMPARISON OF CLASSIFIERS
============================================================

Summary Table:
------------------------------------------------------------
Classifier           Accuracy        Training Time
------------------------------------------------------------
KNN                    38.20%             8.45s
SVM                    42.50%            45.23s
Neural Network         52.80%            78.90s
------------------------------------------------------------

✓ Comparison plot saved
✓ Total execution time: 132.58 seconds
```

## 🎯 Next Steps

After completing this lab:
1. Analyze the confusion matrices to understand misclassifications
2. Compare the trade-offs between accuracy and training time
3. Understand why neural networks outperform classical ML
4. Move on to Lab 3: Batch Normalization and Dropout Study

---

**Note**: This lab uses a subset of CIFAR-10 for faster execution. For production use, train on the full dataset and consider using Convolutional Neural Networks (CNNs) for better performance.