# Lab 3: Batch Normalization and Dropout Study

## 📋 Overview

This lab provides a comprehensive study of two important regularization techniques in deep learning:
- **Batch Normalization**: Normalizes layer inputs to accelerate training
- **Dropout**: Randomly drops neurons during training to prevent overfitting

We compare four model variants to understand their individual and combined effects.

## 🎯 Learning Objectives

By completing this lab, you will:
1. Understand how batch normalization accelerates training
2. Learn how dropout prevents overfitting
3. Compare training dynamics with different regularization techniques
4. Analyze convergence speed and generalization
5. Visualize the impact on training/validation curves
6. Understand the overfitting gap

## 🔧 Prerequisites

- Python 3.8+
- PyTorch
- torchvision
- NumPy
- Matplotlib
- Seaborn

## 📦 Installation

```bash
pip install torch torchvision numpy matplotlib seaborn
```

## 🚀 Running the Lab

```bash
python batchnorm_dropout_study.py
```

Expected execution time: **3-4 minutes**

## 🧪 Models Compared

### 1. Baseline Network
```
Input (3072) → FC(512) → ReLU → FC(256) → ReLU → FC(128) → ReLU → Output(10)
```
- No regularization
- Prone to overfitting
- Slower convergence

### 2. Batch Normalization Network
```
Input → FC → BatchNorm → ReLU → FC → BatchNorm → ReLU → FC → BatchNorm → ReLU → Output
```
- Normalizes activations
- Faster convergence
- More stable training

### 3. Dropout Network
```
Input → FC → ReLU → Dropout(0.5) → FC → ReLU → Dropout(0.5) → FC → ReLU → Dropout(0.5) → Output
```
- Randomly drops 50% of neurons
- Prevents overfitting
- Better generalization

### 4. BatchNorm + Dropout Network
```
Input → FC → BatchNorm → ReLU → Dropout → ... → Output
```
- Combines both techniques
- Best of both worlds
- Optimal performance

## 📊 What is Batch Normalization?

**Batch Normalization** normalizes the inputs of each layer:

```python
# For each mini-batch
mean = batch.mean()
variance = batch.var()
normalized = (batch - mean) / sqrt(variance + epsilon)
output = gamma * normalized + beta  # Learnable parameters
```

**Benefits:**
- Reduces internal covariate shift
- Allows higher learning rates
- Acts as regularization
- Faster convergence
- More stable gradients

**When to use:**
- Deep networks
- When training is unstable
- To speed up convergence
- With higher learning rates

## 🎲 What is Dropout?

**Dropout** randomly sets a fraction of neurons to zero during training:

```python
# During training
mask = torch.bernoulli(torch.ones_like(x) * (1 - dropout_rate))
output = x * mask / (1 - dropout_rate)

# During inference
output = x  # No dropout
```

**Benefits:**
- Prevents overfitting
- Ensemble effect
- Forces redundancy
- Better generalization

**When to use:**
- When overfitting occurs
- With large networks
- Limited training data
- After fully connected layers

## 📁 Output Files

```
outputs/
├── training_comparison.png          # All models training curves
├── overfitting_analysis.png         # Train vs validation accuracy
└── final_accuracy_comparison.png    # Test accuracy comparison
```

## 📈 Expected Results

### Typical Performance:

| Model | Test Accuracy | Overfitting Gap | Convergence Speed |
|-------|---------------|-----------------|-------------------|
| Baseline | 45-50% | 15-20% | Slow |
| BatchNorm | 50-55% | 10-15% | Fast |
| Dropout | 48-52% | 5-10% | Moderate |
| BatchNorm + Dropout | 52-58% | 5-8% | Fast |

### Key Observations:

1. **Batch Normalization**:
   - Fastest convergence
   - Higher training accuracy
   - Moderate overfitting

2. **Dropout**:
   - Slower convergence
   - Lower training accuracy
   - Minimal overfitting
   - Best generalization gap

3. **Combined**:
   - Fast convergence (from BatchNorm)
   - Good generalization (from Dropout)
   - Best overall performance

## 🔍 Understanding the Plots

### Training Comparison
Shows loss and accuracy curves for all models:
- **Training Loss**: How well model fits training data
- **Validation Loss**: How well model generalizes
- **Training Accuracy**: Performance on training set
- **Validation Accuracy**: Performance on unseen data

### Overfitting Analysis
Compares train vs validation accuracy:
- **Small gap**: Good generalization
- **Large gap**: Overfitting
- **Dropout reduces gap**: Better generalization

### Final Accuracy Comparison
Bar chart of test accuracies:
- Shows which regularization works best
- Combined approach usually wins

## 💡 Key Insights

### Batch Normalization Effects:

1. **Faster Training**:
   - Normalizes inputs to each layer
   - Allows higher learning rates
   - Reduces training time by 2-3x

2. **Stability**:
   - Reduces sensitivity to initialization
   - More stable gradients
   - Less likely to diverge

3. **Regularization**:
   - Slight regularization effect
   - Reduces need for dropout
   - Can replace some dropout

### Dropout Effects:

1. **Prevents Overfitting**:
   - Forces network to learn redundant representations
   - Reduces co-adaptation of neurons
   - Ensemble effect

2. **Generalization**:
   - Lower training accuracy
   - Better validation accuracy
   - Smaller overfitting gap

3. **Trade-offs**:
   - Slower convergence
   - Needs more epochs
   - Lower training accuracy

### Combined Approach:

1. **Best of Both**:
   - Fast convergence from BatchNorm
   - Good generalization from Dropout
   - Highest test accuracy

2. **Order Matters**:
   - BatchNorm → ReLU → Dropout
   - This order works best empirically

## 🎓 Exercises

1. **Experiment with dropout rates**:
   ```python
   for rate in [0.2, 0.3, 0.5, 0.7]:
       model = DropoutNetwork(dropout_rate=rate)
       # Train and compare
   ```

2. **Try different BatchNorm positions**:
   ```python
   # Before activation
   x = self.bn(self.fc(x))
   x = self.relu(x)
   
   # After activation
   x = self.relu(self.fc(x))
   x = self.bn(x)
   ```

3. **Add more layers**:
   ```python
   # Deeper network
   self.fc1 = nn.Linear(3072, 1024)
   self.fc2 = nn.Linear(1024, 512)
   self.fc3 = nn.Linear(512, 256)
   self.fc4 = nn.Linear(256, 128)
   self.fc5 = nn.Linear(128, 10)
   ```

4. **Try Layer Normalization**:
   ```python
   self.ln1 = nn.LayerNorm(512)
   # Compare with BatchNorm
   ```

5. **Implement early stopping**:
   ```python
   if val_loss > best_val_loss:
       patience_counter += 1
       if patience_counter > patience:
           break
   ```

## 📚 Additional Resources

- [Batch Normalization Paper](https://arxiv.org/abs/1502.03167)
- [Dropout Paper](https://jmlr.org/papers/v15/srivastava14a.html)
- [Understanding BatchNorm](https://towardsdatascience.com/batch-normalization-in-neural-networks-1ac91516821c)
- [Dropout Explained](https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/)

## 🐛 Troubleshooting

**Issue**: Training is too slow
```python
# Use smaller subset
train_indices = np.random.choice(len(trainset), 2000, replace=False)
```

**Issue**: Models not converging
```python
# Reduce learning rate
optimizer = optim.Adam(model.parameters(), lr=0.0001)
```

**Issue**: Out of memory
```python
# Reduce batch size
train_loader = DataLoader(trainset, batch_size=32, shuffle=True)
```

## ✅ Expected Output

```
============================================================
  LAB 3: BATCH NORMALIZATION AND DROPOUT STUDY
============================================================

Loading CIFAR-10 dataset...
  ✓ Training samples: 4000
  ✓ Validation samples: 1000
  ✓ Test samples: 1000

Training Baseline...
  Epoch [3/15] - Train Loss: 1.8234, Train Acc: 35.60%, Val Loss: 1.9123, Val Acc: 32.40%
  Epoch [6/15] - Train Loss: 1.5234, Train Acc: 47.80%, Val Loss: 1.7234, Val Acc: 38.20%
  ...
  ✓ Training completed
  ✓ Final validation accuracy: 45.30%
  ✓ Test Accuracy: 44.80%

Training Batch Normalization...
  Epoch [3/15] - Train Loss: 1.6234, Train Acc: 42.30%, Val Loss: 1.7123, Val Acc: 40.10%
  ...
  ✓ Training completed
  ✓ Final validation accuracy: 52.60%
  ✓ Test Accuracy: 51.90%

Training Dropout...
  ...
  ✓ Test Accuracy: 49.20%

Training BatchNorm + Dropout...
  ...
  ✓ Test Accuracy: 54.70%

============================================================
  SUMMARY
============================================================

Final Results:
------------------------------------------------------------
Model                          Test Accuracy   Overfitting Gap
------------------------------------------------------------
Baseline                          44.80%           18.50%
Batch Normalization               51.90%           12.30%
Dropout                           49.20%            6.80%
BatchNorm + Dropout               54.70%            7.20%
------------------------------------------------------------

✓ Study completed in 187.45 seconds
```

## 🎯 Next Steps

After completing this lab:
1. Understand when to use each regularization technique
2. Recognize overfitting in your models
3. Apply these techniques to your own projects
4. Move on to Lab 4: Image Labeling Tools

---

**Note**: Results may vary based on random initialization and data splits. The key takeaway is understanding the relative performance and characteristics of each approach.