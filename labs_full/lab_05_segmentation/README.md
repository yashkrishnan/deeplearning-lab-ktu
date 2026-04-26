# Lab 5: Image Segmentation

## 📋 Overview

This lab demonstrates image segmentation techniques using modern deep learning architectures:
- **UNet**: For binary and multi-class segmentation
- **SegNet**: Encoder-decoder architecture
- **Mask R-CNN**: Instance segmentation

## 🎯 Learning Objectives

1. Understand semantic vs instance segmentation
2. Implement UNet architecture from scratch
3. Use pre-trained models for segmentation
4. Evaluate segmentation with IoU and Dice coefficient
5. Visualize segmentation masks

## 🏗️ Architectures

### UNet
- Encoder-decoder with skip connections
- Excellent for medical image segmentation
- Works well with limited data

### SegNet
- Symmetric encoder-decoder
- Uses pooling indices for upsampling
- Memory efficient

### Mask R-CNN
- Extends Faster R-CNN
- Instance segmentation
- Detects and segments individual objects

## 📦 Installation

```bash
pip install torch torchvision segmentation-models-pytorch
# For Mask R-CNN
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html
```

## 🚀 Quick Start

```python
import torch
import segmentation_models_pytorch as smp

# Load pre-trained UNet
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
)

# For binary segmentation
output = model(image)
mask = torch.sigmoid(output) > 0.5
```

## 📊 Evaluation Metrics

### IoU (Intersection over Union)
```python
def iou_score(pred, target):
    intersection = (pred & target).sum()
    union = (pred | target).sum()
    return intersection / union
```

### Dice Coefficient
```python
def dice_coefficient(pred, target):
    intersection = (pred * target).sum()
    return (2 * intersection) / (pred.sum() + target.sum())
```

## 🎓 Exercises

1. Train UNet on custom dataset
2. Compare different encoder backbones
3. Implement data augmentation
4. Fine-tune pre-trained models
5. Visualize feature maps

## 📚 Resources

- [UNet Paper](https://arxiv.org/abs/1505.04597)
- [SegNet Paper](https://arxiv.org/abs/1511.00561)
- [Mask R-CNN Paper](https://arxiv.org/abs/1703.06870)
- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)

## 🎯 Next Steps

Move on to Lab 6: Object Detection