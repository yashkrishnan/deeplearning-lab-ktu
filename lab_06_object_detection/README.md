# Lab 6: Object Detection

## 📋 Overview

This lab covers modern object detection architectures:
- **Single-stage detectors**: YOLO, SSD
- **Two-stage detectors**: Faster R-CNN, Mask R-CNN

## 🎯 Learning Objectives

1. Understand single-stage vs two-stage detectors
2. Implement YOLO for real-time detection
3. Use Faster R-CNN for accurate detection
4. Compare speed vs accuracy trade-offs
5. Perform real-time object detection

## 🏗️ Architectures

### YOLO (You Only Look Once)
- Single-stage detector
- Real-time performance (30+ FPS)
- Divides image into grid
- Predicts bounding boxes and classes simultaneously

**Versions**:
- YOLOv5: Easy to use, PyTorch-based
- YOLOv8: Latest, best performance
- YOLOv9: State-of-the-art

### SSD (Single Shot Detector)
- Multi-scale feature maps
- Faster than two-stage detectors
- Good balance of speed and accuracy

### Faster R-CNN
- Two-stage detector
- Region Proposal Network (RPN)
- Higher accuracy, slower inference
- Better for small objects

## 📦 Installation

```bash
# For YOLO
pip install ultralytics

# For Faster R-CNN
pip install torch torchvision
```

## 🚀 Quick Start

### YOLOv8 Example
```python
from ultralytics import YOLO

# Load pre-trained model
model = YOLO('yolov8n.pt')

# Detect objects
results = model('image.jpg')

# Display results
results[0].show()

# Get bounding boxes
boxes = results[0].boxes
for box in boxes:
    x1, y1, x2, y2 = box.xyxy[0]
    conf = box.conf[0]
    cls = box.cls[0]
    print(f"Class: {cls}, Confidence: {conf:.2f}")
```

### Faster R-CNN Example
```python
import torch
import torchvision

# Load pre-trained model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Detect objects
with torch.no_grad():
    predictions = model([image])

# Get results
boxes = predictions[0]['boxes']
labels = predictions[0]['labels']
scores = predictions[0]['scores']
```

## 📊 Performance Comparison

| Model | mAP | FPS | Use Case |
|-------|-----|-----|----------|
| YOLOv8n | 37.3 | 80+ | Real-time, edge devices |
| YOLOv8m | 50.2 | 50+ | Balanced |
| YOLOv8x | 53.9 | 30+ | High accuracy |
| Faster R-CNN | 42.0 | 5-10 | Accuracy-critical |
| SSD | 25.1 | 40+ | Real-time |

## 🎓 Training Custom Detector

### Prepare Dataset (YOLO format)
```
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

### Train YOLOv8
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.train(
    data='dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)
```

### dataset.yaml
```yaml
path: ./dataset
train: images/train
val: images/val

nc: 3  # number of classes
names: ['person', 'car', 'dog']
```

## 📈 Evaluation Metrics

### mAP (mean Average Precision)
- Standard metric for object detection
- Considers both localization and classification
- mAP@0.5: IoU threshold of 0.5
- mAP@0.5:0.95: Average over IoU thresholds

### Precision & Recall
```python
precision = TP / (TP + FP)
recall = TP / (TP + FN)
```

### IoU (Intersection over Union)
```python
iou = intersection_area / union_area
```

## 🎓 Exercises

1. **Real-time Detection**:
   ```python
   import cv2
   from ultralytics import YOLO
   
   model = YOLO('yolov8n.pt')
   cap = cv2.VideoCapture(0)
   
   while True:
       ret, frame = cap.read()
       results = model(frame)
       annotated = results[0].plot()
       cv2.imshow('Detection', annotated)
       if cv2.waitKey(1) & 0xFF == ord('q'):
           break
   ```

2. **Custom Object Detection**:
   - Collect and label images
   - Train YOLOv8 on custom dataset
   - Evaluate performance

3. **Compare Detectors**:
   - Test YOLO vs Faster R-CNN
   - Measure speed and accuracy
   - Analyze trade-offs

## 📚 Resources

- [YOLO Paper](https://arxiv.org/abs/1506.02640)
- [Faster R-CNN Paper](https://arxiv.org/abs/1506.01497)
- [Ultralytics Documentation](https://docs.ultralytics.com/)
- [Detectron2](https://github.com/facebookresearch/detectron2)

## 🔍 Common Issues

**Low FPS**: Use smaller model (YOLOv8n)
**Low Accuracy**: Use larger model or train longer
**False Positives**: Increase confidence threshold
**Missed Detections**: Lower confidence threshold

## 🎯 Next Steps

Move on to Lab 7: Image Captioning