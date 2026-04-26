# Lab 6 Lite: Object Detection with VOC2012

Lightweight object detection using a simplified YOLO-style architecture on Pascal VOC 2012 dataset.

## Quick Start

```bash
# 1. Copy sample dataset (if not already done)
python3 copy_sample_dataset.py

# 2. Run the lab
python3 object_detection_lite.py
```

## Dataset

This lab uses a **local sample** of 50 images from the Pascal VOC 2012 dataset with three target classes:
- person
- car
- dog

### Dataset Structure
```
data/VOC2012/
├── JPEGImages/      # 50 sample images
└── Annotations/     # 50 XML annotation files
```

### Setting Up the Dataset

The `copy_sample_dataset.py` script copies 50 sample images with target classes from the main VOC2012 dataset:

```bash
python3 copy_sample_dataset.py
```

This will:
- Copy 50 images containing person, car, or dog
- Copy corresponding XML annotation files
- Create the local `data/VOC2012/` directory structure

## Configuration

- **Samples**: 50 images (local dataset)
- **Epochs**: 10
- **Batch Size**: 8
- **Image Size**: 224x224
- **Max Objects**: 5 per image
- **Classes**: 3 (person, car, dog)

## Model Architecture

**Simplified YOLO**:
- Backbone: 4 convolutional blocks with batch normalization
- Detection head: Adaptive pooling + fully connected layers
- Parameters: ~6.8M
- Output: Bounding boxes + class predictions

## Expected Results

- **Training Time**: ~10 seconds on CPU
- **Loss Reduction**: From ~3.0 to ~0.6
- **Outputs**:
  - `output/training_history.png` - Loss curves
  - `output/detection_results.png` - Sample detections

## Performance

With 50 training samples:
- Initial Loss: ~3.09
- Final Loss: ~0.65
- Box Loss: ~0.56
- Class Loss: ~0.09

## Comparison with Original

| Aspect | Original Lab 6 | Lab 6 Lite |
|--------|---------------|------------|
| Dataset Size | ~17,000 images | 50 images |
| Classes | 20 VOC classes | 3 classes |
| Epochs | 30 | 10 |
| Batch Size | 16 | 8 |
| Runtime | ~2-3 hours | ~10 seconds |
| Model Size | Large YOLO | Simplified YOLO |

## Notes

- The local dataset makes this lab completely self-contained
- Training is very fast due to small dataset size
- Results demonstrate object detection concepts effectively
- For better accuracy, use the full dataset in the original lab

## Troubleshooting

### Dataset Not Found
```bash
# Run the copy script to set up the local dataset
python3 copy_sample_dataset.py
```

### Out of Memory
```python
# Reduce batch size in object_detection_lite.py
BATCH_SIZE = 4  # or even 2
```

## Files

- `object_detection_lite.py` - Main training script
- `copy_sample_dataset.py` - Dataset setup script
- `data/VOC2012/` - Local sample dataset (created by copy script)
- `output/` - Generated visualizations

## Learning Objectives

1. Object detection with bounding boxes
2. YOLO-style architecture
3. Multi-task loss (box + classification)
4. XML annotation parsing
5. Detection visualization

---

**Runtime**: ~10 seconds  
**Dataset**: 50 local VOC2012 images  
**Model**: Simplified YOLO (~6.8M parameters)