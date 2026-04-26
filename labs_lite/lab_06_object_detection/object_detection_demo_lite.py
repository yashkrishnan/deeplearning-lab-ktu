#!/usr/bin/env python3
"""
Lab 6: Object Detection with YOLO-style Architecture (LIGHTWEIGHT VERSION)
===========================================================================

Lightweight version optimized for faster training on laptops.
Changes from original:
- Training samples: 500 (from 1000)
- Validation samples: 100 (from 200)
- Epochs: 15 (from 30)
- Image size: 128x128 (from 224x224)
- Max objects: 2 (from 3)
- Backbone channels: Reduced by 50%
- Expected time: ~3-4 minutes (vs ~8-10 minutes)

This program demonstrates:
1. Simple YOLO-style object detection architecture
2. Training on synthetic detection data
3. Bounding box prediction and classification
4. Non-Maximum Suppression (NMS)
5. Evaluation metrics (mAP, precision, recall)
6. Visualization of detection results

Author: Deep Learning Lab
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import time
from tqdm import tqdm

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# Create output directory
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Classes
CLASSES = ['car', 'person', 'bicycle']
NUM_CLASSES = len(CLASSES)

print("=" * 70)
print("Lab 6: Object Detection (LIGHTWEIGHT VERSION)")
print("=" * 70)
print(f"Device: {device}")
print("Training samples: 500, Validation: 100, Epochs: 15")
print("Image size: 128x128, Max objects: 2")
print("Expected training time: ~3-4 minutes")
print("=" * 70)
print()
print("Note: This is a simplified demonstration version.")
print("For full implementation, see object_detection_demo.py")
print()
