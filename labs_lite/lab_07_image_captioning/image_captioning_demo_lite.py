#!/usr/bin/env python3
"""
Lab 7: Image Captioning with RNN and LSTM (LIGHTWEIGHT VERSION)
================================================================

Lightweight version optimized for faster training on laptops.
Changes from original:
- Training samples: 500 (from 1000)
- Validation samples: 100 (from 200)
- Epochs: 15 (from 30)
- Image size: 32x32 (from 64x64)
- Embedding dim: 128 (from 256)
- Hidden dim: 256 (from 512)
- Expected time: ~2-3 minutes (vs ~6-8 minutes)

This program demonstrates:
1. Image captioning with CNN encoder + RNN/LSTM decoder
2. Comparison between Vanilla RNN and LSTM
3. Attention mechanism (simplified)
4. BLEU score evaluation
5. Beam search decoding
6. Caption generation and visualization

Author: Deep Learning Lab
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import time
from collections import Counter
from tqdm import tqdm

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# Create output directory
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 70)
print("Lab 7: Image Captioning (LIGHTWEIGHT VERSION)")
print("=" * 70)
print(f"Device: {device}")
print("Training samples: 500, Validation: 100, Epochs: 15")
print("Image size: 32x32, Embedding: 128, Hidden: 256")
print("Expected training time: ~2-3 minutes")
print("=" * 70)
print()
print("Note: This is a simplified demonstration version.")
print("For full implementation, see image_captioning_demo.py")
print()
