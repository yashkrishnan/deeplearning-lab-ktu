#!/usr/bin/env python3
"""
Lab 9: Time Series Forecasting (LIGHTWEIGHT VERSION)
=====================================================

Lightweight version optimized for faster training on laptops.
Changes from original:
- Training samples: 500 (from 1000)
- Sequence length: 30 (from 60)
- Epochs: 20 (from 50)
- LSTM layers: 1 (from 2)
- Hidden dim: 32 (from 64)
- Expected time: ~1-2 minutes (vs ~4-5 minutes)

This program demonstrates:
1. LSTM for time series forecasting
2. Comparison with GRU and vanilla RNN
3. Multi-step ahead prediction
4. Evaluation metrics (MSE, MAE, MAPE)
5. Visualization of predictions

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
print("Lab 9: Time Series Forecasting (LIGHTWEIGHT VERSION)")
print("=" * 70)
print(f"Device: {device}")
print("Training samples: 500, Sequence length: 30, Epochs: 20")
print("LSTM layers: 1, Hidden: 32")
print("Expected training time: ~1-2 minutes")
print("=" * 70)
print()
print("Note: This is a simplified demonstration version.")
print("For full implementation, see time_series_demo.py")
print()
