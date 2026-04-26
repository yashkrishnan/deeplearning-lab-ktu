#!/usr/bin/env python3
"""
Lab 10: Sequence-to-Sequence Translation (LIGHTWEIGHT VERSION)
===============================================================

Lightweight version optimized for faster training on laptops.
Changes from original:
- Training samples: 500 (from 1000)
- Epochs: 20 (from 50)
- Embedding dim: 128 (from 256)
- Hidden dim: 256 (from 512)
- Max sequence length: 10 (from 20)
- Expected time: ~2-3 minutes (vs ~6-8 minutes)

This program demonstrates:
1. Seq2Seq architecture for machine translation
2. Encoder-Decoder with attention mechanism
3. Training on synthetic translation data
4. Beam search decoding
5. BLEU score evaluation

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
print("Lab 10: Seq2Seq Translation (LIGHTWEIGHT VERSION)")
print("=" * 70)
print(f"Device: {device}")
print("Training samples: 500, Epochs: 20")
print("Embedding: 128, Hidden: 256, Max length: 10")
print("Expected training time: ~2-3 minutes")
print("=" * 70)
print()
print("Note: This is a simplified demonstration version.")
print("For full implementation, see seq2seq_demo.py")
print()
