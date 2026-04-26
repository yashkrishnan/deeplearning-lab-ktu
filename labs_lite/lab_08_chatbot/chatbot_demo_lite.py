#!/usr/bin/env python3
"""
Lab 8: Chatbot with Seq2Seq (LIGHTWEIGHT VERSION)
==================================================

Lightweight version optimized for faster training on laptops.
Changes from original:
- Training samples: 500 (from 1000)
- Epochs: 20 (from 50)
- Embedding dim: 128 (from 256)
- Hidden dim: 256 (from 512)
- Vocabulary: Reduced to most common 500 words
- Expected time: ~2-3 minutes (vs ~5-7 minutes)

This program demonstrates:
1. Seq2Seq architecture for conversational AI
2. Encoder-Decoder with attention
3. Training on synthetic conversation data
4. Response generation
5. Beam search decoding

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
print("Lab 8: Chatbot (LIGHTWEIGHT VERSION)")
print("=" * 70)
print(f"Device: {device}")
print("Training samples: 500, Epochs: 20")
print("Embedding: 128, Hidden: 256, Vocab: 500")
print("Expected training time: ~2-3 minutes")
print("=" * 70)
print()
print("Note: This is a simplified demonstration version.")
print("For full implementation, see chatbot_demo.py")
print()
