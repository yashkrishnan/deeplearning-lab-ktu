#!/usr/bin/env python3
"""
Lab 7 Lite: Image Captioning with RNN and LSTM
===============================================

Lightweight version using Flickr8k dataset with reduced samples and epochs.

Changes from original:
- Uses 2000 real Flickr8k images (local sample dataset)
- 30 epochs (from 30)
- Smaller batch size: 16 (from 32)
- Reduced vocabulary size
- Simplified model architecture
- Uses real image-caption pairs

Dataset: Flickr8k (local sample)
Expected runtime: ~10-15 minutes on CPU

Author: Deep Learning Lab (Lite Version)
"""

import os
import sys
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
from PIL import Image
import pandas as pd
import random

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Create output directory
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Special tokens
PAD_TOKEN = '<PAD>'
START_TOKEN = '<START>'
END_TOKEN = '<END>'
UNK_TOKEN = '<UNK>'

# Lite configuration
NUM_SAMPLES = 2000  # Using local sample dataset
BATCH_SIZE = 16
NUM_EPOCHS = 30  # Full epochs for better caption quality
IMG_SIZE = 128
MAX_CAPTION_LEN = 20
VOCAB_SIZE = 5000  # Vocabulary for 2000 samples


class Flickr8kDataset(Dataset):
    """Load Flickr8k dataset for image captioning."""
    
    def __init__(self, data_root, img_size=128, max_samples=None, max_caption_len=20):
        self.data_root = Path(data_root)
        self.img_size = img_size
        self.max_caption_len = max_caption_len
        
        # Paths
        self.img_dir = self.data_root / "Images"
        captions_file = self.data_root / "captions.txt"
        
        # Load captions
        print("Loading captions...")
        df = pd.read_csv(captions_file)
        
        # Group by image and take first caption for each
        self.samples = []
        seen_images = set()
        
        for _, row in df.iterrows():
            img_name = row['image']
            if img_name not in seen_images:
                img_path = self.img_dir / img_name
                if img_path.exists():
                    caption = row['caption']
                    self.samples.append((img_name, caption))
                    seen_images.add(img_name)
                    
                    if max_samples and len(self.samples) >= max_samples:
                        break
        
        print(f"Loaded {len(self.samples)} image-caption pairs")
        
        # Build vocabulary
        self.build_vocabulary()
    
    def build_vocabulary(self):
        """Build vocabulary from captions."""
        print("Building vocabulary...")
        word_counts = Counter()
        
        for _, caption in self.samples:
            words = caption.lower().split()
            word_counts.update(words)
        
        # Take most common words
        most_common = word_counts.most_common(VOCAB_SIZE - 4)  # Reserve space for special tokens
        
        # Create vocabulary
        self.vocab = [PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN]
        self.vocab.extend([word for word, _ in most_common])
        
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        self.pad_idx = self.word_to_idx[PAD_TOKEN]
        self.start_idx = self.word_to_idx[START_TOKEN]
        self.end_idx = self.word_to_idx[END_TOKEN]
        self.unk_idx = self.word_to_idx[UNK_TOKEN]
        
        print(f"Vocabulary size: {len(self.vocab)}")
    
    def caption_to_indices(self, caption):
        """Convert caption to indices."""
        words = caption.lower().split()
        indices = [self.start_idx]
        
        for word in words:
            idx = self.word_to_idx.get(word, self.unk_idx)
            indices.append(idx)
            if len(indices) >= self.max_caption_len - 1:
                break
        
        indices.append(self.end_idx)
        
        # Pad
        while len(indices) < self.max_caption_len:
            indices.append(self.pad_idx)
        
        return indices[:self.max_caption_len]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_name, caption = self.samples[idx]
        
        # Load and preprocess image
        img_path = self.img_dir / img_name
        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.img_size, self.img_size))
        img = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
        
        # Convert caption to indices
        caption_indices = self.caption_to_indices(caption)
        
        return torch.FloatTensor(img), torch.LongTensor(caption_indices)


class CNNEncoder(nn.Module):
    """CNN encoder for image features."""
    
    def __init__(self, embed_size=256):
        super(CNNEncoder, self).__init__()
        
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.fc = nn.Linear(256, embed_size)
    
    def forward(self, x):
        features = self.cnn(x)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        return features


class LSTMDecoder(nn.Module):
    """LSTM decoder for caption generation."""
    
    def __init__(self, embed_size=256, hidden_size=256, vocab_size=2000, num_layers=1):
        super(LSTMDecoder, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        # features: [batch, embed_size]
        # captions: [batch, seq_len]
        
        # Embed captions
        embeddings = self.embedding(captions)  # [batch, seq_len, embed_size]
        
        # Prepend image features
        features = features.unsqueeze(1)  # [batch, 1, embed_size]
        embeddings = torch.cat([features, embeddings[:, :-1, :]], dim=1)
        
        # LSTM
        lstm_out, _ = self.lstm(embeddings)
        
        # Predict
        outputs = self.fc(lstm_out)
        
        return outputs


class ImageCaptioningModel(nn.Module):
    """Combined encoder-decoder model."""
    
    def __init__(self, embed_size=256, hidden_size=256, vocab_size=2000):
        super(ImageCaptioningModel, self).__init__()
        
        self.encoder = CNNEncoder(embed_size)
        self.decoder = LSTMDecoder(embed_size, hidden_size, vocab_size)
    
    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs


def train_model(model, train_loader, num_epochs, device, pad_idx):
    """Train the captioning model."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    history = {'loss': []}
    
    print("\nTraining Image Captioning Model...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, captions in pbar:
            images = images.to(device)
            captions = captions.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images, captions)
            
            # Reshape for loss computation
            outputs = outputs.view(-1, outputs.size(-1))
            targets = captions.view(-1)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        
        avg_loss = epoch_loss / len(train_loader)
        history['loss'].append(avg_loss)
        
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}")
        sys.stdout.flush()
    
    return history


def generate_caption(model, image, dataset, device, max_length=20):
    """Generate caption for an image."""
    model.eval()
    
    with torch.no_grad():
        # Encode image
        image = image.unsqueeze(0).to(device)
        features = model.encoder(image)
        
        # Start with START token
        caption = [dataset.start_idx]
        
        for _ in range(max_length):
            caption_tensor = torch.LongTensor([caption]).to(device)
            outputs = model.decoder(features, caption_tensor)
            
            # Get last prediction
            predicted = outputs[0, -1, :].argmax().item()
            caption.append(predicted)
            
            if predicted == dataset.end_idx:
                break
        
        # Convert to words
        words = []
        for idx in caption[1:-1]:  # Skip START and END
            if idx in dataset.idx_to_word:
                word = dataset.idx_to_word[idx]
                if word not in [PAD_TOKEN, START_TOKEN, END_TOKEN]:
                    words.append(word)
        
        return ' '.join(words)


def visualize_captions(model, dataset, device, num_samples=4):
    """Visualize generated captions."""
    model.eval()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.ravel()
    
    indices = random.sample(range(len(dataset)), num_samples)
    
    for idx, ax in zip(indices, axes):
        img, caption_indices = dataset[idx]
        
        # Generate caption
        generated = generate_caption(model, img, dataset, device)
        
        # Get ground truth
        gt_words = []
        for idx_val in caption_indices.numpy():
            if idx_val in dataset.idx_to_word:
                word = dataset.idx_to_word[idx_val]
                if word not in [PAD_TOKEN, START_TOKEN, END_TOKEN]:
                    gt_words.append(word)
        gt_caption = ' '.join(gt_words)
        
        # Display
        img_display = img.permute(1, 2, 0).numpy()
        ax.imshow(img_display)
        ax.axis('off')
        ax.set_title(f'Generated: {generated}\n\nGround Truth: {gt_caption}',
                    fontsize=8, wrap=True)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'caption_results.png', dpi=150, bbox_inches='tight')
    print(f"Saved caption visualization to {OUTPUT_DIR / 'caption_results.png'}")
    plt.close()


def plot_training_history(history):
    """Plot training history."""
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], 'b-', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'training_history.png', dpi=150, bbox_inches='tight')
    print(f"Saved training history to {OUTPUT_DIR / 'training_history.png'}")
    plt.close()


def main():
    print("=" * 60)
    print("Lab 7 Lite: Image Captioning with Flickr8k")
    print("=" * 60)
    
    start_time = time.time()
    
    # Check if dataset exists
    data_root = Path("data/flickr8k")
    if not data_root.exists():
        print(f"\nError: Flickr8k dataset not found at {data_root}")
        print("Please run: python3 copy_sample_dataset.py")
        return
    
    # Load dataset
    print(f"\nLoading Flickr8k dataset (max {NUM_SAMPLES} samples)...")
    dataset = Flickr8kDataset(
        data_root=data_root,
        img_size=IMG_SIZE,
        max_samples=NUM_SAMPLES,
        max_caption_len=MAX_CAPTION_LEN
    )
    
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    
    # Create model
    print("\nCreating Image Captioning Model...")
    model = ImageCaptioningModel(
        embed_size=256,
        hidden_size=256,
        vocab_size=len(dataset.vocab)
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    history = train_model(model, train_loader, NUM_EPOCHS, device, dataset.pad_idx)
    
    # Plot training history
    plot_training_history(history)
    
    # Visualize results
    print("\nGenerating caption visualizations...")
    visualize_captions(model, dataset, device, num_samples=4)
    
    elapsed_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Total execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"{'='*60}")
    print(f"\nOutputs saved to: {OUTPUT_DIR.absolute()}")
    print("\nLab 7 Lite completed successfully!")


if __name__ == "__main__":
    main()


