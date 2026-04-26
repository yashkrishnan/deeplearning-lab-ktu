#!/usr/bin/env python3
"""
Lab 10 Lite: Sequence to Sequence Learning
===========================================

Lightweight version using real English-French translation dataset with reduced samples and epochs.

Changes from original:
- Uses 2,000 real translation pairs (from ~175,000)
- 10 epochs (from 30)
- Smaller batch size: 32 (from 64)
- Reduced vocabulary size: 3000 words per language
- Simplified model architecture
- Uses real English-French translation pairs

Dataset: English-French Translation Dataset
Expected runtime: ~8-12 minutes on CPU

Author: Deep Learning Lab (Lite Version)
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
import random
from tqdm import tqdm
import pandas as pd
from collections import Counter
import re

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
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'
UNK_TOKEN = '<UNK>'

# Lite configuration
NUM_SAMPLES = 2000
BATCH_SIZE = 32
NUM_EPOCHS = 10
MAX_LENGTH = 20
VOCAB_SIZE = 3000


class TranslationDataset(Dataset):
    """Load English-French translation dataset."""
    
    def __init__(self, data_file, max_samples=None, max_length=20):
        self.max_length = max_length
        
        # Load translations
        print(f"Loading translation data from {data_file.name}...")
        df = pd.read_csv(data_file)
        
        # Filter by length and take samples
        self.pairs = []
        for _, row in df.iterrows():
            en_text = self.normalize_text(str(row['en']))
            fr_text = self.normalize_text(str(row['fr']))
            
            en_words = en_text.split()
            fr_words = fr_text.split()
            
            if (len(en_words) <= max_length and len(fr_words) <= max_length and
                len(en_words) > 0 and len(fr_words) > 0):
                self.pairs.append((en_text, fr_text))
                
                if max_samples and len(self.pairs) >= max_samples:
                    break
        
        print(f"Loaded {len(self.pairs)} translation pairs")
        
        # Build vocabularies
        self.build_vocabularies()
    
    def normalize_text(self, text):
        """Normalize text."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s']", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    def build_vocabularies(self):
        """Build vocabularies for source and target languages."""
        print("Building vocabularies...")
        
        en_word_counts = Counter()
        fr_word_counts = Counter()
        
        for en_text, fr_text in self.pairs:
            en_word_counts.update(en_text.split())
            fr_word_counts.update(fr_text.split())
        
        # English vocabulary
        en_most_common = en_word_counts.most_common(VOCAB_SIZE - 4)
        self.en_vocab = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
        self.en_vocab.extend([word for word, _ in en_most_common])
        
        self.en_word_to_idx = {word: idx for idx, word in enumerate(self.en_vocab)}
        self.en_idx_to_word = {idx: word for word, idx in self.en_word_to_idx.items()}
        
        # French vocabulary
        fr_most_common = fr_word_counts.most_common(VOCAB_SIZE - 4)
        self.fr_vocab = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
        self.fr_vocab.extend([word for word, _ in fr_most_common])
        
        self.fr_word_to_idx = {word: idx for idx, word in enumerate(self.fr_vocab)}
        self.fr_idx_to_word = {idx: word for word, idx in self.fr_word_to_idx.items()}
        
        # Special token indices
        self.pad_idx = 0
        self.sos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3
        
        print(f"English vocabulary size: {len(self.en_vocab)}")
        print(f"French vocabulary size: {len(self.fr_vocab)}")
    
    def text_to_indices(self, text, vocab, add_sos=False, add_eos=False):
        """Convert text to indices."""
        words = text.split()
        indices = []
        
        if add_sos:
            indices.append(self.sos_idx)
        
        for word in words:
            idx = vocab.get(word, self.unk_idx)
            indices.append(idx)
        
        if add_eos:
            indices.append(self.eos_idx)
        
        # Pad or truncate
        if len(indices) < self.max_length:
            indices += [self.pad_idx] * (self.max_length - len(indices))
        else:
            indices = indices[:self.max_length]
        
        return indices
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        en_text, fr_text = self.pairs[idx]
        
        en_indices = self.text_to_indices(en_text, self.en_word_to_idx, add_eos=True)
        fr_indices = self.text_to_indices(fr_text, self.fr_word_to_idx, add_sos=True, add_eos=True)
        
        return torch.LongTensor(en_indices), torch.LongTensor(fr_indices)


class Encoder(nn.Module):
    """LSTM Encoder."""
    
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, num_layers=2, dropout=0.3):
        super(Encoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
    def forward(self, x):
        # x: [batch, seq_len]
        embedded = self.embedding(x)  # [batch, seq_len, embed_size]
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell


class Decoder(nn.Module):
    """LSTM Decoder with attention."""
    
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, num_layers=2, dropout=0.3):
        super(Decoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_size + hidden_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.attention = nn.Linear(hidden_size * 2, 1)
        
    def forward(self, x, hidden, cell, encoder_outputs):
        # x: [batch, 1]
        embedded = self.embedding(x)  # [batch, 1, embed_size]
        
        # Attention
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)
        
        # Repeat hidden for attention
        hidden_repeated = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)
        
        # Compute attention scores
        attention_input = torch.cat([encoder_outputs, hidden_repeated], dim=2)
        attention_scores = self.attention(attention_input).squeeze(2)
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        # Apply attention
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        
        # Combine with embedding
        lstm_input = torch.cat([embedded, context], dim=2)
        
        # LSTM
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        
        # Predict
        prediction = self.fc(output.squeeze(1))
        
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    """Sequence-to-sequence model with attention."""
    
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_size=128, hidden_size=256):
        super(Seq2Seq, self).__init__()
        
        self.encoder = Encoder(src_vocab_size, embed_size, hidden_size)
        self.decoder = Decoder(tgt_vocab_size, embed_size, hidden_size)
    
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        tgt_vocab_size = self.decoder.fc.out_features
        
        # Encode
        encoder_outputs, hidden, cell = self.encoder(src)
        
        # Decode
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(src.device)
        decoder_input = tgt[:, 0].unsqueeze(1)
        
        for t in range(1, tgt_len):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell, encoder_outputs)
            outputs[:, t, :] = output
            
            # Teacher forcing
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            if use_teacher_forcing:
                decoder_input = tgt[:, t].unsqueeze(1)
            else:
                decoder_input = output.argmax(1).unsqueeze(1)
        
        return outputs


def train_model(model, train_loader, num_epochs, device, pad_idx):
    """Train the translation model."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    history = {'loss': []}
    
    print("\nTraining Seq2Seq Translation Model...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for src, tgt in pbar:
            src = src.to(device)
            tgt = tgt.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(src, tgt, teacher_forcing_ratio=0.5)
            
            # Reshape for loss
            outputs = outputs[:, 1:, :].reshape(-1, outputs.size(-1))
            targets = tgt[:, 1:].reshape(-1)
            
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        
        avg_loss = epoch_loss / len(train_loader)
        history['loss'].append(avg_loss)
        
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}")
    
    return history


def translate(model, src_text, dataset, device, max_length=20):
    """Translate a source sentence."""
    model.eval()
    
    with torch.no_grad():
        # Prepare input
        src_indices = dataset.text_to_indices(src_text, dataset.en_word_to_idx, add_eos=True)
        src_tensor = torch.LongTensor([src_indices]).to(device)
        
        # Encode
        encoder_outputs, hidden, cell = model.encoder(src_tensor)
        
        # Decode
        decoder_input = torch.LongTensor([[dataset.sos_idx]]).to(device)
        translation_indices = []
        
        for _ in range(max_length):
            output, hidden, cell = model.decoder(decoder_input, hidden, cell, encoder_outputs)
            predicted = output.argmax(1)
            
            predicted_idx = predicted.item()
            if predicted_idx == dataset.eos_idx:
                break
            
            translation_indices.append(predicted_idx)
            decoder_input = predicted.unsqueeze(1)
        
        # Convert to text
        translation_words = []
        for idx in translation_indices:
            if idx in dataset.fr_idx_to_word:
                word = dataset.fr_idx_to_word[idx]
                if word not in [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]:
                    translation_words.append(word)
        
        return ' '.join(translation_words)


def demo_translations(model, dataset, device):
    """Demo translations with sample inputs."""
    print("\n" + "="*60)
    print("Translation Demo")
    print("="*60)
    
    # Get some test samples
    test_indices = random.sample(range(len(dataset)), 5)
    
    for idx in test_indices:
        en_text, fr_text = dataset.pairs[idx]
        translation = translate(model, en_text, dataset, device)
        
        print(f"\nEnglish: {en_text}")
        print(f"Reference: {fr_text}")
        print(f"Translation: {translation}")


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
    print("Lab 10 Lite: Seq2Seq Translation (English-French)")
    print("=" * 60)
    
    start_time = time.time()
    
    # Check if dataset exists (in labs_lite directory)
    data_file = Path("data/translation/en-fr.csv")
    if not data_file.exists():
        print(f"\nError: Translation dataset not found at {data_file}")
        print("Please ensure the dataset is downloaded.")
        return
    
    # Load dataset
    print(f"\nLoading translation dataset (max {NUM_SAMPLES} pairs)...")
    dataset = TranslationDataset(
        data_file=data_file,
        max_samples=NUM_SAMPLES,
        max_length=MAX_LENGTH
    )
    
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    
    # Create model
    print("\nCreating Seq2Seq Translation Model...")
    model = Seq2Seq(
        src_vocab_size=len(dataset.en_vocab),
        tgt_vocab_size=len(dataset.fr_vocab),
        embed_size=128,
        hidden_size=256
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    history = train_model(model, train_loader, NUM_EPOCHS, device, dataset.pad_idx)
    
    # Plot training history
    plot_training_history(history)
    
    # Demo translations
    demo_translations(model, dataset, device)
    
    elapsed_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Total execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"{'='*60}")
    print(f"\nOutputs saved to: {OUTPUT_DIR.absolute()}")
    print("\nLab 10 Lite completed successfully!")


if __name__ == "__main__":
    main()


