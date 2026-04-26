#!/usr/bin/env python3
"""
Lab 8 Lite: Chatbot with Bi-directional LSTM
=============================================

Lightweight version using Cornell Movie Dialogs dataset with reduced samples and epochs.

Changes from original:
- Uses 10,000 real dialog pairs (local sample dataset)
- 20 epochs (from 30)
- Smaller batch size: 32 (from 64)
- Reduced vocabulary size: 5000 words
- Simplified model architecture
- Uses real movie dialog conversations

Dataset: Cornell Movie Dialogs Corpus (local sample)
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
import random
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
NUM_SAMPLES = 10000  # Increased for better response quality
BATCH_SIZE = 32
NUM_EPOCHS = 20  # Increased for better training
MAX_LENGTH = 20
VOCAB_SIZE = 5000  # Increased vocabulary


class CornellDialogDataset(Dataset):
    """Load Cornell Movie Dialogs dataset."""
    
    def __init__(self, data_root, max_samples=None, max_length=20):
        self.data_root = Path(data_root)
        self.max_length = max_length
        
        # Load conversations
        print("Loading Cornell Movie Dialogs...")
        self.pairs = self.load_conversations()
        
        if max_samples:
            self.pairs = self.pairs[:max_samples]
        
        print(f"Loaded {len(self.pairs)} conversation pairs")
        
        # Build vocabulary
        self.build_vocabulary()
    
    def load_conversations(self):
        """Load conversation pairs from Cornell dataset."""
        # Load movie lines
        lines_file = self.data_root / "movie_lines.txt"
        lines = {}
        
        with open(lines_file, 'r', encoding='iso-8859-1') as f:
            for line in f:
                parts = line.strip().split(' +++$+++ ')
                if len(parts) == 5:
                    line_id = parts[0]
                    text = parts[4]
                    lines[line_id] = self.normalize_text(text)
        
        # Load conversations
        conv_file = self.data_root / "movie_conversations.txt"
        pairs = []
        
        with open(conv_file, 'r', encoding='iso-8859-1') as f:
            for line in f:
                parts = line.strip().split(' +++$+++ ')
                if len(parts) == 4:
                    line_ids = eval(parts[3])
                    
                    # Create pairs from consecutive lines
                    for i in range(len(line_ids) - 1):
                        if line_ids[i] in lines and line_ids[i+1] in lines:
                            input_text = lines[line_ids[i]]
                            target_text = lines[line_ids[i+1]]
                            
                            # Filter by length
                            if (len(input_text.split()) <= self.max_length and
                                len(target_text.split()) <= self.max_length):
                                pairs.append((input_text, target_text))
        
        return pairs
    
    def normalize_text(self, text):
        """Normalize text."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    def build_vocabulary(self):
        """Build vocabulary from conversations."""
        print("Building vocabulary...")
        word_counts = Counter()
        
        for input_text, target_text in self.pairs:
            word_counts.update(input_text.split())
            word_counts.update(target_text.split())
        
        # Take most common words
        most_common = word_counts.most_common(VOCAB_SIZE - 4)
        
        # Create vocabulary
        self.vocab = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
        self.vocab.extend([word for word, _ in most_common])
        
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        self.pad_idx = self.word_to_idx[PAD_TOKEN]
        self.sos_idx = self.word_to_idx[SOS_TOKEN]
        self.eos_idx = self.word_to_idx[EOS_TOKEN]
        self.unk_idx = self.word_to_idx[UNK_TOKEN]
        
        print(f"Vocabulary size: {len(self.vocab)}")
    
    def text_to_indices(self, text, add_sos=False, add_eos=False):
        """Convert text to indices."""
        words = text.split()
        indices = []
        
        if add_sos:
            indices.append(self.sos_idx)
        
        for word in words:
            idx = self.word_to_idx.get(word, self.unk_idx)
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
        input_text, target_text = self.pairs[idx]
        
        input_indices = self.text_to_indices(input_text)
        target_indices = self.text_to_indices(target_text, add_sos=True, add_eos=True)
        
        return torch.LongTensor(input_indices), torch.LongTensor(target_indices)


class Encoder(nn.Module):
    """Bi-directional LSTM encoder."""
    
    def __init__(self, vocab_size, embed_size=128, hidden_size=256):
        super(Encoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)
    
    def forward(self, x):
        # x: [batch, seq_len]
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded)
        
        # Combine bidirectional hidden states
        hidden = torch.tanh(self.fc(torch.cat([hidden[-2], hidden[-1]], dim=1)))
        
        return outputs, hidden.unsqueeze(0)


class Decoder(nn.Module):
    """LSTM decoder with attention."""
    
    def __init__(self, vocab_size, embed_size=128, hidden_size=256):
        super(Decoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size + hidden_size * 2, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.attention = nn.Linear(hidden_size * 3, 1)
    
    def forward(self, x, hidden, encoder_outputs):
        # x: [batch, 1]
        # hidden: [1, batch, hidden_size]
        # encoder_outputs: [batch, seq_len, hidden_size * 2]
        
        embedded = self.embedding(x)  # [batch, 1, embed_size]
        
        # Attention
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)
        
        # Repeat hidden for each encoder output
        hidden_repeated = hidden.squeeze(0).unsqueeze(1).repeat(1, seq_len, 1)
        
        # Compute attention scores
        attention_input = torch.cat([encoder_outputs, hidden_repeated], dim=2)
        attention_scores = self.attention(attention_input).squeeze(2)
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        # Apply attention
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        
        # Combine with embedding
        lstm_input = torch.cat([embedded, context], dim=2)
        
        # LSTM
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, hidden))
        
        # Predict
        prediction = self.fc(output.squeeze(1))
        
        return prediction, hidden


class Seq2SeqChatbot(nn.Module):
    """Sequence-to-sequence chatbot model."""
    
    def __init__(self, vocab_size, embed_size=128, hidden_size=256):
        super(Seq2SeqChatbot, self).__init__()
        
        self.encoder = Encoder(vocab_size, embed_size, hidden_size)
        self.decoder = Decoder(vocab_size, embed_size, hidden_size)
    
    def forward(self, input_seq, target_seq):
        # Encode
        encoder_outputs, hidden = self.encoder(input_seq)
        
        # Decode
        batch_size = input_seq.size(0)
        target_len = target_seq.size(1)
        vocab_size = self.decoder.fc.out_features
        
        outputs = torch.zeros(batch_size, target_len, vocab_size).to(input_seq.device)
        
        decoder_input = target_seq[:, 0].unsqueeze(1)
        
        for t in range(1, target_len):
            output, hidden = self.decoder(decoder_input, hidden, encoder_outputs)
            outputs[:, t, :] = output
            decoder_input = target_seq[:, t].unsqueeze(1)
        
        return outputs


def train_model(model, train_loader, num_epochs, device, pad_idx):
    """Train the chatbot model."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    history = {'loss': []}
    
    print("\nTraining Chatbot Model...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for input_seq, target_seq in pbar:
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(input_seq, target_seq)
            
            # Reshape for loss
            outputs = outputs[:, 1:, :].reshape(-1, outputs.size(-1))
            targets = target_seq[:, 1:].reshape(-1)
            
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
        sys.stdout.flush()
    
    return history


def generate_response(model, input_text, dataset, device, max_length=20):
    """Generate response for input text."""
    model.eval()
    
    with torch.no_grad():
        # Prepare input
        input_indices = dataset.text_to_indices(input_text)
        input_tensor = torch.LongTensor([input_indices]).to(device)
        
        # Encode
        encoder_outputs, hidden = model.encoder(input_tensor)
        
        # Decode
        decoder_input = torch.LongTensor([[dataset.sos_idx]]).to(device)
        response_indices = []
        
        for _ in range(max_length):
            output, hidden = model.decoder(decoder_input, hidden, encoder_outputs)
            predicted = output.argmax(1)
            
            predicted_idx = predicted.item()
            if predicted_idx == dataset.eos_idx:
                break
            
            response_indices.append(predicted_idx)
            decoder_input = predicted.unsqueeze(1)
        
        # Convert to text
        response_words = []
        for idx in response_indices:
            if idx in dataset.idx_to_word:
                word = dataset.idx_to_word[idx]
                if word not in [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]:
                    response_words.append(word)
        
        return ' '.join(response_words)


def demo_chatbot(model, dataset, device):
    """Demo the chatbot with sample inputs."""
    print("\n" + "="*60)
    print("Chatbot Demo")
    print("="*60)
    
    test_inputs = [
        "hello how are you",
        "what is your name",
        "tell me a joke",
        "goodbye",
        "i need help"
    ]
    
    for input_text in test_inputs:
        response = generate_response(model, input_text, dataset, device)
        print(f"\nInput: {input_text}")
        print(f"Response: {response}")


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
    print("Lab 8 Lite: Chatbot with Cornell Movie Dialogs")
    print("=" * 60)
    
    start_time = time.time()
    
    # Check if dataset exists
    data_root = Path("data/cornell")
    if not data_root.exists():
        print(f"\nError: Cornell dataset not found at {data_root}")
        print("Please run: python3 copy_sample_dataset.py")
        return
    
    # Load dataset
    print(f"\nLoading Cornell Movie Dialogs (max {NUM_SAMPLES} pairs)...")
    dataset = CornellDialogDataset(
        data_root=data_root,
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
    print("\nCreating Seq2Seq Chatbot Model...")
    model = Seq2SeqChatbot(
        vocab_size=len(dataset.vocab),
        embed_size=128,
        hidden_size=256
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    history = train_model(model, train_loader, NUM_EPOCHS, device, dataset.pad_idx)
    
    # Plot training history
    plot_training_history(history)
    
    # Demo chatbot
    demo_chatbot(model, dataset, device)
    
    elapsed_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Total execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"{'='*60}")
    print(f"\nOutputs saved to: {OUTPUT_DIR.absolute()}")
    print("\nLab 8 Lite completed successfully!")


if __name__ == "__main__":
    main()


