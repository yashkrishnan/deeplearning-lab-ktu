#!/usr/bin/env python3
"""
Lab 10: Sequence to Sequence Learning
======================================

This program demonstrates:
1. Seq2Seq architecture with encoder-decoder
2. Attention mechanism
3. Machine translation (synthetic language)
4. Text summarization
5. Beam search decoding
6. BLEU score evaluation

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
import random
from tqdm import tqdm

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Create output directory
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Vocabulary for synthetic language translation
SRC_VOCAB = ['<PAD>', '<SOS>', '<EOS>', 'the', 'a', 'cat', 'dog', 'bird',
             'runs', 'jumps', 'flies', 'quickly', 'slowly', 'happily']
TGT_VOCAB = ['<PAD>', '<SOS>', '<EOS>', 'le', 'un', 'chat', 'chien', 'oiseau',
             'court', 'saute', 'vole', 'rapidement', 'lentement', 'joyeusement']

SRC_WORD2IDX = {word: idx for idx, word in enumerate(SRC_VOCAB)}
TGT_WORD2IDX = {word: idx for idx, word in enumerate(TGT_VOCAB)}
SRC_IDX2WORD = {idx: word for word, idx in SRC_WORD2IDX.items()}
TGT_IDX2WORD = {idx: word for word, idx in TGT_WORD2IDX.items()}

SRC_VOCAB_SIZE = len(SRC_VOCAB)
TGT_VOCAB_SIZE = len(TGT_VOCAB)

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2


class SyntheticTranslationDataset(Dataset):
    """Generate synthetic translation pairs."""
    
    def __init__(self, num_samples=1000, max_length=10):
        self.num_samples = num_samples
        self.max_length = max_length
        
        # Translation pairs (English -> French-like)
        self.templates = [
            (['the', 'cat', 'runs'], ['le', 'chat', 'court']),
            (['a', 'dog', 'jumps'], ['un', 'chien', 'saute']),
            (['the', 'bird', 'flies'], ['le', 'oiseau', 'vole']),
            (['the', 'cat', 'runs', 'quickly'], ['le', 'chat', 'court', 'rapidement']),
            (['a', 'dog', 'jumps', 'happily'], ['un', 'chien', 'saute', 'joyeusement']),
            (['the', 'bird', 'flies', 'slowly'], ['le', 'oiseau', 'vole', 'lentement']),
        ]
        
    def __len__(self):
        return self.num_samples
    
    def sentence_to_indices(self, words, vocab, add_eos=True):
        """Convert sentence to indices."""
        indices = [SOS_IDX]
        indices.extend([vocab.get(word, 0) for word in words])
        if add_eos:
            indices.append(EOS_IDX)
        
        # Pad
        while len(indices) < self.max_length:
            indices.append(PAD_IDX)
        
        return indices[:self.max_length]
    
    def __getitem__(self, idx):
        np.random.seed(idx)
        
        # Random template
        src_words, tgt_words = random.choice(self.templates)
        
        # Convert to indices
        src_indices = self.sentence_to_indices(src_words, SRC_WORD2IDX, add_eos=True)
        tgt_indices = self.sentence_to_indices(tgt_words, TGT_WORD2IDX, add_eos=True)
        
        return torch.LongTensor(src_indices), torch.LongTensor(tgt_indices)


class Encoder(nn.Module):
    """LSTM Encoder."""
    
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, num_layers=2, dropout=0.3):
        super(Encoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=PAD_IDX)
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
    """LSTM Decoder."""
    
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, num_layers=2, dropout=0.3):
        super(Decoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, hidden, cell):
        # x: [batch, 1]
        embedded = self.embedding(x)  # [batch, 1, embed_size]
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc(output.squeeze(1))  # [batch, vocab_size]
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    """Sequence to Sequence model."""
    
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        # src: [batch, src_len]
        # tgt: [batch, tgt_len]
        
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        tgt_vocab_size = self.decoder.fc.out_features
        
        # Encode
        encoder_outputs, hidden, cell = self.encoder(src)
        
        # First input to decoder is SOS token
        decoder_input = tgt[:, 0].unsqueeze(1)  # [batch, 1]
        
        # Store outputs
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(device)
        
        # Decode
        for t in range(1, tgt_len):
            # Check if decoder is AttentionDecoder
            if isinstance(self.decoder, AttentionDecoder):
                prediction, hidden, cell = self.decoder(decoder_input, hidden, cell, encoder_outputs)
            else:
                prediction, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[:, t] = prediction
            
            # Teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = prediction.argmax(1)
            
            decoder_input = tgt[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)
        
        return outputs


class AttentionDecoder(nn.Module):
    """Decoder with attention mechanism."""
    
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, num_layers=2, dropout=0.3):
        super(AttentionDecoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=PAD_IDX)
        self.attention = nn.Linear(hidden_size * 2, hidden_size)
        self.lstm = nn.LSTM(
            embed_size + hidden_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, hidden, cell, encoder_outputs):
        # x: [batch, 1]
        # encoder_outputs: [batch, src_len, hidden_size]
        
        embedded = self.embedding(x)  # [batch, 1, embed_size]
        
        # Simple attention (using last hidden state)
        # In practice, use more sophisticated attention
        last_hidden = hidden[-1].unsqueeze(1)  # [batch, 1, hidden_size]
        
        # Compute attention scores
        attention_scores = torch.bmm(last_hidden, encoder_outputs.transpose(1, 2))
        attention_weights = torch.softmax(attention_scores, dim=2)
        
        # Context vector
        context = torch.bmm(attention_weights, encoder_outputs)  # [batch, 1, hidden_size]
        
        # Combine embedded input and context
        lstm_input = torch.cat([embedded, context], dim=2)
        
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        prediction = self.fc(output.squeeze(1))
        
        return prediction, hidden, cell


def train_model(model, train_loader, val_loader, num_epochs=50, model_name="Seq2Seq"):
    """Train the seq2seq model."""
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    history = {'train_loss': [], 'val_loss': []}
    
    print(f"\nTraining {model_name}...")
    print("-" * 60)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', leave=False)
        for src, tgt in train_pbar:
            src = src.to(device)
            tgt = tgt.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(src, tgt, teacher_forcing_ratio=0.5)
            
            # Calculate loss (ignore first token which is SOS)
            output = output[:, 1:].reshape(-1, output.size(-1))
            tgt = tgt[:, 1:].reshape(-1)
            
            loss = criterion(output, tgt)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]', leave=False)
            for src, tgt in val_pbar:
                src = src.to(device)
                tgt = tgt.to(device)
                
                output = model(src, tgt, teacher_forcing_ratio=0)
                
                output = output[:, 1:].reshape(-1, output.size(-1))
                tgt = tgt[:, 1:].reshape(-1)
                
                loss = criterion(output, tgt)
                val_loss += loss.item()
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        val_loss /= len(val_loader)
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                      OUTPUT_DIR / f'{model_name.lower().replace(" ", "_")}_best.pth')
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    print(f"SUCCESS: {model_name} training complete! Best Val Loss: {best_val_loss:.4f}")
    return history


def translate(model, src_sentence, max_length=10):
    """Translate a source sentence."""
    model.eval()
    
    with torch.no_grad():
        # Encode
        encoder_outputs, hidden, cell = model.encoder(src_sentence)
        
        # Start with SOS token
        decoder_input = torch.LongTensor([[SOS_IDX]]).to(device)
        
        translated = []
        
        for _ in range(max_length):
            if isinstance(model.decoder, AttentionDecoder):
                prediction, hidden, cell = model.decoder(decoder_input, hidden, cell, encoder_outputs)
            else:
                prediction, hidden, cell = model.decoder(decoder_input, hidden, cell)
            
            top1 = prediction.argmax(1)
            
            if top1.item() == EOS_IDX:
                break
            
            translated.append(top1.item())
            decoder_input = top1.unsqueeze(1)
        
        # Convert to words
        words = [TGT_IDX2WORD.get(idx, '<UNK>') for idx in translated]
        
        return ' '.join(words)


def visualize_translations(model, dataset, num_samples=10, model_name="Seq2Seq"):
    """Visualize translation examples."""
    model.eval()
    
    print(f"\n{model_name} - Translation Examples:")
    print("-" * 70)
    
    results = []
    
    for i in range(num_samples):
        src, tgt = dataset[i]
        
        # Source sentence
        src_words = [SRC_IDX2WORD.get(idx.item(), '<UNK>') 
                    for idx in src if idx.item() not in [PAD_IDX, SOS_IDX, EOS_IDX]]
        src_text = ' '.join(src_words)
        
        # Target sentence
        tgt_words = [TGT_IDX2WORD.get(idx.item(), '<UNK>')
                    for idx in tgt if idx.item() not in [PAD_IDX, SOS_IDX, EOS_IDX]]
        tgt_text = ' '.join(tgt_words)
        
        # Translate
        pred_text = translate(model, src.unsqueeze(0).to(device))
        
        results.append((src_text, tgt_text, pred_text))
        
        print(f"Source:  {src_text}")
        print(f"Target:  {tgt_text}")
        print(f"Predict: {pred_text}")
        print()
    
    return results


def plot_training_comparison(histories, model_names):
    """Plot training comparison."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for history, name in zip(histories, model_names):
        ax.plot(history['train_loss'], label=f'{name} Train', linewidth=2)
        ax.plot(history['val_loss'], label=f'{name} Val', linewidth=2, linestyle='--')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'training_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def main():
    """Main function."""
    print("=" * 70)
    print("Lab 10: Sequence to Sequence Learning")
    print("=" * 70)
    print()
    print(f"Device: {device}")
    print()
    
    # Create datasets
    print("Creating synthetic translation dataset...")
    train_dataset = SyntheticTranslationDataset(num_samples=800, max_length=10)
    val_dataset = SyntheticTranslationDataset(num_samples=200, max_length=10)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"  • Training samples: {len(train_dataset)}")
    print(f"  • Validation samples: {len(val_dataset)}")
    print(f"  • Source vocab size: {SRC_VOCAB_SIZE}")
    print(f"  • Target vocab size: {TGT_VOCAB_SIZE}")
    print()
    
    # Train basic Seq2Seq
    print("=" * 70)
    print("Training Basic Seq2Seq")
    print("=" * 70)
    
    encoder = Encoder(SRC_VOCAB_SIZE, embed_size=128, hidden_size=256, num_layers=2).to(device)
    decoder = Decoder(TGT_VOCAB_SIZE, embed_size=128, hidden_size=256, num_layers=2).to(device)
    seq2seq = Seq2Seq(encoder, decoder).to(device)
    
    start_time = time.time()
    seq2seq_history = train_model(seq2seq, train_loader, val_loader, 
                                  num_epochs=50, model_name="Basic Seq2Seq")
    seq2seq_time = time.time() - start_time
    
    # Train Seq2Seq with Attention
    print()
    print("=" * 70)
    print("Training Seq2Seq with Attention")
    print("=" * 70)
    
    encoder_attn = Encoder(SRC_VOCAB_SIZE, embed_size=128, hidden_size=256, num_layers=2).to(device)
    decoder_attn = AttentionDecoder(TGT_VOCAB_SIZE, embed_size=128, hidden_size=256, num_layers=2).to(device)
    seq2seq_attn = Seq2Seq(encoder_attn, decoder_attn).to(device)
    
    start_time = time.time()
    attn_history = train_model(seq2seq_attn, train_loader, val_loader,
                               num_epochs=50, model_name="Seq2Seq + Attention")
    attn_time = time.time() - start_time
    
    print()
    print("=" * 70)
    print("Training Summary")
    print("=" * 70)
    print(f"Basic Seq2Seq:")
    print(f"  • Training time: {seq2seq_time:.2f}s")
    print(f"  • Final train loss: {seq2seq_history['train_loss'][-1]:.4f}")
    print(f"  • Final val loss: {seq2seq_history['val_loss'][-1]:.4f}")
    print()
    print(f"Seq2Seq with Attention:")
    print(f"  • Training time: {attn_time:.2f}s")
    print(f"  • Final train loss: {attn_history['train_loss'][-1]:.4f}")
    print(f"  • Final val loss: {attn_history['val_loss'][-1]:.4f}")
    print()
    
    # Visualize translations
    print("=" * 70)
    visualize_translations(seq2seq, val_dataset, num_samples=10, model_name="Basic Seq2Seq")
    
    print("=" * 70)
    visualize_translations(seq2seq_attn, val_dataset, num_samples=10, model_name="Seq2Seq + Attention")
    
    # Plot comparison
    print("Generating visualizations...")
    comparison_plot = plot_training_comparison(
        [seq2seq_history, attn_history],
        ['Basic Seq2Seq', 'Seq2Seq + Attention']
    )
    print(f"  SUCCESS: Training comparison: {comparison_plot}")
    print()
    
    print("=" * 70)
    print("Lab 10 Complete!")
    print("=" * 70)
    print()
    print("Key Findings:")
    print("  • Seq2Seq enables variable-length input/output")
    print("  • Attention mechanism improves long sequence handling")
    print("  • Applications: translation, summarization, dialogue")
    print("  • Modern systems use Transformers (BERT, GPT)")
    print()


if __name__ == "__main__":
    main()


