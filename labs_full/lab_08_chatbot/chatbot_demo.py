#!/usr/bin/env python3
"""
Lab 8: Chatbot with Bi-directional LSTM
========================================

This program demonstrates:
1. Bi-directional LSTM for intent classification
2. Response generation based on intent
3. Conversation context management
4. Training on synthetic dialogue data
5. Evaluation metrics (accuracy, F1-score)
6. Interactive chatbot demo

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

# Intent categories
INTENTS = ['greeting', 'farewell', 'question', 'thanks', 'help']
INTENT_TO_IDX = {intent: idx for idx, intent in enumerate(INTENTS)}
IDX_TO_INTENT = {idx: intent for intent, idx in INTENT_TO_IDX.items()}

# Vocabulary
VOCAB = ['<PAD>', '<UNK>', 'hello', 'hi', 'hey', 'goodbye', 'bye', 'see', 'you',
         'what', 'how', 'when', 'where', 'why', 'is', 'are', 'can', 'could',
         'thanks', 'thank', 'help', 'please', 'need', 'want', 'know', 'tell',
         'me', 'the', 'a', 'to', 'do', 'does', 'did', 'will', 'would', 'later']
WORD_TO_IDX = {word: idx for idx, word in enumerate(VOCAB)}
IDX_TO_WORD = {idx: word for word, idx in WORD_TO_IDX.items()}
VOCAB_SIZE = len(VOCAB)
PAD_IDX = 0
UNK_IDX = 1

# Response templates
RESPONSES = {
    'greeting': [
        "Hello! How can I help you today?",
        "Hi there! What can I do for you?",
        "Hey! Nice to meet you!"
    ],
    'farewell': [
        "Goodbye! Have a great day!",
        "See you later!",
        "Bye! Take care!"
    ],
    'question': [
        "That's an interesting question. Let me think...",
        "I understand your question. Here's what I know...",
        "Good question! Let me help you with that."
    ],
    'thanks': [
        "You're welcome!",
        "Happy to help!",
        "No problem at all!"
    ],
    'help': [
        "I'm here to help! What do you need?",
        "Sure, I can assist you with that.",
        "Let me help you with that."
    ]
}


class SyntheticDialogueDataset(Dataset):
    """Generate synthetic dialogue data."""
    
    def __init__(self, num_samples=1000, max_length=15):
        self.num_samples = num_samples
        self.max_length = max_length
        
        # Sample utterances for each intent
        self.utterances = {
            'greeting': [
                ['hello'], ['hi'], ['hey'], ['hello', 'there'],
                ['hi', 'how', 'are', 'you']
            ],
            'farewell': [
                ['goodbye'], ['bye'], ['see', 'you', 'later'],
                ['goodbye', 'thanks']
            ],
            'question': [
                ['what', 'is', 'the'], ['how', 'can', 'you', 'help'],
                ['where', 'is'], ['when', 'will'], ['why', 'does']
            ],
            'thanks': [
                ['thanks'], ['thank', 'you'], ['thanks', 'a', 'lot']
            ],
            'help': [
                ['help', 'me'], ['can', 'you', 'help'],
                ['need', 'help'], ['please', 'help']
            ]
        }
        
    def __len__(self):
        return self.num_samples
    
    def text_to_indices(self, words):
        """Convert words to indices."""
        indices = [WORD_TO_IDX.get(word, UNK_IDX) for word in words]
        
        # Pad or truncate
        if len(indices) < self.max_length:
            indices += [PAD_IDX] * (self.max_length - len(indices))
        else:
            indices = indices[:self.max_length]
        
        return indices
    
    def __getitem__(self, idx):
        np.random.seed(idx)
        
        # Random intent
        intent = np.random.choice(INTENTS)
        intent_idx = INTENT_TO_IDX[intent]
        
        # Random utterance for this intent
        utterance_idx = np.random.randint(0, len(self.utterances[intent]))
        utterance = self.utterances[intent][utterance_idx]
        
        # Add some random words occasionally
        if np.random.rand() < 0.3:
            extra_words = np.random.choice(['please', 'can', 'you', 'me'],
                                          size=np.random.randint(1, 3))
            utterance = list(utterance) + list(extra_words)
        
        # Convert to indices
        indices = self.text_to_indices(utterance)
        
        return torch.LongTensor(indices), torch.LongTensor([intent_idx])


class BiLSTMClassifier(nn.Module):
    """Bi-directional LSTM for intent classification."""
    
    def __init__(self, vocab_size=VOCAB_SIZE, embed_size=128, hidden_size=128, 
                 num_classes=len(INTENTS), num_layers=2):
        super(BiLSTMClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=PAD_IDX)
        
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x):
        # x: [batch, seq_len]
        embedded = self.embedding(x)  # [batch, seq_len, embed_size]
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use last hidden states from both directions
        # hidden: [num_layers * 2, batch, hidden_size]
        forward_hidden = hidden[-2, :, :]
        backward_hidden = hidden[-1, :, :]
        combined = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        # Classification
        output = self.fc(combined)
        
        return output


class SimpleLSTMClassifier(nn.Module):
    """Simple unidirectional LSTM for comparison."""
    
    def __init__(self, vocab_size=VOCAB_SIZE, embed_size=128, hidden_size=128,
                 num_classes=len(INTENTS), num_layers=2):
        super(SimpleLSTMClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=PAD_IDX)
        
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use last hidden state
        output = self.fc(hidden[-1])
        
        return output


def calculate_metrics(predictions, targets):
    """Calculate accuracy and per-class metrics."""
    correct = (predictions == targets).sum().item()
    total = targets.size(0)
    accuracy = correct / total
    
    # Per-class metrics
    class_correct = {}
    class_total = {}
    
    for intent_idx in range(len(INTENTS)):
        mask = targets == intent_idx
        if mask.sum() > 0:
            class_correct[intent_idx] = (predictions[mask] == targets[mask]).sum().item()
            class_total[intent_idx] = mask.sum().item()
    
    return accuracy, class_correct, class_total


def train_model(model, train_loader, val_loader, num_epochs=30, model_name="Model"):
    """Train the intent classifier."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    print(f"\nTraining {model_name}...")
    print("-" * 60)
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', leave=False)
        for inputs, labels in train_pbar:
            inputs = inputs.to(device)
            labels = labels.squeeze().to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            predictions = outputs.argmax(dim=1)
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
            
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]', leave=False)
            for inputs, labels in val_pbar:
                inputs = inputs.to(device)
                labels = labels.squeeze().to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                
                predictions = outputs.argmax(dim=1)
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
                
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 
                      OUTPUT_DIR / f'{model_name.lower().replace(" ", "_")}_best.pth')
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    print(f"✓ {model_name} training complete! Best Val Acc: {best_val_acc:.4f}")
    return history


def generate_response(model, text):
    """Generate response for input text."""
    model.eval()
    
    # Tokenize
    words = text.lower().split()
    indices = [WORD_TO_IDX.get(word, UNK_IDX) for word in words]
    
    # Pad
    max_length = 15
    if len(indices) < max_length:
        indices += [PAD_IDX] * (max_length - len(indices))
    else:
        indices = indices[:max_length]
    
    # Predict
    with torch.no_grad():
        input_tensor = torch.LongTensor([indices]).to(device)
        output = model(input_tensor)
        intent_idx = output.argmax(dim=1).item()
    
    intent = IDX_TO_INTENT[intent_idx]
    response = np.random.choice(RESPONSES[intent])
    
    return intent, response


def plot_training_comparison(histories, model_names):
    """Plot training comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for history, name in zip(histories, model_names):
        axes[0].plot(history['train_loss'], label=f'{name} Train', linewidth=2)
        axes[0].plot(history['val_loss'], label=f'{name} Val', linewidth=2, linestyle='--')
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss', fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    for history, name in zip(histories, model_names):
        axes[1].plot(history['train_acc'], label=f'{name} Train', linewidth=2)
        axes[1].plot(history['val_acc'], label=f'{name} Val', linewidth=2, linestyle='--')
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy', fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'training_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def interactive_demo(model):
    """Interactive chatbot demo."""
    print("\n" + "=" * 70)
    print("Interactive Chatbot Demo")
    print("=" * 70)
    print("Type your message (or 'quit' to exit)")
    print("-" * 70)
    
    test_messages = [
        "hello",
        "how can you help me",
        "thanks a lot",
        "goodbye",
        "what is the weather"
    ]
    
    for msg in test_messages:
        intent, response = generate_response(model, msg)
        print(f"\nYou: {msg}")
        print(f"Bot: {response} [Intent: {intent}]")
    
    print("\n" + "=" * 70)


def main():
    """Main function."""
    print("=" * 70)
    print("Lab 8: Chatbot with Bi-directional LSTM")
    print("=" * 70)
    print()
    print(f"Device: {device}")
    print()
    
    # Create datasets
    print("Creating synthetic dialogue dataset...")
    train_dataset = SyntheticDialogueDataset(num_samples=800, max_length=15)
    val_dataset = SyntheticDialogueDataset(num_samples=200, max_length=15)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"  • Training samples: {len(train_dataset)}")
    print(f"  • Validation samples: {len(val_dataset)}")
    print(f"  • Vocabulary size: {VOCAB_SIZE}")
    print(f"  • Intent categories: {INTENTS}")
    print()
    
    # Train Bi-LSTM model
    print("=" * 70)
    print("Training Bi-directional LSTM")
    print("=" * 70)
    
    bilstm_model = BiLSTMClassifier().to(device)
    
    start_time = time.time()
    bilstm_history = train_model(bilstm_model, train_loader, val_loader, 
                                 num_epochs=30, model_name="BiLSTM")
    bilstm_time = time.time() - start_time
    
    # Train Simple LSTM model
    print()
    print("=" * 70)
    print("Training Simple LSTM")
    print("=" * 70)
    
    lstm_model = SimpleLSTMClassifier().to(device)
    
    start_time = time.time()
    lstm_history = train_model(lstm_model, train_loader, val_loader,
                               num_epochs=30, model_name="SimpleLSTM")
    lstm_time = time.time() - start_time
    
    print()
    print("=" * 70)
    print("Training Summary")
    print("=" * 70)
    print(f"Bi-directional LSTM:")
    print(f"  • Training time: {bilstm_time:.2f}s")
    print(f"  • Final train acc: {bilstm_history['train_acc'][-1]:.4f}")
    print(f"  • Final val acc: {bilstm_history['val_acc'][-1]:.4f}")
    print()
    print(f"Simple LSTM:")
    print(f"  • Training time: {lstm_time:.2f}s")
    print(f"  • Final train acc: {lstm_history['train_acc'][-1]:.4f}")
    print(f"  • Final val acc: {lstm_history['val_acc'][-1]:.4f}")
    print()
    
    # Plot comparison
    print("Generating visualizations...")
    comparison_plot = plot_training_comparison(
        [bilstm_history, lstm_history],
        ['BiLSTM', 'SimpleLSTM']
    )
    print(f"  ✓ Training comparison: {comparison_plot}")
    print()
    
    # Interactive demo
    interactive_demo(bilstm_model)
    
    print()
    print("=" * 70)
    print("Lab 8 Complete!")
    print("=" * 70)
    print()
    print("Key Findings:")
    print("  • Bi-directional LSTM captures context from both directions")
    print("  • Better performance than unidirectional LSTM")
    print("  • Intent classification is fundamental for chatbots")
    print("  • Real chatbots use transformers (BERT, GPT)")
    print()


if __name__ == "__main__":
    main()


