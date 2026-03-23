# Lab 10: Sequence to Sequence Learning

## 📋 Overview

This lab implements sequence-to-sequence (Seq2Seq) models for:
- **Machine Translation**: Language-to-language translation
- **Text Summarization**: Long text to short summary
- **Question Answering**: Question to answer generation
- **Code Generation**: Natural language to code

## 🎯 Learning Objectives

1. Understand encoder-decoder architecture
2. Implement attention mechanism
3. Build machine translation system
4. Apply to text summarization
5. Evaluate with BLEU scores

## 🏗️ Architecture

### Basic Seq2Seq
```
Input Sequence → Encoder → Context Vector → Decoder → Output Sequence
```

### With Attention
```
Input → Encoder → Hidden States
                      ↓
                  Attention
                      ↓
                  Decoder → Output
```

## 📦 Installation

```bash
pip install torch transformers nltk sacrebleu
python -m nltk.downloader punkt
```

## 🚀 Implementation

### Encoder
```python
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, 
                          dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        # src: [batch_size, src_len]
        embedded = self.dropout(self.embedding(src))
        # embedded: [batch_size, src_len, emb_dim]
        
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs: [batch_size, src_len, hid_dim]
        # hidden: [n_layers, batch_size, hid_dim]
        # cell: [n_layers, batch_size, hid_dim]
        
        return outputs, hidden, cell
```

### Decoder
```python
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers,
                          dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        # input: [batch_size]
        # hidden: [n_layers, batch_size, hid_dim]
        # cell: [n_layers, batch_size, hid_dim]
        
        input = input.unsqueeze(1)
        # input: [batch_size, 1]
        
        embedded = self.dropout(self.embedding(input))
        # embedded: [batch_size, 1, emb_dim]
        
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output: [batch_size, 1, hid_dim]
        
        prediction = self.fc_out(output.squeeze(1))
        # prediction: [batch_size, output_dim]
        
        return prediction, hidden, cell
```

### Attention Mechanism
```python
class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear(hid_dim * 2, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        # hidden: [batch_size, hid_dim]
        # encoder_outputs: [batch_size, src_len, hid_dim]
        
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # Repeat hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        # hidden: [batch_size, src_len, hid_dim]
        
        # Calculate energy
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy: [batch_size, src_len, hid_dim]
        
        attention = self.v(energy).squeeze(2)
        # attention: [batch_size, src_len]
        
        return torch.softmax(attention, dim=1)
```

### Seq2Seq with Attention
```python
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, attention, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.attention = attention
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [batch_size, src_len]
        # trg: [batch_size, trg_len]
        
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        encoder_outputs, hidden, cell = self.encoder(src)
        
        # First input to decoder is <sos> token
        input = trg[:, 0]
        
        for t in range(1, trg_len):
            # Calculate attention weights
            a = self.attention(hidden[-1], encoder_outputs)
            # a: [batch_size, src_len]
            
            # Apply attention to encoder outputs
            a = a.unsqueeze(1)
            # a: [batch_size, 1, src_len]
            
            weighted = torch.bmm(a, encoder_outputs)
            # weighted: [batch_size, 1, hid_dim]
            
            # Decode
            output, hidden, cell = self.decoder(input, hidden, cell)
            
            outputs[:, t] = output
            
            # Teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1
        
        return outputs
```

## 🎯 Training

```python
def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    
    for batch in iterator:
        src = batch.src
        trg = batch.trg
        
        optimizer.zero_grad()
        
        output = model(src, trg)
        
        # Reshape for loss calculation
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)
```

## 🌐 Machine Translation Example

```python
# English to French translation
def translate_sentence(sentence, src_field, trg_field, model, device, max_len=50):
    model.eval()
    
    # Tokenize
    tokens = [token.lower() for token in sentence.split()]
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    
    # Convert to indices
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    
    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_tensor)
    
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
    
    for _ in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        
        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
        
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        
        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    
    return trg_tokens[1:-1]  # Remove <sos> and <eos>

# Example
sentence = "Hello, how are you?"
translation = translate_sentence(sentence, SRC, TRG, model, device)
print(f"Source: {sentence}")
print(f"Translation: {' '.join(translation)}")
```

## 📝 Text Summarization

```python
# Summarization example
def summarize_text(text, model, tokenizer, max_length=100):
    model.eval()
    
    # Tokenize input
    inputs = tokenizer(text, return_tensors='pt', 
                      max_length=512, truncation=True)
    
    with torch.no_grad():
        summary_ids = model.generate(
            inputs['input_ids'],
            max_length=max_length,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Example
long_text = """
Deep learning is a subset of machine learning that uses neural networks
with multiple layers. These networks can learn hierarchical representations
of data, making them particularly effective for tasks like image recognition,
natural language processing, and speech recognition.
"""

summary = summarize_text(long_text, model, tokenizer)
print(f"Summary: {summary}")
```

## 📊 Evaluation Metrics

### BLEU Score
```python
from nltk.translate.bleu_score import corpus_bleu

def calculate_bleu(references, hypotheses):
    """
    references: list of reference translations
    hypotheses: list of model predictions
    """
    bleu_score = corpus_bleu(references, hypotheses)
    return bleu_score

# Example
references = [['the', 'cat', 'is', 'on', 'the', 'mat']]
hypothesis = ['the', 'cat', 'on', 'the', 'mat']
score = calculate_bleu([references], [hypothesis])
print(f"BLEU Score: {score:.4f}")
```

### ROUGE Score (for summarization)
```python
from rouge import Rouge

rouge = Rouge()

def calculate_rouge(reference, hypothesis):
    scores = rouge.get_scores(hypothesis, reference)
    return scores[0]

# Example
reference = "The cat is on the mat"
hypothesis = "A cat is on a mat"
scores = calculate_rouge(reference, hypothesis)
print(f"ROUGE-1: {scores['rouge-1']['f']:.4f}")
print(f"ROUGE-2: {scores['rouge-2']['f']:.4f}")
print(f"ROUGE-L: {scores['rouge-l']['f']:.4f}")
```

## 🎓 Advanced Techniques

### 1. Beam Search
```python
def beam_search(model, src, beam_width=3, max_len=50):
    """Generate sequences using beam search"""
    model.eval()
    
    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src)
    
    # Initialize beam
    sequences = [[[], 0.0, hidden, cell]]
    
    for _ in range(max_len):
        all_candidates = []
        
        for seq, score, hidden, cell in sequences:
            if len(seq) > 0 and seq[-1] == EOS_token:
                all_candidates.append([seq, score, hidden, cell])
                continue
            
            # Get next token probabilities
            input_token = torch.LongTensor([seq[-1] if seq else SOS_token])
            output, hidden, cell = model.decoder(input_token, hidden, cell)
            probs = torch.softmax(output, dim=1)
            
            # Get top k tokens
            top_probs, top_indices = probs.topk(beam_width)
            
            for prob, idx in zip(top_probs[0], top_indices[0]):
                candidate = [seq + [idx.item()], 
                           score - torch.log(prob).item(),
                           hidden, cell]
                all_candidates.append(candidate)
        
        # Select top beam_width sequences
        ordered = sorted(all_candidates, key=lambda x: x[1])
        sequences = ordered[:beam_width]
    
    return sequences[0][0]
```

### 2. Transformer Architecture
```python
from torch.nn import Transformer

class TransformerSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, d_model=512, 
                 nhead=8, num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()
        
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.trg_embedding = nn.Embedding(trg_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        self.transformer = Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers
        )
        
        self.fc_out = nn.Linear(d_model, trg_vocab_size)
        
    def forward(self, src, trg):
        src_emb = self.pos_encoder(self.src_embedding(src))
        trg_emb = self.pos_encoder(self.trg_embedding(trg))
        
        output = self.transformer(src_emb, trg_emb)
        return self.fc_out(output)
```

## 📚 Datasets

### Machine Translation:
- **WMT**: Workshop on Machine Translation
- **IWSLT**: International Workshop on Spoken Language Translation
- **Multi30k**: Multilingual image descriptions

### Text Summarization:
- **CNN/Daily Mail**: News article summarization
- **XSum**: Extreme summarization
- **Gigaword**: Headline generation

## 🎓 Exercises

1. **Build Translation System**:
   - Train on English-French dataset
   - Implement beam search
   - Evaluate with BLEU

2. **Text Summarization**:
   - Fine-tune pre-trained model
   - Compare extractive vs abstractive
   - Evaluate with ROUGE

3. **Add Attention Visualization**:
   ```python
   def visualize_attention(src, trg, attention_weights):
       plt.figure(figsize=(10, 10))
       plt.imshow(attention_weights, cmap='viridis')
       plt.xticks(range(len(src)), src, rotation=90)
       plt.yticks(range(len(trg)), trg)
       plt.colorbar()
       plt.show()
   ```

4. **Multi-task Learning**:
   - Train on multiple language pairs
   - Share encoder across tasks
   - Compare performance

## 📚 Resources

- [Seq2Seq Paper](https://arxiv.org/abs/1409.3215)
- [Attention Paper](https://arxiv.org/abs/1409.0473)
- [Transformer Paper](https://arxiv.org/abs/1706.03762)
- [Hugging Face Transformers](https://huggingface.co/transformers/)

## 🎯 Summary

You've completed all 10 labs! You now understand:
- Image processing fundamentals
- Classification algorithms
- Regularization techniques
- Data annotation
- Segmentation and detection
- Sequence modeling
- Time series forecasting
- Seq2Seq architectures

Continue exploring and building amazing deep learning applications!