# Lab 7: Image Captioning with RNNs and LSTMs

## 📋 Overview

This lab implements image captioning systems using:
- **Vanilla RNN**: Basic recurrent architecture
- **LSTM**: Long Short-Term Memory networks
- **Attention Mechanism**: Focus on relevant image regions

## 🎯 Learning Objectives

1. Understand encoder-decoder architecture
2. Implement vanilla RNN for caption generation
3. Build LSTM-based captioning model
4. Add attention mechanism
5. Evaluate with BLEU scores

## 🏗️ Architecture

### Encoder-Decoder Framework
```
Image → CNN Encoder → Feature Vector → RNN/LSTM Decoder → Caption
```

### Components:
1. **Encoder**: Pre-trained CNN (ResNet, VGG)
2. **Decoder**: RNN/LSTM for sequence generation
3. **Attention**: Focus on relevant image regions

## 📦 Installation

```bash
pip install torch torchvision transformers nltk
python -m nltk.downloader punkt
```

## 🚀 Implementation

### Vanilla RNN Captioning
```python
import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        
    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.linear(features)
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.rnn(embeddings)
        outputs = self.linear(hiddens)
        return outputs
```

### LSTM Captioning
```python
class DecoderLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderLSTM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
                           batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs
```

### Attention Mechanism
```python
class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(att)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return attention_weighted_encoding, alpha
```

## 📊 Training

```python
# Training loop
encoder = EncoderCNN(embed_size=256)
decoder = DecoderLSTM(embed_size=256, hidden_size=512, 
                      vocab_size=len(vocab))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()),
    lr=0.001
)

for epoch in range(num_epochs):
    for images, captions in train_loader:
        # Forward pass
        features = encoder(images)
        outputs = decoder(features, captions[:, :-1])
        
        # Calculate loss
        loss = criterion(outputs.view(-1, vocab_size), 
                        captions[:, 1:].reshape(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## 🎯 Caption Generation

```python
def generate_caption(image, encoder, decoder, vocab, max_length=20):
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        # Encode image
        features = encoder(image.unsqueeze(0))
        
        # Generate caption
        caption = []
        inputs = features.unsqueeze(1)
        hidden = None
        
        for _ in range(max_length):
            hiddens, hidden = decoder.lstm(inputs, hidden)
            outputs = decoder.linear(hiddens.squeeze(1))
            predicted = outputs.argmax(1)
            
            caption.append(predicted.item())
            if predicted.item() == vocab['<end>']:
                break
                
            inputs = decoder.embed(predicted).unsqueeze(1)
    
    # Convert to words
    words = [vocab.idx2word[idx] for idx in caption]
    return ' '.join(words)
```

## 📈 Evaluation Metrics

### BLEU Score
```python
from nltk.translate.bleu_score import sentence_bleu

def calculate_bleu(reference, candidate):
    reference = [reference.split()]
    candidate = candidate.split()
    
    bleu1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
    bleu2 = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
    bleu3 = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
    
    return bleu1, bleu2, bleu3, bleu4
```

### METEOR, CIDEr
```python
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider

meteor = Meteor()
cider = Cider()

meteor_score, _ = meteor.compute_score(references, candidates)
cider_score, _ = cider.compute_score(references, candidates)
```

## 🎓 Comparison: RNN vs LSTM

| Aspect | Vanilla RNN | LSTM |
|--------|-------------|------|
| Memory | Short-term | Long-term |
| Vanishing Gradient | Yes | No |
| Training Speed | Faster | Slower |
| Caption Quality | Lower | Higher |
| Long Sequences | Poor | Good |

## 📊 Expected Results

### Sample Captions:

**Image**: Dog playing in park
- **RNN**: "a dog is playing"
- **LSTM**: "a brown dog playing with a ball in the park"
- **LSTM + Attention**: "a golden retriever playing fetch with a red ball in a green park"

### BLEU Scores:
- RNN: BLEU-4 ~0.15
- LSTM: BLEU-4 ~0.25
- LSTM + Attention: BLEU-4 ~0.35

## 🎓 Exercises

1. **Compare Architectures**:
   - Train RNN and LSTM on same dataset
   - Compare BLEU scores
   - Analyze caption quality

2. **Add Attention**:
   - Implement attention mechanism
   - Visualize attention weights
   - Compare with baseline

3. **Beam Search**:
   ```python
   def beam_search(encoder, decoder, image, beam_width=3):
       # Implement beam search for better captions
       pass
   ```

4. **Fine-tune Encoder**:
   - Unfreeze CNN layers
   - Train end-to-end
   - Compare results

## 📚 Resources

- [Show and Tell Paper](https://arxiv.org/abs/1411.4555)
- [Show, Attend and Tell](https://arxiv.org/abs/1502.03044)
- [MS COCO Dataset](https://cocodataset.org/)
- [Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)

## 🎯 Next Steps

Move on to Lab 8: Chatbot with Bi-directional LSTMs