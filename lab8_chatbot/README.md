# Lab 8: Chatbot with Bi-directional LSTMs

## 📋 Overview

This lab implements a conversational chatbot using bi-directional LSTMs:
- Intent classification
- Entity recognition
- Response generation
- Context management

## 🎯 Learning Objectives

1. Understand bi-directional LSTMs
2. Implement intent classification
3. Build sequence-to-sequence chatbot
4. Handle conversation context
5. Evaluate chatbot performance

## 🏗️ Architecture

### Bi-directional LSTM
```
Forward LSTM  →→→→→
Input Sequence
Backward LSTM ←←←←←
```

Benefits:
- Captures context from both directions
- Better understanding of sentence meaning
- Improved accuracy for NLP tasks

## 📦 Installation

```bash
pip install torch transformers nltk
python -m nltk.downloader punkt stopwords
```

## 🚀 Implementation

### Intent Classification
```python
import torch
import torch.nn as nn

class IntentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_intents):
        super(IntentClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.bilstm = nn.LSTM(embed_dim, hidden_dim, 
                             bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, num_intents)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.bilstm(embedded)
        # Concatenate final forward and backward hidden states
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        output = self.fc(hidden)
        return output

# Example usage
model = IntentClassifier(vocab_size=10000, embed_dim=128, 
                        hidden_dim=256, num_intents=10)
```

### Seq2Seq Chatbot
```python
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.bilstm = nn.LSTM(embed_dim, hidden_dim, 
                             bidirectional=True, batch_first=True)
        
    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.bilstm(embedded)
        return outputs, hidden, cell

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim * 2, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)
        
    def forward(self, x, hidden, cell):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc(output)
        return prediction, hidden, cell

class Seq2SeqChatbot(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2SeqChatbot, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc.out_features
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size)
        
        encoder_outputs, hidden, cell = self.encoder(src)
        
        # First input to decoder is <sos> token
        input = trg[:, 0]
        
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input.unsqueeze(1), hidden, cell)
            outputs[:, t] = output.squeeze(1)
            
            # Teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(2).squeeze(1)
            input = trg[:, t] if teacher_force else top1
            
        return outputs
```

## 💬 Conversation Management

### Context Tracking
```python
class ConversationContext:
    def __init__(self, max_history=5):
        self.history = []
        self.max_history = max_history
        self.user_info = {}
        
    def add_turn(self, user_input, bot_response):
        self.history.append({
            'user': user_input,
            'bot': bot_response,
            'timestamp': time.time()
        })
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def get_context(self):
        return ' '.join([turn['user'] for turn in self.history])
    
    def update_user_info(self, key, value):
        self.user_info[key] = value
```

## 🎯 Training

```python
# Training loop
encoder = Encoder(vocab_size=10000, embed_dim=256, hidden_dim=512)
decoder = Decoder(vocab_size=10000, embed_dim=256, hidden_dim=512)
model = Seq2SeqChatbot(encoder, decoder)

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for src, trg in train_loader:
        optimizer.zero_grad()
        
        output = model(src, trg)
        
        # Reshape for loss calculation
        output = output[:, 1:].reshape(-1, output.shape[-1])
        trg = trg[:, 1:].reshape(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        
        optimizer.step()
```

## 🗣️ Response Generation

```python
def generate_response(model, input_text, vocab, max_length=50):
    model.eval()
    
    # Tokenize input
    tokens = tokenize(input_text)
    indices = [vocab[token] for token in tokens]
    tensor = torch.LongTensor(indices).unsqueeze(0)
    
    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(tensor)
        
        # Start with <sos> token
        input_token = torch.LongTensor([vocab['<sos>']])
        
        response = []
        for _ in range(max_length):
            output, hidden, cell = model.decoder(
                input_token.unsqueeze(0), hidden, cell
            )
            
            predicted_token = output.argmax(2).item()
            
            if predicted_token == vocab['<eos>']:
                break
                
            response.append(vocab.idx2word[predicted_token])
            input_token = torch.LongTensor([predicted_token])
    
    return ' '.join(response)
```

## 📊 Intent Examples

```python
intents = {
    'greeting': ['hello', 'hi', 'hey', 'good morning'],
    'goodbye': ['bye', 'goodbye', 'see you', 'take care'],
    'thanks': ['thank you', 'thanks', 'appreciate it'],
    'weather': ['weather', 'temperature', 'forecast', 'rain'],
    'time': ['time', 'what time', 'clock'],
    'help': ['help', 'assist', 'support', 'problem']
}

responses = {
    'greeting': ['Hello! How can I help you?', 'Hi there!', 'Hey! What can I do for you?'],
    'goodbye': ['Goodbye!', 'See you later!', 'Take care!'],
    'thanks': ['You\'re welcome!', 'Happy to help!', 'Anytime!'],
    'weather': ['I can help with weather info. What location?'],
    'time': ['The current time is {time}'],
    'help': ['I\'m here to help! What do you need?']
}
```

## 🎓 Evaluation Metrics

### Perplexity
```python
def calculate_perplexity(model, data_loader):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for src, trg in data_loader:
            output = model(src, trg, teacher_forcing_ratio=0)
            loss = criterion(output.view(-1, vocab_size), trg.view(-1))
            total_loss += loss.item()
    
    perplexity = np.exp(total_loss / len(data_loader))
    return perplexity
```

### Response Quality
- Relevance: Does response match intent?
- Coherence: Is response grammatically correct?
- Context: Does it consider conversation history?
- Engagement: Is it natural and engaging?

## 🎓 Exercises

1. **Add Attention Mechanism**:
   ```python
   class AttentionDecoder(nn.Module):
       def __init__(self, vocab_size, embed_dim, hidden_dim):
           super().__init__()
           self.attention = nn.Linear(hidden_dim * 3, 1)
           # ... rest of implementation
   ```

2. **Implement Beam Search**:
   ```python
   def beam_search(model, input_text, beam_width=3):
       # Generate multiple candidate responses
       # Select best based on probability
       pass
   ```

3. **Add Personality**:
   - Formal vs casual responses
   - Emoji usage
   - Response length preferences

4. **Multi-turn Conversations**:
   - Track conversation state
   - Reference previous messages
   - Handle follow-up questions

## 📚 Datasets

### Popular Chatbot Datasets:
- **Cornell Movie Dialogs**: 220k conversations
- **Ubuntu Dialogue Corpus**: Technical support
- **PersonaChat**: Personality-based conversations
- **DailyDialog**: Daily conversations

### Data Format:
```json
{
  "conversations": [
    {
      "user": "Hello, how are you?",
      "bot": "I'm doing great! How can I help you today?"
    },
    {
      "user": "What's the weather like?",
      "bot": "I'd be happy to check the weather. What's your location?"
    }
  ]
}
```

## 🔍 Comparison: Unidirectional vs Bi-directional

| Aspect | Unidirectional | Bi-directional |
|--------|----------------|----------------|
| Context | Forward only | Both directions |
| Accuracy | Lower | Higher |
| Speed | Faster | Slower |
| Use Case | Generation | Classification |

## 📚 Resources

- [Bi-LSTM Paper](https://arxiv.org/abs/1508.01991)
- [Seq2Seq Learning](https://arxiv.org/abs/1409.3215)
- [Attention Mechanism](https://arxiv.org/abs/1409.0473)
- [Rasa Framework](https://rasa.com/)

## 🎯 Next Steps

Move on to Lab 9: Time Series Forecasting