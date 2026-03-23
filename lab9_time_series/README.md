# Lab 9: Time Series Forecasting with Deep Learning

## 📋 Overview

This lab implements time series forecasting using:
- **LSTM Networks**: For sequential data
- **GRU Networks**: Simplified LSTM variant
- **Transformer Models**: Attention-based forecasting
- **CNN-LSTM Hybrid**: Combined approach

## 🎯 Learning Objectives

1. Understand time series data characteristics
2. Implement LSTM for forecasting
3. Handle multivariate time series
4. Evaluate forecasting accuracy
5. Apply to real-world datasets

## 📊 Time Series Concepts

### Components:
- **Trend**: Long-term direction
- **Seasonality**: Periodic patterns
- **Cyclical**: Non-periodic fluctuations
- **Noise**: Random variations

### Stationarity:
- Mean is constant over time
- Variance is constant
- No seasonality

## 📦 Installation

```bash
pip install torch numpy pandas matplotlib scikit-learn statsmodels
```

## 🚀 Implementation

### Data Preparation
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def create_sequences(data, seq_length):
    """Create input-output sequences for training"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Load and preprocess data
df = pd.read_csv('stock_prices.csv')
data = df['Close'].values.reshape(-1, 1)

# Normalize
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)

# Create sequences
seq_length = 60  # Use 60 days to predict next day
X, y = create_sequences(data_normalized, seq_length)

# Train-test split
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
```

### LSTM Model
```python
import torch
import torch.nn as nn

class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Get output from last time step
        out = self.fc(out[:, -1, :])
        return out

# Initialize model
model = LSTMForecaster(input_size=1, hidden_size=50, num_layers=2)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

### GRU Model
```python
class GRUForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(GRUForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                         batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out
```

### Training Loop
```python
def train_model(model, train_loader, criterion, optimizer, num_epochs=100):
    model.train()
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    return losses
```

### Forecasting
```python
def forecast(model, data, steps=30):
    """Multi-step forecasting"""
    model.eval()
    predictions = []
    
    # Start with last sequence from data
    current_seq = data[-seq_length:].copy()
    
    with torch.no_grad():
        for _ in range(steps):
            # Prepare input
            x = torch.FloatTensor(current_seq).unsqueeze(0)
            
            # Predict next value
            pred = model(x)
            predictions.append(pred.item())
            
            # Update sequence (sliding window)
            current_seq = np.append(current_seq[1:], pred.item())
            current_seq = current_seq.reshape(-1, 1)
    
    return np.array(predictions)
```

## 📈 Evaluation Metrics

### Mean Absolute Error (MAE)
```python
def calculate_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))
```

### Root Mean Squared Error (RMSE)
```python
def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))
```

### Mean Absolute Percentage Error (MAPE)
```python
def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
```

### R² Score
```python
from sklearn.metrics import r2_score

r2 = r2_score(y_true, y_pred)
```

## 📊 Visualization

```python
import matplotlib.pyplot as plt

def plot_predictions(y_true, y_pred, title='Forecast vs Actual'):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual', linewidth=2)
    plt.plot(y_pred, label='Predicted', linewidth=2, linestyle='--')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('forecast_results.png', dpi=150)
    plt.show()

def plot_training_loss(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_loss.png', dpi=150)
    plt.show()
```

## 🎓 Use Cases

### 1. Stock Price Prediction
```python
# Load stock data
df = pd.read_csv('AAPL.csv')
data = df['Close'].values

# Train model
model = LSTMForecaster()
# ... training code ...

# Forecast next 30 days
predictions = forecast(model, data, steps=30)
```

### 2. Weather Forecasting
```python
# Multivariate time series
features = ['temperature', 'humidity', 'pressure', 'wind_speed']
data = df[features].values

model = LSTMForecaster(input_size=len(features), hidden_size=100)
```

### 3. Energy Consumption
```python
# Hourly energy consumption
df = pd.read_csv('energy_consumption.csv')
data = df['consumption'].values

# Consider seasonality
model = LSTMForecaster(hidden_size=100, num_layers=3)
```

### 4. Sales Forecasting
```python
# Daily sales data
df = pd.read_csv('sales.csv')
data = df['sales'].values

# Include external features (holidays, promotions)
features = df[['sales', 'is_holiday', 'promotion']].values
```

## 🎓 Advanced Techniques

### 1. Attention Mechanism
```python
class AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        # Calculate attention weights
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        
        # Apply attention
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        output = self.fc(context)
        return output
```

### 2. Ensemble Methods
```python
def ensemble_forecast(models, data):
    predictions = []
    for model in models:
        pred = forecast(model, data)
        predictions.append(pred)
    
    # Average predictions
    ensemble_pred = np.mean(predictions, axis=0)
    return ensemble_pred
```

### 3. Hyperparameter Tuning
```python
from sklearn.model_selection import TimeSeriesSplit

def tune_hyperparameters(X, y):
    tscv = TimeSeriesSplit(n_splits=5)
    
    best_params = None
    best_score = float('inf')
    
    for hidden_size in [32, 64, 128]:
        for num_layers in [1, 2, 3]:
            for lr in [0.001, 0.01, 0.1]:
                # Train and evaluate
                score = cross_validate(X, y, hidden_size, num_layers, lr, tscv)
                
                if score < best_score:
                    best_score = score
                    best_params = (hidden_size, num_layers, lr)
    
    return best_params
```

## 📚 Datasets

### Popular Time Series Datasets:
- **Stock Prices**: Yahoo Finance, Alpha Vantage
- **Weather**: NOAA, OpenWeatherMap
- **Energy**: UCI Machine Learning Repository
- **Sales**: Kaggle competitions
- **Traffic**: Caltrans PeMS

## 🎓 Exercises

1. **Compare Models**:
   - Train LSTM, GRU, and simple RNN
   - Compare accuracy and training time
   - Analyze results

2. **Multivariate Forecasting**:
   - Use multiple input features
   - Predict multiple outputs
   - Handle missing data

3. **Real-time Forecasting**:
   - Implement online learning
   - Update model with new data
   - Deploy as API

4. **Anomaly Detection**:
   - Detect unusual patterns
   - Set threshold for alerts
   - Visualize anomalies

## 📚 Resources

- [LSTM Paper](https://www.bioinf.jku.at/publications/older/2604.pdf)
- [Time Series Forecasting Guide](https://otexts.com/fpp3/)
- [Prophet by Facebook](https://facebook.github.io/prophet/)
- [Statsmodels Documentation](https://www.statsmodels.org/)

## 🎯 Next Steps

Move on to Lab 10: Sequence to Sequence Learning