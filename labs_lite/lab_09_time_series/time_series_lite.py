#!/usr/bin/env python3
"""
Lab 9 Lite: Time Series Forecasting with LSTM
==============================================

Lightweight version using real Energy consumption dataset with reduced samples and epochs.

Changes from original:
- Uses 5,000 real hourly energy readings (from ~120,000)
- 10 epochs (from 30)
- Smaller batch size: 32 (from 64)
- Window size: 24 hours (from 48)
- Forecast horizon: 6 hours (from 12)
- Uses AEP hourly energy consumption data

Dataset: AEP Hourly Energy Consumption
Expected runtime: ~4-6 minutes on CPU

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
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# Create output directory
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Lite configuration
NUM_SAMPLES = 5000
BATCH_SIZE = 32
NUM_EPOCHS = 10
WINDOW_SIZE = 24  # 24 hours
FORECAST_HORIZON = 6  # 6 hours ahead


class EnergyTimeSeriesDataset(Dataset):
    """Load energy consumption time series dataset."""
    
    def __init__(self, data_file, window_size=24, forecast_horizon=6, max_samples=None):
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        
        # Load data
        print(f"Loading energy data from {data_file.name}...")
        df = pd.read_csv(data_file)
        
        # Extract energy values
        energy_col = df.columns[1]  # Second column is energy consumption
        self.data = df[energy_col].values
        
        # Remove NaN values
        self.data = self.data[~np.isnan(self.data)]
        
        # Limit samples
        if max_samples:
            self.data = self.data[:max_samples + window_size + forecast_horizon]
        
        print(f"Loaded {len(self.data)} data points")
        
        # Normalize
        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(self.data.reshape(-1, 1)).flatten()
    
    def __len__(self):
        return len(self.data) - self.window_size - self.forecast_horizon + 1
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.window_size]
        y = self.data[idx + self.window_size:idx + self.window_size + self.forecast_horizon]
        
        return torch.FloatTensor(x).unsqueeze(-1), torch.FloatTensor(y)


class LSTMForecaster(nn.Module):
    """LSTM model for time series forecasting."""
    
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, 
                 forecast_horizon=6, dropout=0.2):
        super(LSTMForecaster, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, forecast_horizon)
        )
        
    def forward(self, x):
        # x: [batch, seq_len, input_size]
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]
        
        # Forecast
        output = self.fc(last_hidden)
        
        return output


class GRUForecaster(nn.Module):
    """GRU model for comparison."""
    
    def __init__(self, input_size=1, hidden_size=64, num_layers=2,
                 forecast_horizon=6, dropout=0.2):
        super(GRUForecaster, self).__init__()
        
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, forecast_horizon)
        )
        
    def forward(self, x):
        gru_out, hidden = self.gru(x)
        last_hidden = gru_out[:, -1, :]
        output = self.fc(last_hidden)
        return output


def train_model(model, train_loader, num_epochs, device, model_name="Model"):
    """Train the forecasting model."""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    history = {'loss': []}
    
    print(f"\nTraining {model_name}...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            
            predictions = model(x)
            loss = criterion(predictions, y)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        
        avg_loss = epoch_loss / len(train_loader)
        history['loss'].append(avg_loss)
        
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}")
    
    return history


def evaluate_model(model, test_loader, device):
    """Evaluate model performance."""
    model.eval()
    criterion = nn.MSELoss()
    
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            
            predictions = model(x)
            loss = criterion(predictions, y)
            
            total_loss += loss.item()
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Calculate metrics
    mse = np.mean((all_predictions - all_targets) ** 2)
    mae = np.mean(np.abs(all_predictions - all_targets))
    rmse = np.sqrt(mse)
    
    return avg_loss, mse, mae, rmse, all_predictions, all_targets


def visualize_predictions(predictions, targets, dataset, num_samples=3):
    """Visualize predictions vs actual values."""
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        if i >= len(predictions):
            break
        
        pred = predictions[i]
        target = targets[i]
        
        # Denormalize
        pred_denorm = dataset.scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
        target_denorm = dataset.scaler.inverse_transform(target.reshape(-1, 1)).flatten()
        
        x = np.arange(len(pred))
        ax.plot(x, target_denorm, 'b-o', label='Actual', linewidth=2)
        ax.plot(x, pred_denorm, 'r--s', label='Predicted', linewidth=2)
        ax.set_xlabel('Hours Ahead')
        ax.set_ylabel('Energy Consumption (MW)')
        ax.set_title(f'Sample {i+1}: Forecast vs Actual')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'forecast_predictions.png', dpi=150, bbox_inches='tight')
    print(f"Saved predictions visualization to {OUTPUT_DIR / 'forecast_predictions.png'}")
    plt.close()


def plot_training_comparison(lstm_history, gru_history):
    """Plot training comparison between LSTM and GRU."""
    plt.figure(figsize=(10, 6))
    plt.plot(lstm_history['loss'], 'b-', label='LSTM', linewidth=2)
    plt.plot(gru_history['loss'], 'r-', label='GRU', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training Loss Comparison: LSTM vs GRU')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'training_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved training comparison to {OUTPUT_DIR / 'training_comparison.png'}")
    plt.close()


def plot_metrics_comparison(lstm_metrics, gru_metrics):
    """Plot metrics comparison."""
    metrics = ['MSE', 'MAE', 'RMSE']
    lstm_vals = [lstm_metrics['mse'], lstm_metrics['mae'], lstm_metrics['rmse']]
    gru_vals = [gru_metrics['mse'], gru_metrics['mae'], gru_metrics['rmse']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, lstm_vals, width, label='LSTM', color='blue', alpha=0.7)
    ax.bar(x + width/2, gru_vals, width, label='GRU', color='red', alpha=0.7)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Value')
    ax.set_title('Performance Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'metrics_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved metrics comparison to {OUTPUT_DIR / 'metrics_comparison.png'}")
    plt.close()


def main():
    print("=" * 60)
    print("Lab 9 Lite: Time Series Forecasting with Energy Data")
    print("=" * 60)
    
    start_time = time.time()
    
    # Check if dataset exists (in labs_lite directory)
    data_file = Path("data/energy/AEP_hourly.csv")
    if not data_file.exists():
        print(f"\nError: Energy dataset not found at {data_file}")
        print("Please ensure the dataset is downloaded.")
        return
    
    # Load dataset
    print(f"\nLoading energy time series (max {NUM_SAMPLES} samples)...")
    dataset = EnergyTimeSeriesDataset(
        data_file=data_file,
        window_size=WINDOW_SIZE,
        forecast_horizon=FORECAST_HORIZON,
        max_samples=NUM_SAMPLES
    )
    
    # Split into train and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train samples: {train_size}, Test samples: {test_size}")
    
    # Train LSTM model
    print("\n" + "="*60)
    print("Training LSTM Model")
    print("="*60)
    lstm_model = LSTMForecaster(
        input_size=1,
        hidden_size=64,
        num_layers=2,
        forecast_horizon=FORECAST_HORIZON,
        dropout=0.2
    )
    print(f"LSTM parameters: {sum(p.numel() for p in lstm_model.parameters()):,}")
    lstm_history = train_model(lstm_model, train_loader, NUM_EPOCHS, device, "LSTM")
    
    # Train GRU model
    print("\n" + "="*60)
    print("Training GRU Model")
    print("="*60)
    gru_model = GRUForecaster(
        input_size=1,
        hidden_size=64,
        num_layers=2,
        forecast_horizon=FORECAST_HORIZON,
        dropout=0.2
    )
    print(f"GRU parameters: {sum(p.numel() for p in gru_model.parameters()):,}")
    gru_history = train_model(gru_model, train_loader, NUM_EPOCHS, device, "GRU")
    
    # Evaluate models
    print("\n" + "="*60)
    print("Evaluating Models")
    print("="*60)
    
    lstm_loss, lstm_mse, lstm_mae, lstm_rmse, lstm_preds, targets = evaluate_model(
        lstm_model, test_loader, device
    )
    print(f"\nLSTM Results:")
    print(f"  Test Loss: {lstm_loss:.4f}")
    print(f"  MSE: {lstm_mse:.4f}")
    print(f"  MAE: {lstm_mae:.4f}")
    print(f"  RMSE: {lstm_rmse:.4f}")
    
    gru_loss, gru_mse, gru_mae, gru_rmse, gru_preds, _ = evaluate_model(
        gru_model, test_loader, device
    )
    print(f"\nGRU Results:")
    print(f"  Test Loss: {gru_loss:.4f}")
    print(f"  MSE: {gru_mse:.4f}")
    print(f"  MAE: {gru_mae:.4f}")
    print(f"  RMSE: {gru_rmse:.4f}")
    
    # Visualizations
    print("\nGenerating visualizations...")
    plot_training_comparison(lstm_history, gru_history)
    
    lstm_metrics = {'mse': lstm_mse, 'mae': lstm_mae, 'rmse': lstm_rmse}
    gru_metrics = {'mse': gru_mse, 'mae': gru_mae, 'rmse': gru_rmse}
    plot_metrics_comparison(lstm_metrics, gru_metrics)
    
    visualize_predictions(lstm_preds, targets, dataset, num_samples=3)
    
    elapsed_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Total execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"{'='*60}")
    print(f"\nOutputs saved to: {OUTPUT_DIR.absolute()}")
    print("\nLab 9 Lite completed successfully!")


if __name__ == "__main__":
    main()


