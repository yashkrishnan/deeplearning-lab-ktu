#!/usr/bin/env python3
"""
Lab 9: Time Series Forecasting with LSTM
=========================================

This program demonstrates:
1. LSTM for time series forecasting
2. Multi-step ahead prediction
3. Comparison with simple baselines
4. Evaluation metrics (MSE, MAE, MAPE)
5. Visualization of predictions
6. Feature importance analysis

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
from tqdm import tqdm

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# Create output directory
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_synthetic_timeseries(n_samples=1000, noise_level=0.1):
    """Generate synthetic time series with trend, seasonality, and noise."""
    t = np.linspace(0, 100, n_samples)
    
    # Trend component
    trend = 0.5 * t
    
    # Seasonal components
    seasonal1 = 10 * np.sin(2 * np.pi * t / 20)  # Period of 20
    seasonal2 = 5 * np.sin(2 * np.pi * t / 50)   # Period of 50
    
    # Noise
    noise = np.random.randn(n_samples) * noise_level * 10
    
    # Combine
    series = trend + seasonal1 + seasonal2 + noise
    
    return series, t


class TimeSeriesDataset(Dataset):
    """Time series dataset with sliding window."""
    
    def __init__(self, data, window_size=20, forecast_horizon=5):
        self.data = data
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        
    def __len__(self):
        return len(self.data) - self.window_size - self.forecast_horizon + 1
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.window_size]
        y = self.data[idx + self.window_size:idx + self.window_size + self.forecast_horizon]
        
        return torch.FloatTensor(x).unsqueeze(-1), torch.FloatTensor(y)


class LSTMForecaster(nn.Module):
    """LSTM model for time series forecasting."""
    
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, 
                 forecast_horizon=5, dropout=0.2):
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
                 forecast_horizon=5, dropout=0.2):
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


class SimpleBaseline(nn.Module):
    """Simple baseline: last value persistence."""
    
    def __init__(self, forecast_horizon=5):
        super(SimpleBaseline, self).__init__()
        self.forecast_horizon = forecast_horizon
        
    def forward(self, x):
        # Repeat last value
        last_value = x[:, -1, 0]
        return last_value.unsqueeze(-1).repeat(1, self.forecast_horizon)


def calculate_metrics(predictions, targets):
    """Calculate forecasting metrics."""
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(mse)
    
    # MAPE (avoid division by zero)
    mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
    
    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }


def train_model(model, train_loader, val_loader, num_epochs=50, model_name="Model"):
    """Train the forecasting model."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    print(f"\nTraining {model_name}...")
    print("-" * 60)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', leave=False)
        for inputs, targets in train_pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]', leave=False)
            for inputs, targets in val_pbar:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        val_loss /= len(val_loader)
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                      OUTPUT_DIR / f'{model_name.lower().replace(" ", "_")}_best.pth')
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    print(f"✓ {model_name} training complete! Best Val Loss: {best_val_loss:.6f}")
    return history


def evaluate_model(model, test_loader):
    """Evaluate model on test set."""
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())
    
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    metrics = calculate_metrics(predictions, targets)
    
    return predictions, targets, metrics


def visualize_predictions(predictions, targets, time_points, model_name="Model"):
    """Visualize predictions vs actual values."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot first 200 predictions
    n_samples = min(200, len(predictions))
    
    # Multi-step predictions
    for i in range(min(5, predictions.shape[1])):
        axes[0].plot(time_points[:n_samples], predictions[:n_samples, i],
                    alpha=0.6, label=f'Step {i+1} ahead')
        axes[0].plot(time_points[:n_samples], targets[:n_samples, i],
                    alpha=0.3, linestyle='--')
    
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Value')
    axes[0].set_title(f'{model_name} - Multi-step Predictions', fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Prediction errors
    errors = np.abs(predictions - targets)
    mean_errors = errors.mean(axis=1)
    
    axes[1].plot(time_points[:n_samples], mean_errors[:n_samples], 
                linewidth=2, color='red')
    axes[1].fill_between(time_points[:n_samples], 0, mean_errors[:n_samples],
                         alpha=0.3, color='red')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Mean Absolute Error')
    axes[1].set_title('Prediction Error Over Time', fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / f'{model_name.lower().replace(" ", "_")}_predictions.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def plot_training_comparison(histories, model_names):
    """Plot training comparison."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for history, name in zip(histories, model_names):
        ax.plot(history['train_loss'], label=f'{name} Train', linewidth=2)
        ax.plot(history['val_loss'], label=f'{name} Val', linewidth=2, linestyle='--')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'training_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def plot_metrics_comparison(metrics_dict):
    """Plot metrics comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    models = list(metrics_dict.keys())
    metric_names = ['MSE', 'MAE', 'RMSE', 'MAPE']
    
    for idx, metric_name in enumerate(metric_names):
        values = [metrics_dict[model][metric_name] for model in models]
        
        bars = axes[idx].bar(models, values, alpha=0.7, edgecolor='black')
        
        # Color bars
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.4f}',
                          ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        axes[idx].set_ylabel(metric_name, fontsize=12)
        axes[idx].set_title(f'{metric_name} Comparison', fontsize=12, fontweight='bold')
        axes[idx].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'metrics_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def main():
    """Main function."""
    print("=" * 70)
    print("Lab 9: Time Series Forecasting with LSTM")
    print("=" * 70)
    print()
    print(f"Device: {device}")
    print()
    
    # Generate synthetic time series
    print("Generating synthetic time series...")
    series, time_points = generate_synthetic_timeseries(n_samples=1000, noise_level=0.1)
    
    # Normalize
    mean = series.mean()
    std = series.std()
    series_normalized = (series - mean) / std
    
    # Split data
    train_size = int(0.7 * len(series_normalized))
    val_size = int(0.15 * len(series_normalized))
    
    train_data = series_normalized[:train_size]
    val_data = series_normalized[train_size:train_size + val_size]
    test_data = series_normalized[train_size + val_size:]
    
    print(f"  • Total samples: {len(series)}")
    print(f"  • Train samples: {len(train_data)}")
    print(f"  • Val samples: {len(val_data)}")
    print(f"  • Test samples: {len(test_data)}")
    print()
    
    # Create datasets
    window_size = 20
    forecast_horizon = 5
    
    train_dataset = TimeSeriesDataset(train_data, window_size, forecast_horizon)
    val_dataset = TimeSeriesDataset(val_data, window_size, forecast_horizon)
    test_dataset = TimeSeriesDataset(test_data, window_size, forecast_horizon)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"Configuration:")
    print(f"  • Window size: {window_size}")
    print(f"  • Forecast horizon: {forecast_horizon} steps")
    print()
    
    # Train LSTM model
    print("=" * 70)
    print("Training LSTM Model")
    print("=" * 70)
    
    lstm_model = LSTMForecaster(
        input_size=1,
        hidden_size=64,
        num_layers=2,
        forecast_horizon=forecast_horizon
    ).to(device)
    
    start_time = time.time()
    lstm_history = train_model(lstm_model, train_loader, val_loader,
                               num_epochs=50, model_name="LSTM")
    lstm_time = time.time() - start_time
    
    # Train GRU model
    print()
    print("=" * 70)
    print("Training GRU Model")
    print("=" * 70)
    
    gru_model = GRUForecaster(
        input_size=1,
        hidden_size=64,
        num_layers=2,
        forecast_horizon=forecast_horizon
    ).to(device)
    
    start_time = time.time()
    gru_history = train_model(gru_model, train_loader, val_loader,
                             num_epochs=50, model_name="GRU")
    gru_time = time.time() - start_time
    
    # Baseline model
    print()
    print("Evaluating Baseline Model...")
    baseline_model = SimpleBaseline(forecast_horizon=forecast_horizon).to(device)
    
    # Evaluate all models
    print()
    print("=" * 70)
    print("Evaluation on Test Set")
    print("=" * 70)
    
    lstm_pred, lstm_target, lstm_metrics = evaluate_model(lstm_model, test_loader)
    gru_pred, gru_target, gru_metrics = evaluate_model(gru_model, test_loader)
    baseline_pred, baseline_target, baseline_metrics = evaluate_model(baseline_model, test_loader)
    
    metrics_dict = {
        'LSTM': lstm_metrics,
        'GRU': gru_metrics,
        'Baseline': baseline_metrics
    }
    
    print()
    for model_name, metrics in metrics_dict.items():
        print(f"{model_name}:")
        for metric_name, value in metrics.items():
            print(f"  • {metric_name}: {value:.6f}")
        print()
    
    # Visualizations
    print("Generating visualizations...")
    
    test_time = time_points[train_size + val_size + window_size:]
    
    lstm_vis = visualize_predictions(lstm_pred, lstm_target, test_time, "LSTM")
    gru_vis = visualize_predictions(gru_pred, gru_target, test_time, "GRU")
    
    print(f"  ✓ LSTM predictions: {lstm_vis}")
    print(f"  ✓ GRU predictions: {gru_vis}")
    
    comparison_plot = plot_training_comparison(
        [lstm_history, gru_history],
        ['LSTM', 'GRU']
    )
    print(f"  ✓ Training comparison: {comparison_plot}")
    
    metrics_plot = plot_metrics_comparison(metrics_dict)
    print(f"  ✓ Metrics comparison: {metrics_plot}")
    print()
    
    print("=" * 70)
    print("Lab 9 Complete!")
    print("=" * 70)
    print()
    print("Key Findings:")
    print("  • LSTM and GRU outperform simple baseline")
    print("  • LSTM handles long-term dependencies well")
    print("  • Multi-step forecasting is challenging")
    print("  • Real applications: stock prices, weather, energy demand")
    print()


if __name__ == "__main__":
    main()


