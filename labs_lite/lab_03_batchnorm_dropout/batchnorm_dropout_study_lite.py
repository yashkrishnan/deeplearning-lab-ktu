"""
Lab 3: Batch Normalization and Dropout Study (LIGHTWEIGHT VERSION)
===================================================================

This is a lightweight version optimized for faster training on laptops.
Changes from original:
- Reduced training samples: 2000 (from 4000)
- Reduced validation samples: 500 (from 1000)
- Reduced test samples: 500 (from 1000)
- Reduced epochs: 8 (from 15)
- Smaller network architecture
- Faster execution time: ~3-4 minutes (vs ~8-10 minutes)

This program demonstrates the effects of batch normalization and dropout
on neural network training and performance. It compares four model variants:
1. Baseline (no regularization)
2. With Batch Normalization only
3. With Dropout only
4. With both Batch Normalization and Dropout

Author: Deep Learning Lab
Date: March 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time

# Set style for better plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class BaselineNetwork(nn.Module):
    """
    Baseline neural network without any regularization (smaller version).
    """
    def __init__(self):
        super(BaselineNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3 * 32 * 32, 256)  # Reduced from 512
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)          # Reduced from 256
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 64)           # Reduced from 128
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)
        return x


class BatchNormNetwork(nn.Module):
    """
    Neural network with Batch Normalization after each layer (smaller version).
    """
    def __init__(self):
        super(BatchNormNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3 * 32 * 32, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.bn1(self.fc1(x)))
        x = self.relu2(self.bn2(self.fc2(x)))
        x = self.relu3(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x


class DropoutNetwork(nn.Module):
    """
    Neural network with Dropout for regularization (smaller version).
    """
    def __init__(self, dropout_rate=0.5):
        super(DropoutNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3 * 32 * 32, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        x = self.dropout3(self.relu3(self.fc3(x)))
        x = self.fc4(x)
        return x


class BatchNormDropoutNetwork(nn.Module):
    """
    Neural network with both Batch Normalization and Dropout (smaller version).
    """
    def __init__(self, dropout_rate=0.5):
        super(BatchNormDropoutNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3 * 32 * 32, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.dropout1(self.relu1(self.bn1(self.fc1(x))))
        x = self.dropout2(self.relu2(self.bn2(self.fc2(x))))
        x = self.dropout3(self.relu3(self.bn3(self.fc3(x))))
        x = self.fc4(x)
        return x


class RegularizationStudyLite:
    """
    Study the effects of batch normalization and dropout on neural networks (lightweight version).
    """
    
    def __init__(self, output_dir='output'):
        """
        Initialize the study.
        
        Args:
            output_dir (str): Directory to save outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"✓ Output directory: {self.output_dir}")
        print(f"✓ Device: {self.device}")
        print(f"✓ LIGHTWEIGHT VERSION - Optimized for laptops")
    
    def load_data(self):
        """
        Load CIFAR-10 dataset with train/validation split (reduced samples).
        
        Returns:
            tuple: Train, validation, and test loaders
        """
        print("\nLoading CIFAR-10 dataset (lightweight)...")
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Load full training set
        full_trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform
        )
        
        # Split into train and validation (80-20 split)
        train_size = int(0.8 * len(full_trainset))
        val_size = len(full_trainset) - train_size
        trainset, valset = torch.utils.data.random_split(
            full_trainset, [train_size, val_size]
        )
        
        # Use smaller subset for faster execution
        train_indices = np.random.choice(len(trainset), 2000, replace=False)
        val_indices = np.random.choice(len(valset), 500, replace=False)
        
        trainset = torch.utils.data.Subset(trainset, train_indices.tolist())
        valset = torch.utils.data.Subset(valset, val_indices.tolist())
        
        # Load test set
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform
        )
        test_indices = np.random.choice(len(testset), 500, replace=False)
        testset = torch.utils.data.Subset(testset, test_indices.tolist())
        
        # Create data loaders
        train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
        val_loader = DataLoader(valset, batch_size=64, shuffle=False)
        test_loader = DataLoader(testset, batch_size=64, shuffle=False)
        
        print(f"  ✓ Training samples: {len(trainset)}")
        print(f"  ✓ Validation samples: {len(valset)}")
        print(f"  ✓ Test samples: {len(testset)}")
        
        return train_loader, val_loader, test_loader
    
    def train_model(self, model, train_loader, val_loader, model_name, num_epochs=8):
        """
        Train a model and track training/validation metrics.
        
        Args:
            model: Neural network model
            train_loader: Training data loader
            val_loader: Validation data loader
            model_name: Name of the model
            num_epochs: Number of training epochs
            
        Returns:
            dict: Training history
        """
        print(f"\nTraining {model_name}...")
        
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            # Calculate metrics
            train_loss = train_loss / len(train_loader)
            train_acc = 100 * train_correct / train_total
            val_loss = val_loss / len(val_loader)
            val_acc = 100 * val_correct / val_total
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            if (epoch + 1) % 2 == 0:
                print(f"  Epoch [{epoch+1}/{num_epochs}] - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        print(f"  ✓ Training completed")
        print(f"  ✓ Final validation accuracy: {history['val_acc'][-1]:.2f}%")
        
        return history, model
    
    def evaluate_model(self, model, test_loader):
        """
        Evaluate model on test set.
        
        Args:
            model: Trained model
            test_loader: Test data loader
            
        Returns:
            float: Test accuracy
        """
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        return accuracy
    
    def plot_comparison(self, histories, model_names):
        """
        Plot comparison of all models.
        
        Args:
            histories: List of training histories
            model_names: List of model names
        """
        print("\nGenerating comparison plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
        
        # Training Loss
        ax = axes[0, 0]
        for i, (history, name) in enumerate(zip(histories, model_names)):
            ax.plot(history['train_loss'], label=name, color=colors[i], linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Validation Loss
        ax = axes[0, 1]
        for i, (history, name) in enumerate(zip(histories, model_names)):
            ax.plot(history['val_loss'], label=name, color=colors[i], linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Validation Loss Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Training Accuracy
        ax = axes[1, 0]
        for i, (history, name) in enumerate(zip(histories, model_names)):
            ax.plot(history['train_acc'], label=name, color=colors[i], linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Training Accuracy Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Validation Accuracy
        ax = axes[1, 1]
        for i, (history, name) in enumerate(zip(histories, model_names)):
            ax.plot(history['val_acc'], label=name, color=colors[i], linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Validation Accuracy Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Comparison plot saved to: {self.output_dir / 'training_comparison.png'}")
    
    def plot_overfitting_analysis(self, histories, model_names):
        """
        Analyze overfitting by comparing train vs validation accuracy.
        
        Args:
            histories: List of training histories
            model_names: List of model names
        """
        print("Generating overfitting analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
        
        for idx, (history, name, color) in enumerate(zip(histories, model_names, colors)):
            row, col = idx // 2, idx % 2
            ax = axes[row, col]
            
            ax.plot(history['train_acc'], label='Training', color=color, linewidth=2)
            ax.plot(history['val_acc'], label='Validation', color=color, 
                   linewidth=2, linestyle='--')
            
            # Calculate overfitting gap
            final_gap = history['train_acc'][-1] - history['val_acc'][-1]
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy (%)')
            ax.set_title(f'{name}\nOverfitting Gap: {final_gap:.2f}%')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'overfitting_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Overfitting analysis saved to: {self.output_dir / 'overfitting_analysis.png'}")
    
    def plot_final_comparison(self, test_accuracies, model_names):
        """
        Plot final test accuracy comparison.
        
        Args:
            test_accuracies: List of test accuracies
            model_names: List of model names
        """
        print("Generating final comparison...")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
        
        bars = ax.bar(model_names, test_accuracies, color=colors)
        ax.set_ylabel('Test Accuracy (%)')
        ax.set_title('Final Test Accuracy Comparison')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}%',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'final_accuracy_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Final comparison saved to: {self.output_dir / 'final_accuracy_comparison.png'}")
    
    def run_study(self):
        """
        Run the complete regularization study.
        """
        print("\n" + "="*60)
        print("  LAB 3: BATCH NORMALIZATION AND DROPOUT (LIGHTWEIGHT)")
        print("="*60)
        
        start_time = time.time()
        
        # Load data
        train_loader, val_loader, test_loader = self.load_data()
        
        # Define models
        models = [
            (BaselineNetwork(), "Baseline"),
            (BatchNormNetwork(), "Batch Normalization"),
            (DropoutNetwork(dropout_rate=0.5), "Dropout"),
            (BatchNormDropoutNetwork(dropout_rate=0.5), "BatchNorm + Dropout")
        ]
        
        # Train all models
        histories = []
        trained_models = []
        test_accuracies = []
        model_names = []
        
        for model, name in models:
            history, trained_model = self.train_model(
                model, train_loader, val_loader, name, num_epochs=8
            )
            test_acc = self.evaluate_model(trained_model, test_loader)
            
            histories.append(history)
            trained_models.append(trained_model)
            test_accuracies.append(test_acc)
            model_names.append(name)
            
            print(f"  ✓ Test Accuracy: {test_acc:.2f}%")
        
        # Generate plots
        self.plot_comparison(histories, model_names)
        self.plot_overfitting_analysis(histories, model_names)
        self.plot_final_comparison(test_accuracies, model_names)
        
        elapsed_time = time.time() - start_time
        
        # Print summary
        print("\n" + "="*60)
        print("  SUMMARY")
        print("="*60)
        print("\nFinal Results:")
        print("-" * 60)
        print(f"{'Model':<30} {'Test Accuracy':<15} {'Overfitting Gap':<15}")
        print("-" * 60)
        for i, name in enumerate(model_names):
            gap = histories[i]['train_acc'][-1] - histories[i]['val_acc'][-1]
            print(f"{name:<30} {test_accuracies[i]:>6.2f}%        {gap:>6.2f}%")
        print("-" * 60)
        
        print(f"\n✓ Study completed in {elapsed_time:.2f} seconds")
        print(f"✓ Output files saved to: {self.output_dir}/")
        print("\nGenerated files:")
        print("  1. training_comparison.png - Training curves comparison")
        print("  2. overfitting_analysis.png - Overfitting analysis")
        print("  3. final_accuracy_comparison.png - Test accuracy comparison")
        print("\n" + "="*60)


def main():
    """
    Main function to run the regularization study.
    """
    study = RegularizationStudyLite(output_dir='output')
    study.run_study()
    
    print("\n✨ Lab 3 (Lightweight) completed! Check the 'output' folder for results.")


if __name__ == "__main__":
    main()


