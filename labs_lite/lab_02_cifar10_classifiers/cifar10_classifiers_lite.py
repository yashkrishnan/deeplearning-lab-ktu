"""
Lab 2: CIFAR-10 Classifiers (LIGHTWEIGHT VERSION)
==================================================

This is a lightweight version optimized for faster training on laptops.
Changes from original:
- Reduced training samples: 2000 (from 5000)
- Reduced test samples: 500 (from 1000)
- Reduced epochs: 5 (from 10)
- Smaller neural network architecture
- Faster execution time: ~2-3 minutes (vs ~5 minutes)

This program implements and compares three different classification approaches on CIFAR-10:
1. K-Nearest Neighbors (KNN) Classifier
2. 3-Layer Neural Network
3. SVM/Softmax Classifier

Author: Deep Learning Lab
Date: March 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from pathlib import Path
import time
import pickle

# For neural network
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms


class CIFAR10ClassifiersLite:
    """
    A lightweight comparison of different classifiers on CIFAR-10 dataset.
    Optimized for faster execution on laptops.
    """
    
    def __init__(self, output_dir='output'):
        """
        Initialize the classifier comparison framework.
        
        Args:
            output_dir (str): Directory to save outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # CIFAR-10 class names
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                           'dog', 'frog', 'horse', 'ship', 'truck']
        
        print(f"✓ Output directory: {self.output_dir}")
        print(f"✓ Device: {self.device}")
        print(f"✓ LIGHTWEIGHT VERSION - Optimized for laptops")
    
    def load_data(self):
        """
        Load and preprocess CIFAR-10 dataset with reduced samples.
        
        Returns:
            tuple: Training and test data
        """
        print("\nLoading CIFAR-10 dataset (lightweight)...")
        
        # Define transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Load CIFAR-10
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform
        )
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform
        )
        
        # Use smaller subset for faster execution
        train_indices = np.random.choice(len(trainset), 2000, replace=False)
        test_indices = np.random.choice(len(testset), 500, replace=False)
        
        trainset = torch.utils.data.Subset(trainset, train_indices.tolist())
        testset = torch.utils.data.Subset(testset, test_indices.tolist())
        
        print(f"  ✓ Training samples: {len(trainset)}")
        print(f"  ✓ Test samples: {len(testset)}")
        
        return trainset, testset
    
    def prepare_data_for_sklearn(self, dataset):
        """
        Prepare data for sklearn classifiers (flatten images).
        
        Args:
            dataset: PyTorch dataset
            
        Returns:
            tuple: (X, y) flattened features and labels
        """
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        data, labels = next(iter(loader))
        
        # Flatten images: (N, 3, 32, 32) -> (N, 3072)
        X = data.numpy().reshape(len(data), -1)
        y = labels.numpy()
        
        return X, y
    
    def train_knn(self, X_train, y_train, X_test, y_test):
        """
        Train and evaluate K-Nearest Neighbors classifier.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            dict: Results including accuracy and predictions
        """
        print("\n" + "="*60)
        print("1. K-NEAREST NEIGHBORS (KNN) CLASSIFIER")
        print("="*60)
        
        start_time = time.time()
        
        # Train KNN with k=3 (reduced from 5 for speed)
        print("Training KNN (k=3)...")
        knn = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
        knn.fit(X_train, y_train)
        
        # Predictions
        y_pred = knn.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        train_time = time.time() - start_time
        
        print(f"  ✓ Training completed in {train_time:.2f} seconds")
        print(f"  ✓ Test Accuracy: {accuracy*100:.2f}%")
        
        return {
            'model': knn,
            'predictions': y_pred,
            'accuracy': accuracy,
            'train_time': train_time,
            'name': 'KNN'
        }
    
    def train_svm(self, X_train, y_train, X_test, y_test):
        """
        Train and evaluate SVM classifier with linear kernel.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            dict: Results including accuracy and predictions
        """
        print("\n" + "="*60)
        print("2. SVM CLASSIFIER (LINEAR KERNEL)")
        print("="*60)
        
        start_time = time.time()
        
        # Train SVM
        print("Training SVM...")
        svm = SVC(kernel='linear', C=1.0, random_state=42)
        svm.fit(X_train, y_train)
        
        # Predictions
        y_pred = svm.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        train_time = time.time() - start_time
        
        print(f"  ✓ Training completed in {train_time:.2f} seconds")
        print(f"  ✓ Test Accuracy: {accuracy*100:.2f}%")
        
        return {
            'model': svm,
            'predictions': y_pred,
            'accuracy': accuracy,
            'train_time': train_time,
            'name': 'SVM'
        }
    
    def train_neural_network(self, trainset, testset):
        """
        Train and evaluate a smaller 3-layer neural network.
        
        Args:
            trainset: Training dataset
            testset: Test dataset
            
        Returns:
            dict: Results including accuracy and predictions
        """
        print("\n" + "="*60)
        print("3. 3-LAYER NEURAL NETWORK (LIGHTWEIGHT)")
        print("="*60)
        
        start_time = time.time()
        
        # Create data loaders
        train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
        test_loader = DataLoader(testset, batch_size=64, shuffle=False)
        
        # Define smaller 3-layer neural network
        class ThreeLayerNN(nn.Module):
            def __init__(self):
                super(ThreeLayerNN, self).__init__()
                self.flatten = nn.Flatten()
                self.fc1 = nn.Linear(3 * 32 * 32, 256)  # Reduced from 512
                self.relu1 = nn.ReLU()
                self.fc2 = nn.Linear(256, 128)          # Reduced from 256
                self.relu2 = nn.ReLU()
                self.fc3 = nn.Linear(128, 10)           # Output layer
            
            def forward(self, x):
                x = self.flatten(x)
                x = self.relu1(self.fc1(x))
                x = self.relu2(self.fc2(x))
                x = self.fc3(x)
                return x
        
        # Initialize model
        model = ThreeLayerNN().to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training
        print("Training Neural Network...")
        print(f"  Architecture: 3072 -> 256 -> 128 -> 10")
        
        num_epochs = 5  # Reduced from 10
        train_losses = []
        train_accuracies = []
        
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100 * correct / total
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_acc)
            
            print(f"  Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
        
        # Evaluation
        model.eval()
        y_pred = []
        y_true = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                y_pred.extend(predicted.cpu().numpy())
                y_true.extend(labels.numpy())
        
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        accuracy = accuracy_score(y_true, y_pred)
        train_time = time.time() - start_time
        
        print(f"  ✓ Training completed in {train_time:.2f} seconds")
        print(f"  ✓ Test Accuracy: {accuracy*100:.2f}%")
        
        # Plot training history
        self.plot_training_history(train_losses, train_accuracies)
        
        return {
            'model': model,
            'predictions': y_pred,
            'accuracy': accuracy,
            'train_time': train_time,
            'name': 'Neural Network',
            'train_losses': train_losses,
            'train_accuracies': train_accuracies
        }
    
    def plot_training_history(self, losses, accuracies):
        """
        Plot training loss and accuracy curves.
        
        Args:
            losses: List of training losses
            accuracies: List of training accuracies
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(losses, 'b-', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(accuracies, 'g-', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training Accuracy')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_history.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Training history saved to: {self.output_dir / 'training_history.png'}")
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        """
        Plot confusion matrix for a classifier.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        filename = f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Confusion matrix saved to: {self.output_dir / filename}")
    
    def compare_results(self, results):
        """
        Compare and visualize results from all classifiers.
        
        Args:
            results: List of result dictionaries from each classifier
        """
        print("\n" + "="*60)
        print("COMPARISON OF CLASSIFIERS")
        print("="*60)
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        names = [r['name'] for r in results]
        accuracies = [r['accuracy'] * 100 for r in results]
        times = [r['train_time'] for r in results]
        
        # Accuracy comparison
        bars1 = ax1.bar(names, accuracies, color=['#3498db', '#e74c3c', '#2ecc71'])
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Test Accuracy Comparison')
        ax1.set_ylim([0, 100])
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%',
                    ha='center', va='bottom')
        
        # Training time comparison
        bars2 = ax2.bar(names, times, color=['#3498db', '#e74c3c', '#2ecc71'])
        ax2.set_ylabel('Training Time (seconds)')
        ax2.set_title('Training Time Comparison')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}s',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'classifier_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Comparison plot saved to: {self.output_dir / 'classifier_comparison.png'}")
        
        # Print summary table
        print("\nSummary Table:")
        print("-" * 60)
        print(f"{'Classifier':<20} {'Accuracy':<15} {'Training Time':<15}")
        print("-" * 60)
        for r in results:
            print(f"{r['name']:<20} {r['accuracy']*100:>6.2f}%        {r['train_time']:>6.2f}s")
        print("-" * 60)
    
    def run_all_classifiers(self):
        """
        Run all classifiers and compare results.
        """
        print("\n" + "="*60)
        print("  LAB 2: CIFAR-10 CLASSIFIERS (LIGHTWEIGHT)")
        print("="*60)
        
        total_start = time.time()
        
        # Load data
        trainset, testset = self.load_data()
        
        # Prepare data for sklearn classifiers
        print("\nPreparing data for classical ML classifiers...")
        X_train, y_train = self.prepare_data_for_sklearn(trainset)
        X_test, y_test = self.prepare_data_for_sklearn(testset)
        print("  ✓ Data prepared (flattened to vectors)")
        
        # Train classifiers
        results = []
        
        # 1. KNN
        knn_results = self.train_knn(X_train, y_train, X_test, y_test)
        self.plot_confusion_matrix(y_test, knn_results['predictions'], knn_results['name'])
        results.append(knn_results)
        
        # 2. SVM
        svm_results = self.train_svm(X_train, y_train, X_test, y_test)
        self.plot_confusion_matrix(y_test, svm_results['predictions'], svm_results['name'])
        results.append(svm_results)
        
        # 3. Neural Network
        nn_results = self.train_neural_network(trainset, testset)
        self.plot_confusion_matrix(y_test, nn_results['predictions'], nn_results['name'])
        results.append(nn_results)
        
        # Compare results
        self.compare_results(results)
        
        total_time = time.time() - total_start
        
        print("\n" + "="*60)
        print("  SUMMARY")
        print("="*60)
        print(f"✓ All classifiers trained and evaluated!")
        print(f"✓ Total execution time: {total_time:.2f} seconds")
        print(f"✓ Output files saved to: {self.output_dir}/")
        print("\nGenerated files:")
        print("  1. training_history.png - Neural network training curves")
        print("  2. confusion_matrix_knn.png - KNN confusion matrix")
        print("  3. confusion_matrix_svm.png - SVM confusion matrix")
        print("  4. confusion_matrix_neural_network.png - NN confusion matrix")
        print("  5. classifier_comparison.png - Performance comparison")
        print("\n" + "="*60)


def main():
    """
    Main function to run the CIFAR-10 classifiers comparison.
    """
    # Create classifier comparison instance
    classifier = CIFAR10ClassifiersLite(output_dir='output')
    
    # Run all classifiers
    classifier.run_all_classifiers()
    
    print("\n✨ Lab 2 (Lightweight) completed! Check the 'output' folder for results.")


if __name__ == "__main__":
    main()


