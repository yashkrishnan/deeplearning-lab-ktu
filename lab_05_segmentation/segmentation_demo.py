#!/usr/bin/env python3
"""
Lab 5: Image Segmentation with UNet
====================================

This program demonstrates:
1. UNet architecture for semantic segmentation
2. Training on synthetic segmentation data
3. Evaluation metrics (IoU, Dice coefficient)
4. Visualization of segmentation results
5. Comparison with simple CNN baseline

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

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create output directory
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SyntheticSegmentationDataset(Dataset):
    """Generate synthetic images with segmentation masks."""
    
    def __init__(self, num_samples=1000, img_size=128):
        self.num_samples = num_samples
        self.img_size = img_size
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Create synthetic image with geometric shapes
        img = np.zeros((3, self.img_size, self.img_size), dtype=np.float32)
        mask = np.zeros((self.img_size, self.img_size), dtype=np.int64)
        
        # Random seed for this sample
        np.random.seed(idx)
        
        # Add 2-4 random shapes
        num_shapes = np.random.randint(2, 5)
        
        for shape_id in range(1, num_shapes + 1):
            shape_type = np.random.choice(['circle', 'rectangle'])
            
            if shape_type == 'circle':
                # Random circle
                cx = np.random.randint(20, self.img_size - 20)
                cy = np.random.randint(20, self.img_size - 20)
                radius = np.random.randint(10, 25)
                
                y, x = np.ogrid[:self.img_size, :self.img_size]
                circle_mask = (x - cx)**2 + (y - cy)**2 <= radius**2
                
                # Random color for image
                color = np.random.rand(3)
                for c in range(3):
                    img[c][circle_mask] = color[c]
                
                # Update segmentation mask
                mask[circle_mask] = shape_id
                
            else:  # rectangle
                x1 = np.random.randint(10, self.img_size - 40)
                y1 = np.random.randint(10, self.img_size - 40)
                w = np.random.randint(20, 40)
                h = np.random.randint(20, 40)
                x2 = min(x1 + w, self.img_size)
                y2 = min(y1 + h, self.img_size)
                
                # Random color for image
                color = np.random.rand(3)
                for c in range(3):
                    img[c, y1:y2, x1:x2] = color[c]
                
                # Update segmentation mask
                mask[y1:y2, x1:x2] = shape_id
        
        # Add some noise
        img += np.random.randn(3, self.img_size, self.img_size) * 0.05
        img = np.clip(img, 0, 1)
        
        return torch.FloatTensor(img), torch.LongTensor(mask)


class UNet(nn.Module):
    """UNet architecture for semantic segmentation."""
    
    def __init__(self, in_channels=3, num_classes=5):
        super(UNet, self).__init__()
        
        # Encoder (downsampling)
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder (upsampling)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        # Output layer
        self.out = nn.Conv2d(64, num_classes, 1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
    def conv_block(self, in_channels, out_channels):
        """Convolutional block with two conv layers."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        return self.out(dec1)


class SimpleCNN(nn.Module):
    """Simple CNN baseline for comparison."""
    
    def __init__(self, in_channels=3, num_classes=5):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, 1)
        )
    
    def forward(self, x):
        return self.features(x)


def calculate_iou(pred, target, num_classes):
    """Calculate Intersection over Union (IoU) for each class."""
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    
    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls
        
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        
        if union == 0:
            iou = 1.0  # If no ground truth and no prediction, perfect score
        else:
            iou = intersection / union
        
        ious.append(iou.item())
    
    return ious


def calculate_dice(pred, target, num_classes):
    """Calculate Dice coefficient for each class."""
    dice_scores = []
    pred = pred.view(-1)
    target = target.view(-1)
    
    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls
        
        intersection = (pred_cls & target_cls).sum().float()
        dice = (2 * intersection) / (pred_cls.sum() + target_cls.sum()).float()
        
        if (pred_cls.sum() + target_cls.sum()) == 0:
            dice = 1.0
        
        dice_scores.append(dice.item())
    
    return dice_scores


def train_model(model, train_loader, val_loader, num_epochs=20, model_name="Model"):
    """Train the segmentation model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_iou': [],
        'val_dice': []
    }
    
    print(f"\nTraining {model_name}...")
    print("-" * 60)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        # Progress bar for training
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', leave=False)
        for images, masks in train_pbar:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_ious = []
        all_dice = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]', leave=False)
            for images, masks in val_pbar:
                images, masks = images.to(device), masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                # Calculate metrics
                pred = outputs.argmax(dim=1)
                ious = calculate_iou(pred, masks, num_classes=5)
                dice = calculate_dice(pred, masks, num_classes=5)
                
                all_ious.append(ious)
                all_dice.append(dice)
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        val_loss /= len(val_loader)
        mean_iou = np.mean(all_ious)
        mean_dice = np.mean(all_dice)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(mean_iou)
        history['val_dice'].append(mean_dice)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), OUTPUT_DIR / f'{model_name.lower().replace(" ", "_")}_best.pth')
        
        # Print progress every epoch
        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"IoU: {mean_iou:.4f}, Dice: {mean_dice:.4f}")
    
    print(f"✓ {model_name} training complete!")
    return history


def visualize_predictions(model, dataset, num_samples=5, model_name="Model"):
    """Visualize segmentation predictions."""
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    
    with torch.no_grad():
        for i in range(num_samples):
            image, mask = dataset[i]
            image_input = image.unsqueeze(0).to(device)
            
            output = model(image_input)
            pred = output.argmax(dim=1).squeeze().cpu().numpy()
            
            # Display image
            axes[i, 0].imshow(image.permute(1, 2, 0).numpy())
            axes[i, 0].set_title('Input Image')
            axes[i, 0].axis('off')
            
            # Display ground truth
            axes[i, 1].imshow(mask.numpy(), cmap='tab10', vmin=0, vmax=4)
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            # Display prediction
            axes[i, 2].imshow(pred, cmap='tab10', vmin=0, vmax=4)
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')
    
    plt.suptitle(f'{model_name} - Segmentation Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    output_path = OUTPUT_DIR / f'{model_name.lower().replace(" ", "_")}_predictions.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def plot_training_history(histories, model_names):
    """Plot training history comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = [
        ('train_loss', 'Training Loss'),
        ('val_loss', 'Validation Loss'),
        ('val_iou', 'Validation IoU'),
        ('val_dice', 'Validation Dice Coefficient')
    ]
    
    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        for history, name in zip(histories, model_names):
            ax.plot(history[metric], label=name, linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
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
    print("Lab 5: Image Segmentation with UNet")
    print("=" * 70)
    print()
    print(f"Device: {device}")
    print()
    
    # Create datasets
    print("Creating synthetic segmentation dataset...")
    train_dataset = SyntheticSegmentationDataset(num_samples=800, img_size=128)
    val_dataset = SyntheticSegmentationDataset(num_samples=200, img_size=128)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    print(f"  • Training samples: {len(train_dataset)}")
    print(f"  • Validation samples: {len(val_dataset)}")
    print(f"  • Image size: 128x128")
    print(f"  • Number of classes: 5 (background + 4 shapes)")
    print()
    
    # Initialize models
    unet = UNet(in_channels=3, num_classes=5).to(device)
    simple_cnn = SimpleCNN(in_channels=3, num_classes=5).to(device)
    
    print("Model Architectures:")
    print(f"  • UNet parameters: {sum(p.numel() for p in unet.parameters()):,}")
    print(f"  • Simple CNN parameters: {sum(p.numel() for p in simple_cnn.parameters()):,}")
    print()
    
    # Train models
    num_epochs = 20
    
    start_time = time.time()
    unet_history = train_model(unet, train_loader, val_loader, num_epochs, "UNet")
    unet_time = time.time() - start_time
    
    start_time = time.time()
    cnn_history = train_model(simple_cnn, train_loader, val_loader, num_epochs, "Simple CNN")
    cnn_time = time.time() - start_time
    
    print()
    print("Training Summary:")
    print("-" * 60)
    print(f"UNet:")
    print(f"  • Training time: {unet_time:.2f}s")
    print(f"  • Final Val Loss: {unet_history['val_loss'][-1]:.4f}")
    print(f"  • Final Val IoU: {unet_history['val_iou'][-1]:.4f}")
    print(f"  • Final Val Dice: {unet_history['val_dice'][-1]:.4f}")
    print()
    print(f"Simple CNN:")
    print(f"  • Training time: {cnn_time:.2f}s")
    print(f"  • Final Val Loss: {cnn_history['val_loss'][-1]:.4f}")
    print(f"  • Final Val IoU: {cnn_history['val_iou'][-1]:.4f}")
    print(f"  • Final Val Dice: {cnn_history['val_dice'][-1]:.4f}")
    print()
    
    # Visualize predictions
    print("Generating visualizations...")
    unet_vis = visualize_predictions(unet, val_dataset, num_samples=5, model_name="UNet")
    cnn_vis = visualize_predictions(simple_cnn, val_dataset, num_samples=5, model_name="Simple CNN")
    
    print(f"  ✓ UNet predictions: {unet_vis}")
    print(f"  ✓ Simple CNN predictions: {cnn_vis}")
    print()
    
    # Plot training history
    history_plot = plot_training_history(
        [unet_history, cnn_history],
        ['UNet', 'Simple CNN']
    )
    print(f"  ✓ Training comparison: {history_plot}")
    print()
    
    print("=" * 70)
    print("Lab 5 Complete!")
    print("=" * 70)
    print()
    print("Key Findings:")
    print("  • UNet with skip connections performs better than simple CNN")
    print("  • Skip connections help preserve spatial information")
    print("  • IoU and Dice coefficient are standard segmentation metrics")
    print("  • UNet is the standard architecture for medical image segmentation")
    print()


if __name__ == "__main__":
    main()


