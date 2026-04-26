#!/usr/bin/env python3
"""
Lab 5: Image Segmentation with UNet (FULL VERSION)
===================================================

Full version using Oxford-IIIT Pet Dataset with scaled-up parameters.

Full version parameters (vs lite):
- Training samples: 800 (vs 100)
- Validation samples: 200 (vs 20)
- Epochs: 20 (vs 10)
- Image size: 128x128 (vs 64x64)
- UNet channels: 64->128->256->512 (vs 32->64->128->256)
- Binary segmentation: foreground (pet) vs background

Dataset: Oxford-IIIT Pet Dataset (real images)

This program demonstrates:
1. UNet architecture for semantic segmentation
2. Training on real pet images
3. Evaluation metrics (IoU, Dice coefficient)
4. Visualization of segmentation results
5. Comparison with simple CNN baseline

Author: Deep Learning Lab
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import time
from tqdm import tqdm
from PIL import Image
import random

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Create output directory
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Full version configuration
NUM_TRAIN_SAMPLES = 800
NUM_VAL_SAMPLES = 200
NUM_EPOCHS = 20
IMG_SIZE = 128
BATCH_SIZE = 16


class PetSegmentationDataset(Dataset):
    """Load real pet images and create simple binary segmentation masks."""

    def __init__(self, data_dir, num_samples=800, img_size=128):
        self.img_size = img_size
        self.data_dir = Path(data_dir)

        # Find all jpg images
        img_dir = self.data_dir / "images" / "images"
        if not img_dir.exists():
            img_dir = self.data_dir / "images"

        all_images = list(img_dir.glob("*.jpg"))

        if not all_images:
            raise FileNotFoundError(
                f"No images found in {img_dir}\n"
                "Please ensure the Pet dataset is downloaded."
            )

        # Randomly sample images
        random.shuffle(all_images)
        self.image_paths = all_images[:num_samples]

        print(f"Loaded {len(self.image_paths)} pet images from {img_dir}")

    def __len__(self):
        return len(self.image_paths)

    def create_simple_mask(self, img_array):
        """
        Create a simple binary mask using color-based segmentation.
        Assumes pet is in center and has different color than background.
        """
        # Convert to grayscale for simplicity
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array

        # Use Otsu's method approximation - threshold at mean
        threshold = np.mean(gray)

        # Create binary mask (1 for foreground/pet, 0 for background)
        mask = (gray > threshold).astype(np.int64)

        # Apply morphological operations to clean up mask
        from scipy import ndimage

        # Fill holes
        mask = ndimage.binary_fill_holes(mask).astype(np.int64)

        # Remove small objects
        mask = ndimage.binary_opening(mask, iterations=2).astype(np.int64)

        return mask

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')

        # Resize
        img = img.resize((self.img_size, self.img_size))
        img_array = np.array(img).astype(np.float32) / 255.0

        # Create simple binary mask
        mask = self.create_simple_mask(img_array)

        # Convert to torch tensors
        # Image: [C, H, W]
        img_tensor = torch.FloatTensor(img_array.transpose(2, 0, 1))

        # Mask: [H, W]
        mask_tensor = torch.LongTensor(mask)

        return img_tensor, mask_tensor


class UNet(nn.Module):
    """Full UNet architecture for semantic segmentation.

    Full version uses deeper channels: 64->128->256->512
    """

    def __init__(self, in_channels=3, num_classes=2):
        super(UNet, self).__init__()

        # Encoder (downsampling) - full channels
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)

        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)

        # Decoder (upsampling)
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

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc3))

        # Decoder with skip connections
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        # Output
        out = self.out(dec1)

        return out


class SimpleCNN(nn.Module):
    """Simple CNN baseline for comparison."""

    def __init__(self, in_channels=3, num_classes=2):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1)
        )

    def forward(self, x):
        return self.features(x)


def calculate_iou(pred, target, num_classes=2):
    """Calculate Intersection over Union (IoU)."""
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls

        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()

        if union == 0:
            iou = 1.0
        else:
            iou = intersection / union

        ious.append(iou.item())

    return np.mean(ious)


def calculate_dice(pred, target, num_classes=2):
    """Calculate Dice coefficient."""
    dice_scores = []
    pred = pred.view(-1)
    target = target.view(-1)

    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls

        intersection = (pred_cls & target_cls).sum().float()
        dice = (2.0 * intersection) / (pred_cls.sum() + target_cls.sum() + 1e-8)

        dice_scores.append(dice.item())

    return np.mean(dice_scores)


def train_model(model, train_loader, val_loader, num_epochs, device, model_name="Model"):
    """Train the segmentation model."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_iou': [],
        'val_dice': []
    }

    print(f"\nTraining {model_name}...")
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        val_iou = 0
        val_dice = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                preds = outputs.argmax(dim=1)
                val_iou += calculate_iou(preds, masks)
                val_dice += calculate_dice(preds, masks)

        val_loss /= len(val_loader)
        val_iou /= len(val_loader)
        val_dice /= len(val_loader)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        history['val_dice'].append(val_dice)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
              f"IoU={val_iou:.4f}, Dice={val_dice:.4f}")
        sys.stdout.flush()

        scheduler.step()

    return history


def visualize_results(model, dataset, device, num_samples=4):
    """Visualize segmentation results."""
    model.eval()

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))

    indices = random.sample(range(len(dataset)), num_samples)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            img, mask = dataset[idx]

            # Get prediction
            img_batch = img.unsqueeze(0).to(device)
            output = model(img_batch)
            pred = output.argmax(dim=1)[0].cpu().numpy()

            # Display
            img_display = img.permute(1, 2, 0).numpy()

            axes[i, 0].imshow(img_display)
            axes[i, 0].set_title('Input Image')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(mask.numpy(), cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(pred, cmap='gray')
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'segmentation_results.png', dpi=100, bbox_inches='tight')
    print(f"Saved results to {OUTPUT_DIR / 'segmentation_results.png'}")
    plt.close()


def plot_training_history(unet_history, cnn_history):
    """Plot training history comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Training loss
    axes[0, 0].plot(unet_history['train_loss'], 'b-', label='UNet')
    axes[0, 0].plot(cnn_history['train_loss'], 'r-', label='Simple CNN')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Validation loss
    axes[0, 1].plot(unet_history['val_loss'], 'b-', label='UNet')
    axes[0, 1].plot(cnn_history['val_loss'], 'r-', label='Simple CNN')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # IoU
    axes[1, 0].plot(unet_history['val_iou'], 'b-', label='UNet')
    axes[1, 0].plot(cnn_history['val_iou'], 'r-', label='Simple CNN')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('IoU')
    axes[1, 0].set_title('Validation IoU')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Dice
    axes[1, 1].plot(unet_history['val_dice'], 'b-', label='UNet')
    axes[1, 1].plot(cnn_history['val_dice'], 'r-', label='Simple CNN')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Dice')
    axes[1, 1].set_title('Validation Dice Coefficient')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'training_history.png', dpi=100, bbox_inches='tight')
    print(f"Saved training history to {OUTPUT_DIR / 'training_history.png'}")
    plt.close()


def main():
    print("="*60)
    print("Lab 5: Image Segmentation with Real Pet Images (FULL)")
    print("="*60)

    start_time = time.time()

    # Check if dataset exists
    data_dir = Path("data/pets")
    if not data_dir.exists():
        print(f"\nError: Pet dataset not found at {data_dir}")
        print("Please ensure the dataset is downloaded.")
        return

    # Load datasets
    print(f"\nLoading Pet dataset (train={NUM_TRAIN_SAMPLES}, val={NUM_VAL_SAMPLES}, "
          f"img_size={IMG_SIZE}x{IMG_SIZE})...")
    train_dataset = PetSegmentationDataset(data_dir, num_samples=NUM_TRAIN_SAMPLES, img_size=IMG_SIZE)
    val_dataset = PetSegmentationDataset(data_dir, num_samples=NUM_VAL_SAMPLES, img_size=IMG_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Train UNet (full version)
    print("\n" + "="*60)
    print("Training Full UNet (64->128->256->512 channels)")
    print("="*60)
    unet = UNet(in_channels=3, num_classes=2)
    print(f"UNet parameters: {sum(p.numel() for p in unet.parameters()):,}")
    unet_history = train_model(unet, train_loader, val_loader, NUM_EPOCHS, device, "UNet")

    # Train Simple CNN
    print("\n" + "="*60)
    print("Training Simple CNN Baseline")
    print("="*60)
    cnn = SimpleCNN(in_channels=3, num_classes=2)
    print(f"CNN parameters: {sum(p.numel() for p in cnn.parameters()):,}")
    cnn_history = train_model(cnn, train_loader, val_loader, NUM_EPOCHS, device, "Simple CNN")

    # Plot comparison
    print("\nGenerating visualizations...")
    plot_training_history(unet_history, cnn_history)
    visualize_results(unet, val_dataset, device, num_samples=4)

    # Final comparison
    print("\n" + "="*60)
    print("Final Results")
    print("="*60)
    print(f"UNet - Final IoU: {unet_history['val_iou'][-1]:.4f}, "
          f"Dice: {unet_history['val_dice'][-1]:.4f}")
    print(f"Simple CNN - Final IoU: {cnn_history['val_iou'][-1]:.4f}, "
          f"Dice: {cnn_history['val_dice'][-1]:.4f}")

    elapsed_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Total execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"{'='*60}")
    print(f"\nOutputs saved to: {OUTPUT_DIR.absolute()}")
    print("\nLab 5 completed successfully!")


if __name__ == "__main__":
    main()
