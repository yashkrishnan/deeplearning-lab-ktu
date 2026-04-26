#!/usr/bin/env python3
"""
Lab 6: Object Detection with YOLO-style Architecture (FULL VERSION)
====================================================================

Full version using VOC2012 dataset with scaled-up parameters.

Full version parameters (vs lite):
- Training samples: 1000 (vs 50)
- Validation samples: 200 (separate)
- Epochs: 30 (vs 10)
- Image size: 224x224
- Max objects: 3
- Full backbone channels: 32->64->128->256->512

Dataset: Pascal VOC 2012
Expected runtime: ~30-60 minutes on CPU

Author: Deep Learning Lab
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import time
from tqdm import tqdm
import xml.etree.ElementTree as ET
from PIL import Image
import random

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Create output directory
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# VOC Classes - using subset for training
VOC_CLASSES = ['person', 'car', 'dog']
NUM_CLASSES = len(VOC_CLASSES)
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(VOC_CLASSES)}

# Full version configuration
NUM_TRAIN_SAMPLES = 1000
NUM_VAL_SAMPLES = 200
BATCH_SIZE = 16
NUM_EPOCHS = 30
IMG_SIZE = 224
MAX_OBJECTS = 3


class VOCDetectionDataset(Dataset):
    """Load VOC2012 dataset for object detection."""

    def __init__(self, voc_root, img_size=224, max_samples=None, max_objects=3):
        self.voc_root = Path(voc_root)
        self.img_size = img_size
        self.max_objects = max_objects

        # Paths
        self.img_dir = self.voc_root / "JPEGImages"
        self.ann_dir = self.voc_root / "Annotations"

        # Get all annotation files
        all_ann_files = sorted(list(self.ann_dir.glob("*.xml")))

        # Filter to only include images with our target classes
        self.samples = []
        for ann_file in all_ann_files:
            if self._has_target_classes(ann_file):
                self.samples.append(ann_file.stem)
                if max_samples and len(self.samples) >= max_samples:
                    break

        print(f"Loaded {len(self.samples)} images with target classes: {VOC_CLASSES}")

    def _has_target_classes(self, ann_file):
        """Check if annotation contains any of our target classes."""
        try:
            tree = ET.parse(ann_file)
            root = tree.getroot()
            for obj in root.findall('object'):
                name = obj.find('name').text
                if name in VOC_CLASSES:
                    return True
        except Exception:
            pass
        return False

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id = self.samples[idx]

        # Load image
        img_path = self.img_dir / f"{img_id}.jpg"
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size

        # Resize image
        img = img.resize((self.img_size, self.img_size))
        img = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0

        # Load annotations
        ann_path = self.ann_dir / f"{img_id}.xml"
        tree = ET.parse(ann_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in VOC_CLASSES:
                continue

            if len(boxes) >= self.max_objects:
                break

            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)

            # Normalize coordinates
            x_center = ((xmin + xmax) / 2) / orig_w
            y_center = ((ymin + ymax) / 2) / orig_h
            width = (xmax - xmin) / orig_w
            height = (ymax - ymin) / orig_h

            # Clip to [0, 1]
            x_center = np.clip(x_center, 0, 1)
            y_center = np.clip(y_center, 0, 1)
            width = np.clip(width, 0, 1)
            height = np.clip(height, 0, 1)

            boxes.append([x_center, y_center, width, height])
            labels.append(CLASS_TO_IDX[name])

        # Pad to max_objects
        while len(boxes) < self.max_objects:
            boxes.append([0, 0, 0, 0])
            labels.append(-1)

        boxes = np.array(boxes[:self.max_objects], dtype=np.float32)
        labels = np.array(labels[:self.max_objects], dtype=np.int64)

        return torch.FloatTensor(img), torch.FloatTensor(boxes), torch.LongTensor(labels)


class YOLODetector(nn.Module):
    """Full YOLO-style detector with deeper backbone (32->64->128->256->512)."""

    def __init__(self, num_classes=3, max_objects=3):
        super(YOLODetector, self).__init__()
        self.num_classes = num_classes
        self.max_objects = max_objects

        # Full backbone
        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112x112

            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56x56

            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28x28

            # Block 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14x14

            # Block 5 (full version - deeper)
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 7x7
        )

        # Detection head (larger for full version)
        self.detector = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, max_objects * (4 + num_classes))
        )

    def forward(self, x):
        features = self.backbone(x)
        output = self.detector(features)

        # Reshape to [batch, max_objects, 4 + num_classes]
        batch_size = x.size(0)
        output = output.view(batch_size, self.max_objects, 4 + self.num_classes)

        # Split into boxes and class predictions
        boxes = torch.sigmoid(output[:, :, :4])  # Normalize to [0, 1]
        class_logits = output[:, :, 4:]

        return boxes, class_logits


def detection_loss(pred_boxes, pred_classes, target_boxes, target_labels, num_classes):
    """Combined loss for object detection."""
    batch_size = pred_boxes.size(0)

    # Mask for valid objects (label != -1)
    valid_mask = (target_labels != -1).float()

    # Box loss (only for valid objects)
    box_loss = nn.functional.mse_loss(
        pred_boxes * valid_mask.unsqueeze(-1),
        target_boxes * valid_mask.unsqueeze(-1),
        reduction='sum'
    ) / (valid_mask.sum() + 1e-6)

    # Classification loss
    target_labels_masked = target_labels.clone()
    target_labels_masked[target_labels == -1] = 0  # Set invalid to 0 for loss computation

    class_loss = nn.functional.cross_entropy(
        pred_classes.view(-1, num_classes),
        target_labels_masked.view(-1),
        reduction='none'
    )
    class_loss = (class_loss.view(batch_size, -1) * valid_mask).sum() / (valid_mask.sum() + 1e-6)

    return box_loss + class_loss, box_loss, class_loss


def train_model(model, train_loader, num_epochs, device):
    """Train the detection model."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    history = {'loss': [], 'box_loss': [], 'class_loss': []}

    print("\nTraining Full YOLO Detector...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_box_loss = 0
        epoch_class_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, boxes, labels in pbar:
            images = images.to(device)
            boxes = boxes.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            pred_boxes, pred_classes = model(images)
            loss, box_loss, class_loss = detection_loss(
                pred_boxes, pred_classes, boxes, labels, NUM_CLASSES
            )

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_box_loss += box_loss.item()
            epoch_class_loss += class_loss.item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'box': f'{box_loss.item():.4f}',
                'cls': f'{class_loss.item():.4f}'
            })

        scheduler.step()

        avg_loss = epoch_loss / len(train_loader)
        avg_box_loss = epoch_box_loss / len(train_loader)
        avg_class_loss = epoch_class_loss / len(train_loader)

        history['loss'].append(avg_loss)
        history['box_loss'].append(avg_box_loss)
        history['class_loss'].append(avg_class_loss)

        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Box={avg_box_loss:.4f}, Class={avg_class_loss:.4f}")

    return history


def visualize_detections(model, dataset, device, num_samples=4):
    """Visualize detection results."""
    model.eval()

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.ravel()

    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    with torch.no_grad():
        for idx, ax in zip(indices, axes):
            img, boxes, labels = dataset[idx]

            # Get predictions
            img_batch = img.unsqueeze(0).to(device)
            pred_boxes, pred_classes = model(img_batch)
            pred_boxes = pred_boxes[0].cpu().numpy()
            pred_labels = torch.argmax(pred_classes[0], dim=-1).cpu().numpy()

            # Display image
            img_display = img.permute(1, 2, 0).numpy()
            ax.imshow(img_display)

            # Draw predictions (only confident ones)
            for i in range(len(pred_boxes)):
                if pred_labels[i] >= 0 and pred_labels[i] < NUM_CLASSES:
                    x, y, w, h = pred_boxes[i]
                    if w > 0.05 and h > 0.05:  # Filter small boxes
                        x1 = (x - w/2) * IMG_SIZE
                        y1 = (y - h/2) * IMG_SIZE
                        width = w * IMG_SIZE
                        height = h * IMG_SIZE

                        rect = patches.Rectangle(
                            (x1, y1), width, height,
                            linewidth=2, edgecolor='red', facecolor='none'
                        )
                        ax.add_patch(rect)

                        label_text = VOC_CLASSES[pred_labels[i]]
                        ax.text(x1, y1-5, label_text, color='red',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

            ax.axis('off')
            ax.set_title(f'Sample {idx}')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'detection_results.png', dpi=150, bbox_inches='tight')
    print(f"Saved detection visualization to {OUTPUT_DIR / 'detection_results.png'}")
    plt.close()


def plot_training_history(history):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Total loss
    axes[0].plot(history['loss'], 'b-', label='Total Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Component losses
    axes[1].plot(history['box_loss'], 'r-', label='Box Loss')
    axes[1].plot(history['class_loss'], 'g-', label='Class Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Loss Components')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'training_history.png', dpi=150, bbox_inches='tight')
    print(f"Saved training history to {OUTPUT_DIR / 'training_history.png'}")
    plt.close()


def main():
    print("=" * 60)
    print("Lab 6: Object Detection with VOC2012 (FULL VERSION)")
    print("=" * 60)

    start_time = time.time()

    # Check if dataset exists
    voc_root = Path("data/voc/VOC2012")
    if not voc_root.exists():
        print(f"\nError: VOC2012 dataset not found at {voc_root}")
        print("Please ensure the dataset is available.")
        return

    # Load training dataset
    print(f"\nLoading VOC2012 training dataset (max {NUM_TRAIN_SAMPLES} samples)...")
    train_dataset = VOCDetectionDataset(
        voc_root=voc_root,
        img_size=IMG_SIZE,
        max_samples=NUM_TRAIN_SAMPLES,
        max_objects=MAX_OBJECTS
    )

    # Load validation dataset (separate slice)
    print(f"Loading VOC2012 validation dataset (max {NUM_VAL_SAMPLES} samples)...")
    val_dataset = VOCDetectionDataset(
        voc_root=voc_root,
        img_size=IMG_SIZE,
        max_samples=NUM_TRAIN_SAMPLES + NUM_VAL_SAMPLES,
        max_objects=MAX_OBJECTS
    )
    # Use only the val portion (samples beyond train set)
    val_dataset.samples = val_dataset.samples[NUM_TRAIN_SAMPLES:]
    print(f"  Val samples: {len(val_dataset.samples)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    # Create model
    print("\nCreating Full YOLO Detector (32->64->128->256->512 backbone)...")
    model = YOLODetector(num_classes=NUM_CLASSES, max_objects=MAX_OBJECTS)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train model
    history = train_model(model, train_loader, NUM_EPOCHS, device)

    # Plot training history
    plot_training_history(history)

    # Visualize results
    print("\nGenerating detection visualizations...")
    visualize_detections(model, train_dataset, device, num_samples=4)

    elapsed_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Total execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"{'='*60}")
    print(f"\nOutputs saved to: {OUTPUT_DIR.absolute()}")
    print("\nLab 6 completed successfully!")


if __name__ == "__main__":
    main()
