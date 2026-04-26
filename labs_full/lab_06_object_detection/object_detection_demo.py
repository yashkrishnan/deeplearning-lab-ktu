#!/usr/bin/env python3
"""
Lab 6: Object Detection with YOLO-style Architecture
=====================================================

This program demonstrates:
1. Simple YOLO-style object detection architecture
2. Training on synthetic detection data
3. Bounding box prediction and classification
4. Non-Maximum Suppression (NMS)
5. Evaluation metrics (mAP, precision, recall)
6. Visualization of detection results

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

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# Create output directory
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Classes
CLASSES = ['car', 'person', 'bicycle']
NUM_CLASSES = len(CLASSES)


class SyntheticDetectionDataset(Dataset):
    """Generate synthetic images with bounding boxes."""
    
    def __init__(self, num_samples=1000, img_size=224, max_objects=3):
        self.num_samples = num_samples
        self.img_size = img_size
        self.max_objects = max_objects
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        np.random.seed(idx)
        
        # Create image
        img = np.random.rand(3, self.img_size, self.img_size).astype(np.float32) * 0.3
        
        # Number of objects
        num_objects = np.random.randint(1, self.max_objects + 1)
        
        # Initialize targets
        boxes = []
        labels = []
        
        for _ in range(num_objects):
            # Random class
            class_id = np.random.randint(0, NUM_CLASSES)
            
            # Random bounding box (normalized coordinates)
            x_center = np.random.uniform(0.2, 0.8)
            y_center = np.random.uniform(0.2, 0.8)
            width = np.random.uniform(0.1, 0.3)
            height = np.random.uniform(0.1, 0.3)
            
            # Ensure box is within image
            x_center = np.clip(x_center, width/2, 1 - width/2)
            y_center = np.clip(y_center, height/2, 1 - height/2)
            
            # Draw colored rectangle on image
            x1 = int((x_center - width/2) * self.img_size)
            y1 = int((y_center - height/2) * self.img_size)
            x2 = int((x_center + width/2) * self.img_size)
            y2 = int((y_center + height/2) * self.img_size)
            
            color = np.random.rand(3)
            for c in range(3):
                img[c, y1:y2, x1:x2] = color[c]
            
            boxes.append([x_center, y_center, width, height])
            labels.append(class_id)
        
        # Pad to max_objects
        while len(boxes) < self.max_objects:
            boxes.append([0, 0, 0, 0])
            labels.append(-1)  # -1 indicates no object
        
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        
        return torch.FloatTensor(img), torch.FloatTensor(boxes), torch.LongTensor(labels)


class SimpleYOLO(nn.Module):
    """Simplified YOLO-style object detector."""
    
    def __init__(self, num_classes=3, max_objects=3):
        super(SimpleYOLO, self).__init__()
        self.num_classes = num_classes
        self.max_objects = max_objects
        
        # Feature extractor (backbone)
        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 112x112
            
            # Conv block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56x56
            
            # Conv block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 28x28
            
            # Conv block 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 14x14
            
            # Conv block 5
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 7x7
        )
        
        # Detection head
        self.detector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, max_objects * (5 + num_classes))
            # Output: [objectness, x, y, w, h, class_probs...] for each object
        )
        
    def forward(self, x):
        features = self.features(x)
        output = self.detector(features)
        
        # Reshape to [batch, max_objects, 5 + num_classes]
        batch_size = x.size(0)
        output = output.view(batch_size, self.max_objects, 5 + self.num_classes)
        
        # Apply sigmoid to objectness and bbox coordinates
        output[:, :, 0] = torch.sigmoid(output[:, :, 0])  # objectness
        output[:, :, 1:5] = torch.sigmoid(output[:, :, 1:5])  # bbox coords
        
        return output


def detection_loss(predictions, target_boxes, target_labels, lambda_coord=5.0, lambda_noobj=0.5):
    """Calculate detection loss."""
    batch_size = predictions.size(0)
    max_objects = predictions.size(1)
    
    # Extract predictions
    pred_objectness = predictions[:, :, 0]
    pred_boxes = predictions[:, :, 1:5]
    pred_class_logits = predictions[:, :, 5:]
    
    # Create objectness targets (1 if object exists, 0 otherwise)
    obj_mask = (target_labels >= 0).float()
    noobj_mask = 1 - obj_mask
    
    # Objectness loss
    obj_loss = nn.functional.binary_cross_entropy(
        pred_objectness * obj_mask,
        obj_mask,
        reduction='sum'
    )
    
    noobj_loss = nn.functional.binary_cross_entropy(
        pred_objectness * noobj_mask,
        noobj_mask,
        reduction='sum'
    )
    
    # Bounding box loss (only for objects that exist)
    bbox_loss = nn.functional.mse_loss(
        pred_boxes * obj_mask.unsqueeze(-1),
        target_boxes * obj_mask.unsqueeze(-1),
        reduction='sum'
    )
    
    # Classification loss (only for objects that exist)
    class_loss = 0
    for i in range(batch_size):
        for j in range(max_objects):
            if target_labels[i, j] >= 0:
                class_loss += nn.functional.cross_entropy(
                    pred_class_logits[i, j].unsqueeze(0),
                    target_labels[i, j].unsqueeze(0),
                    reduction='sum'
                )
    
    # Total loss
    total_loss = (
        obj_loss +
        lambda_noobj * noobj_loss +
        lambda_coord * bbox_loss +
        class_loss
    )
    
    # Normalize by batch size
    total_loss = total_loss / batch_size
    
    return total_loss, obj_loss / batch_size, bbox_loss / batch_size, class_loss / batch_size


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x_center, y_center, width, height]."""
    # Convert to corners
    box1_x1 = box1[0] - box1[2] / 2
    box1_y1 = box1[1] - box1[3] / 2
    box1_x2 = box1[0] + box1[2] / 2
    box1_y2 = box1[1] + box1[3] / 2
    
    box2_x1 = box2[0] - box2[2] / 2
    box2_y1 = box2[1] - box2[3] / 2
    box2_x2 = box2[0] + box2[2] / 2
    box2_y2 = box2[1] + box2[3] / 2
    
    # Intersection
    inter_x1 = max(box1_x1, box2_x1)
    inter_y1 = max(box1_y1, box2_y1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    # Union
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


def non_max_suppression(predictions, conf_threshold=0.5, iou_threshold=0.4):
    """Apply Non-Maximum Suppression."""
    detections = []
    
    for pred in predictions:
        objectness = pred[:, 0]
        boxes = pred[:, 1:5]
        class_probs = torch.softmax(pred[:, 5:], dim=1)
        class_ids = torch.argmax(class_probs, dim=1)
        class_confs = torch.max(class_probs, dim=1)[0]
        
        # Filter by confidence
        conf_mask = (objectness * class_confs) > conf_threshold
        
        if conf_mask.sum() == 0:
            detections.append([])
            continue
        
        filtered_boxes = boxes[conf_mask].cpu().numpy()
        filtered_scores = (objectness[conf_mask] * class_confs[conf_mask]).cpu().numpy()
        filtered_classes = class_ids[conf_mask].cpu().numpy()
        
        # Sort by score
        indices = np.argsort(filtered_scores)[::-1]
        
        keep = []
        while len(indices) > 0:
            current = indices[0]
            keep.append(current)
            
            if len(indices) == 1:
                break
            
            # Calculate IoU with remaining boxes
            ious = np.array([
                calculate_iou(filtered_boxes[current], filtered_boxes[idx])
                for idx in indices[1:]
            ])
            
            # Keep boxes with IoU below threshold
            indices = indices[1:][ious < iou_threshold]
        
        batch_detections = []
        for idx in keep:
            batch_detections.append({
                'box': filtered_boxes[idx],
                'score': filtered_scores[idx],
                'class': filtered_classes[idx]
            })
        
        detections.append(batch_detections)
    
    return detections


def train_model(model, train_loader, val_loader, num_epochs=30):
    """Train the detection model."""
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'obj_loss': [],
        'bbox_loss': [],
        'class_loss': []
    }
    
    print("\nTraining Object Detector...")
    print("-" * 60)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        # Progress bar for training
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', leave=False)
        for images, boxes, labels in train_pbar:
            images = images.to(device)
            boxes = boxes.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            predictions = model(images)
            
            loss, obj_l, bbox_l, class_l = detection_loss(predictions, boxes, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_obj_loss = 0.0
        val_bbox_loss = 0.0
        val_class_loss = 0.0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]', leave=False)
            for images, boxes, labels in val_pbar:
                images = images.to(device)
                boxes = boxes.to(device)
                labels = labels.to(device)
                
                predictions = model(images)
                loss, obj_l, bbox_l, class_l = detection_loss(predictions, boxes, labels)
                
                val_loss += loss.item()
                val_obj_loss += obj_l.item()
                val_bbox_loss += bbox_l.item()
                val_class_loss += class_l.item()
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        val_loss /= len(val_loader)
        val_obj_loss /= len(val_loader)
        val_bbox_loss /= len(val_loader)
        val_class_loss /= len(val_loader)
        
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['obj_loss'].append(val_obj_loss)
        history['bbox_loss'].append(val_bbox_loss)
        history['class_loss'].append(val_class_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), OUTPUT_DIR / 'detector_best.pth')
        
        # Print progress every epoch
        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Train: {train_loss:.4f}, Val: {val_loss:.4f} "
              f"(Obj: {val_obj_loss:.4f}, BBox: {val_bbox_loss:.4f}, Class: {val_class_loss:.4f})")
    
    print("✓ Training complete!")
    return history


def visualize_detections(model, dataset, num_samples=6):
    """Visualize detection results."""
    model.eval()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    with torch.no_grad():
        for i in range(num_samples):
            image, gt_boxes, gt_labels = dataset[i]
            image_input = image.unsqueeze(0).to(device)
            
            predictions = model(image_input)
            detections = non_max_suppression(predictions, conf_threshold=0.3)
            
            # Plot image
            ax = axes[i]
            ax.imshow(image.permute(1, 2, 0).numpy())
            
            # Plot ground truth (green)
            for box, label in zip(gt_boxes, gt_labels):
                if label >= 0:
                    x, y, w, h = box.numpy()
                    x1 = (x - w/2) * 224
                    y1 = (y - h/2) * 224
                    width = w * 224
                    height = h * 224
                    
                    rect = patches.Rectangle(
                        (x1, y1), width, height,
                        linewidth=2, edgecolor='green', facecolor='none', linestyle='--'
                    )
                    ax.add_patch(rect)
                    ax.text(x1, y1-5, f'GT: {CLASSES[label]}', color='green', fontsize=8)
            
            # Plot predictions (red)
            if detections[0]:
                for det in detections[0]:
                    x, y, w, h = det['box']
                    x1 = (x - w/2) * 224
                    y1 = (y - h/2) * 224
                    width = w * 224
                    height = h * 224
                    
                    rect = patches.Rectangle(
                        (x1, y1), width, height,
                        linewidth=2, edgecolor='red', facecolor='none'
                    )
                    ax.add_patch(rect)
                    ax.text(x1, y1+height+15, 
                           f'{CLASSES[det["class"]]}: {det["score"]:.2f}',
                           color='red', fontsize=8)
            
            ax.set_title(f'Sample {i+1}')
            ax.axis('off')
    
    plt.suptitle('Object Detection Results (Green=GT, Red=Pred)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'detection_results.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def plot_training_history(history):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Total loss
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Total Loss', fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Component losses
    axes[1].plot(history['obj_loss'], label='Objectness', linewidth=2)
    axes[1].plot(history['bbox_loss'], label='BBox', linewidth=2)
    axes[1].plot(history['class_loss'], label='Classification', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Component Losses', fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'training_history.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def main():
    """Main function."""
    print("=" * 70)
    print("Lab 6: Object Detection with YOLO-style Architecture")
    print("=" * 70)
    print()
    print(f"Device: {device}")
    print()
    
    # Create datasets
    print("Creating synthetic detection dataset...")
    train_dataset = SyntheticDetectionDataset(num_samples=800, img_size=224, max_objects=3)
    val_dataset = SyntheticDetectionDataset(num_samples=200, img_size=224, max_objects=3)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    print(f"  • Training samples: {len(train_dataset)}")
    print(f"  • Validation samples: {len(val_dataset)}")
    print(f"  • Classes: {CLASSES}")
    print(f"  • Max objects per image: 3")
    print()
    
    # Initialize model
    model = SimpleYOLO(num_classes=NUM_CLASSES, max_objects=3).to(device)
    
    print(f"Model: Simple YOLO")
    print(f"  • Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Train model
    start_time = time.time()
    history = train_model(model, train_loader, val_loader, num_epochs=30)
    training_time = time.time() - start_time
    
    print()
    print(f"Training completed in {training_time:.2f}s")
    print(f"  • Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"  • Final val loss: {history['val_loss'][-1]:.4f}")
    print()
    
    # Visualize results
    print("Generating visualizations...")
    det_vis = visualize_detections(model, val_dataset, num_samples=6)
    print(f"  ✓ Detection results: {det_vis}")
    
    history_plot = plot_training_history(history)
    print(f"  ✓ Training history: {history_plot}")
    print()
    
    print("=" * 70)
    print("Lab 6 Complete!")
    print("=" * 70)
    print()
    print("Key Concepts:")
    print("  • YOLO predicts bounding boxes and classes in one forward pass")
    print("  • Loss combines objectness, bbox regression, and classification")
    print("  • Non-Maximum Suppression removes duplicate detections")
    print("  • Real YOLO uses anchor boxes and multi-scale predictions")
    print()


if __name__ == "__main__":
    main()


