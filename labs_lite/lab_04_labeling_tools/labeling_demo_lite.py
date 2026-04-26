#!/usr/bin/env python3
"""
Lab 4: Image Labeling Tools Demonstration (LIGHTWEIGHT VERSION - REAL DATASET)
===============================================================================

This is a lightweight version using REAL road sign images with existing annotations.
Changes from original:
- Uses 30 real road sign images (from practice_images dataset)
- Reads existing Pascal VOC XML annotations
- Converts to YOLO and COCO formats
- Reduced DPI: 100 (from 150) - faster saving
- Expected time: ~3-5 seconds

This program demonstrates:
1. Loading real images with existing annotations
2. Reading Pascal VOC XML format
3. Converting between annotation formats (YOLO, COCO, Pascal VOC)
4. Visualizing annotations
5. Generating annotation statistics

Author: Deep Learning Lab
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import random
from pathlib import Path
import xml.etree.ElementTree as ET
import glob

# Create output directories
OUTPUT_DIR = Path("output")
IMAGES_DIR = OUTPUT_DIR / "images"
ANNOTATIONS_DIR = OUTPUT_DIR / "annotations"

for dir_path in [OUTPUT_DIR, IMAGES_DIR, ANNOTATIONS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Path to dataset (in labs_lite directory)
DATASET_DIR = Path("data/practice_images")
DATASET_IMAGES = DATASET_DIR / "images"
DATASET_ANNOTATIONS = DATASET_DIR / "annotations"


def parse_pascal_voc_xml(xml_path):
    """Parse Pascal VOC XML annotation file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Get image info
    size = root.find('size')
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)
    filename = root.find('filename').text
    
    # Get all objects
    annotations = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        
        x1 = int(bndbox.find('xmin').text)
        y1 = int(bndbox.find('ymin').text)
        x2 = int(bndbox.find('xmax').text)
        y2 = int(bndbox.find('ymax').text)
        
        annotations.append({
            'class': name,
            'bbox': [x1, y1, x2, y2],
            'width': x2 - x1,
            'height': y2 - y1
        })
    
    return filename, img_width, img_height, annotations


def get_all_classes(annotation_files):
    """Extract all unique classes from annotation files."""
    classes = set()
    for xml_file in annotation_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for obj in root.findall('object'):
            classes.add(obj.find('name').text)
    return sorted(list(classes))


def convert_to_yolo_format(annotations, img_width, img_height, classes):
    """Convert annotations to YOLO format (class x_center y_center width height)."""
    yolo_annotations = []
    
    for ann in annotations:
        if ann['class'] not in classes:
            continue
        class_id = classes.index(ann['class'])
        x1, y1, x2, y2 = ann['bbox']
        
        # Convert to YOLO format (normalized)
        x_center = ((x1 + x2) / 2) / img_width
        y_center = ((y1 + y2) / 2) / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        
        yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    return yolo_annotations


def convert_to_coco_format(annotations, image_id, img_width, img_height, classes):
    """Convert annotations to COCO format."""
    coco_annotations = []
    
    for idx, ann in enumerate(annotations):
        if ann['class'] not in classes:
            continue
        x1, y1, x2, y2 = ann['bbox']
        width = x2 - x1
        height = y2 - y1
        
        coco_ann = {
            'id': idx,
            'image_id': image_id,
            'category_id': classes.index(ann['class']) + 1,
            'bbox': [x1, y1, width, height],
            'area': width * height,
            'iscrowd': 0
        }
        coco_annotations.append(coco_ann)
    
    return coco_annotations


def visualize_annotations(image_path, annotations, output_path, classes):
    """Visualize bounding box annotations on image."""
    img = Image.open(image_path)
    fig, ax = plt.subplots(1, figsize=(10, 7))
    ax.imshow(img)
    
    # Generate colors for each class
    np.random.seed(42)
    colors = {cls: tuple(np.random.rand(3)) for cls in classes}
    
    for ann in annotations:
        if ann['class'] not in classes:
            continue
        x1, y1, x2, y2 = ann['bbox']
        width = x2 - x1
        height = y2 - y1
        
        # Create rectangle patch
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2,
            edgecolor=colors[ann['class']],
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label
        ax.text(
            x1, y1 - 5,
            ann['class'],
            color='white',
            fontsize=10,
            bbox=dict(facecolor=colors[ann['class']], alpha=0.8)
        )
    
    ax.axis('off')
    plt.title('Annotated Image', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def generate_annotation_statistics(all_annotations, classes):
    """Generate statistics about annotations."""
    class_counts = {cls: 0 for cls in classes}
    total_objects = 0
    bbox_sizes = []
    
    for annotations in all_annotations:
        for ann in annotations:
            if ann['class'] in classes:
                class_counts[ann['class']] += 1
                total_objects += 1
                bbox_sizes.append(ann['width'] * ann['height'])
    
    return {
        'total_objects': int(total_objects),
        'class_counts': {k: int(v) for k, v in class_counts.items()},
        'avg_bbox_size': float(np.mean(bbox_sizes)) if bbox_sizes else 0.0,
        'min_bbox_size': float(np.min(bbox_sizes)) if bbox_sizes else 0.0,
        'max_bbox_size': float(np.max(bbox_sizes)) if bbox_sizes else 0.0
    }


def plot_class_distribution(stats, output_path, classes):
    """Plot class distribution."""
    class_list = list(stats['class_counts'].keys())
    counts = list(stats['class_counts'].values())
    
    # Generate colors
    np.random.seed(42)
    colors = [tuple(np.random.rand(3)) for _ in class_list]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(class_list, counts, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Object Class', fontsize=14, fontweight='bold')
    ax.set_ylabel('Count', fontsize=14, fontweight='bold')
    ax.set_title('Object Class Distribution', fontsize=16, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def main():
    """Main function to demonstrate labeling workflows."""
    print("=" * 70)
    print("Lab 4: Image Labeling Tools (LIGHTWEIGHT - REAL DATASET)")
    print("=" * 70)
    print()
    
    # Check if dataset exists
    if not DATASET_DIR.exists():
        print(f"ERROR: Dataset not found at {DATASET_DIR}")
        print("Please ensure the road sign dataset is available.")
        return
    
    # Get all annotation files
    all_xml_files = sorted(glob.glob(str(DATASET_ANNOTATIONS / "*.xml")))
    
    if len(all_xml_files) == 0:
        print(f"ERROR: No annotation files found in {DATASET_ANNOTATIONS}")
        return
    
    # Select 30 random annotation files for lightweight version
    num_images = min(30, len(all_xml_files))
    selected_xml_files = random.sample(all_xml_files, num_images)
    
    print(f"Using {num_images} real road sign images with annotations...")
    print(f"Dataset location: {DATASET_DIR}")
    print()
    
    # Get all classes from selected files
    classes = get_all_classes(selected_xml_files)
    print(f"Found {len(classes)} object classes: {', '.join(classes)}")
    print()
    
    all_annotations = []
    coco_dataset = {
        'images': [],
        'annotations': [],
        'categories': [{'id': i+1, 'name': cls} for i, cls in enumerate(classes)]
    }
    
    annotation_id = 0
    
    for i, xml_path in enumerate(selected_xml_files):
        # Parse XML annotation
        filename, img_width, img_height, annotations = parse_pascal_voc_xml(xml_path)
        
        # Find corresponding image
        img_path = DATASET_IMAGES / filename
        if not img_path.exists():
            print(f"  ⚠ Image not found: {filename}, skipping...")
            continue
        
        # Copy image to output
        img = Image.open(img_path)
        output_img_path = IMAGES_DIR / f"image_{i:03d}.png"
        img.save(output_img_path)
        
        all_annotations.append(annotations)
        
        # Add to COCO dataset
        coco_dataset['images'].append({
            'id': i,
            'file_name': f"image_{i:03d}.png",
            'width': img_width,
            'height': img_height,
            'original_filename': filename
        })
        
        # Convert to different formats
        # 1. YOLO format
        yolo_annotations = convert_to_yolo_format(annotations, img_width, img_height, classes)
        yolo_path = ANNOTATIONS_DIR / f"image_{i:03d}.txt"
        with open(yolo_path, 'w') as f:
            f.write('\n'.join(yolo_annotations))
        
        # 2. COCO format (accumulate)
        coco_anns = convert_to_coco_format(annotations, i, img_width, img_height, classes)
        for ann in coco_anns:
            ann['id'] = annotation_id
            annotation_id += 1
            coco_dataset['annotations'].append(ann)
        
        # 3. Pascal VOC format (copy original)
        import shutil
        voc_path = ANNOTATIONS_DIR / f"image_{i:03d}.xml"
        shutil.copy(xml_path, voc_path)
        
        # Visualize first 3 images
        if i < 3:
            vis_path = OUTPUT_DIR / f"annotated_image_{i:03d}.png"
            visualize_annotations(output_img_path, annotations, vis_path, classes)
        
        print(f"  ✓ Image {i+1}/{num_images}: {filename} - {len(annotations)} objects")
    
    # Save COCO format JSON
    coco_path = ANNOTATIONS_DIR / "annotations_coco.json"
    with open(coco_path, 'w') as f:
        json.dump(coco_dataset, f, indent=2)
    
    print()
    print("Annotation format conversion complete!")
    print(f"  • YOLO format: {ANNOTATIONS_DIR}/*.txt")
    print(f"  • COCO format: {coco_path}")
    print(f"  • Pascal VOC format: {ANNOTATIONS_DIR}/*.xml")
    print()
    
    # Generate statistics
    print("Generating annotation statistics...")
    stats = generate_annotation_statistics(all_annotations, classes)
    
    print()
    print("Annotation Statistics:")
    print("-" * 50)
    print(f"Total images: {num_images}")
    print(f"Total objects: {stats['total_objects']}")
    print(f"Average objects per image: {stats['total_objects'] / num_images:.2f}")
    print()
    print("Class distribution:")
    for cls, count in stats['class_counts'].items():
        if stats['total_objects'] > 0:
            percentage = (count / stats['total_objects']) * 100
            print(f"  • {cls:15s}: {count:3d} ({percentage:5.1f}%)")
    print()
    print(f"Bounding box statistics:")
    print(f"  • Average size: {stats['avg_bbox_size']:.0f} pixels²")
    print(f"  • Min size: {stats['min_bbox_size']:.0f} pixels²")
    print(f"  • Max size: {stats['max_bbox_size']:.0f} pixels²")
    print()
    
    # Plot class distribution
    dist_path = OUTPUT_DIR / "class_distribution.png"
    plot_class_distribution(stats, dist_path, classes)
    print(f"✓ Class distribution plot saved: {dist_path}")
    print()
    
    # Save statistics to JSON
    stats_path = OUTPUT_DIR / "annotation_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Statistics saved: {stats_path}")
    print()
    
    # Create classes.txt for YOLO
    classes_path = ANNOTATIONS_DIR / "classes.txt"
    with open(classes_path, 'w') as f:
        f.write('\n'.join(classes))
    print(f"✓ Class names saved: {classes_path}")
    print()
    
    print("=" * 70)
    print("Demonstration Complete!")
    print("=" * 70)
    print()
    print("Optimizations applied:")
    print("  • Images: 30 real road sign images")
    print("  • Uses existing Pascal VOC annotations")
    print("  • DPI: 100 (vs 150)")
    print("  • Visualizes first 3 annotated images")
    print()
    print("Output files:")
    print(f"  • Images: {IMAGES_DIR}/")
    print(f"  • Annotations: {ANNOTATIONS_DIR}/")
    print(f"  • Visualizations: {OUTPUT_DIR}/annotated_image_*.png")
    print(f"  • Class distribution: {dist_path}")
    print()
    print("Format Examples:")
    print()
    print("1. YOLO Format (normalized coordinates):")
    print("   class_id x_center y_center width height")
    if (ANNOTATIONS_DIR / 'image_000.txt').exists():
        print(f"   Example: {open(ANNOTATIONS_DIR / 'image_000.txt').readline().strip()}")
    print()
    print("2. COCO Format (JSON with absolute coordinates):")
    print("   See: annotations/annotations_coco.json")
    print()
    print("3. Pascal VOC Format (XML with absolute coordinates):")
    print("   See: annotations/image_000.xml")
    print()


if __name__ == "__main__":
    main()


