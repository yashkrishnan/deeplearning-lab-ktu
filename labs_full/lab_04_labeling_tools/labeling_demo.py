#!/usr/bin/env python3
"""
Lab 4: Image Labeling Tools Demonstration
==========================================

This program demonstrates:
1. Creating synthetic images for annotation
2. Simulating bounding box annotations
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
from PIL import Image, ImageDraw, ImageFont
import random
from pathlib import Path

# Create output directories
OUTPUT_DIR = Path("output")
IMAGES_DIR = OUTPUT_DIR / "images"
ANNOTATIONS_DIR = OUTPUT_DIR / "annotations"

for dir_path in [OUTPUT_DIR, IMAGES_DIR, ANNOTATIONS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Define classes for synthetic dataset
CLASSES = ['car', 'person', 'bicycle', 'dog', 'cat']
CLASS_COLORS = {
    'car': (255, 0, 0),
    'person': (0, 255, 0),
    'bicycle': (0, 0, 255),
    'dog': (255, 255, 0),
    'cat': (255, 0, 255)
}


def create_synthetic_image(width=640, height=480, num_objects=5):
    """Create a synthetic image with colored rectangles representing objects."""
    img = Image.new('RGB', (width, height), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    annotations = []
    
    for _ in range(num_objects):
        # Random object class
        obj_class = random.choice(CLASSES)
        color = CLASS_COLORS[obj_class]
        
        # Random bounding box
        x1 = random.randint(0, width - 100)
        y1 = random.randint(0, height - 100)
        w = random.randint(50, min(150, width - x1))
        h = random.randint(50, min(150, height - y1))
        x2 = x1 + w
        y2 = y1 + h
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], fill=color, outline=(0, 0, 0), width=2)
        
        # Add label text
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        except:
            font = ImageFont.load_default()
        
        draw.text((x1 + 5, y1 + 5), obj_class, fill=(255, 255, 255), font=font)
        
        annotations.append({
            'class': obj_class,
            'bbox': [x1, y1, x2, y2],
            'width': w,
            'height': h
        })
    
    return img, annotations


def convert_to_yolo_format(annotations, img_width, img_height):
    """Convert annotations to YOLO format (class x_center y_center width height)."""
    yolo_annotations = []
    
    for ann in annotations:
        class_id = CLASSES.index(ann['class'])
        x1, y1, x2, y2 = ann['bbox']
        
        # Convert to YOLO format (normalized)
        x_center = ((x1 + x2) / 2) / img_width
        y_center = ((y1 + y2) / 2) / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        
        yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    return yolo_annotations


def convert_to_coco_format(annotations, image_id, img_width, img_height):
    """Convert annotations to COCO format."""
    coco_annotations = []
    
    for idx, ann in enumerate(annotations):
        x1, y1, x2, y2 = ann['bbox']
        width = x2 - x1
        height = y2 - y1
        
        coco_ann = {
            'id': idx,
            'image_id': image_id,
            'category_id': CLASSES.index(ann['class']) + 1,
            'bbox': [x1, y1, width, height],
            'area': width * height,
            'iscrowd': 0
        }
        coco_annotations.append(coco_ann)
    
    return coco_annotations


def convert_to_pascal_voc_format(annotations, filename, img_width, img_height):
    """Convert annotations to Pascal VOC XML format."""
    xml_content = f"""<annotation>
    <folder>images</folder>
    <filename>{filename}</filename>
    <size>
        <width>{img_width}</width>
        <height>{img_height}</height>
        <depth>3</depth>
    </size>
"""
    
    for ann in annotations:
        x1, y1, x2, y2 = ann['bbox']
        xml_content += f"""    <object>
        <name>{ann['class']}</name>
        <bndbox>
            <xmin>{x1}</xmin>
            <ymin>{y1}</ymin>
            <xmax>{x2}</xmax>
            <ymax>{y2}</ymax>
        </bndbox>
    </object>
"""
    
    xml_content += "</annotation>"
    return xml_content


def visualize_annotations(image_path, annotations, output_path):
    """Visualize bounding box annotations on image."""
    img = Image.open(image_path)
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)
    
    for ann in annotations:
        x1, y1, x2, y2 = ann['bbox']
        width = x2 - x1
        height = y2 - y1
        
        # Create rectangle patch
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2,
            edgecolor=tuple(c/255 for c in CLASS_COLORS[ann['class']]),
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label
        ax.text(
            x1, y1 - 5,
            ann['class'],
            color='white',
            fontsize=12,
            bbox=dict(facecolor=tuple(c/255 for c in CLASS_COLORS[ann['class']]), alpha=0.8)
        )
    
    ax.axis('off')
    plt.title('Annotated Image', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_annotation_statistics(all_annotations):
    """Generate statistics about annotations."""
    class_counts = {cls: 0 for cls in CLASSES}
    total_objects = 0
    bbox_sizes = []
    
    for annotations in all_annotations:
        for ann in annotations:
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


def plot_class_distribution(stats, output_path):
    """Plot class distribution."""
    classes = list(stats['class_counts'].keys())
    counts = list(stats['class_counts'].values())
    colors = [tuple(c/255 for c in CLASS_COLORS[cls]) for cls in classes]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(classes, counts, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Object Class', fontsize=14, fontweight='bold')
    ax.set_ylabel('Count', fontsize=14, fontweight='bold')
    ax.set_title('Object Class Distribution', fontsize=16, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Main function to demonstrate labeling workflows."""
    print("=" * 70)
    print("Lab 4: Image Labeling Tools Demonstration")
    print("=" * 70)
    print()
    
    # Configuration
    num_images = 10
    img_width, img_height = 640, 480
    
    print(f"Creating {num_images} synthetic images with annotations...")
    print()
    
    all_annotations = []
    coco_dataset = {
        'images': [],
        'annotations': [],
        'categories': [{'id': i+1, 'name': cls} for i, cls in enumerate(CLASSES)]
    }
    
    annotation_id = 0
    
    for i in range(num_images):
        # Create synthetic image
        img, annotations = create_synthetic_image(img_width, img_height, num_objects=random.randint(3, 7))
        
        # Save image
        img_filename = f"image_{i:03d}.jpg"
        img_path = IMAGES_DIR / img_filename
        img.save(img_path)
        
        all_annotations.append(annotations)
        
        # Add to COCO dataset
        coco_dataset['images'].append({
            'id': i,
            'file_name': img_filename,
            'width': img_width,
            'height': img_height
        })
        
        # Convert to different formats
        # 1. YOLO format
        yolo_annotations = convert_to_yolo_format(annotations, img_width, img_height)
        yolo_path = ANNOTATIONS_DIR / f"image_{i:03d}.txt"
        with open(yolo_path, 'w') as f:
            f.write('\n'.join(yolo_annotations))
        
        # 2. COCO format (accumulate)
        coco_anns = convert_to_coco_format(annotations, i, img_width, img_height)
        for ann in coco_anns:
            ann['id'] = annotation_id
            annotation_id += 1
            coco_dataset['annotations'].append(ann)
        
        # 3. Pascal VOC format
        voc_xml = convert_to_pascal_voc_format(annotations, img_filename, img_width, img_height)
        voc_path = ANNOTATIONS_DIR / f"image_{i:03d}.xml"
        with open(voc_path, 'w') as f:
            f.write(voc_xml)
        
        # Visualize first 3 images
        if i < 3:
            vis_path = OUTPUT_DIR / f"annotated_image_{i:03d}.png"
            visualize_annotations(img_path, annotations, vis_path)
        
        print(f"  SUCCESS: Image {i+1}/{num_images}: {len(annotations)} objects")
    
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
    stats = generate_annotation_statistics(all_annotations)
    
    print()
    print("Annotation Statistics:")
    print("-" * 50)
    print(f"Total images: {num_images}")
    print(f"Total objects: {stats['total_objects']}")
    print(f"Average objects per image: {stats['total_objects'] / num_images:.2f}")
    print()
    print("Class distribution:")
    for cls, count in stats['class_counts'].items():
        percentage = (count / stats['total_objects']) * 100
        print(f"  • {cls:10s}: {count:3d} ({percentage:5.1f}%)")
    print()
    print(f"Bounding box statistics:")
    print(f"  • Average size: {stats['avg_bbox_size']:.0f} pixels²")
    print(f"  • Min size: {stats['min_bbox_size']:.0f} pixels²")
    print(f"  • Max size: {stats['max_bbox_size']:.0f} pixels²")
    print()
    
    # Plot class distribution
    dist_path = OUTPUT_DIR / "class_distribution.png"
    plot_class_distribution(stats, dist_path)
    print(f"SUCCESS: Class distribution plot saved: {dist_path}")
    print()
    
    # Save statistics to JSON
    stats_path = OUTPUT_DIR / "annotation_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"SUCCESS: Statistics saved: {stats_path}")
    print()
    
    # Create classes.txt for YOLO
    classes_path = ANNOTATIONS_DIR / "classes.txt"
    with open(classes_path, 'w') as f:
        f.write('\n'.join(CLASSES))
    print(f"SUCCESS: Class names saved: {classes_path}")
    print()
    
    print("=" * 70)
    print("Demonstration Complete!")
    print("=" * 70)
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


