# Lab 4: Image Labeling Tools Familiarization

## 📋 Overview

This lab provides a comprehensive guide to popular image annotation tools used for creating labeled datasets for object detection, segmentation, and classification tasks. You'll learn about different tools, their features, and how to use them effectively.

## 🎯 Learning Objectives

By completing this lab, you will:
1. Understand the importance of data annotation
2. Learn about popular labeling tools (LabelImg, CVAT, VGG Image Annotator)
3. Create annotations for object detection
4. Create annotations for segmentation
5. Convert between different annotation formats (COCO, YOLO, Pascal VOC)
6. Understand best practices for data labeling

## 🔧 Popular Labeling Tools

### 1. LabelImg
**Best for**: Object detection (bounding boxes)

**Features**:
- Simple and lightweight
- Supports YOLO and Pascal VOC formats
- Keyboard shortcuts for fast labeling
- Cross-platform (Windows, Mac, Linux)

**Installation**:
```bash
pip install labelImg
```

**Usage**:
```bash
labelImg
# Or specify directory
labelImg [IMAGE_PATH] [PRE-DEFINED CLASS FILE]
```

**Keyboard Shortcuts**:
- `w`: Create bounding box
- `d`: Next image
- `a`: Previous image
- `Ctrl+s`: Save
- `Ctrl+d`: Delete box

### 2. CVAT (Computer Vision Annotation Tool)
**Best for**: Complex annotations, team collaboration

**Features**:
- Web-based interface
- Supports multiple annotation types
- Team collaboration
- Video annotation
- AI-assisted labeling
- Multiple export formats

**Installation**:
```bash
# Using Docker
docker run -it --rm -p 8080:8080 openvino/cvat
```

**Access**: http://localhost:8080

**Supported Formats**:
- COCO
- YOLO
- Pascal VOC
- TFRecord
- CVAT XML

### 3. VGG Image Annotator (VIA)
**Best for**: Quick annotations, no installation needed

**Features**:
- Browser-based (no installation)
- Offline capable
- Multiple annotation types
- Lightweight
- Export to JSON/CSV

**Access**: https://www.robots.ox.ac.uk/~vgg/software/via/

**Annotation Types**:
- Bounding boxes
- Polygons
- Points
- Circles
- Polylines

### 4. Labelme
**Best for**: Segmentation tasks

**Features**:
- Polygon annotations
- Python-based
- JSON output
- Easy to use

**Installation**:
```bash
pip install labelme
```

**Usage**:
```bash
labelme
# Or specify directory
labelme [IMAGE_DIR]
```

### 5. Roboflow
**Best for**: End-to-end dataset management

**Features**:
- Cloud-based
- Auto-labeling with AI
- Data augmentation
- Format conversion
- Team collaboration
- Version control

**Access**: https://roboflow.com

## 📊 Annotation Formats

### 1. YOLO Format
```
# Format: <class_id> <x_center> <y_center> <width> <height>
# All values normalized to [0, 1]
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.15 0.2
```

**File Structure**:
```
dataset/
├── images/
│   ├── img1.jpg
│   └── img2.jpg
└── labels/
    ├── img1.txt
    └── img2.txt
```

### 2. Pascal VOC Format (XML)
```xml
<annotation>
  <filename>image.jpg</filename>
  <size>
    <width>640</width>
    <height>480</height>
  </size>
  <object>
    <name>cat</name>
    <bndbox>
      <xmin>100</xmin>
      <ymin>150</ymin>
      <xmax>300</xmax>
      <ymax>400</ymax>
    </bndbox>
  </object>
</annotation>
```

### 3. COCO Format (JSON)
```json
{
  "images": [
    {
      "id": 1,
      "file_name": "image.jpg",
      "width": 640,
      "height": 480
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 150, 200, 250],
      "area": 50000,
      "segmentation": [[x1,y1,x2,y2,...]]
    }
  ],
  "categories": [
    {"id": 1, "name": "cat"}
  ]
}
```

## 🛠️ Format Conversion Script

Create a Python script to convert between formats:

```python
# convert_annotations.py
import json
import xml.etree.ElementTree as ET
from pathlib import Path

def yolo_to_voc(yolo_file, image_width, image_height, class_names):
    """Convert YOLO format to Pascal VOC XML"""
    root = ET.Element('annotation')
    
    with open(yolo_file, 'r') as f:
        for line in f:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            
            # Convert normalized coordinates to absolute
            x_center *= image_width
            y_center *= image_height
            width *= image_width
            height *= image_height
            
            xmin = int(x_center - width/2)
            ymin = int(y_center - height/2)
            xmax = int(x_center + width/2)
            ymax = int(y_center + height/2)
            
            obj = ET.SubElement(root, 'object')
            ET.SubElement(obj, 'name').text = class_names[int(class_id)]
            bndbox = ET.SubElement(obj, 'bndbox')
            ET.SubElement(bndbox, 'xmin').text = str(xmin)
            ET.SubElement(bndbox, 'ymin').text = str(ymin)
            ET.SubElement(bndbox, 'xmax').text = str(xmax)
            ET.SubElement(bndbox, 'ymax').text = str(ymax)
    
    return ET.tostring(root, encoding='unicode')

def voc_to_coco(voc_dir, output_file):
    """Convert Pascal VOC to COCO format"""
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Implementation here
    with open(output_file, 'w') as f:
        json.dump(coco_format, f, indent=2)
```

## 📝 Labeling Best Practices

### 1. Consistency
- Use consistent naming conventions
- Follow the same labeling guidelines
- Maintain uniform quality across dataset

### 2. Quality Over Quantity
- Accurate labels are more important than many labels
- Review and validate annotations
- Use multiple annotators for critical data

### 3. Class Balance
- Ensure balanced representation of classes
- Avoid extreme class imbalance
- Consider data augmentation for rare classes

### 4. Edge Cases
- Label difficult examples
- Include occluded objects
- Handle partial visibility
- Document ambiguous cases

### 5. Annotation Guidelines
Create a document specifying:
- What to label
- How to handle edge cases
- Bounding box rules (tight vs. loose)
- Occlusion handling
- Class definitions

## 🎓 Practical Exercise

### Exercise 1: Object Detection Labeling

1. **Download sample images**:
```bash
# Create sample dataset
mkdir -p sample_dataset/images
# Add your images to this folder
```

2. **Label with LabelImg**:
```bash
labelImg sample_dataset/images
```

3. **Create classes file** (`classes.txt`):
```
person
car
dog
cat
```

4. **Label guidelines**:
- Draw tight bounding boxes
- Include partially visible objects
- Label all instances
- Use consistent class names

### Exercise 2: Segmentation Labeling

1. **Use Labelme**:
```bash
labelme sample_dataset/images
```

2. **Create polygon annotations**:
- Click to create polygon vertices
- Right-click to complete polygon
- Label each polygon with class name

3. **Export annotations**:
```bash
labelme_json_to_dataset annotation.json
```

### Exercise 3: Format Conversion

Convert YOLO annotations to COCO format:

```python
import json
from pathlib import Path

def convert_yolo_to_coco(yolo_dir, output_file, image_dir):
    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 0, "name": "person"},
            {"id": 1, "name": "car"}
        ]
    }
    
    annotation_id = 0
    for img_id, label_file in enumerate(Path(yolo_dir).glob("*.txt")):
        # Read image dimensions
        img_file = Path(image_dir) / f"{label_file.stem}.jpg"
        # Add to coco format
        
    with open(output_file, 'w') as f:
        json.dump(coco, f, indent=2)
```

## 🔍 Quality Assurance

### Validation Checklist:
- [ ] All images are labeled
- [ ] No duplicate annotations
- [ ] Bounding boxes are accurate
- [ ] Class labels are correct
- [ ] Annotations follow guidelines
- [ ] Edge cases are handled
- [ ] Format is correct

### Automated Checks:
```python
def validate_annotations(annotation_dir):
    """Validate annotation quality"""
    issues = []
    
    for ann_file in Path(annotation_dir).glob("*.txt"):
        with open(ann_file) as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split()
                
                # Check format
                if len(parts) != 5:
                    issues.append(f"{ann_file}:{line_num} - Invalid format")
                
                # Check values in range [0, 1]
                values = [float(x) for x in parts[1:]]
                if not all(0 <= v <= 1 for v in values):
                    issues.append(f"{ann_file}:{line_num} - Values out of range")
    
    return issues
```

## 📚 Additional Resources

- [LabelImg GitHub](https://github.com/tzutalin/labelImg)
- [CVAT Documentation](https://opencv.github.io/cvat/)
- [VIA Documentation](https://www.robots.ox.ac.uk/~vgg/software/via/)
- [Labelme GitHub](https://github.com/wkentaro/labelme)
- [Roboflow Tutorials](https://roboflow.com/tutorials)
- [COCO Format Specification](https://cocodataset.org/#format-data)

## 🎯 Summary

### Tool Selection Guide:

| Use Case | Recommended Tool | Why |
|----------|-----------------|-----|
| Quick bounding boxes | LabelImg | Simple, fast, offline |
| Team project | CVAT | Collaboration features |
| Segmentation | Labelme | Polygon support |
| No installation | VIA | Browser-based |
| Production pipeline | Roboflow | End-to-end solution |

### Key Takeaways:
1. Choose the right tool for your task
2. Maintain consistency in labeling
3. Document your annotation guidelines
4. Validate annotations regularly
5. Convert formats as needed
6. Consider automation for large datasets

## 🎯 Next Steps

After completing this lab:
1. Practice with different labeling tools
2. Create your own annotated dataset
3. Implement format conversion scripts
4. Move on to Lab 5: Image Segmentation

---

**Note**: This lab is primarily educational. For production use, consider using professional annotation services or tools with quality control features.