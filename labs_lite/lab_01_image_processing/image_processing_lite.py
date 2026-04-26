"""
Lab 1: Basic Image Processing Operations (LIGHTWEIGHT VERSION)
===============================================================

This is a lightweight version optimized for faster execution on laptops.
Changes from original:
- Uses REAL Face Mask dataset images (not synthetic)
- Image size: 300x300 (from 400x400) - 44% fewer pixels
- Reduced DPI: 100 (from 150) - faster saving
- Smaller figure sizes for faster rendering
- Expected time: ~2-3 seconds (vs ~4-5 seconds)

This program demonstrates fundamental image processing techniques including:
1. Histogram Equalization - Enhance image contrast
2. Thresholding - Binary and adaptive thresholding
3. Edge Detection - Sobel and Canny edge detection
4. Data Augmentation - Rotation, flipping, scaling, etc.
5. Morphological Operations - Erosion, dilation, opening, closing

Dataset: Face Mask Detection Dataset (real images)

Learning Objectives:
- Understand basic image preprocessing techniques
- Learn how to enhance image quality
- Master edge detection algorithms
- Apply data augmentation for deep learning
- Use morphological operations for image analysis

Author: Deep Learning Lab
Date: March 2026
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import random

class ImageProcessor:
    """
    A comprehensive image processing toolkit for educational purposes.
    Demonstrates various fundamental image processing operations.
    """
    
    def __init__(self, output_dir='output'):
        """
        Initialize the image processor.
        
        Args:
            output_dir (str): Directory to save output images
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        print(f"✓ Output directory created: {self.output_dir}")
    
    def create_synthetic_image(self):
        """
        Create a synthetic grayscale image for demonstration.
        
        Returns:
            numpy.ndarray: Synthetic grayscale image (300x300)
        """
        print("  Creating synthetic image (dataset not available)...")
        
        # Create a 300x300 grayscale image
        img = np.zeros((300, 300), dtype=np.uint8)
        
        # Add some geometric shapes for interesting features
        # Circle
        cv2.circle(img, (150, 150), 80, 200, -1)
        
        # Rectangle
        cv2.rectangle(img, (50, 50), (120, 120), 150, -1)
        
        # Triangle (using polygon)
        pts = np.array([[250, 50], [200, 150], [280, 150]], np.int32)
        cv2.fillPoly(img, [pts], 180)
        
        # Add some noise for texture
        noise = np.random.normal(0, 15, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
        
        return img
    
    def load_real_image(self):
        """
        Load a real image from the Face Mask dataset.
        Falls back to synthetic image if dataset is not available.
        
        Returns:
            numpy.ndarray: Real or synthetic grayscale image from dataset
        """
        # Path to Face Mask dataset (in labs_lite directory)
        data_dir = Path("data/sample_images/Face Mask Dataset")
        
        # Try to load from WithoutMask folder first, then WithMask
        image_dirs = [
            data_dir / "Train" / "WithoutMask",
            data_dir / "Validation" / "WithMask"
        ]
        
        # Find available images
        available_images = []
        for img_dir in image_dirs:
            if img_dir.exists():
                available_images.extend(list(img_dir.glob("*.png")))
        
        if not available_images:
            print(f"  NOTE: Face Mask dataset not found at {data_dir}")
            print("  Falling back to synthetic image generation...")
            return self.create_synthetic_image()
        
        # Select a random image
        img_path = random.choice(available_images)
        print(f"  Loading image: {img_path.name}")
        
        # Load and convert to grayscale
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize to 300x300 for consistency
        img = cv2.resize(img, (300, 300))
        
        return img
    
    def histogram_equalization(self, img):
        """
        Apply histogram equalization to enhance image contrast.
        
        Histogram equalization redistributes pixel intensities to improve
        contrast, especially useful for images with poor lighting.
        
        Args:
            img (numpy.ndarray): Input grayscale image
            
        Returns:
            numpy.ndarray: Equalized image
        """
        print("\n1. Applying Histogram Equalization...")
        
        # Apply histogram equalization
        equalized = cv2.equalizeHist(img)
        
        # Calculate histograms
        hist_original = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist_equalized = cv2.calcHist([equalized], [0], None, [256], [0, 256])
        
        # Visualize results
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        axes[0, 0].imshow(img, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(equalized, cmap='gray')
        axes[0, 1].set_title('Equalized Image')
        axes[0, 1].axis('off')
        
        axes[1, 0].plot(hist_original)
        axes[1, 0].set_title('Original Histogram')
        axes[1, 0].set_xlabel('Pixel Intensity')
        axes[1, 0].set_ylabel('Frequency')
        
        axes[1, 1].plot(hist_equalized)
        axes[1, 1].set_title('Equalized Histogram')
        axes[1, 1].set_xlabel('Pixel Intensity')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '1_histogram_equalization.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        print("  ✓ Histogram equalization complete")
        print(f"  ✓ Saved to: {self.output_dir / '1_histogram_equalization.png'}")
        
        return equalized
    
    def thresholding(self, img):
        """
        Apply various thresholding techniques.
        
        Thresholding converts grayscale images to binary images,
        useful for segmentation and object detection.
        
        Args:
            img (numpy.ndarray): Input grayscale image
            
        Returns:
            dict: Dictionary of thresholded images
        """
        print("\n2. Applying Thresholding Techniques...")
        
        # Simple binary threshold
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        
        # Otsu's thresholding (automatic threshold selection)
        _, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Adaptive thresholding (local threshold)
        adaptive_mean = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        adaptive_gaussian = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Visualize results
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        
        axes[0, 0].imshow(img, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(binary, cmap='gray')
        axes[0, 1].set_title('Binary Threshold (127)')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(otsu, cmap='gray')
        axes[0, 2].set_title("Otsu's Threshold")
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(adaptive_mean, cmap='gray')
        axes[1, 0].set_title('Adaptive Mean Threshold')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(adaptive_gaussian, cmap='gray')
        axes[1, 1].set_title('Adaptive Gaussian Threshold')
        axes[1, 1].axis('off')
        
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '2_thresholding.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        print("  ✓ Thresholding complete")
        print(f"  ✓ Saved to: {self.output_dir / '2_thresholding.png'}")
        
        return {
            'binary': binary,
            'otsu': otsu,
            'adaptive_mean': adaptive_mean,
            'adaptive_gaussian': adaptive_gaussian
        }
    
    def edge_detection(self, img):
        """
        Apply edge detection algorithms.
        
        Edge detection identifies boundaries in images, crucial for
        object recognition and image segmentation.
        
        Args:
            img (numpy.ndarray): Input grayscale image
            
        Returns:
            dict: Dictionary of edge-detected images
        """
        print("\n3. Applying Edge Detection...")
        
        # Sobel edge detection (X and Y gradients)
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        sobel_combined = cv2.magnitude(sobel_x, sobel_y)
        
        # Normalize for display
        sobel_x = cv2.convertScaleAbs(sobel_x)
        sobel_y = cv2.convertScaleAbs(sobel_y)
        sobel_combined = cv2.convertScaleAbs(sobel_combined)
        
        # Canny edge detection
        canny = cv2.Canny(img, 50, 150)
        
        # Laplacian edge detection
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        laplacian = cv2.convertScaleAbs(laplacian)
        
        # Visualize results
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        
        axes[0, 0].imshow(img, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(sobel_x, cmap='gray')
        axes[0, 1].set_title('Sobel X')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(sobel_y, cmap='gray')
        axes[0, 2].set_title('Sobel Y')
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(sobel_combined, cmap='gray')
        axes[1, 0].set_title('Sobel Combined')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(canny, cmap='gray')
        axes[1, 1].set_title('Canny Edge Detection')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(laplacian, cmap='gray')
        axes[1, 2].set_title('Laplacian')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '3_edge_detection.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        print("  ✓ Edge detection complete")
        print(f"  ✓ Saved to: {self.output_dir / '3_edge_detection.png'}")
        
        return {
            'sobel_x': sobel_x,
            'sobel_y': sobel_y,
            'sobel_combined': sobel_combined,
            'canny': canny,
            'laplacian': laplacian
        }
    
    def data_augmentation(self, img):
        """
        Apply data augmentation techniques.
        
        Data augmentation artificially increases dataset size by creating
        modified versions of images, improving model generalization.
        
        Args:
            img (numpy.ndarray): Input grayscale image
            
        Returns:
            dict: Dictionary of augmented images
        """
        print("\n4. Applying Data Augmentation...")
        
        h, w = img.shape
        
        # Rotation
        M_rotate = cv2.getRotationMatrix2D((w/2, h/2), 30, 1.0)
        rotated = cv2.warpAffine(img, M_rotate, (w, h))
        
        # Horizontal flip
        flipped_h = cv2.flip(img, 1)
        
        # Vertical flip
        flipped_v = cv2.flip(img, 0)
        
        # Scaling (zoom in)
        scale = 1.2
        M_scale = cv2.getRotationMatrix2D((w/2, h/2), 0, scale)
        scaled = cv2.warpAffine(img, M_scale, (w, h))
        
        # Translation
        M_translate = np.float32([[1, 0, 50], [0, 1, 30]])
        translated = cv2.warpAffine(img, M_translate, (w, h))
        
        # Brightness adjustment
        brightened = cv2.convertScaleAbs(img, alpha=1.3, beta=30)
        
        # Gaussian blur
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        
        # Add noise
        noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
        noisy = cv2.add(img, noise)
        
        # Visualize results
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        
        augmentations = [
            (img, 'Original'),
            (rotated, 'Rotated (30°)'),
            (flipped_h, 'Horizontal Flip'),
            (flipped_v, 'Vertical Flip'),
            (scaled, 'Scaled (1.2x)'),
            (translated, 'Translated'),
            (brightened, 'Brightened'),
            (blurred, 'Gaussian Blur'),
            (noisy, 'Added Noise')
        ]
        
        for idx, (aug_img, title) in enumerate(augmentations):
            row, col = idx // 3, idx % 3
            axes[row, col].imshow(aug_img, cmap='gray')
            axes[row, col].set_title(title)
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '4_data_augmentation.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        print("  ✓ Data augmentation complete")
        print(f"  ✓ Saved to: {self.output_dir / '4_data_augmentation.png'}")
        
        return {
            'rotated': rotated,
            'flipped_h': flipped_h,
            'flipped_v': flipped_v,
            'scaled': scaled,
            'translated': translated,
            'brightened': brightened,
            'blurred': blurred,
            'noisy': noisy
        }
    
    def morphological_operations(self, img):
        """
        Apply morphological operations.
        
        Morphological operations process images based on shapes,
        useful for noise removal and shape analysis.
        
        Args:
            img (numpy.ndarray): Input grayscale image
            
        Returns:
            dict: Dictionary of morphologically processed images
        """
        print("\n5. Applying Morphological Operations...")
        
        # Create binary image for morphological operations
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        
        # Define kernel for morphological operations
        kernel = np.ones((5, 5), np.uint8)
        
        # Erosion - removes small white noises, shrinks objects
        erosion = cv2.erode(binary, kernel, iterations=1)
        
        # Dilation - increases object area, fills small holes
        dilation = cv2.dilate(binary, kernel, iterations=1)
        
        # Opening - erosion followed by dilation (removes noise)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Closing - dilation followed by erosion (fills holes)
        closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Gradient - difference between dilation and erosion (outlines)
        gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
        
        # Top hat - difference between input and opening
        tophat = cv2.morphologyEx(binary, cv2.MORPH_TOPHAT, kernel)
        
        # Black hat - difference between closing and input
        blackhat = cv2.morphologyEx(binary, cv2.MORPH_BLACKHAT, kernel)
        
        # Visualize results
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        
        operations = [
            (img, 'Original'),
            (binary, 'Binary'),
            (erosion, 'Erosion'),
            (dilation, 'Dilation'),
            (opening, 'Opening'),
            (closing, 'Closing'),
            (gradient, 'Gradient'),
            (tophat, 'Top Hat'),
            (blackhat, 'Black Hat')
        ]
        
        for idx, (op_img, title) in enumerate(operations):
            row, col = idx // 3, idx % 3
            axes[row, col].imshow(op_img, cmap='gray')
            axes[row, col].set_title(title)
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '5_morphological_operations.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        print("  ✓ Morphological operations complete")
        print(f"  ✓ Saved to: {self.output_dir / '5_morphological_operations.png'}")
        
        return {
            'binary': binary,
            'erosion': erosion,
            'dilation': dilation,
            'opening': opening,
            'closing': closing,
            'gradient': gradient,
            'tophat': tophat,
            'blackhat': blackhat
        }
    
    def run_all_operations(self):
        """
        Run all image processing operations in sequence.
        Demonstrates the complete workflow of image processing techniques.
        """
        print("\n" + "="*60)
        print("  LAB 1: IMAGE PROCESSING (LIGHTWEIGHT)")
        print("="*60)
        
        start_time = time.time()
        
        # Load real image from Face Mask dataset (or create synthetic)
        print("\nLoading image...")
        img = self.load_real_image()
        cv2.imwrite(str(self.output_dir / '0_original_image.png'), img)
        print(f"  SUCCESS: Image loaded: {img.shape}")
        
        # Run all operations
        self.histogram_equalization(img)
        self.thresholding(img)
        self.edge_detection(img)
        self.data_augmentation(img)
        self.morphological_operations(img)
        
        elapsed_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("  SUMMARY")
        print("="*60)
        print(f"✓ All operations completed successfully!")
        print(f"✓ Total execution time: {elapsed_time:.2f} seconds")
        print(f"✓ Output files saved to: {self.output_dir}/")
        print("\nOptimizations applied:")
        print("  • Image size: 300x300 (vs 400x400) - 44% fewer pixels")
        print("  • Figure sizes reduced by 20%")
        print("  • DPI: 100 (vs 150)")
        print("\nGenerated files:")
        print("  1. 0_original_image.png - Sample input image")
        print("  2. 1_histogram_equalization.png - Contrast enhancement")
        print("  3. 2_thresholding.png - Binary segmentation")
        print("  4. 3_edge_detection.png - Edge detection methods")
        print("  5. 4_data_augmentation.png - Augmentation techniques")
        print("  6. 5_morphological_operations.png - Morphological transforms")
        print("\n" + "="*60)


def main():
    """
    Main function to run the image processing lab.
    """
    # Create processor instance
    processor = ImageProcessor(output_dir='output')
    
    # Run all operations
    processor.run_all_operations()
    
    print("\n✨ Lab 1 completed! Check the 'outputs' folder for results.")


if __name__ == "__main__":
    main()


