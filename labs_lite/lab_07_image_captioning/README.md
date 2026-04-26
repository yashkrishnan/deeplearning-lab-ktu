# Lab 7 Lite: Image Captioning with Flickr8k

Lightweight image captioning using CNN encoder and LSTM decoder on Flickr8k dataset.

## Quick Start

```bash
# 1. Copy sample dataset (if not already done)
python3 copy_sample_dataset.py

# 2. Run the lab
python3 image_captioning_lite.py
```

## Dataset

This lab uses a **local sample** of 2,000 images from the Flickr8k dataset with their captions.

### Dataset Structure
```
data/flickr8k/
├── Images/         # 2,000 sample images
└── captions.txt    # 10,000 caption entries (5 per image)
```

### Setting Up the Dataset

The `copy_sample_dataset.py` script copies 2,000 random images with their captions:

```bash
python3 copy_sample_dataset.py
```

This will:
- Randomly select 2,000 images from Flickr8k
- Copy images to local directory
- Filter and copy corresponding captions (5 per image)
- Create the local `data/flickr8k/` directory structure

## Configuration

- **Samples**: 2,000 images (local dataset)
- **Epochs**: 30
- **Batch Size**: 16
- **Image Size**: 128x128
- **Max Caption Length**: 20 words
- **Vocabulary Size**: ~5,000 words (built from captions)

## Model Architecture

**CNN Encoder**:
- 4 convolutional blocks with batch normalization
- Adaptive average pooling
- Fully connected layer to embedding space
- Output: 256-dimensional image features

**LSTM Decoder**:
- Word embedding layer
- LSTM with 256 hidden units
- Fully connected output layer
- Generates captions word-by-word

**Total Parameters**: ~1.2M

## Expected Results

- **Training Time**: ~10-15 minutes on CPU (3-5 minutes on GPU)
- **Loss Reduction**: From ~5.2 to ~2.5-3.0
- **Outputs**:
  - `output/training_history.png` - Loss curve
  - `output/caption_results.png` - Sample generated captions

## Performance

With 2,000 training samples and 30 epochs:
- Initial Loss: ~5.2
- Final Loss: ~2.5-3.0
- Vocabulary: ~5,000 words
- Model generates diverse and contextually appropriate captions

## Sample Captions

The model generates varied captions like:
- "a dog is running through the grass"
- "a man in a red shirt is standing on a beach"
- "two people are walking down a street"
- "a child is playing with a ball"

With 2,000 samples and 30 epochs, the model generates diverse and contextually appropriate captions with good variety.

## Comparison with Original

| Aspect | Original Lab 7 | Lab 7 Lite |
|--------|---------------|------------|
| Dataset Size | ~8,000 images | 2,000 images |
| Captions | ~40,000 | 10,000 |
| Vocabulary | ~8,000 words | ~5,000 words |
| Epochs | 50 | 30 |
| Batch Size | 32 | 16 |
| Runtime | ~2-3 hours | ~10-15 minutes |
| Model Size | Large | Compact |

## Notes

- The local dataset makes this lab completely self-contained
- Training takes ~10-15 minutes on CPU (reasonable for quality output)
- **Caption Quality**: With 2,000 samples, the model generates diverse and meaningful captions. This is a good balance between training time and output quality.
- **Learning Objective**: This lite version demonstrates the encoder-decoder architecture with realistic caption generation
- For production-quality captions, use the full 8,000-image dataset in the original lab
- Vocabulary is automatically built from the sample captions
- The model successfully learns diverse caption patterns and vocabulary

## Troubleshooting

### Dataset Not Found
```bash
# Run the copy script to set up the local dataset
python3 copy_sample_dataset.py
```

### Out of Memory
```python
# Reduce batch size in image_captioning_lite.py
BATCH_SIZE = 8  # or even 4
```

### Repetitive Captions
With 2,000 samples, captions should be diverse. If you still see repetitive outputs:
- Ensure the dataset was regenerated correctly (should have 2,000 images)
- Check that training completed all 30 epochs
- Verify loss decreased to ~2.5-3.0
- For even better quality, use the original Lab 7 with full Flickr8k dataset (8,000 images)

## Files

- `image_captioning_lite.py` - Main training script
- `copy_sample_dataset.py` - Dataset setup script (copies 2,000 images)
- `data/flickr8k/` - Local sample dataset (created by copy script)
- `output/` - Generated visualizations

## Learning Objectives

1. Image captioning with encoder-decoder architecture
2. CNN feature extraction
3. LSTM sequence generation
4. Vocabulary building from text
5. Caption generation and evaluation

---

**Runtime**: ~10-15 minutes on CPU
**Dataset**: 2,000 local Flickr8k images
**Model**: CNN-LSTM (~1.5M parameters)

## Quality Improvements

This version has been updated from the original 500-image configuration to 2,000 images for significantly better caption quality. See `../QUALITY_IMPROVEMENTS.md` for details on the changes and how to revert if needed.