# Lab 8 Lite: Chatbot with Cornell Movie Dialogs

Lightweight chatbot using Seq2Seq architecture with bi-directional LSTM encoder and attention-based decoder on Cornell Movie Dialogs dataset.

## Quick Start

```bash
# 1. Copy sample dataset (if not already done)
python3 copy_sample_dataset.py

# 2. Run the lab
python3 chatbot_lite.py
```

## Dataset

This lab uses a **local sample** of Cornell Movie Dialogs dataset with 50,000 movie lines.

### Dataset Structure
```
data/cornell/
├── movie_lines.txt              # 50,000 dialog lines
├── movie_conversations.txt      # Conversation structure
├── movie_characters_metadata.txt
└── movie_titles_metadata.txt
```

### Setting Up the Dataset

The `copy_sample_dataset.py` script copies sample dialog data:

```bash
python3 copy_sample_dataset.py
```

This will:
- Copy first 50,000 lines from movie_lines.txt
- Copy all conversation structure files
- Create the local `data/cornell/` directory structure

## Configuration

- **Samples**: 10,000 conversation pairs (local dataset)
- **Epochs**: 20
- **Batch Size**: 32
- **Max Length**: 20 words per utterance
- **Vocabulary Size**: 5,000 words

## Model Architecture

**Encoder (Bi-directional LSTM)**:
- Word embedding layer (128 dimensions)
- Bi-directional LSTM (256 hidden units)
- Combines forward and backward hidden states

**Decoder (LSTM with Attention)**:
- Word embedding layer (128 dimensions)
- Attention mechanism over encoder outputs
- LSTM decoder (256 hidden units)
- Output layer to vocabulary

**Total Parameters**: ~3.4M

## Expected Results

- **Training Time**: ~10-15 minutes on CPU (3-5 minutes on GPU)
- **Loss Reduction**: From ~6.0 to ~3.0-3.5
- **Outputs**:
  - `output/training_history.png` - Loss curve
  - Console demo with sample conversations

## Performance

With 10,000 training pairs and 20 epochs:
- Initial Loss: ~6.0
- Final Loss: ~3.0-3.5
- Vocabulary: 5,000 words
- Model generates varied and contextually appropriate responses

## Response Quality

With 10,000 training pairs and 20 epochs, the model generates varied responses that demonstrate understanding of conversational patterns. While responses are simpler than production chatbots, they show contextual awareness and diversity.

### What the Lite Version Demonstrates

The lite version successfully demonstrates:
- Seq2Seq architecture (Encoder-Decoder)
- Bi-directional LSTM encoding
- Attention mechanism
- Training process and loss reduction
- Dialog pair processing
- Vocabulary building from conversations
- Varied response generation

### For Production-Quality Responses

For even better response quality:
1. Use the original Lab 8 with full Cornell dataset (220,000+ pairs)
2. Train for 30+ epochs
3. Consider using pre-trained models or transformer architectures

## Comparison with Original

| Aspect | Original Lab 8 | Lab 8 Lite |
|--------|---------------|------------|
| Dataset Size | ~220,000 pairs | 10,000 pairs |
| Dialog Lines | ~300,000 | 50,000 |
| Vocabulary | ~10,000 words | 5,000 words |
| Epochs | 30 | 20 |
| Batch Size | 64 | 32 |
| Runtime | ~2-3 hours | ~10-15 minutes |
| Model Size | Large | Compact |

## Notes

- The local dataset makes this lab completely self-contained
- Training takes ~10-15 minutes on CPU (reasonable for quality output)
- **Response Quality**: With 10,000 samples, the model generates varied and contextually appropriate responses
- **Learning Objective**: This lite version demonstrates the Seq2Seq architecture with realistic dialog generation
- For production-quality responses, use the full dataset in the original lab
- Vocabulary is automatically built from the dialog corpus
- The model successfully learns diverse conversational patterns

## Troubleshooting

### Dataset Not Found
```bash
# Run the copy script to set up the local dataset
python3 copy_sample_dataset.py
```

### Out of Memory
```python
# Reduce batch size in chatbot_lite.py
BATCH_SIZE = 16  # or even 8
```

### Repetitive Responses
With 10,000 samples, responses should be varied. If you still see repetitive outputs:
- Ensure the dataset was regenerated correctly (should have 50,000 lines)
- Check that training completed all 20 epochs
- Verify loss decreased to ~3.0-3.5
- For even better quality, use the original Lab 8 with full Cornell dataset (220,000+ pairs)

## Files

- `chatbot_lite.py` - Main training script
- `copy_sample_dataset.py` - Dataset setup script (copies 50k lines)
- `data/cornell/` - Local sample dataset (created by copy script)
- `output/` - Generated visualizations

## Learning Objectives

1. Sequence-to-sequence architecture
2. Bi-directional LSTM encoding
3. Attention mechanism
4. Dialog pair processing
5. Vocabulary building from text
6. Response generation

---

**Runtime**: ~10-15 minutes on CPU
**Dataset**: 10,000 local Cornell dialog pairs
**Model**: Seq2Seq with Attention (~3.4M parameters)

## Quality Improvements

This version has been updated from the original 2,000-pair configuration to 10,000 pairs for significantly better response quality and diversity. See `../QUALITY_IMPROVEMENTS.md` for details on the changes and how to revert if needed.