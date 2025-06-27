# GPT-2 From Scratch

A clean PyTorch implementation of GPT-2 (Generative Pre-trained Transformer 2) built from scratch, featuring training on FineWeb dataset, HellaSwag evaluation, and distributed training support.

## Overview

This repository contains a minimal yet complete implementation of GPT-2 that closely follows the original architecture. The model is trained from scratch using the FineWeb dataset and evaluated on the HellaSwag benchmark for common sense reasoning.

**Key Features:**
- 🚀 **Clean Implementation** - Pure PyTorch implementation without heavy dependencies
- 📊 **FineWeb Training** - High-quality web text dataset for training
- 🧠 **HellaSwag Evaluation** - Common sense reasoning benchmark
- ⚡ **Distributed Training** - Multi-GPU support with PyTorch DDP
- 🎯 **Exact GPT-2 Architecture** - Matches OpenAI's GPT-2 specifications
- 💾 **Model Checkpointing** - Save and resume training
- 🔧 **Optimized Training** - Mixed precision, gradient accumulation, and more

## Architecture Details

The implementation includes all key GPT-2 components:

- **Causal Self-Attention**: Multi-head attention with causal masking
- **MLP Blocks**: Feed-forward networks with GELU activation
- **Layer Normalization**: Applied before attention and MLP layers
- **Positional Embeddings**: Learned position encodings
- **Weight Tying**: Shared weights between input and output embeddings

### Model Configurations

| Model Size | Parameters | Layers | Heads | Embedding Dim |
|------------|------------|--------|-------|---------------|
| GPT-2 124M | 124M       | 12     | 12    | 768          |
| GPT-2 350M | 350M       | 24     | 16    | 1024         |
| GPT-2 774M | 774M       | 36     | 20    | 1280         |
| GPT-2 1.5B | 1.5B       | 48     | 25    | 1600         |

## Installation

```bash
# Clone the repository
git clone https://github.com/BrijeshSukhadiya/GPT-2.git
cd GPT-2

# Install dependencies
pip install torch tiktoken numpy transformers
```

### Requirements

- Python 3.8+
- PyTorch 2.0+
- tiktoken (for GPT-2 tokenization)
- numpy
- transformers (for loading pretrained weights)

## Quick Start

### Training from Scratch

```bash
# Single GPU training
python train_gpt2.py

# Multi-GPU distributed training (8 GPUs)
torchrun --standalone --nproc_per_node=8 train_gpt2.py
```

### Generate Text

```python
import torch
import tiktoken
from train_gpt2 import GPT, GPTConfig

# Load trained model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = GPT(GPTConfig())
checkpoint = torch.load('log/model_19073.pt', map_location=device)
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

# Initialize tokenizer
enc = tiktoken.get_encoding("gpt2")

# Generate text
prompt = "The future of artificial intelligence is"
tokens = enc.encode(prompt)
tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

# Generate with top-k sampling
with torch.no_grad():
    for _ in range(50):  # generate 50 tokens
        logits, _ = model(tokens)
        logits = logits[:, -1, :]  # get last token logits
        probs = torch.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices, -1, ix)
        tokens = torch.cat((tokens, xcol), dim=1)

# Decode and print
generated_text = enc.decode(tokens[0].tolist())
print(generated_text)
```

### Load Pretrained GPT-2 Weights

```python
# Load OpenAI's pretrained GPT-2
model = GPT.from_pretrained("gpt2")  # or gpt2-medium, gpt2-large, gpt2-xl
model.to(device)

# Generate text with pretrained model
# ... (same generation code as above)
```

## File Structure

```
GPT-2/
├── train_gpt2.py        # Main training script with GPT model implementation
├── hellaswag.py         # HellaSwag evaluation utilities
├── fineweb.py           # FineWeb dataset processing
├── pretrained.ipynb     # Jupyter notebook for pretrained model experiments
├── input.txt           # Sample input text
├── text.txt            # Additional text data
├── edu_fineweb10B/     # FineWeb dataset directory (created during training)
├── log/                # Training logs and model checkpoints
└── README.md           # This file
```

## Training Configuration

The training script uses the following optimized settings:

```python
# Model Configuration
config = GPTConfig(
    block_size=1024,        # sequence length
    vocab_size=50304,       # vocabulary size (padded for efficiency)
    n_layer=12,             # number of transformer blocks
    n_head=12,              # number of attention heads
    n_embd=768              # embedding dimension
)

# Training Hyperparameters
total_batch_size = 524288   # effective batch size
B = 16                      # micro batch size
T = 1024                    # sequence length
max_lr = 6e-4              # peak learning rate
min_lr = max_lr * 0.1      # minimum learning rate
warmup_steps = 715         # warmup steps
max_steps = 19073          # total training steps
```

## Features

### 1. Distributed Training Support

```bash
# Multi-node, multi-GPU training
torchrun --nnodes=2 --nproc_per_node=8 train_gpt2.py
```

The implementation automatically detects DDP environment variables and configures distributed training.

### 2. HellaSwag Evaluation

The model is evaluated on HellaSwag every 250 steps to measure common sense reasoning:

### 3. Mixed Precision Training

```python
# Automatic mixed precision with bfloat16
with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
    logits, loss = model(x, y)
```

### 4. Gradient Accumulation

```python
# Accumulate gradients over multiple micro-batches
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
```

### 5. Learning Rate Scheduling

```python
def get_lr(it):
    # Warmup + Cosine decay schedule
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)
```

## Data Processing

### FineWeb Dataset

The training uses the FineWeb dataset, a high-quality web text dataset:

```python
# Data is tokenized and stored as .npy files
# Each shard contains pre-tokenized sequences
data_root = "edu_fineweb10B"
```

### Custom Dataset

To train on your own data:

1. Prepare your text data in a single file
2. Tokenize using tiktoken
3. Save as numpy arrays in the expected format
4. Update the data path in `train_gpt2.py`

## Performance Optimization

### Memory Optimization
- **Mixed Precision**: Uses bfloat16 to reduce memory usage
- **Gradient Checkpointing**: Can be enabled for large models
- **Efficient Attention**: Uses PyTorch's `scaled_dot_product_attention`

### Training Speed
- **Fused AdamW**: Automatically uses fused optimizer when available
- **Compiled Model**: Optional `torch.compile` support
- **Gradient Accumulation**: Simulates large batch sizes

## Monitoring and Logging

The training script logs:
- **Training Loss**: Logged every step
- **Validation Loss**: Every 250 steps
- **HellaSwag Accuracy**: Every 250 steps
- **Generated Samples**: Every 250 steps
- **Model Checkpoints**: Every 5000 steps
## Advanced Usage

### Custom Model Sizes

```python
# Create custom model configuration
custom_config = GPTConfig(
    block_size=2048,    # longer sequences
    vocab_size=50304,
    n_layer=24,         # deeper model
    n_head=16,
    n_embd=1024
)
model = GPT(custom_config)
```

### Resume Training

```python
# Resume from checkpoint
checkpoint = torch.load('log/model_10000.pt')
model.load_state_dict(checkpoint['model'])
start_step = checkpoint['step']
```

### Fine-tuning

```python
# Load pretrained model and fine-tune
model = GPT.from_pretrained("gpt2")
# ... (setup optimizer and data loader for your specific task)
```

## Troubleshooting

### Common Issues

**CUDA Out of Memory**
```bash
# Reduce batch size
B = 8  # instead of 16

# Or increase gradient accumulation
# This maintains the same effective batch size
```

**Slow Training**
```bash
# Enable model compilation (may have compatibility issues)
use_compile = True

# Use more GPUs
torchrun --nproc_per_node=8 train_gpt2.py
```

**HellaSwag Evaluation Failing**
```bash
# Disable compilation for evaluation
use_compile = False
```
## Acknowledgments

- **OpenAI** for the original GPT-2 and GPT-3 research and architecture
- **Andrej Karpathy** for educational content that inspired clean implementations
- **PyTorch Team** for the excellent deep learning framework
- **HuggingFace** for the transformers library and model weights

---

**Built with ❤️ by [Brijesh Sukhadiya](https://github.com/BrijeshSukhadiya)**

*A clean, educational implementation of GPT-2 for understanding transformer architectures and training large language models from scratch.*
