# Installation Guide

Complete guide for setting up Python and required dependencies for Deep Learning Labs on Windows, Ubuntu, and macOS.

---

## Table of Contents
- [Python Installation](#python-installation)
  - [Windows](#windows)
  - [Ubuntu/Linux](#ubuntulinux)
  - [macOS](#macos)
- [Virtual Environment Setup](#virtual-environment-setup)
- [Installing Dependencies](#installing-dependencies)
- [GPU Support (Optional)](#gpu-support-optional)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

---

## Python Installation

### Windows

#### Method 1: Official Python Installer (Recommended)

1. **Download Python**
   - Visit [python.org/downloads](https://www.python.org/downloads/)
   - Download Python 3.8 or later (3.10+ recommended)

2. **Install Python**
   ```
   - Run the installer
   - CHECK "Add Python to PATH"
   - Click "Install Now"
   - Wait for installation to complete
   ```

3. **Verify Installation**
   ```cmd
   python --version
   pip --version
   ```

#### Method 2: Microsoft Store

1. Open Microsoft Store
2. Search for "Python 3.10" or later
3. Click "Get" to install
4. Verify in Command Prompt:
   ```cmd
   python --version
   ```

#### Method 3: Anaconda (For Data Science)

1. Download [Anaconda](https://www.anaconda.com/download)
2. Run installer with default settings
3. Open Anaconda Prompt and verify:
   ```cmd
   python --version
   conda --version
   ```

---

### Ubuntu/Linux

#### Method 1: APT Package Manager (Ubuntu/Debian)

1. **Update Package List**
   ```bash
   sudo apt update
   sudo apt upgrade -y
   ```

2. **Install Python 3**
   ```bash
   sudo apt install python3 python3-pip python3-venv -y
   ```

3. **Verify Installation**
   ```bash
   python3 --version
   pip3 --version
   ```

4. **Create Aliases (Optional)**
   ```bash
   echo "alias python=python3" >> ~/.bashrc
   echo "alias pip=pip3" >> ~/.bashrc
   source ~/.bashrc
   ```

#### Method 2: Deadsnakes PPA (Latest Python Versions)

1. **Add PPA Repository**
   ```bash
   sudo apt update
   sudo apt install software-properties-common -y
   sudo add-apt-repository ppa:deadsnakes/ppa -y
   sudo apt update
   ```

2. **Install Python 3.10 (or later)**
   ```bash
   sudo apt install python3.10 python3.10-venv python3.10-dev -y
   ```

3. **Install pip**
   ```bash
   curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
   ```

#### Method 3: Build from Source

```bash
# Install dependencies
sudo apt install build-essential zlib1g-dev libncurses5-dev \
  libgdbm-dev libnss3-dev libssl-dev libreadline-dev \
  libffi-dev libsqlite3-dev wget libbz2-dev -y

# Download Python
cd /tmp
wget https://www.python.org/ftp/python/3.10.11/Python-3.10.11.tgz
tar -xf Python-3.10.11.tgz
cd Python-3.10.11

# Configure and install
./configure --enable-optimizations
make -j $(nproc)
sudo make altinstall

# Verify
python3.10 --version
```

---

### macOS

#### Method 1: Homebrew (Recommended)

1. **Install Homebrew** (if not installed)
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install Python**
   ```bash
   brew install python@3.10
   ```

3. **Verify Installation**
   ```bash
   python3 --version
   pip3 --version
   ```

4. **Create Aliases (Optional)**
   ```bash
   echo "alias python=python3" >> ~/.zshrc
   echo "alias pip=pip3" >> ~/.zshrc
   source ~/.zshrc
   ```

#### Method 2: Official Python Installer

1. Download from [python.org/downloads/macos](https://www.python.org/downloads/macos/)
2. Run the `.pkg` installer
3. Follow installation wizard
4. Verify in Terminal:
   ```bash
   python3 --version
   ```

#### Method 3: Anaconda

1. Download [Anaconda for macOS](https://www.anaconda.com/download)
2. Run the installer
3. Follow installation instructions
4. Verify:
   ```bash
   python --version
   conda --version
   ```

---

## Virtual Environment Setup

### Using venv (Built-in)

#### Windows
```cmd
# Navigate to project directory
cd path\to\dl-lab

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Deactivate when done
deactivate
```

#### Ubuntu/Linux & macOS
```bash
# Navigate to project directory
cd ~/path/to/dl-lab

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Deactivate when done
deactivate
```

### Using Conda

```bash
# Create environment
conda create -n dl-lab python=3.10 -y

# Activate environment
conda activate dl-lab

# Deactivate when done
conda deactivate
```

---

## Installing Dependencies

### Option 1: Full Installation (All Features)

```bash
# Activate your virtual environment first!

# Install all dependencies
pip install -r requirements.txt

# Verify installation
pip list
```

### Option 2: Minimal Installation (Core Features Only)

```bash
# Activate your virtual environment first!

# Install minimal dependencies
pip install -r requirements-minimal.txt

# Verify installation
pip list
```

### Manual Installation (If needed)

```bash
# Core dependencies
pip install torch torchvision torchaudio
pip install numpy matplotlib opencv-python pillow
pip install scikit-learn scipy tqdm

# Optional: For specific labs
pip install transformers  # For NLP labs
pip install tensorboard   # For visualization
```

---

## GPU Support (Optional)

### NVIDIA GPU (CUDA)

#### Windows & Linux

1. **Check GPU Compatibility**
   ```bash
   nvidia-smi
   ```

2. **Install CUDA Toolkit**
   - Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
   - Follow installation instructions

3. **Install PyTorch with CUDA**
   ```bash
   # For CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # For CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Verify GPU Support**
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA version: {torch.version.cuda}")
   print(f"GPU: {torch.cuda.get_device_name(0)}")
   ```

### Apple Silicon (M1/M2/M3)

```bash
# PyTorch with MPS (Metal Performance Shaders) support
pip install torch torchvision torchaudio

# Verify MPS support
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

---

## Verification

### Test Python Installation

```bash
python --version
pip --version
```

### Test Package Installation

```python
# Create test script: test_installation.py
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

print("✓ All packages imported successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"OpenCV version: {cv2.__version__}")
```

Run the test:
```bash
python test_installation.py
```

### Run Sample Lab

```bash
# Test with Lab 1
cd lab_01_image_processing
python image_processing.py
```

---

## Troubleshooting

### Common Issues

#### 1. "python: command not found"

**Solution:**
- Windows: Reinstall Python and check "Add to PATH"
- Linux/macOS: Use `python3` instead of `python`

#### 2. "pip: command not found"

**Solution:**
```bash
# Linux/macOS
python3 -m ensurepip --upgrade

# Windows
python -m ensurepip --upgrade
```

#### 3. Permission Denied (Linux/macOS)

**Solution:**
```bash
# Don't use sudo with pip in virtual environment
# If outside venv, use --user flag
pip install --user package_name
```

#### 4. SSL Certificate Error

**Solution:**
```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org package_name
```

#### 5. Slow Download Speed

**Solution:**
```bash
# Use a mirror (example: Tsinghua mirror)
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple package_name
```

#### 6. Conflicting Dependencies

**Solution:**
```bash
# Create fresh virtual environment
python -m venv fresh_venv
source fresh_venv/bin/activate  # or fresh_venv\Scripts\activate on Windows
pip install -r requirements.txt
```

#### 7. CUDA Out of Memory

**Solution:**
- Reduce batch size in training scripts
- Use CPU instead: Set `device = 'cpu'` in code
- Close other GPU-intensive applications

#### 8. Import Error: No module named 'cv2'

**Solution:**
```bash
pip uninstall opencv-python opencv-python-headless
pip install opencv-python
```

---

## Additional Resources

### Documentation
- [Python Official Docs](https://docs.python.org/3/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [NumPy Documentation](https://numpy.org/doc/)
- [OpenCV Documentation](https://docs.opencv.org/)

### Learning Resources
- [Python Tutorial](https://docs.python.org/3/tutorial/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Deep Learning Book](https://www.deeplearningbook.org/)

### Community Support
- [Stack Overflow](https://stackoverflow.com/questions/tagged/python)
- [PyTorch Forums](https://discuss.pytorch.org/)
- [Reddit r/learnpython](https://www.reddit.com/r/learnpython/)

---

## Quick Reference

### Essential Commands

```bash
# Check versions
python --version
pip --version

# Create virtual environment
python -m venv venv

# Activate venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# List installed packages
pip list

# Update pip
pip install --upgrade pip

# Deactivate venv
deactivate
```

---

## Next Steps

After successful installation:

1.  Verify all packages are installed
2.  Read [GETTING_STARTED.md](GETTING_STARTED.md)
3.  Review [README.md](README.md) for lab overview
4.  Start with Lab 1: Image Processing
5.  Follow the learning path through all 10 labs

---

**Need Help?** Check the [Troubleshooting](#troubleshooting) section or create an issue in the repository.