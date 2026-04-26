# Advanced Tab Guide

## Overview

The Advanced tab in the web interface provides UI-based controls for common setup and management tasks. This eliminates the need to run shell scripts manually from the terminal.

## Features

### 1. Install Full Requirements
- **Purpose**: Install all dependencies from `requirements.txt`
- **Use Case**: Setting up the environment for full lab features
- **Command**: `pip install -r requirements.txt`
- **Duration**: 5-10 minutes depending on internet speed

### 2. Install Minimal Requirements
- **Purpose**: Install minimal dependencies from `requirements-minimal.txt`
- **Use Case**: Setting up for lite labs only (faster, smaller footprint)
- **Command**: `pip install -r requirements-minimal.txt`
- **Duration**: 2-3 minutes

### 3. Download All Datasets
- **Purpose**: Download all required datasets (~2GB total)
- **Use Case**: Preparing data for all labs at once
- **Command**: `bash download_all_datasets.sh`
- **Duration**: 10-20 minutes depending on internet speed
- **Datasets Included**:
  - CIFAR-10 (Lab 2)
  - Oxford-IIIT Pets (Lab 5)
  - COCO subset (Lab 6)
  - Flickr8k (Lab 7)
  - Cornell Movie Dialogs (Lab 8)
  - Stock prices (Lab 9)
  - English-French translation (Lab 10)

### 4. Interactive Dataset Download
- **Purpose**: Choose which datasets to download
- **Use Case**: Selective dataset download to save time/space
- **Command**: `bash download_datasets_interactive.sh`
- **Duration**: Varies based on selection

### 5. Run All Full Labs
- **Purpose**: Execute all 10 full labs sequentially
- **Use Case**: Complete lab run for comprehensive results
- **Command**: `bash run_all_labs.sh`
- **Duration**: 25-30 minutes
- **Requirements**: All datasets must be downloaded first

### 6. Run All Lite Labs
- **Purpose**: Execute all 10 lightweight labs sequentially
- **Use Case**: Quick testing or resource-constrained environments
- **Command**: `bash labs_lite/run_lite_labs.sh`
- **Duration**: 10-15 minutes
- **Requirements**: Minimal requirements and datasets

## How to Use

1. **Navigate to Advanced Tab**: Click the "Advanced" tab in the web interface
2. **Select Task**: Click on any task card to start execution
3. **Monitor Progress**: Check the terminal where the Flask server is running for real-time output
4. **Task Output**: The console in the Advanced tab will show confirmation messages

## Important Notes

### Background Execution
- All tasks run in the background
- The web interface will confirm task start but won't show live output
- Monitor the terminal where Flask is running for detailed progress

### Prerequisites
- **For Install Tasks**: Internet connection required
- **For Download Tasks**: Sufficient disk space (~2GB for all datasets)
- **For Run Tasks**: Dependencies and datasets must be installed/downloaded first

### Task Dependencies
Recommended execution order:
1. Install Full Requirements OR Install Minimal Requirements
2. Download All Datasets OR Interactive Dataset Download
3. Run All Full Labs OR Run All Lite Labs

### Error Handling
If a task fails:
1. Check the terminal output for error messages
2. Ensure prerequisites are met
3. Verify internet connection (for downloads/installs)
4. Check disk space availability

## Technical Details

### API Endpoint
- **URL**: `/api/setup/<task_id>`
- **Method**: POST
- **Response**: JSON with message, command, and note

### Task IDs
- `install-full`: Full requirements installation
- `install-minimal`: Minimal requirements installation
- `download-datasets`: All datasets download
- `download-interactive`: Interactive dataset selection
- `run-all-full`: Run all full labs
- `run-all-lite`: Run all lite labs

### Implementation
Tasks are executed using Python's `subprocess.Popen()` in daemon threads, allowing the web interface to remain responsive while tasks run in the background.

## Troubleshooting

### Task Not Starting
- Verify Flask server is running
- Check browser console for JavaScript errors
- Ensure proper network connectivity

### Task Running But No Output
- This is normal - check the terminal where Flask is running
- Tasks execute in the background
- Output appears in the server terminal, not the web interface

### Permission Errors
- Ensure write permissions in the project directory
- On Unix systems, shell scripts may need execute permissions:
  ```bash
  chmod +x download_all_datasets.sh
  chmod +x run_all_labs.sh
  chmod +x labs_lite/run_lite_labs.sh
  ```

## Future Enhancements

Potential improvements for the Advanced tab:
- Real-time output streaming to web interface
- Progress bars for long-running tasks
- Task queue management
- Ability to cancel running tasks
- Task history and logs
- Email notifications on completion

## Related Documentation

- [Quick Setup Guide](QUICK_SETUP.md) - Manual setup instructions
- [Installation Guide](INSTALLATION_GUIDE.md) - Detailed installation steps
- [Datasets Guide](DATASETS.md) - Information about datasets
- [Web Interface Guide](WEB_INTERFACE_GUIDE.md) - General web interface usage