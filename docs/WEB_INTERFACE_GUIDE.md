# Web Interface for Deep Learning Labs

A complete web-based interface has been created to run and visualize all lab programs through your browser.

## What's Included

The web interface provides:

- Interactive dashboard with 3 sections: Full Labs, Lite Labs, and Documentation
- All 10 full lab programs + All 10 lightweight versions (20 labs total)
- One-click execution of any lab program
- Real-time console output display
- Automatic visualization of generated images
- Built-in documentation viewer with markdown rendering
- Run/Stop controls for each lab
- Modern dark theme with cyan/blue accents
- Clean, responsive UI

## Quick Start

### 1. Install Flask

```bash
cd web_interface
pip3 install -r requirements.txt
```

### 2. Start the Server

```bash
./start_server.sh
```

Or manually:
```bash
python3 app.py
```

### 3. Open in Browser

Navigate to: **http://localhost:5000**

## Features

### Main Dashboard
- View all 10 labs at a glance
- Click any lab card to open its interface
- Beautiful gradient design with hover effects

### Lab Interface
Each lab page includes:

1. **Control Panel**
   - Run button to start the lab
   - Stop button to terminate execution
   - Refresh button to reload outputs
   - Status indicator (Idle/Running/Completed/Failed)

2. **Console Output**
   - Real-time output from the running program
   - Auto-scrolling to latest output
   - Error messages highlighted in red
   - Monospace font for readability

3. **Generated Results**
   - Automatic display of output images
   - Responsive grid layout
   - File name and size information
   - Click images to view full size

## Usage Example

1. Open http://localhost:5000 in your browser
2. Click on "Lab 1: Image Processing"
3. Click the "Run Lab" button
4. Watch the console output in real-time
5. View generated images automatically when complete

## Supported Labs

All 20 labs are fully supported:

### Full Labs (10)
1. **Image Processing** - Basic operations (< 30 seconds)
2. **CIFAR-10 Classifiers** - KNN, SVM, Neural Network (2-3 minutes)
3. **Batch Normalization & Dropout** - Regularization study (2-3 minutes)
4. **Image Labeling Tools** - Annotation demo (< 30 seconds)
5. **Image Segmentation** - UNet implementation (3-4 minutes)
6. **Object Detection** - YOLO-style detector (3-4 minutes)
7. **Image Captioning** - RNN/LSTM captioning (3-4 minutes)
8. **Chatbot** - Bi-directional LSTM (2-3 minutes)
9. **Time Series Forecasting** - LSTM/GRU models (2-3 minutes)
10. **Sequence to Sequence** - Encoder-decoder with attention (3-4 minutes)

### Lite Labs (10)
All 10 labs available in lightweight versions with 2-3x faster execution:
1. **Image Processing (Lite)** - Fast operations (< 15 seconds)
2. **CIFAR-10 Classifiers (Lite)** - Reduced dataset (1-2 minutes)
3. **Batch Normalization & Dropout (Lite)** - Quick comparison (1-2 minutes)
4. **Image Labeling Tools (Lite)** - Fast demo (< 15 seconds)
5. **Image Segmentation (Lite)** - Smaller dataset (1-2 minutes)
6. **Object Detection (Lite)** - Fast detector (1-2 minutes)
7. **Image Captioning (Lite)** - Quick captioning (1-2 minutes)
8. **Chatbot (Lite)** - Fast BiLSTM (1-2 minutes)
9. **Time Series Forecasting (Lite)** - Quick forecasting (1-2 minutes)
10. **Sequence to Sequence (Lite)** - Fast translation (1-2 minutes)

## Technical Details

### Architecture
- **Backend**: Flask (Python web framework)
- **Frontend**: HTML5, CSS3, JavaScript (vanilla)
- **Process Management**: Python subprocess
- **Real-time Updates**: AJAX polling (1-second intervals)

### File Structure
```
web_interface/
├── app.py                    # Flask application
├── start_server.sh           # Server startup script
├── requirements.txt          # Python dependencies
├── README.md                 # Detailed documentation
└── templates/
    ├── index.html           # Main dashboard
    └── lab.html             # Individual lab page
```

### API Endpoints
- `GET /` - Main dashboard
- `GET /lab/<lab_id>` - Individual lab page
- `POST /api/run/<lab_id>` - Start lab execution
- `POST /api/stop/<lab_id>` - Stop running lab
- `GET /api/status/<lab_id>` - Get execution status
- `GET /api/outputs/<lab_id>` - List output files
- `GET /api/output/<lab_id>/<filename>` - Serve output file

## Screenshots

### Main Dashboard
- Grid of 10 lab cards
- Purple gradient background
- Hover effects on cards
- Quick start instructions

### Lab Interface
- Split-screen layout
- Control panel at top
- Console output on left
- Results gallery on right
- Real-time status updates

## Advantages

1. **No Command Line Required** - Everything through the browser
2. **Visual Feedback** - See outputs immediately
3. **Easy to Use** - Click and run
4. **Real-time Monitoring** - Watch progress live
5. **Organized Results** - All outputs in one place
6. **Multi-Lab Support** - Switch between labs easily

## Requirements

- Python 3.8 or higher
- Flask 3.0.0
- All lab dependencies (PyTorch, OpenCV, etc.)
- Modern web browser (Chrome, Firefox, Safari, Edge)

## Troubleshooting

### Port Already in Use
Change the port in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)
```

### Flask Not Found
```bash
pip3 install flask
```

### Lab Dependencies Missing
```bash
cd ..
pip3 install -r requirements.txt
```

### Images Not Displaying
1. Wait for lab to complete
2. Click "Refresh Outputs"
3. Check browser console for errors

## Security Note

This interface is designed for local development and educational use:
- Runs on localhost only
- No authentication required
- Direct file system access
- Process execution without sandboxing

**Do not expose to the internet without proper security measures.**

## Browser Compatibility

Tested on:
- Chrome/Chromium 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Performance Tips

1. Run one lab at a time
2. Close browser tab when not in use (stops polling)
3. Use lightweight versions in `labs_lite/` for faster execution
4. Clear old outputs periodically

## Next Steps

1. Start the server: `cd web_interface && ./start_server.sh`
2. Open browser: http://localhost:5000
3. Try Lab 1 (fastest) to test the interface
4. Explore other labs as needed

## Additional Documentation

- See `web_interface/README.md` for detailed documentation
- See main `README.md` for lab descriptions
- See individual lab README files for specific details

---

**Created**: April 2026  
**Purpose**: Educational deep learning labs with web interface  
**License**: Academic use