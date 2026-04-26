# Deep Learning Labs - Web Interface

A Flask-based web interface for running and visualizing deep learning lab programs.

## Features

- Interactive web dashboard with 3 sections: Full Labs, Lite Labs, and Documentation
- All 10 full lab programs + All 10 lightweight versions
- Real-time console output display
- Automatic visualization of generated images and results
- Built-in documentation viewer with markdown rendering
- Modern dark theme with cyan/blue accents
- Clean, responsive UI design
- Run/Stop controls for each lab
- Status monitoring and progress tracking

## Quick Start

### 1. Install Dependencies

```bash
cd web_interface
pip install -r requirements.txt
```

Note: You also need the main lab dependencies installed. From the parent directory:
```bash
pip install -r ../requirements.txt
```

### 2. Start the Server

```bash
python app.py
```

The server will start on `http://localhost:5000`

### 3. Open in Browser

Navigate to `http://localhost:5000` in your web browser.

## Usage

### Main Dashboard

The main page displays all 10 labs as cards. Click on any lab card to open its interface.

### Lab Interface

Each lab has its own page with:

- **Control Panel**: Run, Stop, and Refresh buttons
- **Console Output**: Real-time output from the running program
- **Generated Results**: Automatic display of output images and visualizations

### Running a Lab

1. Click on a lab card from the main dashboard
2. Click the "Run Lab" button
3. Watch the console output in real-time
4. View generated results automatically when the lab completes

### Stopping a Lab

If a lab is taking too long or you want to stop it:
1. Click the "Stop" button while the lab is running
2. The process will be terminated gracefully

### Viewing Results

- Results are displayed automatically when a lab completes
- Click "Refresh Outputs" to manually reload the results
- Images are displayed in a responsive grid
- Click on images to view them in full size

## Lab Programs

The web interface supports all 10 labs:

1. **Lab 1**: Image Processing - Basic operations and transformations
2. **Lab 2**: CIFAR-10 Classifiers - KNN, SVM, and Neural Network comparison
3. **Lab 3**: Batch Normalization & Dropout - Regularization techniques study
4. **Lab 4**: Image Labeling Tools - Annotation and format conversion
5. **Lab 5**: Image Segmentation - UNet architecture implementation
6. **Lab 6**: Object Detection - YOLO-style detector
7. **Lab 7**: Image Captioning - CNN encoder + RNN/LSTM decoder
8. **Lab 8**: Chatbot - Bi-directional LSTM for conversational AI
9. **Lab 9**: Time Series Forecasting - LSTM and GRU models
10. **Lab 10**: Sequence to Sequence - Encoder-decoder with attention

## Technical Details

### Architecture

- **Backend**: Flask (Python web framework)
- **Frontend**: HTML, CSS, JavaScript (vanilla)
- **Process Management**: Python subprocess module
- **Real-time Updates**: AJAX polling for status updates

### File Structure

```
web_interface/
├── app.py                 # Flask application
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── templates/
    ├── index.html        # Main dashboard
    └── lab.html          # Individual lab page
```

### API Endpoints

- `GET /` - Main dashboard
- `GET /lab/<lab_id>` - Individual lab page
- `POST /api/run/<lab_id>` - Start a lab program
- `POST /api/stop/<lab_id>` - Stop a running lab
- `GET /api/status/<lab_id>` - Get lab execution status
- `GET /api/outputs/<lab_id>` - List output files
- `GET /api/output/<lab_id>/<filename>` - Serve output file

## Configuration

Lab configurations are defined in `app.py` in the `LABS_CONFIG` dictionary. Each lab has:

- `name`: Display name
- `script`: Path to the Python script (relative to parent directory)
- `output_dir`: Directory where outputs are saved
- `description`: Brief description of the lab

## Troubleshooting

### Port Already in Use

If port 5000 is already in use, modify the last line in `app.py`:

```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Change to 5001 or any available port
```

### Lab Not Running

1. Check that the lab script exists in the parent directory
2. Verify all dependencies are installed
3. Check the console output for error messages
4. Ensure you have write permissions for output directories

### Images Not Displaying

1. Click "Refresh Outputs" to reload
2. Check that the lab completed successfully
3. Verify output files exist in the lab's output directory
4. Check browser console for errors

### Slow Performance

- Labs run in real-time and may take several minutes
- Close other applications to free up resources
- Consider using GPU if available (automatically detected)
- Some labs (2, 3, 5, 6, 7, 9, 10) may take 2-5 minutes

## Development

### Running in Debug Mode

Debug mode is enabled by default. To disable:

```python
app.run(debug=False, host='0.0.0.0', port=5000)
```

### Adding New Labs

To add a new lab:

1. Add entry to `LABS_CONFIG` in `app.py`
2. Ensure the script path and output directory are correct
3. Restart the server

### Customizing the UI

- Edit `templates/index.html` for the main dashboard
- Edit `templates/lab.html` for individual lab pages
- CSS is embedded in the HTML files for simplicity

## Security Notes

This web interface is designed for local development and educational use:

- Runs on localhost by default
- No authentication or authorization
- Direct file system access
- Process execution without sandboxing

**Do not expose this server to the internet without proper security measures.**

## Browser Compatibility

Tested and working on:
- Chrome/Chromium 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Performance Tips

1. Run one lab at a time for best performance
2. Close the browser tab when not in use to stop polling
3. Clear old output files periodically to save disk space
4. Use the lightweight versions in `labs_lite/` for faster execution

## License

This web interface is part of the Deep Learning Labs educational project.
Created for academic learning purposes.

## Support

For issues or questions:
1. Check the main project README.md
2. Review lab-specific README files
3. Check console output for error messages

---

**Last Updated**: April 2026