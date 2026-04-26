#!/bin/bash

# Start the Deep Learning Labs Web Interface
# This script starts the Flask server

echo "============================================================"
echo "Deep Learning Labs - Web Interface"
echo "============================================================"
echo ""

# Check if Flask is installed
if ! python3 -c "import flask" 2>/dev/null; then
    echo "Flask is not installed. Installing dependencies..."
    pip3 install -r requirements.txt
    echo ""
fi

# Check if parent dependencies are installed
if ! python3 -c "import torch" 2>/dev/null; then
    echo "WARNING: PyTorch not found. Lab programs may not work."
    echo "Install dependencies from parent directory:"
    echo "  cd .. && pip3 install -r requirements.txt"
    echo ""
fi

echo "Starting Flask server..."
echo "Open your browser and navigate to: http://localhost:5001"
echo ""
echo "Press Ctrl+C to stop the server"
echo "============================================================"
echo ""

# Start the Flask app
python3 app.py


