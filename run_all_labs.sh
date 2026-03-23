#!/bin/bash

# Deep Learning Lab - Run All Labs Script
# This script runs all lab programs sequentially

echo "=========================================="
echo "  DEEP LEARNING LAB - RUNNING ALL LABS"
echo "=========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

echo "Python version:"
python3 --version
echo ""

# Function to run a lab
run_lab() {
    lab_num=$1
    lab_name=$2
    lab_dir=$3
    lab_script=$4
    
    echo "=========================================="
    echo "  LAB $lab_num: $lab_name"
    echo "=========================================="
    
    if [ -f "$lab_dir/$lab_script" ]; then
        cd "$lab_dir" || exit
        echo "Running: python3 $lab_script"
        python3 "$lab_script"
        exit_code=$?
        cd - > /dev/null || exit
        
        if [ $exit_code -eq 0 ]; then
            echo "✓ Lab $lab_num completed successfully"
        else
            echo "✗ Lab $lab_num failed with exit code $exit_code"
            return $exit_code
        fi
    else
        echo "⚠ Lab $lab_num script not found: $lab_dir/$lab_script"
        echo "  (This lab may be documentation-only)"
    fi
    
    echo ""
    sleep 2
}

# Start timer
start_time=$(date +%s)

# Run all labs
run_lab 1 "Basic Image Processing" "lab1_image_processing" "image_processing.py"
run_lab 2 "CIFAR-10 Classifiers" "lab2_cifar10_classifiers" "cifar10_classifiers.py"
run_lab 3 "Batch Normalization & Dropout" "lab3_batchnorm_dropout" "batchnorm_dropout_study.py"
run_lab 4 "Labeling Tools Demo" "lab4_labeling_tools" "labeling_demo.py"
run_lab 5 "Image Segmentation" "lab5_segmentation" "segmentation_demo.py"
run_lab 6 "Object Detection" "lab6_object_detection" "object_detection_demo.py"
run_lab 7 "Image Captioning" "lab7_image_captioning" "image_captioning_demo.py"
run_lab 8 "Chatbot" "lab8_chatbot" "chatbot_demo.py"
run_lab 9 "Time Series Forecasting" "lab9_time_series" "time_series_demo.py"
run_lab 10 "Sequence to Sequence" "lab10_seq2seq" "seq2seq_demo.py"

# End timer
end_time=$(date +%s)
elapsed=$((end_time - start_time))
minutes=$((elapsed / 60))
seconds=$((elapsed % 60))

echo "=========================================="
echo "  ALL LABS COMPLETED"
echo "=========================================="
echo "Total execution time: ${minutes}m ${seconds}s"
echo ""
echo "Check individual lab 'outputs' directories for results"
echo ""


