#!/bin/bash
# Script to run all lightweight lab programs
# This runs the optimized versions that train faster on laptops

echo "========================================================================"
echo "Running All Lightweight Lab Programs"
echo "========================================================================"
echo ""
echo "These are optimized versions that train 2-3x faster than originals"
echo "Perfect for quick testing and learning on regular laptops"
echo ""

# Function to run a lab
run_lab() {
    local lab_dir=$1
    local script=$2
    local lab_name=$3
    
    echo "------------------------------------------------------------------------"
    echo "Running: $lab_name"
    echo "------------------------------------------------------------------------"
    
    if [ -f "$lab_dir/$script" ]; then
        cd "$lab_dir" || exit
        python3 "$script"
        cd ..
        echo ""
        echo "✓ $lab_name completed!"
        echo ""
    else
        echo "⚠ Skipping $lab_name - file not found: $lab_dir/$script"
        echo ""
    fi
}

# Track start time
start_time=$(date +%s)

# Run Lab 1
run_lab "lab_01_image_processing" "image_processing_lite.py" "Lab 1: Image Processing (Lightweight)"

# Run Lab 2
run_lab "lab_02_cifar10_classifiers" "cifar10_classifiers_lite.py" "Lab 2: CIFAR-10 Classifiers (Lightweight)"

# Run Lab 3
run_lab "lab_03_batchnorm_dropout" "batchnorm_dropout_study_lite.py" "Lab 3: Batch Normalization & Dropout (Lightweight)"

# Run Lab 4
run_lab "lab_04_labeling_tools" "labeling_demo_lite.py" "Lab 4: Labeling Tools (Lightweight)"

# Run Lab 5
run_lab "lab_05_segmentation" "segmentation_demo_lite.py" "Lab 5: Image Segmentation (Lightweight)"

# Run Lab 6 (if full implementation exists)
run_lab "lab_06_object_detection" "object_detection_demo_lite.py" "Lab 6: Object Detection (Lightweight)"

# Run Lab 7 (if full implementation exists)
run_lab "lab_07_image_captioning" "image_captioning_demo_lite.py" "Lab 7: Image Captioning (Lightweight)"

# Run Lab 8 (if full implementation exists)
run_lab "lab_08_chatbot" "chatbot_demo_lite.py" "Lab 8: Chatbot (Lightweight)"

# Run Lab 9 (if full implementation exists)
run_lab "lab_09_time_series" "time_series_demo_lite.py" "Lab 9: Time Series (Lightweight)"

# Run Lab 10 (if full implementation exists)
run_lab "lab_10_seq2seq" "seq2seq_demo_lite.py" "Lab 10: Seq2Seq (Lightweight)"

# Calculate total time
end_time=$(date +%s)
total_time=$((end_time - start_time))
minutes=$((total_time / 60))
seconds=$((total_time % 60))

echo "========================================================================"
echo "All Lightweight Labs Completed!"
echo "========================================================================"
echo ""
echo "Total execution time: ${minutes}m ${seconds}s"
echo ""
echo "Output files saved in respective lab directories:"
echo "  - lab_02_cifar10_classifiers/outputs_lite/"
echo "  - lab_03_batchnorm_dropout/outputs_lite/"
echo "  - lab_05_segmentation/output_lite/"
echo ""
echo "Note: Labs 6-10 are placeholder versions. For full implementations,"
echo "see LIGHTWEIGHT_LABS_README.md for instructions."
echo ""


