"""
Deep Learning Labs - Web Interface
==================================

A Flask-based web interface to run and display outputs from deep learning lab programs.

Author: Deep Learning Lab
Date: April 2026
"""

from flask import Flask, render_template, jsonify, send_from_directory, request
import subprocess
import os
import json
import threading
import sys
from pathlib import Path
from datetime import datetime

app = Flask(__name__)

# Configuration
BASE_DIR = Path(__file__).parent.parent
LABS_CONFIG = {
    'lab_01': {
        'name': 'Image Processing',
        'script': 'labs_full/lab_01_image_processing/image_processing.py',
        'output_dir': 'labs_full/lab_01_image_processing/outputs',
        'description': 'Basic image processing operations including histogram equalization, edge detection, and morphological operations',
        'category': 'full'
    },
    'lab_02': {
        'name': 'CIFAR-10 Classifiers',
        'script': 'labs_full/lab_02_cifar10_classifiers/cifar10_classifiers.py',
        'output_dir': 'labs_full/lab_02_cifar10_classifiers/outputs',
        'description': 'Compare KNN, SVM, and Neural Network classifiers on CIFAR-10 dataset',
        'category': 'full'
    },
    'lab_03': {
        'name': 'Batch Normalization & Dropout',
        'script': 'labs_full/lab_03_batchnorm_dropout/batchnorm_dropout_study.py',
        'output_dir': 'labs_full/lab_03_batchnorm_dropout/outputs',
        'description': 'Study the effects of batch normalization and dropout on neural network training',
        'category': 'full'
    },
    'lab_04': {
        'name': 'Image Labeling Tools',
        'script': 'labs_full/lab_04_labeling_tools/labeling_demo.py',
        'output_dir': 'labs_full/lab_04_labeling_tools/output',
        'description': 'Demonstrate image annotation and format conversion tools',
        'category': 'full'
    },
    'lab_05': {
        'name': 'Image Segmentation',
        'script': 'labs_full/lab_05_segmentation/segmentation_demo.py',
        'output_dir': 'labs_full/lab_05_segmentation/output',
        'description': 'UNet architecture for semantic segmentation',
        'category': 'full'
    },
    'lab_06': {
        'name': 'Object Detection',
        'script': 'labs_full/lab_06_object_detection/object_detection_demo.py',
        'output_dir': 'labs_full/lab_06_object_detection/output',
        'description': 'YOLO-style object detection with bounding boxes',
        'category': 'full'
    },
    'lab_07': {
        'name': 'Image Captioning',
        'script': 'labs_full/lab_07_image_captioning/image_captioning_demo.py',
        'output_dir': 'labs_full/lab_07_image_captioning/output',
        'description': 'CNN encoder + RNN/LSTM decoder for image captioning',
        'category': 'full'
    },
    'lab_08': {
        'name': 'Chatbot',
        'script': 'labs_full/lab_08_chatbot/chatbot_demo.py',
        'output_dir': 'labs_full/lab_08_chatbot/output',
        'description': 'Bi-directional LSTM for conversational AI',
        'category': 'full'
    },
    'lab_09': {
        'name': 'Time Series Forecasting',
        'script': 'labs_full/lab_09_time_series/time_series_demo.py',
        'output_dir': 'labs_full/lab_09_time_series/output',
        'description': 'LSTM and GRU for time series prediction',
        'category': 'full'
    },
    'lab_10': {
        'name': 'Sequence to Sequence',
        'script': 'labs_full/lab_10_seq2seq/seq2seq_demo.py',
        'output_dir': 'labs_full/lab_10_seq2seq/output',
        'description': 'Encoder-decoder with attention for machine translation',
        'category': 'full'
    },
    'lab_01_lite': {
        'name': 'Image Processing',
        'script': 'labs_lite/lab_01_image_processing/image_processing_lite.py',
        'output_dir': 'labs_lite/lab_01_image_processing/output',
        'description': 'Fast image processing operations',
        'category': 'lite'
    },
    'lab_02_lite': {
        'name': 'CIFAR-10 Classifiers',
        'script': 'labs_lite/lab_02_cifar10_classifiers/cifar10_classifiers_lite.py',
        'output_dir': 'labs_lite/lab_02_cifar10_classifiers/output',
        'description': '2-3x faster training with reduced dataset',
        'category': 'lite'
    },
    'lab_03_lite': {
        'name': 'Batch Normalization & Dropout',
        'script': 'labs_lite/lab_03_batchnorm_dropout/batchnorm_dropout_study_lite.py',
        'output_dir': 'labs_lite/lab_03_batchnorm_dropout/output',
        'description': 'Quick comparison of regularization techniques',
        'category': 'lite'
    },
    'lab_04_lite': {
        'name': 'Image Labeling Tools',
        'script': 'labs_lite/lab_04_labeling_tools/labeling_demo_lite.py',
        'output_dir': 'labs_lite/lab_04_labeling_tools/output',
        'description': 'Fast annotation demo',
        'category': 'lite'
    },
    'lab_05_lite': {
        'name': 'Image Segmentation',
        'script': 'labs_lite/lab_05_segmentation/segmentation_demo_lite.py',
        'output_dir': 'labs_lite/lab_05_segmentation/output',
        'description': 'Faster UNet training with smaller dataset',
        'category': 'lite'
    },
    'lab_06_lite': {
        'name': 'Object Detection',
        'script': 'labs_lite/lab_06_object_detection/object_detection_demo_lite.py',
        'output_dir': 'labs_lite/lab_06_object_detection/output',
        'description': 'Fast YOLO-style detector',
        'category': 'lite'
    },
    'lab_07_lite': {
        'name': 'Image Captioning',
        'script': 'labs_lite/lab_07_image_captioning/image_captioning_demo_lite.py',
        'output_dir': 'labs_lite/lab_07_image_captioning/output',
        'description': 'Quick RNN/LSTM captioning',
        'category': 'lite'
    },
    'lab_08_lite': {
        'name': 'Chatbot',
        'script': 'labs_lite/lab_08_chatbot/chatbot_demo_lite.py',
        'output_dir': 'labs_lite/lab_08_chatbot/output',
        'description': 'Fast BiLSTM chatbot',
        'category': 'lite'
    },
    'lab_09_lite': {
        'name': 'Time Series Forecasting',
        'script': 'labs_lite/lab_09_time_series/time_series_demo_lite.py',
        'output_dir': 'labs_lite/lab_09_time_series/output',
        'description': 'Quick LSTM/GRU forecasting',
        'category': 'lite'
    },
    'lab_10_lite': {
        'name': 'Sequence to Sequence',
        'script': 'labs_lite/lab_10_seq2seq/seq2seq_demo_lite.py',
        'output_dir': 'labs_lite/lab_10_seq2seq/output',
        'description': 'Fast encoder-decoder with attention',
        'category': 'lite'
    }
}

DOCS_CONFIG = {
    'readme': {
        'name': 'Main README',
        'file': 'README.md',
        'description': 'Complete project documentation and overview'
    },
    'quick_setup': {
        'name': 'Dataset Setup Guide',
        'file': 'docs/DATASET_SETUP_GUIDE.md',
        'description': 'Step-by-step guide for downloading and setting up datasets'
    },
    'installation': {
        'name': 'Installation Guide',
        'file': 'docs/INSTALLATION_GUIDE.md',
        'description': 'Step-by-step installation instructions'
    },
    'getting_started': {
        'name': 'Getting Started',
        'file': 'docs/GETTING_STARTED.md',
        'description': 'Quick start guide for beginners'
    },
    'datasets': {
        'name': 'Datasets Guide',
        'file': 'docs/DATASETS.md',
        'description': 'Information about datasets used in labs'
    },
    'lightweight_labs': {
        'name': 'Lightweight Labs',
        'file': 'labs_lite/README.md',
        'description': 'Guide to faster, lightweight lab versions'
    },
    'web_interface': {
        'name': 'Web Interface Guide',
        'file': 'docs/WEB_INTERFACE_GUIDE.md',
        'description': 'How to use this web interface'
    },
    'advanced_tab': {
        'name': 'Advanced Tab Guide',
        'file': 'docs/ADVANCED_TAB_GUIDE.md',
        'description': 'UI-based setup and management tasks'
    }
}

# ── Dataset requirements ──────────────────────────────────────────────────────
# check_type: 'file_exists' | 'dir_not_empty'
DATASET_REQUIREMENTS = {
    # Full labs
    'lab_01': {
        'check': 'labs_full/lab_01_image_processing/data/sample_images',
        'check_type': 'dir_not_empty',
        'kaggle': 'ashishjangra27/face-mask-12k-images-dataset',
        'download_to': 'labs_full/lab_01_image_processing/data/sample_images',
        'size': '~1 GB',
    },
    'lab_02': {
        'check': 'labs_full/lab_02_cifar10_classifiers/data/cifar-10-batches-py',
        'check_type': 'dir_not_empty',
        'auto': True,
        'auto_note': 'CIFAR-10 downloads automatically when the lab runs',
    },
    'lab_03': {
        'check': 'labs_full/lab_03_batchnorm_dropout/data/cifar-10-batches-py',
        'check_type': 'dir_not_empty',
        'auto': True,
        'auto_note': 'CIFAR-10 downloads automatically when the lab runs',
    },
    'lab_04': {
        'check': 'labs_full/lab_04_labeling_tools/data/practice_images',
        'check_type': 'dir_not_empty',
        'kaggle': 'andrewmvd/road-sign-detection',
        'download_to': 'labs_full/lab_04_labeling_tools/data/practice_images',
        'size': '~100 MB',
    },
    'lab_05': {
        'check': 'labs_full/lab_05_segmentation/data/pets',
        'check_type': 'dir_not_empty',
        'kaggle': 'tanlikesmath/the-oxfordiiit-pet-dataset',
        'download_to': 'labs_full/lab_05_segmentation/data/pets',
        'size': '~800 MB',
    },
    'lab_06': {
        'check': 'labs_full/lab_06_object_detection/data/voc',
        'check_type': 'dir_not_empty',
        'kaggle': 'huanghanchina/pascal-voc-2012',
        'download_to': 'labs_full/lab_06_object_detection/data/voc',
        'size': '~3.6 GB',
    },
    'lab_07': {
        'check': 'labs_full/lab_07_image_captioning/data/flickr8k/captions.txt',
        'check_type': 'file_exists',
        'kaggle': 'adityajn105/flickr8k',
        'download_to': 'labs_full/lab_07_image_captioning/data/flickr8k',
        'size': '~1 GB',
    },
    'lab_08': {
        'check': 'labs_full/lab_08_chatbot/data/cornell',
        'check_type': 'dir_not_empty',
        'kaggle': 'rajathmc/cornell-moviedialog-corpus',
        'download_to': 'labs_full/lab_08_chatbot/data/cornell',
        'size': '~10 MB',
    },
    'lab_09': {
        'check': 'labs_full/lab_09_time_series/data/energy',
        'check_type': 'dir_not_empty',
        'kaggle': 'robikscube/hourly-energy-consumption',
        'download_to': 'labs_full/lab_09_time_series/data/energy',
        'size': '~50 MB',
    },
    'lab_10': {
        'check': 'labs_full/lab_10_seq2seq/data/translation/en-fr.csv',
        'check_type': 'file_exists',
        'kaggle': 'dhruvildave/en-fr-translation-dataset',
        'download_to': 'labs_full/lab_10_seq2seq/data/translation',
        'size': '~50 MB',
    },
    # Lite labs
    'lab_01_lite': {
        'check': 'labs_lite/lab_01_image_processing/data/sample_images',
        'check_type': 'dir_not_empty',
        'kaggle': 'ashishjangra27/face-mask-12k-images-dataset',
        'download_to': 'labs_lite/lab_01_image_processing/data/sample_images',
        'size': '~1 GB',
    },
    'lab_02_lite': {
        'check': 'labs_lite/lab_02_cifar10_classifiers/data/cifar-10-batches-py',
        'check_type': 'dir_not_empty',
        'auto': True,
        'auto_note': 'CIFAR-10 downloads automatically when the lab runs',
    },
    'lab_03_lite': {
        'check': 'labs_lite/lab_03_batchnorm_dropout/data/cifar-10-batches-py',
        'check_type': 'dir_not_empty',
        'auto': True,
        'auto_note': 'CIFAR-10 downloads automatically when the lab runs',
    },
    'lab_04_lite': {
        'check': 'labs_lite/lab_04_labeling_tools/data/practice_images',
        'check_type': 'dir_not_empty',
        'kaggle': 'andrewmvd/road-sign-detection',
        'download_to': 'labs_lite/lab_04_labeling_tools/data/practice_images',
        'size': '~100 MB',
    },
    'lab_05_lite': {
        'check': 'labs_lite/lab_05_segmentation/data/pets',
        'check_type': 'dir_not_empty',
        'kaggle': 'tanlikesmath/the-oxfordiiit-pet-dataset',
        'download_to': 'labs_lite/lab_05_segmentation/data/pets',
        'size': '~800 MB',
    },
    'lab_06_lite': {
        'check': 'labs_lite/lab_06_object_detection/data/voc',
        'check_type': 'dir_not_empty',
        'kaggle': 'huanghanchina/pascal-voc-2012',
        'download_to': 'labs_lite/lab_06_object_detection/data/voc',
        'size': '~3.6 GB',
    },
    'lab_07_lite': {
        'check': 'labs_lite/lab_07_image_captioning/data/flickr8k/captions.txt',
        'check_type': 'file_exists',
        'kaggle': 'adityajn105/flickr8k',
        'download_to': 'labs_lite/lab_07_image_captioning/data/flickr8k',
        'size': '~1 GB',
    },
    'lab_08_lite': {
        'check': 'labs_lite/lab_08_chatbot/data/cornell',
        'check_type': 'dir_not_empty',
        'kaggle': 'rajathmc/cornell-moviedialog-corpus',
        'download_to': 'labs_lite/lab_08_chatbot/data/cornell',
        'size': '~10 MB',
    },
    'lab_09_lite': {
        'check': 'labs_lite/lab_09_time_series/data/energy',
        'check_type': 'dir_not_empty',
        'kaggle': 'robikscube/hourly-energy-consumption',
        'download_to': 'labs_lite/lab_09_time_series/data/energy',
        'size': '~50 MB',
    },
    'lab_10_lite': {
        'check': 'labs_lite/lab_10_seq2seq/data/translation/en-fr.csv',
        'check_type': 'file_exists',
        'kaggle': 'dhruvildave/en-fr-translation-dataset',
        'download_to': 'labs_lite/lab_10_seq2seq/data/translation',
        'size': '~50 MB',
    },
}

_IGNORE_NAMES = {'.DS_Store', '.gitkeep', '.gitignore', 'Thumbs.db'}


def check_dataset_ready(lab_id):
    """Return (ready: bool, auto: bool)."""
    req = DATASET_REQUIREMENTS.get(lab_id)
    if not req:
        return False, False
    auto = req.get('auto', False)
    path = BASE_DIR / req['check']
    if req['check_type'] == 'file_exists':
        ready = path.is_file()
    else:  # dir_not_empty
        ready = (
            path.is_dir()
            and any(f.name not in _IGNORE_NAMES for f in path.iterdir())
        )
    return ready, auto


# ── In-memory process state ───────────────────────────────────────────────────
# Store running processes
running_processes = {}
process_outputs = {}
setup_task_statuses = {}
setup_task_processes = {}
setup_task_outputs = {}
download_statuses = {}    # lab_id → {status, start_time, ...}
download_outputs = {}     # lab_id → [lines]
download_processes = {}   # lab_id → subprocess.Popen

@app.route('/')
def index():
    """Main page with lab selection"""
    return render_template('index.html', labs=LABS_CONFIG, docs=DOCS_CONFIG)

@app.route('/docs')
def docs_page():
    """Documentation page"""
    return render_template('docs.html', docs=DOCS_CONFIG)

@app.route('/api/doc/<doc_id>')
def get_doc(doc_id):
    """Get documentation content"""
    if doc_id not in DOCS_CONFIG:
        return jsonify({'error': 'Document not found'}), 404
    
    doc = DOCS_CONFIG[doc_id]
    doc_path = BASE_DIR / doc['file']
    
    if not doc_path.exists():
        return jsonify({'error': 'Document file not found'}), 404
    
    try:
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return jsonify({
            'name': doc['name'],
            'content': content,
            'description': doc['description']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/lab/<lab_id>')
def lab_page(lab_id):
    """Individual lab page"""
    if lab_id not in LABS_CONFIG:
        return "Lab not found", 404
    
    lab = LABS_CONFIG[lab_id]
    return render_template('lab.html', lab_id=lab_id, lab=lab)

@app.route('/api/run/<lab_id>', methods=['POST'])
def run_lab(lab_id):
    """Run a lab program"""
    if lab_id not in LABS_CONFIG:
        return jsonify({'error': 'Lab not found'}), 404
    
    if lab_id in running_processes and running_processes[lab_id].poll() is None:
        return jsonify({'error': 'Lab is already running'}), 400
    
    lab = LABS_CONFIG[lab_id]
    script_path = BASE_DIR / lab['script']
    
    if not script_path.exists():
        return jsonify({'error': 'Script not found'}), 404
    
    # Initialize output storage
    process_outputs[lab_id] = {
        'stdout': [],
        'stderr': [],
        'status': 'running',
        'start_time': datetime.now().isoformat()
    }
    
    # Run the script in a separate thread
    def run_script():
        try:
            process = subprocess.Popen(
                ['python', str(script_path)],
                cwd=str(script_path.parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            running_processes[lab_id] = process
            
            # Read output line by line
            for line in process.stdout:
                process_outputs[lab_id]['stdout'].append(line.strip())
            
            for line in process.stderr:
                process_outputs[lab_id]['stderr'].append(line.strip())
            
            process.wait()
            
            process_outputs[lab_id]['status'] = 'completed' if process.returncode == 0 else 'failed'
            process_outputs[lab_id]['return_code'] = process.returncode
            process_outputs[lab_id]['end_time'] = datetime.now().isoformat()
            
        except Exception as e:
            process_outputs[lab_id]['status'] = 'error'
            process_outputs[lab_id]['error'] = str(e)
    
    thread = threading.Thread(target=run_script)
    thread.daemon = True
    thread.start()
    
    return jsonify({'message': 'Lab started', 'lab_id': lab_id})

@app.route('/api/status/<lab_id>')
def get_status(lab_id):
    """Get the status and output of a running lab"""
    if lab_id not in process_outputs:
        return jsonify({'status': 'not_started'})
    
    return jsonify(process_outputs[lab_id])

@app.route('/api/outputs/<lab_id>')
def get_outputs(lab_id):
    """Get list of output files for a lab"""
    if lab_id not in LABS_CONFIG:
        return jsonify({'error': 'Lab not found'}), 404
    
    lab = LABS_CONFIG[lab_id]
    output_dir = BASE_DIR / lab['output_dir']
    
    if not output_dir.exists():
        return jsonify({'files': []})
    
    files = []
    for file_path in output_dir.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.txt', '.json']:
            relative_path = file_path.relative_to(output_dir)
            files.append({
                'name': file_path.name,
                'path': str(relative_path),
                'type': file_path.suffix[1:],
                'size': file_path.stat().st_size
            })
    
    return jsonify({'files': files})

@app.route('/api/output/<lab_id>/<path:filename>')
def serve_output(lab_id, filename):
    """Serve output files"""
    if lab_id not in LABS_CONFIG:
        return "Lab not found", 404
    
    lab = LABS_CONFIG[lab_id]
    output_dir = BASE_DIR / lab['output_dir']
    
    return send_from_directory(output_dir, filename)

@app.route('/api/stop/<lab_id>', methods=['POST'])
def stop_lab(lab_id):
    """Stop a running lab"""
    if lab_id in running_processes:
        process = running_processes[lab_id]
        if process.poll() is None:
            process.terminate()
            process_outputs[lab_id]['status'] = 'stopped'
            return jsonify({'message': 'Lab stopped'})
    
    return jsonify({'error': 'Lab not running'}), 400

@app.route('/api/setup/<task_id>', methods=['POST'])
def run_setup_task(task_id):
    """Run setup tasks like installing requirements or downloading datasets"""
    
    SETUP_TASKS = {
        'install-full': {
            'command': f'{sys.executable} -m pip install -r requirements.txt',
            'description': 'Installing full requirements',
            'interactive': False
        },
        'install-minimal': {
            'command': f'{sys.executable} -m pip install -r requirements-minimal.txt',
            'description': 'Installing minimal requirements',
            'interactive': False
        },
        'download-datasets': {
            'command': 'bash download_datasets_interactive.sh',
            'description': 'Downloading all datasets',
            'interactive': False
        },
        'download-interactive': {
            'command': 'bash download_datasets_interactive.sh',
            'description': 'Interactive dataset download',
            'interactive': True
        },
        'run-all-full': {
            'command': 'bash labs_full/run_all_labs.sh',
            'description': 'Running all full labs',
            'interactive': False
        },
        'run-all-lite': {
            'command': 'bash labs_lite/run_all_labs.sh',
            'description': 'Running all lite labs',
            'interactive': False
        }
    }
    
    if task_id not in SETUP_TASKS:
        return jsonify({'error': 'Invalid task ID'}), 400
    
    # Check if task is already running
    if task_id in setup_task_statuses and setup_task_statuses[task_id]['status'] == 'Running':
        return jsonify({'error': 'Task is already running'}), 400
    
    task = SETUP_TASKS[task_id]
    
    # Initialize task status
    setup_task_statuses[task_id] = {
        'status': 'Running',
        'start_time': datetime.now().isoformat(),
        'command': task['command']
    }
    
    try:
        # Initialize output storage
        setup_task_outputs[task_id] = {
            'stdout': [],
            'stderr': [],
            'combined': []
        }
        
        # Run the command in the background
        def run_command():
            try:
                process = subprocess.Popen(
                    task['command'],
                    shell=True,
                    cwd=str(BASE_DIR),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1
                )

                setup_task_processes[task_id] = process
                setup_task_outputs[task_id]['combined'].append(
                    f'[SYSTEM] Task started: {task["command"]}'
                )

                # Read stdout and stderr concurrently to avoid pipe deadlock
                def read_stream(stream, prefix):
                    for line in stream:
                        line = line.rstrip('\n')
                        if line:
                            setup_task_outputs[task_id][
                                'stdout' if prefix == 'OUT' else 'stderr'
                            ].append(line)
                            setup_task_outputs[task_id]['combined'].append(
                                f'[{prefix}] {line}'
                            )

                t_out = threading.Thread(
                    target=read_stream, args=(process.stdout, 'OUT'), daemon=True
                )
                t_err = threading.Thread(
                    target=read_stream, args=(process.stderr, 'ERR'), daemon=True
                )
                t_out.start()
                t_err.start()
                t_out.join()
                t_err.join()
                process.wait()

                # Update status based on return code
                if setup_task_statuses[task_id]['status'] == 'Stopped':
                    pass  # Already marked stopped by stop endpoint
                elif process.returncode == 0:
                    setup_task_statuses[task_id]['status'] = 'Completed'
                    setup_task_outputs[task_id]['combined'].append(
                        '[SYSTEM] Task completed successfully'
                    )
                else:
                    setup_task_statuses[task_id]['status'] = 'Failed'
                    setup_task_outputs[task_id]['combined'].append(
                        f'[SYSTEM] Task failed with exit code {process.returncode}'
                    )

                setup_task_statuses[task_id]['end_time'] = datetime.now().isoformat()
                setup_task_statuses[task_id]['return_code'] = process.returncode

            except Exception as e:
                print(f"Error running task {task_id}: {e}")
                setup_task_statuses[task_id]['status'] = 'Error'
                setup_task_statuses[task_id]['error'] = str(e)
                setup_task_outputs[task_id]['combined'].append(f'[ERROR] {str(e)}')
        
        thread = threading.Thread(target=run_command)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'message': f'{task["description"]} started successfully',
            'command': task['command'],
            'interactive': task['interactive'],
            'note': 'Check your terminal for progress updates'
        })
    except Exception as e:
        setup_task_statuses[task_id]['status'] = 'Error'
        return jsonify({'error': str(e)}), 500

@app.route('/api/setup/statuses')
def get_setup_statuses():
    """Get status of all setup tasks"""
    # Return status for all known tasks
    all_statuses = {}
    task_ids = ['install-full', 'install-minimal', 'download-datasets',
                'download-interactive', 'run-all-full', 'run-all-lite']
    
    for task_id in task_ids:
        if task_id in setup_task_statuses:
            all_statuses[task_id] = setup_task_statuses[task_id]
        else:
            all_statuses[task_id] = {'status': 'Ready'}
    
    return jsonify(all_statuses)

@app.route('/api/setup/<task_id>/output')
def get_task_output(task_id):
    """Get real-time output for a specific task"""
    if task_id not in setup_task_outputs:
        return jsonify({'output': [], 'status': 'not_started'})
    
    status = setup_task_statuses.get(task_id, {}).get('status', 'Unknown')
    return jsonify({
        'output': setup_task_outputs[task_id]['combined'],
        'status': status
    })

@app.route('/api/setup/<task_id>/stop', methods=['POST'])
def stop_task(task_id):
    """Stop a running setup task"""
    if task_id not in setup_task_processes:
        return jsonify({'error': 'Task not running'}), 400
    
    process = setup_task_processes[task_id]
    if process.poll() is None:
        process.terminate()
        setup_task_statuses[task_id]['status'] = 'Stopped'
        setup_task_outputs[task_id]['combined'].append('[SYSTEM] Task stopped by user')
        return jsonify({'message': 'Task stopped successfully'})
    
    return jsonify({'error': 'Task already completed'}), 400

@app.route('/api/dataset-status')
def get_all_dataset_statuses():
    """Return dataset readiness for every lab."""
    result = {}
    for lab_id in LABS_CONFIG:
        ready, auto = check_dataset_ready(lab_id)
        req = DATASET_REQUIREMENTS.get(lab_id, {})
        dl_status = download_statuses.get(lab_id, {}).get('status', 'idle')
        result[lab_id] = {
            'ready': ready,
            'auto': auto,
            'auto_note': req.get('auto_note', ''),
            'size': req.get('size', ''),
            'downloading': dl_status == 'Running',
            'download_status': dl_status,
        }
    return jsonify(result)


@app.route('/api/download/<lab_id>', methods=['POST'])
def download_dataset(lab_id):
    """Kick off a Kaggle download for a single lab (non-blocking)."""
    if lab_id not in LABS_CONFIG:
        return jsonify({'error': 'Lab not found'}), 404

    req = DATASET_REQUIREMENTS.get(lab_id)
    if not req:
        return jsonify({'error': 'No dataset config for this lab'}), 400
    if req.get('auto'):
        return jsonify({'error': 'Dataset auto-downloads when the lab runs'}), 400

    # Prevent parallel download of the same lab
    if download_statuses.get(lab_id, {}).get('status') == 'Running':
        return jsonify({'error': 'Download already in progress'}), 400

    # Already have the data
    ready, _ = check_dataset_ready(lab_id)
    if ready:
        return jsonify({'error': 'Dataset already available'}), 400

    dest = BASE_DIR / req['download_to']
    dest.mkdir(parents=True, exist_ok=True)

    download_statuses[lab_id] = {'status': 'Running', 'start_time': datetime.now().isoformat()}
    download_outputs[lab_id] = []

    def run_download():
        try:
            cmd = f'kaggle datasets download "{req["kaggle"]}" -p "{dest}" --unzip'
            process = subprocess.Popen(
                cmd, shell=True, cwd=str(BASE_DIR),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1
            )
            download_processes[lab_id] = process
            download_outputs[lab_id].append(
                f'[SYSTEM] Downloading {req["kaggle"]} ({req.get("size", "")})…'
            )

            def read_stream(stream, prefix):
                for line in stream:
                    line = line.rstrip('\n')
                    if line:
                        download_outputs[lab_id].append(f'[{prefix}] {line}')

            t_out = threading.Thread(target=read_stream, args=(process.stdout, 'OUT'), daemon=True)
            t_err = threading.Thread(target=read_stream, args=(process.stderr, 'ERR'), daemon=True)
            t_out.start(); t_err.start()
            t_out.join(); t_err.join()
            process.wait()

            if download_statuses[lab_id]['status'] == 'Stopped':
                pass
            elif process.returncode == 0:
                download_statuses[lab_id]['status'] = 'Completed'
                download_outputs[lab_id].append('[SYSTEM] Dataset downloaded successfully')
            else:
                download_statuses[lab_id]['status'] = 'Failed'
                download_outputs[lab_id].append(
                    f'[SYSTEM] Download failed (exit code {process.returncode})'
                )
            download_statuses[lab_id]['end_time'] = datetime.now().isoformat()
        except Exception as e:
            download_statuses[lab_id]['status'] = 'Error'
            download_outputs[lab_id].append(f'[ERROR] {e}')

    threading.Thread(target=run_download, daemon=True).start()
    return jsonify({'message': 'Download started', 'lab_id': lab_id})


@app.route('/api/download/<lab_id>/output')
def get_download_output(lab_id):
    """Real-time output lines for an in-progress download."""
    if lab_id not in LABS_CONFIG:
        return jsonify({'error': 'Lab not found'}), 404
    ready, auto = check_dataset_ready(lab_id)
    return jsonify({
        'output': download_outputs.get(lab_id, []),
        'status': download_statuses.get(lab_id, {}).get('status', 'idle'),
        'ready': ready,
    })


@app.route('/api/download/<lab_id>/stop', methods=['POST'])
def stop_download(lab_id):
    """Cancel a running download."""
    proc = download_processes.get(lab_id)
    if not proc or proc.poll() is not None:
        return jsonify({'error': 'No active download for this lab'}), 400
    proc.terminate()
    download_statuses[lab_id]['status'] = 'Stopped'
    download_outputs[lab_id].append('[SYSTEM] Download cancelled by user')
    return jsonify({'message': 'Download stopped'})


if __name__ == '__main__':
    print("=" * 60)
    print("Deep Learning Labs - Web Interface")
    print("=" * 60)
    print("\nStarting server...")
    print("Open your browser and navigate to: http://localhost:5001")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5001)


