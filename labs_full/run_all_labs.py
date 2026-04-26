#!/usr/bin/env python3
"""
Run all full labs sequentially using the same Python interpreter as the caller.
Streams each lab's output live to stdout so the web UI can capture it.
"""

import sys
import subprocess
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent  # labs_full/

LABS = [
    ('01', 'Basic Image Processing',      'lab_01_image_processing',   'image_processing.py'),
    ('02', 'CIFAR-10 Classifiers',         'lab_02_cifar10_classifiers', 'cifar10_classifiers.py'),
    ('03', 'Batch Normalization & Dropout','lab_03_batchnorm_dropout',   'batchnorm_dropout_study.py'),
    ('04', 'Labeling Tools Demo',          'lab_04_labeling_tools',      'labeling_demo.py'),
    ('05', 'Image Segmentation',           'lab_05_segmentation',        'segmentation_demo.py'),
    ('06', 'Object Detection',             'lab_06_object_detection',    'object_detection_demo.py'),
    ('07', 'Image Captioning',             'lab_07_image_captioning',    'image_captioning_demo.py'),
    ('08', 'Chatbot',                      'lab_08_chatbot',             'chatbot_demo.py'),
    ('09', 'Time Series Forecasting',      'lab_09_time_series',         'time_series_demo.py'),
    ('10', 'Sequence to Sequence',         'lab_10_seq2seq',             'seq2seq_demo.py'),
]

SEP = '=' * 60

def run_lab(num, name, lab_dir, script):
    script_path = BASE_DIR / lab_dir / script
    print(f'\n{SEP}', flush=True)
    print(f'  LAB {num}: {name}', flush=True)
    print(SEP, flush=True)

    if not script_path.is_file():
        print(f'⚠  Script not found: {script_path}', flush=True)
        return True  # skip, don't abort

    import os
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env['TQDM_DISABLE'] = '1'

    proc = subprocess.Popen(
        [sys.executable, '-u', script],
        cwd=str(script_path.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,   # merge stderr into stdout for unified stream
        text=True,
        bufsize=1,
        env=env,
    )

    for line in proc.stdout:
        print(line, end='', flush=True)
    proc.wait()

    if proc.returncode == 0:
        print(f'\n✓ Lab {num} completed successfully', flush=True)
        return True
    else:
        print(f'\n✗ Lab {num} failed with exit code {proc.returncode}', flush=True)
        return False


def main():
    print(SEP, flush=True)
    print('  DEEP LEARNING LABS — Run All Full Labs', flush=True)
    print(f'  Python: {sys.executable}', flush=True)
    print(SEP, flush=True)

    t_start = time.time()
    passed, failed = [], []

    for num, name, lab_dir, script in LABS:
        ok = run_lab(num, name, lab_dir, script)
        (passed if ok else failed).append(f'Lab {num}: {name}')

    elapsed = int(time.time() - t_start)
    mins, secs = divmod(elapsed, 60)

    print(f'\n{SEP}', flush=True)
    print('  SUMMARY', flush=True)
    print(SEP, flush=True)
    print(f'  Passed : {len(passed)}', flush=True)
    print(f'  Failed : {len(failed)}', flush=True)
    print(f'  Time   : {mins}m {secs}s', flush=True)

    if failed:
        print('\nFailed labs:', flush=True)
        for f in failed:
            print(f'  - {f}', flush=True)
        sys.exit(1)
    else:
        print('\nAll full labs completed successfully!', flush=True)


if __name__ == '__main__':
    main()
