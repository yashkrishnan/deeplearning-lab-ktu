#!/usr/bin/env python3
"""
Smart dataset downloader — skips datasets that are already present.
Run from the project root or from web_interface/; BASE_DIR is resolved automatically.
"""

import os
import sys
import subprocess
from pathlib import Path

# ── Resolve project root ──────────────────────────────────────────────────────
# This file lives at <project_root>/web_interface/download_datasets_smart.py
BASE_DIR = Path(__file__).resolve().parent.parent

# ── Dataset requirements ──────────────────────────────────────────────────────
_IGNORE_NAMES = {'.DS_Store', '.gitkeep', '.gitignore', 'Thumbs.db'}

DATASET_REQUIREMENTS = {
    # ── Full labs ─────────────────────────────────────────────────────────────
    'lab_01': {
        'label': 'Lab 01 Full — Face Mask Dataset',
        'check': 'labs_full/lab_01_image_processing/data/sample_images',
        'check_type': 'dir_not_empty',
        'kaggle': 'ashishjangra27/face-mask-12k-images-dataset',
        'download_to': 'labs_full/lab_01_image_processing/data/sample_images',
        'size': '~1 GB',
    },
    'lab_02': {
        'label': 'Lab 02 Full — CIFAR-10',
        'check': 'labs_full/lab_02_cifar10_classifiers/data/cifar-10-batches-py',
        'check_type': 'dir_not_empty',
        'auto': True,
        'auto_note': 'Auto-downloads when the lab runs (via torchvision)',
    },
    'lab_03': {
        'label': 'Lab 03 Full — CIFAR-10',
        'check': 'labs_full/lab_03_batchnorm_dropout/data/cifar-10-batches-py',
        'check_type': 'dir_not_empty',
        'auto': True,
        'auto_note': 'Auto-downloads when the lab runs (via torchvision)',
    },
    'lab_04': {
        'label': 'Lab 04 Full — Road Sign Detection',
        'check': 'labs_full/lab_04_labeling_tools/data/practice_images',
        'check_type': 'dir_not_empty',
        'kaggle': 'andrewmvd/road-sign-detection',
        'download_to': 'labs_full/lab_04_labeling_tools/data/practice_images',
        'size': '~100 MB',
    },
    'lab_05': {
        'label': 'Lab 05 Full — Oxford-IIIT Pets',
        'check': 'labs_full/lab_05_segmentation/data/pets',
        'check_type': 'dir_not_empty',
        'kaggle': 'tanlikesmath/the-oxfordiiit-pet-dataset',
        'download_to': 'labs_full/lab_05_segmentation/data/pets',
        'size': '~800 MB',
    },
    'lab_06': {
        'label': 'Lab 06 Full — Pascal VOC 2012',
        'check': 'labs_full/lab_06_object_detection/data/voc',
        'check_type': 'dir_not_empty',
        'kaggle': 'huanghanchina/pascal-voc-2012',
        'download_to': 'labs_full/lab_06_object_detection/data/voc',
        'size': '~3.6 GB',
    },
    'lab_07': {
        'label': 'Lab 07 Full — Flickr8k',
        'check': 'labs_full/lab_07_image_captioning/data/flickr8k/captions.txt',
        'check_type': 'file_exists',
        'kaggle': 'adityajn105/flickr8k',
        'download_to': 'labs_full/lab_07_image_captioning/data/flickr8k',
        'size': '~1 GB',
    },
    'lab_08': {
        'label': 'Lab 08 Full — Cornell Movie Dialogs',
        'check': 'labs_full/lab_08_chatbot/data/cornell',
        'check_type': 'dir_not_empty',
        'kaggle': 'rajathmc/cornell-moviedialog-corpus',
        'download_to': 'labs_full/lab_08_chatbot/data/cornell',
        'size': '~10 MB',
    },
    'lab_09': {
        'label': 'Lab 09 Full — Hourly Energy Consumption',
        'check': 'labs_full/lab_09_time_series/data/energy',
        'check_type': 'dir_not_empty',
        'kaggle': 'robikscube/hourly-energy-consumption',
        'download_to': 'labs_full/lab_09_time_series/data/energy',
        'size': '~50 MB',
    },
    'lab_10': {
        'label': 'Lab 10 Full — EN-FR Translation',
        'check': 'labs_full/lab_10_seq2seq/data/translation/en-fr.csv',
        'check_type': 'file_exists',
        'kaggle': 'dhruvildave/en-fr-translation-dataset',
        'download_to': 'labs_full/lab_10_seq2seq/data/translation',
        'size': '~50 MB',
    },
    # ── Lite labs ─────────────────────────────────────────────────────────────
    'lab_01_lite': {
        'label': 'Lab 01 Lite — Face Mask Dataset',
        'check': 'labs_lite/lab_01_image_processing/data/sample_images',
        'check_type': 'dir_not_empty',
        'kaggle': 'ashishjangra27/face-mask-12k-images-dataset',
        'download_to': 'labs_lite/lab_01_image_processing/data/sample_images',
        'size': '~1 GB',
    },
    'lab_02_lite': {
        'label': 'Lab 02 Lite — CIFAR-10',
        'check': 'labs_lite/lab_02_cifar10_classifiers/data/cifar-10-batches-py',
        'check_type': 'dir_not_empty',
        'auto': True,
        'auto_note': 'Auto-downloads when the lab runs (via torchvision)',
    },
    'lab_03_lite': {
        'label': 'Lab 03 Lite — CIFAR-10',
        'check': 'labs_lite/lab_03_batchnorm_dropout/data/cifar-10-batches-py',
        'check_type': 'dir_not_empty',
        'auto': True,
        'auto_note': 'Auto-downloads when the lab runs (via torchvision)',
    },
    'lab_04_lite': {
        'label': 'Lab 04 Lite — Road Sign Detection',
        'check': 'labs_lite/lab_04_labeling_tools/data/practice_images',
        'check_type': 'dir_not_empty',
        'kaggle': 'andrewmvd/road-sign-detection',
        'download_to': 'labs_lite/lab_04_labeling_tools/data/practice_images',
        'size': '~100 MB',
    },
    'lab_05_lite': {
        'label': 'Lab 05 Lite — Oxford-IIIT Pets',
        'check': 'labs_lite/lab_05_segmentation/data/pets',
        'check_type': 'dir_not_empty',
        'kaggle': 'tanlikesmath/the-oxfordiiit-pet-dataset',
        'download_to': 'labs_lite/lab_05_segmentation/data/pets',
        'size': '~800 MB',
    },
    'lab_06_lite': {
        'label': 'Lab 06 Lite — Pascal VOC 2012',
        'check': 'labs_lite/lab_06_object_detection/data/voc',
        'check_type': 'dir_not_empty',
        'kaggle': 'huanghanchina/pascal-voc-2012',
        'download_to': 'labs_lite/lab_06_object_detection/data/voc',
        'size': '~3.6 GB',
    },
    'lab_07_lite': {
        'label': 'Lab 07 Lite — Flickr8k',
        'check': 'labs_lite/lab_07_image_captioning/data/flickr8k/captions.txt',
        'check_type': 'file_exists',
        'kaggle': 'adityajn105/flickr8k',
        'download_to': 'labs_lite/lab_07_image_captioning/data/flickr8k',
        'size': '~1 GB',
    },
    'lab_08_lite': {
        'label': 'Lab 08 Lite — Cornell Movie Dialogs',
        'check': 'labs_lite/lab_08_chatbot/data/cornell',
        'check_type': 'dir_not_empty',
        'kaggle': 'rajathmc/cornell-moviedialog-corpus',
        'download_to': 'labs_lite/lab_08_chatbot/data/cornell',
        'size': '~10 MB',
    },
    'lab_09_lite': {
        'label': 'Lab 09 Lite — Hourly Energy Consumption',
        'check': 'labs_lite/lab_09_time_series/data/energy',
        'check_type': 'dir_not_empty',
        'kaggle': 'robikscube/hourly-energy-consumption',
        'download_to': 'labs_lite/lab_09_time_series/data/energy',
        'size': '~50 MB',
    },
    'lab_10_lite': {
        'label': 'Lab 10 Lite — EN-FR Translation',
        'check': 'labs_lite/lab_10_seq2seq/data/translation/en-fr.csv',
        'check_type': 'file_exists',
        'kaggle': 'dhruvildave/en-fr-translation-dataset',
        'download_to': 'labs_lite/lab_10_seq2seq/data/translation',
        'size': '~50 MB',
    },
}


def _is_ready(req: dict) -> bool:
    """Return True if the dataset is already present on disk."""
    path = BASE_DIR / req['check']
    if req['check_type'] == 'file_exists':
        return path.is_file()
    # dir_not_empty
    if not path.is_dir():
        return False
    return any(f.name not in _IGNORE_NAMES for f in path.iterdir())


def _download(lab_id: str, req: dict) -> bool:
    """
    Download a Kaggle dataset via the kaggle CLI.
    Returns True on success, False on failure.
    """
    dest = BASE_DIR / req['download_to']
    dest.mkdir(parents=True, exist_ok=True)

    cmd = [
        'kaggle', 'datasets', 'download',
        '--unzip',
        '-p', str(dest),
        req['kaggle'],
    ]
    print(f"  → Running: {' '.join(cmd)}", flush=True)

    result = subprocess.run(cmd, cwd=str(BASE_DIR))
    return result.returncode == 0


def main():
    print("=" * 60, flush=True)
    print("Smart Dataset Downloader", flush=True)
    print(f"Project root: {BASE_DIR}", flush=True)
    print("=" * 60, flush=True)

    skipped = []
    downloaded = []
    failed = []
    auto_skipped = []

    for lab_id, req in DATASET_REQUIREMENTS.items():
        label = req.get('label', lab_id)

        # Auto-download datasets (e.g. CIFAR-10 via torchvision)
        if req.get('auto'):
            auto_skipped.append(label)
            print(f"[AUTO]  {label}: {req.get('auto_note', 'downloads automatically')}", flush=True)
            continue

        if _is_ready(req):
            skipped.append(label)
            print(f"[SKIP]  {label}: already present", flush=True)
        else:
            size = req.get('size', '?')
            print(f"\n[DOWN]  {label} ({size})", flush=True)
            ok = _download(lab_id, req)
            if ok:
                downloaded.append(label)
                print(f"[OK]    {label}: download complete", flush=True)
            else:
                failed.append(label)
                print(f"[FAIL]  {label}: download failed — check kaggle credentials", flush=True)

    # Summary
    print("\n" + "=" * 60, flush=True)
    print("Summary", flush=True)
    print("=" * 60, flush=True)
    print(f"  Already present  : {len(skipped)}", flush=True)
    print(f"  Auto (skipped)   : {len(auto_skipped)}", flush=True)
    print(f"  Downloaded       : {len(downloaded)}", flush=True)
    print(f"  Failed           : {len(failed)}", flush=True)

    if failed:
        print("\nFailed datasets:", flush=True)
        for name in failed:
            print(f"  - {name}", flush=True)
        print("\nMake sure kaggle.json is set up: https://www.kaggle.com/docs/api", flush=True)
        sys.exit(1)
    else:
        print("\nAll datasets are ready!", flush=True)


if __name__ == '__main__':
    main()
