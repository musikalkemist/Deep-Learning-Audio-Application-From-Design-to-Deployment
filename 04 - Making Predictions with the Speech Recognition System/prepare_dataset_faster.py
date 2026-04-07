"""
Parallelized MFCC Feature Extraction Pipeline for Audio Datasets.

This module automates the conversion of raw .wav audio files into Mel-frequency 
Cepstral Coefficients (MFCCs) for deep learning applications. It is optimized 
for high-throughput processing of large datasets (e.g., Google Speech Commands).

Key Features:
- Parallel Processing: Leverages `ProcessPoolExecutor` to distribute DSP 
  workloads across all available CPU cores, bypassing the Python GIL.
- Uniform Input Shaping: Enforces consistent sample lengths through 
  truncation to ensure compatibility with downstream neural network architectures.
- Data Serialization: Aggregates features, labels, and file mappings into a 
  single JSON artifact for efficient model ingestion.

Note on Processing:
- Defaults to Librosa's standard resampling (22,050 Hz) unless sr=None is specified.
- Current configuration assumes a target duration of 1 second at 22,050 Hz.

Usage:
    Adjust DATASET_PATH and CLASS_FOLDER as needed and execute the script.
    Ensure the worker function remains in the global scope for pickling compatibility.
"""

import librosa
import numpy as np
import os
import json
from functools import partial
from concurrent.futures.process import BrokenProcessPool, ProcessPoolExecutor
from pathlib import Path

DATASET_PATH = "Speech_Commands_dataset"
CLASS_FOLDER = Path(__file__).parent.name # Dynamically determine the class folder based on the script's location
JSON_PATH = f"{CLASS_FOLDER}/data.json"
SAMPLES_TO_CONSIDER = 22050

# Processing logic wrapper for a single file, designed for parallel execution
def process_single_file(file_path, label_index, num_mfcc, n_fft, hop_length):
    """
    Worker function to process a single audio file.
    Returns a tuple of (MFCCs, label, file_path) or None if invalid.
    """
    try:
        # Load audio file and slice it to ensure length consistency among different files
        signal, sample_rate = librosa.load(file_path)
        
        # drop audio files with less than pre-decided number of samples
        if len(signal) >= SAMPLES_TO_CONSIDER:

            # ensure consistency of the length of the signal
            signal = signal[:SAMPLES_TO_CONSIDER]

            # Extract MFCCs
            mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                                    hop_length=hop_length)

            # Return as list for JSON serialization compatibility
            return mfccs.T.tolist(), label_index, file_path
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    return None

def preprocess_dataset(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512):
    # Initialize the data container
    data = {
        "mapping": [],
        "labels": [],
        "MFCCs": [],
        "files": []
    }

    # MAIN EXECUTION LOGIC ================================================
    # 1. Gather all file paths and label indices first
    list_of_paths = []
    list_of_labels = []
    mapping = []

    # 1. Path Gathering with verification
    try:
        for i, (dirpath, _, filenames) in enumerate(os.walk(DATASET_PATH)):
            if dirpath != DATASET_PATH:
                label = os.path.basename(dirpath)
                mapping.append(label)
                for f in filenames:
                    if f.endswith(".wav"):
                        list_of_paths.append(os.path.join(dirpath, f))
                        list_of_labels.append(i - 1)

        print(f"Found {len(list_of_paths)} files across {len(mapping)} classes.")
    except Exception as e:
        print(f"CRITICAL: Failed to traverse directory: {e}")
        return

    # 2. Execute in parallel
    try:
        print(f"Starting parallel execution with {os.cpu_count()} workers...")
        with ProcessPoolExecutor() as executor:
            worker = partial(
                process_single_file, 
                num_mfcc=13, 
                n_fft=2048, 
                hop_length=512
            )
            
            # map returns results in the order of the input lists
            results = list(executor.map(worker, list_of_paths, list_of_labels))
        print("Parallel processing complete.")
    except BrokenProcessPool as e:
        print(f"CRITICAL: The process pool terminated unexpectedly: {e}")
        # This usually happens if a child process is killed by the OS (e.g., Out of Memory)
    except Exception as e:
        print(f"An unexpected error occurred during parallel execution: {e}")
    
    # 3. Filter out None results and unpack
    for res in results:
        if res:
            mfcc, label_idx, path = res
            data["MFCCs"].append(mfcc)
            data["labels"].append(label_idx)
            data["files"].append(path)
            data["mapping"] = mapping

    if not data["MFCCs"]:
        print("WARNING: No data was processed successfully. Check file paths and SAMPLES_TO_CONSIDER.")
        return

    try:
        with open(json_path, "w") as fp:
            json.dump(data, fp, indent=4)
        print(f"SUCCESS: Dataset saved to {json_path}")
    except IOError as e:
        print(f"CRITICAL: Failed to write JSON to disk: {e}")

if __name__ == "__main__":
    preprocess_dataset(DATASET_PATH, JSON_PATH)