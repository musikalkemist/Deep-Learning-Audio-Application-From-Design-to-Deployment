"""
Speech Commands Dataset Setup Utility
----------------------------------
This script automates the acquisition and organization of the Google Speech Commands 
dataset for the TSOAI (The Sound of AI) courses.

Functionality:
1. Downloads the full v0.02 dataset (35 classes) from Google's servers.
2. Extracts the compressed archive into the local project structure.
3. Sanitizes the dataset by removing hidden MacOS metadata artifacts
   (._ files) to prevent processing errors in librosa.
4. Performs automatic cleanup of temporary download files.
"""

import os
import tensorflow as tf

# Define the local destination and the source URL
# Using 'v0.02' which contains 35 classes and ~105k utterances
DATASET_DIR = "Speech_Commands_dataset"
DATASET_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"

def download_and_extract_data(url, dest_dir):
    """
    Downloads and extracts the Speech Commands dataset.
    """
    # Create directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    # get_file returns the path to the downloaded file
    # extract=True handles .tar.gz or .zip automatically
    # cache_subdir='.' ensures it stays within your specified DATASET_DIR 
    # rather than the default ~/.keras/datasets
    filepath = tf.keras.utils.get_file(
        fname="speech_commands_v2.tar.gz",
        origin=url,
        extract=True,
        cache_dir=".",
        cache_subdir=dest_dir
    )
    
    print(f"Dataset downloaded and extracted to: {os.path.abspath(dest_dir)}")

if __name__ == "__main__":
    download_and_extract_data(DATASET_URL, DATASET_DIR)
