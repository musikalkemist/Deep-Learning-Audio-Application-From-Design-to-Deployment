"""
Speech Commands Dataset Setup Utility
---------------------------------------------------------

This script automates the acquisition and organization of the Google Speech Commands dataset for the TSOAI (The Sound of AI) courses,
optimized for specific subset extraction to save disk space and reduce processing overhead.

Functionality:
1. Orchestrated Download: Utilizes TensorFlow's `get_file` to fetch the complete v0.02 archive (35 classes) if not present locally.
2. Selective Extraction: Instead of a full unpack, it filters the tarball to extract only the user-defined TARGET_CLASSES (default: up, down, left, right).
3. State-Based Execution: Employs a status-check mechanism (Status 0-2) to avoid redundant downloads or extractions based on the current filesystem state.
4. Path Preservation: Maintains the original directory structure required by the TSOAI course pipelines.
Note: This script preserves the compressed archive (.tar.gz) after extraction for verification.
"""

import os
import tarfile
import tensorflow as tf

# Define the local destination and the source URL
# Using 'v0.02' which contains 35 classes and ~105k utterances
DATASET_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
DATASET_DIR = "Speech_Commands_dataset"
FILENAME = "speech_commands_v2.tar.gz"
TARGET_CLASSES = {'up', 'down', 'left', 'right'}

def check_downloaded_data(dest_dir, dataset_path):
    """
    Checks if the target classes are already present in the destination directory.
    If all target classes are found, it assumes the dataset is already downloaded and extracted.
    """
    existing_classes = set(os.listdir(dest_dir))
    if TARGET_CLASSES.issubset(existing_classes):
        print("Dataset already exists with target classes. Skipping download.")
        return 0
    elif os.path.exists(dataset_path):
        print("Dataset archive found but target classes are missing. Extracting...")
        return 1
    return 2

def download_and_extract_data(url, filename, dest_dir, target_classes=None):
    """
    Manages Speech Commands dataset download and extraction with checks to avoid redundant operations.
    """
    status = 2  # Default to needing download
    dataset_path = os.path.join(dest_dir, filename)
    print(f"Targeting archive file at: {os.path.abspath(dataset_path)}")
    
    # 1. Download only (extract=False)
    def download_dataset():
        """
        Downloads the Speech Commands dataset.
        """
        try:
            # `get_file` returns the path to the downloaded file
            # cache_subdir='.' ensures it stays within your specified DATASET_DIR 
            # rather than the default ~/.keras/datasets
            # if extract=True handles .tar.gz or .zip automatically,
            # but we set it to False here to control extraction ourselves.
            archive_path = tf.keras.utils.get_file(
                fname=filename,
                origin=url,
                extract=False,
                cache_dir=".",
                cache_subdir=dest_dir
            )
            return archive_path
        except Exception as e:
            print(f"Error during download: {e}")
            return None
    
    # 2. Selective extraction
    def extract_dataset(archive_path):
        """
        Extracts only the target classes from the downloaded archive.
        """
        try:
            with tarfile.open(archive_path, 'r:gz') as tar:
                # DEBUG: Print the first 5 filenames to verify internal structure
                # print(f"Sample paths in tar: {tar.getnames()[:5]}")

                # Filter members: only those in target folders and not the root background_noise
                members = [
                    m for m in tar.getmembers() 
                    if any(m.name.startswith(f"./{c}/") or m.name.startswith(f"{c}/")
                    for c in target_classes)
                ]

                # DEBUG: Check if we actually found matches
                # print(f"Matches found for target classes: {len(members)}")
                if not members:
                    print("Warning: No files matched the target class criteria.")
                    return False
                
                tar.extractall(path=dest_dir, members=members) 

            print(f"Dataset downloaded and extracted to: {os.path.abspath(dest_dir)}")
            return True
        
        except Exception as e:
            print(f"Error during extraction: {e}")
            return False

    # Create directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    else:
        status = check_downloaded_data(dest_dir, dataset_path)
    
    match status:
        # STATUS 0: Dataset already exists, no action needed
        case 0:
            print("Dataset already exists. Skipping download.")
            return
        # STATUS 1: Dataset archive exists but target classes are missing, proceed to extraction
        case 1:
            sucess = extract_dataset(dataset_path)
            if sucess:
                return
        # STATUS 2: Dataset archive does not exist, proceed to download
        case 2:
            print("Dataset archive not found. Downloading...")
            archive_path = download_dataset()
            sucess = extract_dataset(archive_path)
            if sucess:
                return
        case _:
            raise ValueError("Unexpected status code from check_downloaded_data")
    
    print(f"Extraction failed. You can try extracting the dataset manually: {dataset_path}")

if __name__ == "__main__":
    download_and_extract_data(DATASET_URL, FILENAME, DATASET_DIR, TARGET_CLASSES)