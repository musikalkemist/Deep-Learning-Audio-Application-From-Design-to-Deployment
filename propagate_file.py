"""
File Content Propagation Utility
------------------------------
This script ensures content consistency across a project structure by propagating
a source file's data to all other instances with the same filename.

Functionality:
1. Traverses the directory tree recursively from the current working directory.
2. Identifies matching filenames while pruning specific project and environment 
   folders (e.g., data, test, slides, .venv) to ensure safety and performance.
3. Replaces target files programmatically using shutil.copy2 to preserve original 
   file metadata.
4. Provides real-time CLI feedback using relative paths and an update counter.
5. Protects the source file from self-overwriting via absolute path normalization.
"""

import os
import shutil
import argparse

# Configuration
FOLDERS_TO_AVOID = ['data','test', 'slides',            # Porject folders
                    '.git', '.venv', '__pycache__']     # Safety environment

def sync_files():

    parser = argparse.ArgumentParser(description="Sync a file across a directory tree.")
    parser.add_argument("-f", "--file", required=True, help="Relative path to the source file")
    args = parser.parse_args()

    # Normalize source path
    src_path = os.path.abspath(args.file)
    if not os.path.exists(src_path):
        print(f"Error: Source file {args.file} not found.")
        return
    filename = os.path.basename(src_path)
    root_dir = os.getcwd() # Run from current working directory

    counter = 1
    for root, dirs, files in os.walk(root_dir):
        # Modify dirs in-place to skip avoided folders
        # This prevents os.walk from even entering these directories
        dirs[:] = [d for d in dirs if d not in FOLDERS_TO_AVOID]
        
        for f in files:
            if f == filename:
                target_path = os.path.abspath(os.path.join(root, f))
                print_path = os.path.relpath(os.path.join(root, f))
                
                # Skip if it's the source file itself
                if target_path == src_path:
                    continue
                
                try:
                    shutil.copy2(src_path, target_path)
                    print(f"{counter}- Updated: {print_path}")
                    counter += 1
                except Exception as e:
                    print(f"Error copying to {target_path}: {e}")
    print("Done!")

if __name__ == "__main__":
    sync_files()