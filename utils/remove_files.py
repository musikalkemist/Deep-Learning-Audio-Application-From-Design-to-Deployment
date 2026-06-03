"""
This script is designed to remove all files with the same name as a specified source file across a directory tree,
while avoiding certain folders. It uses Git to untrack the files instead of deleting them, which is safer in a
version-controlled environment. The script takes the source file as a command-line argument and traverses the
directory tree, removing any matching files it finds, except for those in specified folders to avoid.
"""

import os
import shutil
import argparse

# Configuration
FOLDERS_TO_AVOID = ['data','test', 'slides',            # Porject folders
                    '.git', '.venv', '__pycache__']     # Safety environment

DEBUG = False # Set to True to enable debug output

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
                
                import subprocess

                try:
                    # We pass the command as a list to avoid shell injection issues
                    result = subprocess.run(
                        ["git", "rm", "--cached", target_path],
                        check=True,         # Raises an error if the command fails
                        capture_output=True, # Captures stdout and stderr
                        text=True           # Returns output as a string instead of bytes
                    )
                    print(f"{counter}- Removed: {print_path}")
                    if DEBUG: print(f"DEBUG: {result.stdout}")
                    counter += 1

                except subprocess.CalledProcessError as e:
                    print(f"ERROR: Command failed with return code {e.returncode}")
                    print(f"Details: {e.stderr}")
                    
    print("Done!")

if __name__ == "__main__":
    sync_files()