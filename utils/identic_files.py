"""
JSON Semantic Equality Verifier
------------------------------
This script compares two JSON files to determine if they are structurally and
content-wise identical, independent of formatting, indentation, or key ordering.

Functionality:
1. Opens and reads two target JSON files safely within a context manager.
2. Parses the raw text into native Python data structures using the json module.
3. Evaluates semantic equality (deep comparison of keys, values, and nesting) 
   rather than a superficial binary or string-based check.
4. Includes robust exception handling to catch and report file I/O operations 
   and syntax errors caused by malformed JSON.

Usage: Execute the script directly after configuring the target file paths.
Action: Evaluates the two specified paths and prints a boolean result to the console.
"""
import json

def are_json_files_identical(file1_path, file2_path):
    try:
        with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2:
            data1 = json.load(f1)
            data2 = json.load(f2)
            
        return data1 == data2
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error reading or parsing files: {e}")
        return False

# Usage
file_a = "04 - Making Predictions with the Speech Recognition System/data.json"
file_b = "06 - Deploying the Speech Recognition System with uWSGI/code/data.json"
print(f"Identical: {are_json_files_identical(file_a, file_b)}")