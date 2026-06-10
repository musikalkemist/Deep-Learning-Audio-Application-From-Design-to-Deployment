#!/bin/bash

# Exit on any error
set -e

echo "--------------------------------------------------------"
echo "Deep Learning Audio App - Dataset Initialization"
echo "--------------------------------------------------------"

# 1. Download and extract the dataset subset
echo "Step 1: Downloading Speech Commands dataset (up, down, left, right)..."
python utils/dataset_downloader.py

# 2. Run the preprocessor
echo ""
echo "Step 2: Preprocessing audio data (extracting MFCCs)..."
# We run it from the root but target the script in Chapter 02
python "02 - Preparing the Dataset/prepare_dataset_faster.py"

# 3. Propagate the generated data.json
echo ""
echo "Step 3: Propagating data.json to all chapter folders..."
if [ -f "02 - Preparing the Dataset/data.json" ]; then
    python utils/propagate_files.py -f "02 - Preparing the Dataset/data.json"
else
    echo "Error: data.json was not generated in 02 - Preparing the Dataset/"
    exit 1
fi

echo ""
echo "--------------------------------------------------------"
echo "Initialization Complete!"
echo "Dataset is ready and data.json has been propagated."
echo "--------------------------------------------------------"
