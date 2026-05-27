# Speech Recognition System - Flask API Deployment

This directory contains the code to deploy the trained keyword spotting model as a simple Flask API for local testing.

## Project Structure

- `server.py`: The Flask application script that handles POST requests.
- `keyword_spotting_service.py`: Singleton service for making predictions.
- `client.py`: A simple Python client to send requests to the API.
- `train.py`: Script to train the model.
- `prepare_dataset_faster.py` / `prepare_dataset.py`: Scripts for dataset preprocessing.
- `test/`: Sample audio files for testing.
- `model.keras`: The trained Keras model.

## Requirements

- Python 3
- Libraries: `Flask`, `librosa`, `tensorflow`, `numpy`, `requests` (for the client)

## How to Run

### 1. Training the Model (Optional)

If you want to re-train the model locally:
1.  Ensure the `Speech_Commands_dataset` is present in the root of the project.
2.  Preprocess the dataset:
    ```bash
    python prepare_dataset_faster.py
    ```
3.  Train the model:
    ```bash
    python train.py
    ```

### 2. Running the Flask Server

To start the API:
```bash
python server.py
```
The server will start on `http://127.0.0.1:5000` (default Flask port).

### 3. Testing the API

1.  Open a **separate terminal** session.
2.  Run the client script:
    ```bash
    python client.py
    ```

The client will send a sample file to the server and print the predicted keyword.

## Troubleshooting

- **Dependencies:** Ensure all required packages from the project's main `requirements.txt` are installed.
- **Port:** Ensure port 5000 is available on your machine.
