# Speech Recognition System - Docker & NGINX Deployment

This directory contains the code to train a keyword spotting model and deploy it as a scalable Flask API using uWSGI, NGINX, and Docker.

## Project Structure

- `classifier/`: Scripts for dataset preprocessing and model training.
- `flask/`: Flask application code and uWSGI configuration.
- `nginx/`: NGINX configuration to act as a reverse proxy.
- `test/`: Sample audio files for testing.
- `docker-compose.yml`: Docker Compose configuration to orchestrate the services.
- `client.py`: A simple Python client to send requests to the deployed API.

## Requirements

- Docker
- Docker Compose
- Python 3 (for running the client or local training)

## How to Run

### 1. Training the Model (Optional)

If you want to re-train the model, follow these steps:

1.  Ensure the `Speech_Commands_dataset` is present in the root of the project (parent of this directory).
2.  Navigate to the `classifier` directory:
    ```bash
    cd classifier
    ```
3.  Preprocess the dataset:
    ```bash
    python prepare_dataset_faster.py
    ```
4.  Train the model:
    ```bash
    python train.py
    ```
5.  Copy the generated `model.keras` to the `../flask/` directory.

### 2. Deploying with Docker Compose

To build and start the API and NGINX proxy:

1.  From this directory (`07 - Deploying.../code/`), run:
    ```bash
    docker-compose up --build
    ```
2.  This will start two containers:
    - `flask`: The Flask API running with uWSGI on port 900 (internal).
    - `nginx`: The NGINX reverse proxy listening on host port 80.

### 3. Testing the API

Once the containers are running, you can test the API using the provided client:

1.  Install the `requests` library if you haven't:
    ```bash
    pip install requests
    ```
2.  Run the client:
    ```bash
    python client.py
    ```

The client sends `test/left.wav` to the API and prints the predicted keyword.

## Troubleshooting

- **Ports:** Ensure port 80 is not being used by another process on your host machine.
- **Audio Libraries:** The Docker image installs necessary audio libraries (`libsndfile1`, etc.) for `librosa` to function correctly inside the container.
