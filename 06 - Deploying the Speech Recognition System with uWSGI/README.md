# Speech Recognition System - uWSGI Deployment

This directory contains the code to deploy the keyword spotting system using uWSGI, which provides a more robust and production-ready environment for serving the Flask application.

## Project Structure

- `code/`: Contains the application source code.
    - `app.ini`: uWSGI configuration file.
    - `server.py`: The Flask application script.
    - `keyword_spotting_service.py`: Singleton service for inference.
    - `client.py`: Python client to test the server.
    - `model.keras`: The trained Keras model.
    - `test/`: Sample audio files for testing.

## Requirements

- Python 3
- uWSGI
- Libraries: `Flask`, `librosa`, `tensorflow`, `numpy`, `requests`

## How to Run

### 1. Navigating to the Source

Move into the `code` directory:
```bash
cd code
```

### 2. Starting the uWSGI Server

To start the application with uWSGI using the provided configuration:
```bash
uwsgi app.ini
```
The server will be configured according to the settings in `app.ini`.

### 3. Testing the Server

1.  Open a **separate terminal** session.
2.  Run the client script:
    ```bash
    python client.py
    ```

The client sends a request to the uWSGI server and displays the prediction.

## Troubleshooting

- **uWSGI Installation:** You may need to install the uWSGI development headers on your system (e.g., `sudo apt-get install uwsgi-plugin-python3` on Ubuntu).
- **Socket/Port:** Check `app.ini` to verify the socket or port the server is listening on.
