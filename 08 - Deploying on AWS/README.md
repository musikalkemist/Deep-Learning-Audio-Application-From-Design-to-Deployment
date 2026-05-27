# Speech Recognition System - AWS Deployment

This directory contains the necessary components to deploy the keyword spotting system to an AWS EC2 instance using Docker and NGINX.

## Directory Structure

- `local/`:
    - `classifier/`: Scripts for local preprocessing and training.
    - `test/`: Sample audio files for testing.
    - `client.py`: Python client to test the API (requires updating the server IP).
- `server/`:
    - `flask/`: Flask application code, uWSGI config, and the trained model.
    - `nginx/`: NGINX configuration for reverse proxying.
    - `docker-compose.yml`: Orchestrates the Flask and NGINX services.
- `init.sh`: Shell script to automate the setup of the AWS EC2 instance.
- `speech_recognition.pem`: Private key for SSH access to the AWS instance (Note: Keep this secure!).

## Deployment Workflow

### 1. Local Preparation (Optional)

If you wish to train your own model:
1.  Navigate to `local/classifier/`.
2.  Follow the preprocessing and training steps (similar to Chapter 07).
3.  Copy the resulting `model.keras` to `server/flask/`.

### 2. Setting up the AWS EC2 Instance

1.  Launch an [EC2 instance](https://console.aws.amazon.com/ec2/v2/home?p=pm&c=ec2&z=1&trk=59c7f7f8-aea1-4544-afbc-ced8fcaad0b6&sc_channel=ps) (e.g., Ubuntu Server 18.04 LTS or newer).
2.  Ensure your Security Group allows inbound traffic on:
    - **SSH (Port 22):** For management.
    - **Custom TCP (Port 80):** For the NGINX proxy (which forwards to port 1234 internally).
    *Note: The `nginx.conf` listens on port 1234, but `docker-compose.yml` maps host port 80 to container port 1234.*

### 3. Transferring Files to AWS

Use `scp` to upload the `server/` directory and `init.sh` to your instance:
```bash
scp -i "speech_recognition.pem" -r server/ init.sh ubuntu@<AWS-INSTANCE-PUBLIC-IP>:~/
```

### 4. Initializing the Server

1.  SSH into your instance:
    ```bash
    ssh -i "speech_recognition.pem" ubuntu@<AWS-INSTANCE-PUBLIC-IP>
    ```
2.  Run the initialization script:
    ```bash
    chmod +x init.sh
    ./init.sh
    ```
    This script will:
    - Update the system.
    - Install Docker and Docker Compose.
    - Start the Docker services.
    - Build and run the containers defined in `server/docker-compose.yml`.

### 5. Testing the Remote API

1.  On your **local machine**, open `local/client.py`.
2.  Update the `URL` variable with your EC2 instance's public IP:
    ```python
    URL = "http://<AWS-INSTANCE-PUBLIC-IP>/predict"
    ```
3.  Run the client:
    ```bash
    python local/client.py
    ```

## Security Note

- **Private Key:** Never share your `.pem` file or commit it to a public repository.
- **Firewall:** Only open necessary ports in your AWS Security Group.
